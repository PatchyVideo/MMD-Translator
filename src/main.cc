#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

#include <torch/script.h> // One-stop header.
#include <torch/cuda.h>
#include <ATen/cuda/CUDAMultiStreamGuard.h>

#include <ATen/ATen.h>

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <fstream>
#include <chrono>
#include <queue>
#include <optional>

#include "utf_utils.h"

//#define VERBOSE

std::vector<char32_t> g_alphabet;

void CreateAlphabet(std::string const& filename)
{
	std::ifstream file(filename, std::ios::binary | std::ios::ate);
	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<char> buffer(size);
	if (file.read(buffer.data(), size))
	{
		char32_t* tmp(new char32_t[size]);
		memset(tmp, 0, size);

		auto num_chars(uu::UtfUtils::SseConvert(reinterpret_cast<unsigned char const*>(&*buffer.cbegin()), reinterpret_cast<unsigned char const*>(&*buffer.cend()), tmp + 1));
		if (num_chars <= 0)
		{
			throw std::runtime_error("utf8 read failed");
		}
		g_alphabet = std::vector<char32_t>(tmp, tmp + num_chars + 1);
		std::cout << "Alphabet read, total " << static_cast<std::size_t>(num_chars) << " chars\n";

		delete[] tmp;
	}
	else
		throw std::runtime_error("failed to read alphabet file");
}

template <typename T>
T VAMax(T a, T b)
{
	if (a > b)
		return a;
	else
		return b;
}

template <typename T, typename ... Args>
T VAMax(T a, T b, Args ... args)
{
	return VAMax(VAMax(a, b), std::forward<Args>(args)...);
}

template <typename T>
T VAMin(T a, T b)
{
	if (a < b)
		return a;
	else
		return b;
}

template <typename T, typename ... Args>
T VAMin(T a, T b, Args ... args)
{
	return VAMin(VAMin(a, b), std::forward<Args>(args)...);
}

// from http://reedbeta.com/blog/python-like-enumerate-in-cpp17/
template <typename T,
	typename TIter = decltype(std::begin(std::declval<T>())),
	typename = decltype(std::end(std::declval<T>()))>
	constexpr auto enumerate(T&& iterable)
{
	struct iterator
	{
		size_t i;
		TIter iter;
		bool operator != (const iterator& other) const { return iter != other.iter; }
		void operator ++ () { ++i; ++iter; }
		auto operator * () const { return std::tie(i, *iter); }
	};
	struct iterable_wrapper
	{
		T iterable;
		auto begin() { return iterator{ 0, std::begin(iterable) }; }
		auto end() { return iterator{ 0, std::end(iterable) }; }
	};
	return iterable_wrapper{ std::forward<T>(iterable) };
}

struct BBox
{
	std::int32_t x, y, width, height;
	BBox() noexcept :x(0), y(0), width(0), height(0) {}
	BBox(std::int32_t x, std::int32_t y, std::int32_t width, std::int32_t height) noexcept :x(x), y(y), width(width), height(height) {}
	auto left() const noexcept { return x; }
	auto top() const noexcept { return y; }
	auto right() const noexcept { return x + width; }
	auto bottom() const noexcept { return y + height; }
	cv::Rect rect() const noexcept { return cv::Rect(x, y, width, height); }
	void crop(std::int32_t max_width, std::int32_t max_height) noexcept
	{
		x = std::max(x, 0);
		y = std::max(y, 0);
		width = std::min(width, max_width - x);
		height = std::min(height, max_height - y);
	}
	void merge(BBox const& other) noexcept
	{
		auto left1(left()), top1(top()), right1(right()), bottom1(bottom());
		auto left2(other.left()), top2(other.top()), right2(other.right()), bottom2(other.bottom());
		auto left3(std::min(left1, left2)), top3(std::min(top1, top2)), right3(std::max(right1, right2)), bottom3(std::max(bottom1, bottom2));
		x = left3;
		y = top3;
		width = right3 - left3;
		height = bottom3 - top3;
	}
	bool contains(BBox const& a) const noexcept
	{
		return x <= a.x && y <= a.y && right() >= a.right() && bottom() >= a.bottom();
	}
	void scale(float scale) noexcept
	{
		x = static_cast<std::int32_t>(scale * static_cast<float>(x));
		y = static_cast<std::int32_t>(scale * static_cast<float>(y));
		width = static_cast<std::int32_t>(scale * static_cast<float>(width));
		height = static_cast<std::int32_t>(scale * static_cast<float>(height));
	}
	float IoU(BBox const& other) const noexcept
	{
		auto xA(std::max(left(), other.left()));
		auto yA(std::max(top(), other.top()));
		auto xB(std::min(right(), other.right()));
		auto yB(std::min(bottom(), other.bottom()));

		auto interArea(std::max(0, xB - xA + 1) * std::max(0, yB - yA + 1));

		auto boxAArea(width * height);
		auto boxBArea(other.width * other.height);

		auto iou(float(interArea) / float(boxAArea + boxBArea - interArea));
		return std::max(0.0f, std::min(iou, 1.0f));
	}
};

std::unique_ptr<torch::jit::script::Module> CreateModel(std::string const& filename)
{
	try
	{
		auto ptr(std::make_unique<torch::jit::script::Module>(torch::jit::load(filename)));
		ptr->to(at::kCUDA);
		ptr->eval();
		return ptr;
	} catch (...)
	{
		return nullptr;
	}
}

std::tuple<float, std::int64_t, std::int64_t> ResizeKeepAspectRatio(cv::cuda::GpuMat const& img, cv::cuda::GpuMat& ret, cv::cuda::GpuMat& resized, std::uint32_t canvas_size)
{
	float old_width(static_cast<float>(img.size().width)), old_height(static_cast<float>(img.size().height));
	float ratio(static_cast<float>(canvas_size) / std::max(old_width, old_height));
	std::int64_t target_width(static_cast<std::int64_t>(std::round(ratio * old_width))), target_height(static_cast<std::int64_t>(std::round(ratio * old_height)));
	cv::cuda::resize(img, resized, cv::Size(target_width, target_height), 0.0, 0.0, cv::InterpolationFlags::INTER_AREA);
	std::int64_t target_width_32x(target_width % 32 == 0 ? target_width : (target_width + (32 - (target_width % 32))));
	std::int64_t target_height_32x(target_height % 32 == 0 ? target_height : (target_height + (32 - (target_height % 32))));
	cv::cuda::copyMakeBorder(resized, ret, 0, target_height_32x - target_height, 0, target_width_32x - target_width, cv::BORDER_REPLICATE);
	return { ratio, target_width_32x, target_height_32x };
}

std::vector<BBox> MergeBBoxes(std::vector<BBox> const& detects)
{
	std::vector<BBox> result;
	std::uint32_t num_detects(detects.size());
	result.reserve(num_detects);

	std::vector<std::uint32_t> parents(num_detects);
	std::iota(parents.begin(), parents.end(), 0);

	auto find([&parents](std::uint32_t a) {
		while (parents[a] != a)
		{
			parents[a] = parents[parents[a]];
			a = parents[a];
		}
		return a;
	});
	auto unite([&parents, &find](std::uint32_t a, std::uint32_t b) {
		parents[find(a)] = find(b);
	});

	auto can_merge([](BBox const& a, BBox const& b) {
		if (a.contains(b) || b.contains(a))
			return true;
		auto char_size(std::min(a.height, b.height));
		if (std::abs((a.y + a.height / 2) - (b.y + b.height / 2)) * 1.5 > char_size)
		{
			if (std::abs(a.height - b.height) > char_size)
				return false;
			if (std::abs(a.y - b.y) * 2 > char_size)
				return false;
		}
		if (a.x < b.x)
		{
			if (std::abs(a.right() - b.x) > char_size)
				return false;
			else
				return true;
		}
		else
		{
			if (std::abs(b.right() - a.x) > char_size)
				return false;
			else
				return true;
		}
		return false;
	});

	for (std::uint32_t i(0); i < num_detects; ++i)
		for (std::uint32_t j(i + 1); j < num_detects; ++j)
		{
			if (can_merge(detects[i], detects[j]))
				unite(i, j);
		}

	bool compression(true);
	while (compression)
	{
		compression = false;
		for (std::uint32_t i(0); i < num_detects; ++i)
		{
			auto root(i);
			while (parents[root] != parents[parents[root]])
			{
				parents[root] = parents[parents[root]];
				root = parents[root];
				compression = true;
			}
		}
	}

	std::unordered_map<std::uint32_t, BBox> m;
	for (std::uint32_t cur(0); cur < num_detects; ++cur)
	{
		auto root(parents[cur]);
		if (m.count(root))
			m[root].merge(detects[cur]);
		else
			m[root] = detects[cur];
	}
	for (auto const& item : m)
		result.emplace_back(item.second);

	return result;
}

cv::Ptr<cv::cuda::Filter> g_morph_filter_open;
cv::Ptr<cv::cuda::Filter> g_morph_filter_close;


void CreateImageMorphologyResource()
{
	cv::Mat open_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
	cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(23, 23));

	g_morph_filter_open = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8U, open_kernel);
	g_morph_filter_close = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8U, close_kernel);
}

void FilterTextScore(cv::cuda::GpuMat& img)
{
	//g_morph_filter_open->apply(img, img);
	//g_morph_filter_close->apply(img, img);
}

std::vector<BBox> CreateBBoxesFromScoreMaps(cv::cuda::GpuMat const& region_score, cv::cuda::GpuMat const& affinity_score, float text_threshold, float link_threshold, float low_text)
{
	// create text and link binary map
	static cv::cuda::GpuMat text_score_fp32, link_score_fp32;
	static cv::cuda::GpuMat text_score, link_score;
	cv::cuda::threshold(region_score, text_score_fp32, low_text, 1, cv::ThresholdTypes::THRESH_BINARY);
	cv::cuda::threshold(affinity_score, link_score_fp32, link_threshold, 1, cv::ThresholdTypes::THRESH_BINARY);
	text_score_fp32.convertTo(text_score, CV_8U, 255.0f);
	link_score_fp32.convertTo(link_score, CV_8U, 255.0f);

	// link text regions
	static cv::cuda::GpuMat text_link_comb;
	cv::cuda::bitwise_or(text_score, link_score, text_link_comb);

	FilterTextScore(text_link_comb);

	static cv::Mat text_link_comb_cpu;
	text_link_comb.download(text_link_comb_cpu);

#ifdef VERBOSE
	cv::imshow("text regions", text_link_comb_cpu);
#endif

	// find connected text regions
	static cv::Mat labels;
	cv::Mat stats;
	cv::Mat centroids;
	int num_labels(cv::connectedComponentsWithStats(text_link_comb_cpu, labels, stats, centroids, 4));

	cv::cuda::GpuMat labels_gpu;
	labels_gpu.upload(labels);

	std::vector<BBox> result;
	result.reserve(num_labels);
	for (int i(0); i < num_labels; ++i)
	{
		auto area(stats.at<int>(cv::Point(4, i)));
		if (area < 10)
			continue;

		static cv::cuda::GpuMat mask;
		cv::cuda::compare(labels_gpu, i, mask, cv::CMP_EQ);
		static cv::cuda::GpuMat mask_fp32;
		mask.convertTo(mask_fp32, CV_32F);
		static cv::cuda::GpuMat masked;
		cv::cuda::multiply(mask_fp32, text_score_fp32, masked);
		double min_v, max_v;
		cv::cuda::minMaxLoc(masked, &min_v, &max_v, nullptr, nullptr);
		if (static_cast<float>(max_v) < text_threshold * 255.0f)
			continue;

		auto x(stats.at<int>(cv::Point(0, i)));
		auto y(stats.at<int>(cv::Point(1, i)));
		auto w(stats.at<int>(cv::Point(2, i)));
		auto h(stats.at<int>(cv::Point(3, i)));
		//auto sum(cv::cuda::sum(masked));
		//auto ratio(sum.val[0] / static_cast<double>(w * h));
		//std::cout << "ratio=" << ratio << "\n";
		//if (ratio < 50)
		//	continue;

		if (x >= text_link_comb.cols || y >= text_link_comb.rows || x < 0 || y < 0 || w <= 0 || h <= 0)
			continue;

		auto niter(static_cast<std::int32_t>(std::sqrt(area * std::min(w, h) / (w * h)) * 2));
		auto extend(niter + static_cast<std::int32_t>(static_cast<float>(std::min(w, h)) * 0.07f));

		result.emplace_back(
			static_cast<std::int32_t>(x - extend) * 2,
			static_cast<std::int32_t>(y - extend) * 2,
			static_cast<std::int32_t>(w + extend * 2) * 2,
			static_cast<std::int32_t>(h + extend * 2) * 2
		);
	}

	return MergeBBoxes(MergeBBoxes(result));
}

/// <summary>
/// Perform OCR
/// </summary>
/// <param name="model">OCR model</param>
/// <param name="input">Input image tensor (on GPU), shape [N, 3, 32, W] of type fp32 range [-1, 1]</param>
/// <returns>vector of strings</returns>
std::vector<std::u32string> PerformOCR(std::unique_ptr<torch::jit::script::Module> const& model, at::Tensor const& input)
{
	std::int64_t num_imgs(input.size(0));
	std::vector<std::u32string> result(num_imgs);

	at::Tensor probs_torch(model->forward({ input }).toTensor()); // shape [N, W/4, alphabet_size]
	auto [topk_values, topk_indices] = probs_torch.topk(1, 2);
	topk_values = topk_values.cpu();
	topk_indices = topk_indices.cpu();

	// CTC top-1 decode
	// TODO: use top-5 min cost flow decode
	for (std::int64_t i(0); i < num_imgs; ++i)
	{
		std::u32string cur_string;
		std::int64_t width(topk_indices.size(1));
		std::vector<std::int64_t> t(width);
		std::int64_t char_index(topk_indices.index({ i, 0, 0 }).item<int64_t>());
		t[0] = char_index;
		if (char_index != 0)
			cur_string.append(1, g_alphabet[char_index]);
		for (std::int64_t j(1); j < width; ++j)
		{
			std::int64_t char_index(topk_indices.index({ i, j, 0 }).item<int64_t>());
			t[j] = char_index;
			if (char_index != 0 && char_index != t[j - 1])
				cur_string.append(1, g_alphabet[char_index]);
		}
		result[i] = cur_string;
	}

	return result;
}

at::Tensor ExtractTextRegion(at::Tensor const& frame, std::vector<BBox> const& bboxes)
{
	std::int64_t max_width(-1);
	std::int64_t num_boxes(bboxes.size());
	for (auto const& box : bboxes)
	{
		auto ratio(static_cast<float>(box.width) / static_cast<float>(box.height));
		auto new_width(static_cast<std::int64_t>(32.0f * ratio));
		max_width = std::max(max_width, new_width);
	}

	at::Tensor extracted_region(at::zeros({ num_boxes, 3, 32, max_width }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)));
	for (std::int64_t i(0); i < num_boxes; ++i)
	{
		auto const& box(bboxes[i]);
		auto ratio(static_cast<float>(box.width) / static_cast<float>(box.height));
		auto new_width(static_cast<std::int64_t>(32.0f * ratio));
		//std::cout << "new_width: " << new_width << "\n";
		//std::cout << "box: " << box.left() << "," << box.right() << "," << box.top() << "," << box.bottom() << "\n";
		extracted_region.index_put_(
			{ i, torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, new_width) },
			torch::upsample_bilinear2d(
				frame.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(box.top(), box.bottom()), torch::indexing::Slice(box.left(), box.right()) }),
				{ 32, new_width },
				false
			)
			.squeeze_(0)
		);
	}

	return extracted_region;
}

at::Tensor VStackImages(at::Tensor const& batch)
{
	auto width(batch.size(3));
	auto height(batch.size(2));
	auto N(batch.size(0));
	at::Tensor out(at::zeros({ 3, N * height, width }, torch::TensorOptions().dtype(batch.dtype()).device(batch.device())));
	for (int64_t i(0); i < N; ++i)
	{
		out.index_put_(
			{ torch::indexing::Slice(), torch::indexing::Slice(i * height, (i + 1) * height), torch::indexing::Slice() },
			batch.index({ i, torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice() })
		);
	}
	return out;
}

std::string u32tou8(std::u32string const& s)
{
	uu::UtfUtils::char8_t* tmp(new uu::UtfUtils::char8_t[s.size() * 4]);
	uu::UtfUtils::char8_t* pos = tmp;
	for (auto cdpt : s)
	{
		uu::UtfUtils::GetCodeUnits(cdpt, pos);
	}
	auto ret = std::string(tmp, pos);
	delete[] tmp;
	return ret;
}

struct FPSCounter
{
	std::size_t frame_counter;
	std::size_t frame_since_last_update;
	std::chrono::high_resolution_clock::time_point last_time;
	float fps;
	FPSCounter() :frame_counter(0), frame_since_last_update(0), last_time(std::chrono::high_resolution_clock::now()), fps(0.0f)
	{

	}
	bool Update()
	{
		++frame_counter;
		++frame_since_last_update;
		auto now(std::chrono::high_resolution_clock::now());
		auto elpased(std::chrono::duration_cast<std::chrono::microseconds>(now - last_time));
		if (elpased.count() > 1000000)
		{
			fps = 1000000.0f * static_cast<float>(frame_since_last_update) / static_cast<float>(elpased.count());

			last_time = std::chrono::high_resolution_clock::now();
			frame_since_last_update = 0;

			return true;
		}
		return false;
	}

	float GetFPS() const noexcept { return fps; }
};

std::optional<std::pair<at::Tensor, std::vector<BBox>>> DetectAndExtractTextRegions(
	std::unique_ptr<torch::jit::script::Module> const& craft,
	std::unique_ptr<torch::jit::script::Module> const& ocr,
	cv::cuda::GpuMat const& frame_fp32,
	std::tuple<float, std::int64_t, std::int64_t> const& resize_ret)
{
	auto [ratio, resized_width, resized_height] = resize_ret;
	ratio = 1.0f / ratio;

	// convert frame from cv::Mat (H, W, C) to at::Tensor (N, C, H, W)
	auto frame_torch = torch::from_blob(frame_fp32.data, { 1, resized_height, resized_width, 3 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).permute({ 0, 3, 1, 2 });

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(frame_torch);

	// Execute the model and turn its output into a tensor.
	at::Tensor frame_scores_torch = craft->forward(inputs).toTensor();

	auto region_score_torch(frame_scores_torch.index({ 0, 0, torch::indexing::Slice(), torch::indexing::Slice() }));
	region_score_torch = region_score_torch.clamp(0, 1);

	auto affinity_score_torch(frame_scores_torch.index({ 0, 1, torch::indexing::Slice(), torch::indexing::Slice() }));
	affinity_score_torch = affinity_score_torch.clamp(0, 1);

	// region_score is of shape [H, W, 1] of fp32 range [0, 1]
	cv::cuda::GpuMat region_score(region_score_torch.size(0), region_score_torch.size(1), CV_32F, region_score_torch.data_ptr());

	// affinity_score is of shape [H, W, 1] of fp32 range [0, 1]
	cv::cuda::GpuMat affinity_score(affinity_score_torch.size(0), affinity_score_torch.size(1), CV_32F, affinity_score_torch.data_ptr());

#ifdef VERBOSE
	cv::Mat rs, as;
	region_score.download(rs);
	rs.convertTo(rs, CV_8UC1, 255);
	affinity_score.download(as);
	as.convertTo(as, CV_8UC1, 255);

	cv::Mat rs_colorful, as_colorful;
	cv::applyColorMap(rs, rs_colorful, cv::COLORMAP_JET);
	cv::applyColorMap(as, as_colorful, cv::COLORMAP_JET);

	cv::imshow("text score", rs_colorful);
	cv::imshow("link score", as_colorful);
#endif

	auto bboxes(CreateBBoxesFromScoreMaps(region_score, affinity_score, 0.8f, 0.1f, 0.5f));
	for (auto& box : bboxes)
		box.crop(resized_width, resized_height);

	if (bboxes.size() > 0)
	{
		return { { ExtractTextRegion(frame_torch, bboxes), bboxes } };
	}
	else
		return {};
}

//std::vector<std::u32string> DetectAndOCR(
//	std::uint32_t raw_width,
//	std::uint32_t raw_height,
//	std::unique_ptr<torch::jit::script::Module> const& craft,
//	std::unique_ptr<torch::jit::script::Module> const& ocr,
//	cv::cuda::GpuMat const& frame_fp32,
//	std::tuple<float, std::int64_t, std::int64_t> const& resize_ret)
//{
//	auto [ratio, resized_width, resized_height] = resize_ret;
//	ratio = 1.0f / ratio;
//	// convert frame from cv::Mat (H, W, C) to at::Tensor (N, C, H, W)
//	auto frame_torch = torch::from_blob(frame_fp32.data, { 1, resized_height, resized_width, 3 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).permute({ 0, 3, 1, 2 });
//
//	std::vector<torch::jit::IValue> inputs;
//	inputs.push_back(frame_torch);
//
//	// Execute the model and turn its output into a tensor.
//	at::Tensor frame_scores_torch = craft->forward(inputs).toTensor();
//
//	auto region_score_torch(frame_scores_torch.index({ 0, 0, torch::indexing::Slice(), torch::indexing::Slice() }));
//	region_score_torch = region_score_torch.clamp(0, 1);
//
//	auto affinity_score_torch(frame_scores_torch.index({ 0, 1, torch::indexing::Slice(), torch::indexing::Slice() }));
//	affinity_score_torch = affinity_score_torch.clamp(0, 1);
//
//	//std::cout << "DBG: " << __LINE__ << "\n";
//	// region_score is of shape [H, W, 1] of fp32 range [0, 1]
//	cv::cuda::GpuMat region_score(region_score_torch.size(0), region_score_torch.size(1), CV_32F, region_score_torch.data_ptr());
//	//std::memcpy((void*)region_score.data, region_score_torch.data_ptr(), sizeof(float) * region_score_torch.numel());
//
//	// affinity_score is of shape [H, W, 1] of fp32 range [0, 1]
//	cv::cuda::GpuMat affinity_score(affinity_score_torch.size(0), affinity_score_torch.size(1), CV_32F, affinity_score_torch.data_ptr());
//	//std::memcpy((void*)affinity_score.data, affinity_score_torch.data_ptr(), sizeof(float) * affinity_score_torch.numel());
//
//	//cv::Mat rs, as;
//	//region_score.convertTo(rs, CV_8UC1, 255);
//	//affinity_score.convertTo(as, CV_8UC1, 255);
//
//	//cv::Mat rs_colorful, as_colorful;
//	//cv::applyColorMap(rs, rs_colorful, cv::COLORMAP_JET);
//	//cv::applyColorMap(as, as_colorful, cv::COLORMAP_JET);
//
//	//cv::imshow("rs_colorful", rs_colorful);
//	//cv::imshow("as_colorful", as_colorful);
//
//	auto bboxes(CreateBBoxesFromScoreMapsGPU(region_score, affinity_score, 0.8f, 0.1f, 0.3f, ratio, ratio));
//	for (auto& box : bboxes)
//		box.crop(raw_width, raw_height);
//
//	//cv::Mat rendered_frame;
//	//frame_fp32.convertTo(rendered_frame, CV_8UC3, 127.5, 127.5);
//	//cv::cvtColor(rendered_frame, rendered_frame, cv::COLOR_RGB2BGR);
//	//for (auto const& box : bboxes)
//	//{
//	//	cv:rectangle(rendered_frame, box.rect(), cv::Scalar(255, 0, 0), 3);
//	//}
//
//	//cv::imshow("video", rendered_frame);
//
//	if (bboxes.size() > 0)
//	{
//		// scale to GPU frame size
//		auto bboxes_std(bboxes);
//		for (auto& box : bboxes_std)
//			box.scale(1.0f / ratio);
//
//		auto text_regions(ExtractTextRegion(frame_torch, bboxes_std));
//
//		//std::cout << text_regions.sizes() << "\n";
//		//auto text_image_torch(VStackImages(text_regions).permute({ 1, 2, 0 }).add_(1.0f).mul_(0.5f * 255.0f).to(torch::kU8).cpu());
//		//std::cout << text_image_torch.sizes() << "\n";
//
//		//cv::Mat text_image(text_image_torch.size(0), text_image_torch.size(1), CV_8UC3);
//		//std::memcpy((void*)text_image.data, text_image_torch.data_ptr(), sizeof(uint8_t) * text_image_torch.numel());
//
//		//cv::imshow("text_regions", text_image);
//
//		return {};// PerformOCR(ocr, text_regions);
//	}
//	else
//		return {};
//}

//cv::Scalar CalculateMSSIM(const cv::Mat& i1, const cv::Mat& i2)
//{
//	using namespace cv;
//	const double C1 = 6.5025, C2 = 58.5225;
//	/***************************** INITS **********************************/
//	int d = CV_32F;
//
//	Mat I1, I2;
//	i1.convertTo(I1, d);           // cannot calculate on one byte large values
//	i2.convertTo(I2, d);
//
//	Mat I2_2 = I2.mul(I2);        // I2^2
//	Mat I1_2 = I1.mul(I1);        // I1^2
//	Mat I1_I2 = I1.mul(I2);        // I1 * I2
//
//	/*************************** END INITS **********************************/
//
//	Mat mu1, mu2;   // PRELIMINARY COMPUTING
//	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
//	GaussianBlur(I2, mu2, Size(11, 11), 1.5);
//
//	Mat mu1_2 = mu1.mul(mu1);
//	Mat mu2_2 = mu2.mul(mu2);
//	Mat mu1_mu2 = mu1.mul(mu2);
//
//	Mat sigma1_2, sigma2_2, sigma12;
//
//	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
//	sigma1_2 -= mu1_2;
//
//	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
//	sigma2_2 -= mu2_2;
//
//	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
//	sigma12 -= mu1_mu2;
//
//	///////////////////////////////// FORMULA ////////////////////////////////
//	Mat t1, t2, t3;
//
//	t1 = 2 * mu1_mu2 + C1;
//	t2 = 2 * sigma12 + C2;
//	t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
//
//	t1 = mu1_2 + mu2_2 + C1;
//	t2 = sigma1_2 + sigma2_2 + C2;
//	t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
//
//	Mat ssim_map;
//	divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;
//
//	Scalar mssim = mean(ssim_map); // mssim = average of ssim map
//	return mssim;
//}

/// <summary>
/// Find a min cost assignment using successive shortest path algorithm
/// From https://cp-algorithms.com/graph/min_cost_flow.html
/// </summary>
/// <param name="cost">row major cost matrix</param>
/// <param name="rows">rows</param>
/// <param name="cols">cols</param>
/// <returns>row assignment</returns>
std::vector<int> FindMinCostAssignment(int const* costmat, int rows, int cols)
{
	struct Edge
	{
		int from, to, capacity, cost;
		Edge(int from, int to, int capacity, int cost) :from(from), to(to), capacity(capacity), cost(cost)
		{

		}
	};
	// step 1: build graph
	std::vector<Edge> edges;
	edges.reserve(rows * cols + rows + cols);
	for (int i(0); i < rows; ++i)
		edges.emplace_back(0, i + 2, 1, 0);
	for (int i(0); i < cols; ++i)
		edges.emplace_back(i + 2 + rows, 1, 1, 0);
	for (int i(0); i < rows; ++i)
		for (int j(0); j < cols; ++j)
			edges.emplace_back(i + 2, j + 2 + rows, 1, costmat[i * cols + j]);
	// step 2: run SSP
	std::vector<std::vector<int>> adj, cost, capacity;
	auto shortest_paths([&adj, &cost, &capacity](int n, int v0, std::vector<int>& d, std::vector<int>& p) {
		d.assign(n, std::numeric_limits<int>::max());
		d[v0] = 0;
		std::vector<char> inq(n, 0);
		std::queue<int> q;
		q.push(v0);
		p.assign(n, -1);

		while (!q.empty())
		{
			int u = q.front();
			q.pop();
			inq[u] = 0;
			for (int v : adj[u])
			{
				if (capacity[u][v] > 0 && d[v] > d[u] + cost[u][v])
				{
					d[v] = d[u] + cost[u][v];
					p[v] = u;
					if (!inq[v])
					{
						inq[v] = 1;
						q.push(v);
					}
				}
			}
		}
	});
	auto min_cost_flow([&adj, &cost, &capacity, &shortest_paths](int N, std::vector<Edge> edges, int K, int s, int t) {
		adj.assign(N, std::vector<int>());
		cost.assign(N, std::vector<int>(N, 0));
		capacity.assign(N, std::vector<int>(N, 0));
		for (Edge e : edges)
		{
			adj[e.from].push_back(e.to);
			adj[e.to].push_back(e.from);
			cost[e.from][e.to] = e.cost;
			cost[e.to][e.from] = -e.cost;
			capacity[e.from][e.to] = e.capacity;
		}

		int flow = 0;
		int cost = 0;
		std::vector<int> d, p;
		while (flow < K)
		{
			shortest_paths(N, s, d, p);
			if (d[t] == std::numeric_limits<int>::max())
				break;

			// find max flow on that path
			int f = K - flow;
			int cur = t;
			while (cur != s)
			{
				f = std::min(f, capacity[p[cur]][cur]);
				cur = p[cur];
			}

			// apply flow
			flow += f;
			cost += f * d[t];
			cur = t;
			while (cur != s)
			{
				capacity[p[cur]][cur] -= f;
				capacity[cur][p[cur]] += f;
				cur = p[cur];
			}
		}

		if (flow < K)
			return -1;
		else
			return cost;
	});
	auto flow(std::min(rows, cols));
	auto flowcost(min_cost_flow(rows + cols + 2, edges, flow, 0, 1));
	if (flowcost < 0)
		throw std::runtime_error("network flow not satisfied");
	// step 3: find edge with non-zero flow(zero capacity)
	std::vector<int> row_assignment(rows, -1);
	for (int i(0); i < rows; ++i)
	{
		int row(i + 2);
		for (int j(0); j < cols; ++j)
		{
			int col(j + 2 + rows);
			if (capacity[row][col] == 0)
			{
				row_assignment[row - 2] = col - 2 - rows;
				break;
			}
		}
	}
	return row_assignment;
}

bool DetectTextRegionChange(at::Tensor const& a, std::vector<BBox> const& abox, at::Tensor const& b, std::vector<BBox> const& bbox, float iou_threshold, float l1_threshold, float cost_scale = 1000.0f)
{
	auto M(abox.size()), N(bbox.size());
	if (M != N)
	{
#ifdef VERBOSE
		std::cout << "changd due to M=" << M << ",N=" << N << "\n";
#endif
		return true;
	}

	auto costmat(std::make_unique<int[]>(M * N));

	for (std::size_t i(0); i < M; ++i)
		for (std::size_t j(0); j < N; ++j)
		{
			costmat[i * N + j] = static_cast<int>((1.0f - abox[i].IoU(bbox[j])) * cost_scale);
		}
	auto row_assignments(FindMinCostAssignment(costmat.get(), M, N));
	for (std::size_t i(0); i < M; ++i)
	{
		auto j(row_assignments[i]);
		if (costmat[i * N + j] > cost_scale * (1.0f - iou_threshold))
		{
#ifdef VERBOSE
			std::cout << "changd due to iou=" << 1.0 - static_cast<float>(costmat[i * N + j]) / cost_scale << "\n";
#endif
			return true;
		}

		// for the sake of simplicity and speed, only red channel is compared, everything is resized to 32x32
		// TODO: use surf key point match
		auto region_a(a.index({ static_cast<int64_t>(i), 0, torch::indexing::Slice(), torch::indexing::Slice() }).unsqueeze_(0).unsqueeze_(0));
		auto region_b(torch::upsample_bilinear2d(b.index({ static_cast<int64_t>(j), 0, torch::indexing::Slice(), torch::indexing::Slice() }).unsqueeze_(0).unsqueeze_(0), { region_a.size(2), region_a.size(3) }, false));
		auto l1_dist(torch::l1_loss(region_a, region_b).item<float>());

		if (l1_dist > l1_threshold)
		{
#ifdef VERBOSE
			std::cout << "changd due to l1_dist=" << l1_dist << "\n";
#endif
			return true;
		}
	}
	return false;
}

//struct BufferMSSIM                                     // Optimized GPU versions
//{   // Data allocations are very expensive on GPU. Use a buffer to solve: allocate once reuse later.
//	cv::cuda::GpuMat gs, t1, t2;
//
//	cv::cuda::GpuMat I1_2, I2_2, I1_I2;
//	std::vector<cv::cuda::GpuMat> vI1, vI2;
//
//	cv::cuda::GpuMat mu1, mu2;
//	cv::cuda::GpuMat mu1_2, mu2_2, mu1_mu2;
//
//	cv::cuda::GpuMat sigma1_2, sigma2_2, sigma12;
//	cv::cuda::GpuMat t3;
//
//	cv::cuda::GpuMat ssim_map;
//
//	cv::cuda::GpuMat buf;
//
//	cv::Ptr<cv::cuda::Filter> filter;
//
//	BufferMSSIM() :filter(cv::cuda::createGaussianFilter(CV_32F, CV_32F, cv::Size(11, 11), 1.5, 0, cv::BORDER_DEFAULT, cv::BORDER_DEFAULT))
//	{
//
//	}
//};
//
//auto getMSSIM_GPU_optimized(cv::cuda::GpuMat const& i1, cv::cuda::GpuMat const& i2, BufferMSSIM& b)
//{
//	const float C1 = 6.5025f, C2 = 58.5225f;
//	/***************************** INITS **********************************/
//
//	b.t1 = i1;
//	b.t2 = i2;
//
//	cv::cuda::split(b.t1, b.vI1);
//	cv::cuda::split(b.t2, b.vI2);
//	double mssim(0.0f);
//
//	for (int i = 0; i < b.t1.channels(); ++i)
//	{
//		cv::cuda::multiply(b.vI2[i], b.vI2[i], b.I2_2);        // I2^2
//		cv::cuda::multiply(b.vI1[i], b.vI1[i], b.I1_2);        // I1^2
//		cv::cuda::multiply(b.vI1[i], b.vI2[i], b.I1_I2);       // I1 * I2
//
//		b.filter->apply(b.vI1[i], b.mu1);
//		b.filter->apply(b.vI2[i], b.mu2);
//
//		cv::cuda::multiply(b.mu1, b.mu1, b.mu1_2);
//		cv::cuda::multiply(b.mu2, b.mu2, b.mu2_2);
//		cv::cuda::multiply(b.mu1, b.mu2, b.mu1_mu2);
//
//		b.filter->apply(b.I1_2, b.sigma1_2);
//		cv::cuda::subtract(b.sigma1_2, b.mu1_2, b.sigma1_2, cv::cuda::GpuMat(), -1);
//		//b.sigma1_2 -= b.mu1_2;  - This would result in an extra data transfer operation
//
//		b.filter->apply(b.I2_2, b.sigma2_2);
//		cv::cuda::subtract(b.sigma2_2, b.mu2_2, b.sigma2_2, cv::cuda::GpuMat(), -1);
//		//b.sigma2_2 -= b.mu2_2;
//
//		b.filter->apply(b.I1_I2, b.sigma12);
//		cv::cuda::subtract(b.sigma12, b.mu1_mu2, b.sigma12, cv::cuda::GpuMat(), -1);
//		//b.sigma12 -= b.mu1_mu2;
//
//		//here too it would be an extra data transfer due to call of operator*(Scalar, Mat)
//		cv::cuda::multiply(b.mu1_mu2, 2, b.t1, 1, -1); //b.t1 = 2 * b.mu1_mu2 + C1;
//		cv::cuda::add(b.t1, C1, b.t1, cv::cuda::GpuMat(), -1);
//		cv::cuda::multiply(b.sigma12, 2, b.t2, 1, -1); //b.t2 = 2 * b.sigma12 + C2;
//		cv::cuda::add(b.t2, C2, b.t2, cv::cuda::GpuMat(), -12);
//
//		cv::cuda::multiply(b.t1, b.t2, b.t3, 1, -1);     // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
//
//		cv::cuda::add(b.mu1_2, b.mu2_2, b.t1, cv::cuda::GpuMat(), -1);
//		cv::cuda::add(b.t1, C1, b.t1, cv::cuda::GpuMat(), -1);
//
//		cv::cuda::add(b.sigma1_2, b.sigma2_2, b.t2, cv::cuda::GpuMat(), -1);
//		cv::cuda::add(b.t2, C2, b.t2, cv::cuda::GpuMat(), -1);
//
//		cv::cuda::multiply(b.t1, b.t2, b.t1, 1, -1);     // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
//		cv::cuda::divide(b.t3, b.t1, b.ssim_map, 1, -1);      // ssim_map =  t3./t1;
//
//		auto s = cv::cuda::sum(b.ssim_map, b.buf);
//		mssim += s.val[0] / (b.ssim_map.rows * b.ssim_map.cols);
//
//	}
//	return mssim / static_cast<double>(b.t1.channels());
//}

std::vector<char32_t> g_spaces;
void CreateSpaceCharacterU32()
{
	std::string const spaces("':.\n\r[] \t\v\f{}-_■=+`~!@#$%^&*();'\", <> / ? \\ | －＞＜。，《》【】　？！￥…（）、：；·「」『』〔〕［］｛｝｟｠〉〈〖〗〘〙〚〛゠＝‥※＊〽〓〇＂“”‘’＃＄％＆＇＋．／＠＼＾＿｀｜～｡｢｣､･ｰﾟ￠￡￢￣￤￨￩￪￫￬￭￮・◆◊→←↑↓↔—'");
	char32_t* tmp(new char32_t[spaces.size()]);
	memset(tmp, 0, spaces.size());

	auto num_chars(uu::UtfUtils::SseConvert(reinterpret_cast<unsigned char const*>(&*spaces.cbegin()), reinterpret_cast<unsigned char const*>(&*spaces.cend()), tmp));
	if (num_chars <= 0)
	{
		throw std::runtime_error("utf8 read failed");
	}
	g_spaces = std::vector<char32_t>(tmp, tmp + num_chars);
}

std::vector<std::u32string> FilterSpaceCharacters(std::vector<std::u32string> const& raw)
{
	std::vector<std::u32string> result;
	result.reserve(raw.size());
	for (auto const& s : raw)
	{
		std::vector<char32_t> allowed_chars;
		allowed_chars.reserve(s.size());
		std::copy_if(s.cbegin(), s.cend(), std::back_inserter(allowed_chars), [](char32_t cdpt) {
			return std::find(g_spaces.cbegin(), g_spaces.cend(), cdpt) == g_spaces.cend();
		});
		if (allowed_chars.size())
			result.emplace_back(allowed_chars.cbegin(), allowed_chars.cend());
	}
	return result;
}

std::vector<std::u32string> FilterSpaceOnlyString(std::vector<std::u32string> const& raw)
{
	std::vector<std::u32string> result;
	result.reserve(raw.size());
	for (auto const& s : raw)
	{
		std::vector<char32_t> allowed_chars;
		allowed_chars.reserve(s.size());
		std::copy_if(s.cbegin(), s.cend(), std::back_inserter(allowed_chars), [](char32_t cdpt) {
			return std::find(g_spaces.cbegin(), g_spaces.cend(), cdpt) == g_spaces.cend();
		});
		if (allowed_chars.size())
			result.emplace_back(s);
	}
	return result;
}

// from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#C++
float EditDistanceRatio(const std::u32string& s1, const std::u32string& s2)
{
	std::size_t const len1 = s1.size(), len2 = s2.size();
	std::vector<std::vector<unsigned int>> d(len1 + 1, std::vector<unsigned int>(len2 + 1));

	d[0][0] = 0;
	for (unsigned int i = 1; i <= len1; ++i) d[i][0] = i;
	for (unsigned int i = 1; i <= len2; ++i) d[0][i] = i;

	for (unsigned int i = 1; i <= len1; ++i)
		for (unsigned int j = 1; j <= len2; ++j)
			d[i][j] = std::min({ d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? 0 : 1) });
	return (static_cast<float>(len1 + len2) - static_cast<float>(d[len1][len2])) / static_cast<float>(len1 + len2);
}

struct SubtitleSegment
{
	std::vector<std::u32string> texts, filtered_texts;
	std::size_t start, end;

	SubtitleSegment(std::vector<std::u32string> texts, std::size_t start, std::size_t end) :texts(texts), start(start), end(end)
	{
		filtered_texts = FilterSpaceCharacters(texts);
	}

	SubtitleSegment():start(0),end(0)
	{

	}

	std::size_t duration() const noexcept
	{
		return end - start;
	}
};

std::vector<int> GetSegmentsAssignment(SubtitleSegment const& a, SubtitleSegment const& b, float cost_scale = 1000.0f)
{
	auto M(a.filtered_texts.size()), N(b.filtered_texts.size());
	assert(M == N);
	auto costmat(std::make_unique<int[]>(M * N));

	for (std::size_t i(0); i < M; ++i)
		for (std::size_t j(0); j < N; ++j)
		{
			costmat[i * N + j] = static_cast<int>((1.0f - EditDistanceRatio(a.filtered_texts[i], b.filtered_texts[j])) * cost_scale);
		}
	return FindMinCostAssignment(costmat.get(), M, N);
}

SubtitleSegment MergeSubtitleGroup(std::vector<SubtitleSegment> segs)
{
	std::sort(segs.begin(), segs.end(), [](auto const& a, auto const& b) {return a.start < b.start; });

	std::vector<std::vector<int>> assignments(segs.size() - 1, std::vector<int>());
	for (std::size_t i(1); i < segs.size(); ++i)
		assignments[i - 1] = GetSegmentsAssignment(segs[i - 1], segs[i]);

	std::vector<std::u32string> result_texts;
	for (auto const& [txt_idx, txt] : enumerate(segs.front().texts))
	{
		std::unordered_map<std::u32string, std::size_t> text_duration_map;
		text_duration_map[txt] = segs.front().duration();
		auto cur_txt_idx(txt_idx);
		for (std::size_t i(1); i < segs.size(); ++i)
		{
			cur_txt_idx = assignments[i - 1][cur_txt_idx];
			auto const& cur_text(segs[i].texts[cur_txt_idx]);
			text_duration_map[cur_text] += segs[i].duration();
		}
		std::u32string longest_text;
		std::size_t longest_duration(0);
		for (auto const& [cur_text, cur_duration] : text_duration_map)
			if (cur_duration > longest_duration)
			{
				longest_duration = cur_duration;
				longest_text = cur_text;
			}
		result_texts.emplace_back(longest_text);
	}

	return SubtitleSegment(result_texts, segs.front().start, segs.back().end);
}

std::vector<SubtitleSegment> MergeSubtitleSegments(std::vector<SubtitleSegment> const& segs, float levenshtein_threshold)
{
	std::vector<std::uint32_t> parents(segs.size());
	std::iota(parents.begin(), parents.end(), 0);

	auto find([&parents](std::uint32_t a) {
		while (parents[a] != a)
		{
			parents[a] = parents[parents[a]];
			a = parents[a];
		}
		return a;
	});
	auto unite([&parents, &find](std::uint32_t a, std::uint32_t b) {
		parents[find(a)] = find(b);
	});

	auto can_merge([&segs, levenshtein_threshold](std::size_t i, std::size_t j) {
		if (segs[i].filtered_texts.size() != segs[j].filtered_texts.size())
			return false;
		auto const& row_assignments(GetSegmentsAssignment(segs[i], segs[j]));
		for (std::size_t a(0); a < row_assignments.size(); ++a)
		{
			auto b(row_assignments[a]);
			if (EditDistanceRatio(segs[i].filtered_texts[a], segs[j].filtered_texts[b]) < levenshtein_threshold)
				return false;
		}
		return true;
	});

	for (std::size_t i(1); i < segs.size(); ++i)
	{
		if (can_merge(i - 1, i))
			unite(i - 1, i);
	}

	bool compression(true);
	while (compression)
	{
		compression = false;
		for (std::uint32_t i(0); i < segs.size(); ++i)
		{
			auto root(i);
			while (parents[root] != parents[parents[root]])
			{
				parents[root] = parents[parents[root]];
				root = parents[root];
				compression = true;
			}
		}
	}

	std::unordered_map<std::uint32_t, std::vector<SubtitleSegment>> segment_groups;
	for (std::uint32_t cur(0); cur < segs.size(); ++cur)
	{
		auto root(find(cur));
		segment_groups[root].emplace_back(segs[cur]);
	}
	std::vector<SubtitleSegment> result;
	for (auto const& [root, item] : segment_groups)
		result.emplace_back(MergeSubtitleGroup(item));
	std::sort(result.begin(), result.end(), [](auto const& a, auto const& b) {return a.start < b.start; });
	return result;
}

struct SRTGenerator
{
	float levenshtein_threshold;
	float cost_scale;
	std::size_t subtitle_duration_threshold;

	std::size_t last_millisecond;
	std::vector<std::u32string> last_text;

	std::ofstream ofs;

	std::vector<SubtitleSegment> segments;

	SRTGenerator(std::string const& filename, float levenshtein_threshold = 0.8f, float cost_scale = 1000.0f) :levenshtein_threshold(levenshtein_threshold), cost_scale(cost_scale), subtitle_duration_threshold(100), last_millisecond(0)
	{
		ofs = std::ofstream(filename);
		if (ofs.fail())
			throw std::runtime_error("failed to create srt file");
	}

	~SRTGenerator()
	{
		ofs.flush();
		ofs.close();
	}

	void AppendSubtitle(std::vector<std::u32string> const& texts, std::size_t frame_counter, std::size_t millisecond)
	{
		segments.emplace_back(last_text, last_millisecond, millisecond);
		last_millisecond = millisecond;
		last_text = texts;
	}

	void Generate()
	{
		auto merged_stage1(MergeSubtitleSegments(segments, levenshtein_threshold));

		decltype(segments) filtered_stage2;
		std::copy_if(merged_stage1.cbegin(), merged_stage1.cend(), std::back_inserter(filtered_stage2), [this](SubtitleSegment const& s) {return s.duration() > this->subtitle_duration_threshold; });

		auto result(MergeSubtitleSegments(filtered_stage2, levenshtein_threshold));

		std::size_t text_counter(1);
		for (auto const& seg : result)
		{
			if (seg.texts.size() == 0)
				continue;
			ofs << text_counter << "\n";
			ofs << mil2str(seg.start) << " --> " << mil2str(seg.end) << "\n";
			for (auto const& s : seg.texts)
			{
				auto su8(u32tou8(s));
				ofs << su8 << "\n";
			}
			ofs << "\n";
			++text_counter;
		}
	}

	std::string mil2str(std::size_t ms)
	{
		int sec(ms / 1000);
		int remain_ms(ms - sec * 1000);
		int minutes = sec / 60;
		int remain_sec = sec - minutes * 60;
		int hrs = minutes / 60;
		int remain_minutes = minutes - hrs * 60;
		char tmp[64];
		std::sprintf(tmp, "%02d:%02d:%02d,%03d", hrs, remain_minutes, remain_sec, remain_ms);
		return std::string(tmp);
	}
};

//struct SRTGenerator2
//{
//	float levenshtein_threshold;
//	float cost_scale;
//	std::size_t subtitle_last_threshold;
//
//	std::vector<std::u32string> last_texts;
//	std::vector<std::u32string> last_filtered_texts;
//
//	std::size_t last_text_millisecond;
//	std::size_t text_counter;
//	std::vector<std::u32string> saved_texts;
//
//	std::ofstream ofs;
//
//	SRTGenerator2(std::string const& filename, float levenshtein_threshold = 0.8f, float cost_scale = 1000.0f) :levenshtein_threshold(levenshtein_threshold), cost_scale(cost_scale), subtitle_last_threshold(100), last_text_millisecond(0), text_counter(0)
//	{
//		ofs = std::ofstream(filename);
//		if (ofs.fail())
//			throw std::runtime_error("failed to create srt file");
//	}
//
//	~SRTGenerator2()
//	{
//		ofs.flush();
//		ofs.close();
//	}
//
//	std::string mil2str(std::size_t ms)
//	{
//		int sec(ms / 1000);
//		int remain_ms(ms - sec * 1000);
//		int minutes = sec / 60;
//		int remain_sec = sec - minutes * 60;
//		int hrs = minutes / 60;
//		int remain_minutes = minutes - hrs * 60;
//		char tmp[64];
//		std::sprintf(tmp, "%02d:%02d:%02d,%03d", hrs, remain_minutes, remain_sec, remain_ms);
//		return std::string(tmp);
//	}
//
//	void EmitSRT(std::vector<std::u32string> const& texts, std::size_t millisecond)
//	{
//		if (text_counter > 0 && saved_texts.size() > 0)
//		{
//			if (millisecond - last_text_millisecond > 100)
//			{
//				// saved_texts from last_text_millisecond to millisecond
//				ofs << text_counter << "\n";
//				ofs << mil2str(last_text_millisecond) << " --> " << mil2str(millisecond) << "\n";
//				for (auto const& s : saved_texts)
//				{
//					auto su8(u32tou8(s));
//					ofs << su8 << "\n";
//				}
//				ofs << std::endl; // flush
//			}
//			else
//			{
//				// discard subtitle lasts less than 100ms
//				last_text_millisecond = millisecond;
//				saved_texts = texts;
//				return;
//			}
//		}
//		last_text_millisecond = millisecond;
//		++text_counter;
//		saved_texts = texts;
//	}
//
//	void AppendSubtitle(std::vector<std::u32string> const& texts, std::size_t frame_counter, std::size_t millisecond)
//	{
//		bool text_changed(false);
//		std::vector<std::u32string> filtered_texts(FilterSpaceCharacters(texts));
//
//		if (filtered_texts.size() != last_filtered_texts.size())
//		{
//
//			if (millisecond - last_text_millisecond > 100)
//			{
//				EmitSRT(texts, millisecond);
//			}
//			last_filtered_texts = filtered_texts;
//			last_texts = texts;
//			return;
//		}
//
//		// N == M
//		auto M(last_filtered_texts.size()), N(filtered_texts.size());
//		auto costmat(std::make_unique<int[]>(M * N));
//
//		for (std::size_t i(0); i < M; ++i)
//			for (std::size_t j(0); j < N; ++j)
//			{
//				costmat[i * N + j] = static_cast<int>((1.0f - EditDistanceRatio(last_filtered_texts[i], filtered_texts[j])) * cost_scale);
//			}
//		auto row_assignments(FindMinCostAssignment(costmat.get(), M, N));
//		for (std::size_t i(0); i < M; ++i)
//		{
//			auto j(row_assignments[i]);
//			if (costmat[i * N + j] > cost_scale * (1.0f - levenshtein_threshold))
//			{
//				text_changed = true;
//				break;
//			}
//		}
//
//		if (text_changed)
//		{
//			EmitSRT(texts, millisecond);
//			last_texts = texts;
//			last_filtered_texts = filtered_texts;
//		}
//		else
//		{
//			// keep longest text
//			for (std::size_t i(0); i < M; ++i)
//			{
//				auto j(row_assignments[i]);
//				if (texts[j].size() > last_texts[i].size())
//					last_texts[i] = texts[j];
//			}
//			last_filtered_texts = FilterSpaceCharacters(last_texts);
//		}
//	}
//};

void ConvertToRGB(cv::cuda::GpuMat& img, cv::Ptr<cv::cuda::CLAHE> gpu_clahe)
{
	static std::vector<cv::cuda::GpuMat> ycrcb_split;
	cv::cuda::cvtColor(img, img, cv::COLOR_BGR2YCrCb);
	cv::cuda::split(img, ycrcb_split);
	gpu_clahe->apply(ycrcb_split[0], ycrcb_split[0]);
	cv::cuda::merge(ycrcb_split, img);
	cv::cuda::cvtColor(img, img, cv::COLOR_YCrCb2RGB);
}

void ConvertToRGB(cv::Mat& img)
{
	static std::vector<cv::Mat> ycrcb_split;
	cv::cvtColor(img, img, cv::COLOR_BGR2YCrCb);
	cv::split(img, ycrcb_split);
	cv::equalizeHist(ycrcb_split[0], ycrcb_split[0]);
	cv::merge(ycrcb_split, img);
	cv::cvtColor(img, img, cv::COLOR_YCrCb2RGB);
}

float ImageL1Distance(cv::cuda::GpuMat const& a, cv::cuda::GpuMat const& b)
{
	static cv::cuda::GpuMat tmp;
	cv::cuda::absdiff(a, b, tmp);
	auto ret(cv::cuda::sum(tmp));
	return static_cast<float>(ret.val[0] + ret.val[1] + ret.val[2]) / static_cast<float> (a.size().width * a.size().height * 3.0f);
}

std::string GetPathName(std::string const& s)
{
#ifdef _WIN32
	char sep = '\\';
#else
	char sep = '/';
#endif

	size_t i = s.rfind(sep, s.length());
	if (i != std::string::npos)
	{
		return(s.substr(0, i + 1));
	}

	return("");
}

bool ShowDevices()
{
	int nDevices;

	auto ret(cudaGetDeviceCount(&nDevices));
	if (nDevices <= 0 || nDevices > 1000 || ret == cudaErrorNoDevice)
	{
		std::cout << "No CUDA capable device found!\n";
		throw std::runtime_error("No CUDA capable device found!");
	}
	if (ret == cudaErrorInsufficientDriver)
	{
		std::cout << "Driver out-of-date, please update your driver!\n";
		throw std::runtime_error("Driver out-of-date, please update your driver!");
	}
	if (ret == cudaErrorInitializationError)
	{
		std::cout << "Failed to initialize CUDA runtime!\n";
		throw std::runtime_error("Failed to initialize CUDA runtime!");
	}
	for (int i = 0; i < nDevices; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Device VRAM (MB): %d\n", static_cast<int>(prop.totalGlobalMem / 1024));
		printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("  Core Clock Rate (MHz): %d\n",
			   prop.clockRate / 1000);
		printf("  Memory Clock Rate (MHz): %d\n",
			   prop.memoryClockRate / 1000);
		printf("  Memory Bus Width (bits): %d\n",
			   prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			   2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
	}
	return true;
}

int main(int argc, char** argv) try {
	if (argc != 2)
		return 1;

	ShowDevices();

	std::cout << "Reading video file " << argv[1] << "\n";
	auto path(GetPathName(argv[0]));

	std::string video_file_name(argv[1]);
	std::string srt_file_name(video_file_name + ".srt");

	std::cout << "Saving srt file to " << srt_file_name << "\n";

	CreateAlphabet(path + "alphabet.txt");
	CreateSpaceCharacterU32();
	//CreateImageMorphologyResource();

	auto clahe(cv::cuda::createCLAHE(2.0, cv::Size(8, 8)));

	torch::NoGradGuard no_grad_scope;
	torch::jit::setGraphExecutorOptimize(true);

	auto craft(CreateModel(path + "detect.pth"));
	if (!craft)
		throw std::runtime_error("failed to load detect.pth");
	auto ocr(CreateModel(path + "ocr.pth"));
	if (!craft)
		throw std::runtime_error("failed to load ocr.pth");

	std::cout << "Loading models... done\n";

	auto cap{ cv::VideoCapture(argv[1]) };
	//cv::namedWindow("video", cv::WINDOW_AUTOSIZE);
	auto raw_width(static_cast<std::uint32_t>(cap.get(cv::CAP_PROP_FRAME_WIDTH))), raw_height(static_cast<std::uint32_t>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
	auto num_frames(static_cast<std::uint32_t>(cap.get(cv::CAP_PROP_FRAME_COUNT)));
	std::cout << "Video width: " << raw_width << " height: " << raw_height << " frame count: " << num_frames << "\n";

	std::cout << "Video loaded\n";

	FPSCounter fps;
	cv::Mat current_frame_cpu;


	cv::cuda::GpuMat tmp_resized;

	cv::cuda::GpuMat current_frame_raw_gpu;

	// fp32
	cv::cuda::GpuMat current_frame;
	// fp32
	cv::cuda::GpuMat last_frame;

	cv::cuda::GpuMat current_frame_resized;
	cv::cuda::GpuMat current_frame_filtered;

	std::vector<cv::cuda::GpuMat> rgbsplit;

	at::Tensor current_text_regions;
	at::Tensor last_text_regions;

	std::vector<BBox> current_text_bboxes;
	std::vector<BBox> last_text_bboxes;

	std::uint32_t frame_counter(0);

	SRTGenerator output(srt_file_name);

	char tmp[128];
	std::size_t cur_ms(0);

	while (cap.isOpened())
	{
		// read raw video frame
		if (!cap.read(current_frame_cpu))
			break;

		cur_ms = static_cast<std::size_t>(cap.get(cv::CAP_PROP_POS_MSEC));

		//ConvertToRGB(current_frame_cpu);
		//cv::cvtColor(current_frame_cpu, current_frame_cpu, cv::COLOR_RGB2BGR);

		++frame_counter;

		// upload to GPU
		current_frame_raw_gpu.upload(current_frame_cpu);

		// resize to fit
		auto resize_ret(ResizeKeepAspectRatio(current_frame_raw_gpu, current_frame_resized, tmp_resized, 640));
		cv::cuda::bilateralFilter(current_frame_resized, current_frame_filtered, 17, 80, 80);
		// BGR to RGB
		cv::cuda::cvtColor(current_frame_filtered, current_frame_filtered, cv::COLOR_BGR2RGB);

		cv::cuda::split(current_frame_filtered, rgbsplit);
		clahe->apply(rgbsplit[0], rgbsplit[0]);
		clahe->apply(rgbsplit[1], rgbsplit[1]);
		clahe->apply(rgbsplit[2], rgbsplit[2]);
		cv::cuda::merge(rgbsplit, current_frame_filtered);

		current_frame_filtered.convertTo(current_frame, CV_32FC3, 1.0 / 127.5, -1.0);

		// check for frame change
		bool frame_changed(false);
		if (last_frame.empty())
			frame_changed = true;
		else
		{
			// calculate L1 dist
			auto diff(ImageL1Distance(last_frame, current_frame));
			if (diff > 2e-4f)
				frame_changed = true;
			//std::cout << "diff=" << diff << "\n";
		}
		current_frame.copyTo(last_frame);

		if (frame_changed)
		{
			auto detect_result(DetectAndExtractTextRegions(craft, ocr, current_frame, resize_ret));

			bool text_region_changed(false);
			if (detect_result)
			{
				// we have text at current frame
				current_text_regions = detect_result.value().first;
				current_text_bboxes = detect_result.value().second;
				text_region_changed = DetectTextRegionChange(last_text_regions, last_text_bboxes, current_text_regions, current_text_bboxes, 0.6f, 0.01);
				last_text_regions = current_text_regions;
				last_text_bboxes = current_text_bboxes;
			}
			else
			{
				// no text detect at current frame
				if (last_text_bboxes.size() != 0)
					text_region_changed = true; // from text in last from to no text in current frame
				//if (last_text_regions.sizes().size())
				//	last_text_regions.zero_();
				last_text_bboxes.clear();
			}

			if (text_region_changed)
			{
#ifdef VERBOSE
				std::cout << "text_region_changed at " << frame_counter << "\n";
#endif
				if (detect_result)
				{
					auto ocr_texts(FilterSpaceOnlyString(PerformOCR(ocr, current_text_regions)));
					output.AppendSubtitle(ocr_texts, frame_counter, cur_ms);
				}
				else
					output.AppendSubtitle({}, frame_counter, cur_ms);
			}

#ifdef VERBOSE
			for (auto box : last_text_bboxes)
			{
				box.scale(1.0f / std::get<0>(resize_ret));
				cv:rectangle(current_frame_cpu, box.rect(), cv::Scalar(255, 0, 0), 2);
			}
			cv::imshow("video", current_frame_cpu);
#endif
		}



		//auto ocr_strings(DetectAndOCR(raw_width, raw_height, craft, ocr, current_frame_filtered, resize_ret));

		//std::cout << "=========================================\n";
		//for (auto const& s : ocr_strings)
		//{
		//	auto su8(u32tou8(s));
		//	std::cout << su8 << "\n";
		//}
		//std::cout << "=========================================\n";

		//if(last_frame.empty()||CalculateMSSIM(last_frame,))

		//cv::Mat frame_raw_blured;
		//cv::GaussianBlur(frame_raw, frame_raw_blured, cv::Size(5, 5), 8, 8);
		//cv::imshow("video blur", frame_raw_blured);

		if (fps.Update())
		{
			double prog(static_cast<double>(frame_counter) / static_cast<double>(num_frames) * 100.0);
			std::size_t eta_ms(static_cast<std::size_t>(static_cast<double>(num_frames - frame_counter) / fps.GetFPS() * 1000.0));
			std::sprintf(tmp, "[%.2lf%%] %u/%u fps=%.2f ETA=%s", prog, frame_counter, num_frames, fps.GetFPS(), output.mil2str(eta_ms).c_str());
#ifdef VERBOSE
			std::cout << tmp << "\n";
#else
			std::cout << tmp << "                                                            \r";
#endif
		}
#ifdef VERBOSE
		if (cv::waitKey(1) == 'q')
			break;
#endif
	}
	output.AppendSubtitle({}, frame_counter, cur_ms);
	double prog(static_cast<double>(frame_counter) / static_cast<double>(num_frames) * 100.0);
	std::sprintf(tmp, "[100%%] %u/%u fps=%.2f", frame_counter, frame_counter, fps.GetFPS());
	std::cout << tmp << "\n";
	std::cout << "Generating subtitles" << std::endl;
	output.Generate();

	return 0;
} catch (std::exception const& e)
{
	std::cerr << "Exception: " << e.what() << std::endl;
	std::getchar();
}
