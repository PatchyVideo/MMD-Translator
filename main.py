
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import cv2

import imgproc
import craft_utils

from craft import CRAFT
from ocr_ctc import OCR

from scipy.optimize import linear_sum_assignment
from skimage import metrics

from PIL import Image, ImageDraw, ImageFont
import itertools
import time
import networkx as nx
import argparse

from collections import OrderedDict
def copyStateDict(state_dict):
	if list(state_dict.keys())[0].startswith("module"):
		start_idx = 1
	else:
		start_idx = 0
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = ".".join(k.split(".")[start_idx:])
		new_state_dict[name] = v
	return new_state_dict

parser = argparse.ArgumentParser(description='Generate text bboxes given a video file')
parser.add_argument('--video', default='', type=str, help='video file')
parser.add_argument('--language', default='zh-cn', type=str, help='language to translate to')
parser.add_argument('--out', default='', type=str, help='output srt file')
parser.add_argument('--out_translated', default='', type=str, help='output translated srt file')
parser.add_argument('--text_threshold', default=0.51, type=float, help='text_threshold')
parser.add_argument('--link_threshold', default=0.1, type=float, help='link_threshold')
parser.add_argument('--low_text', default=0.5, type=float, help='low_text')
parser.add_argument('--ssim_threshold', default=0.31, type=float, help='ssim_threshold')
parser.add_argument('--iou_threshold', default=0.6, type=float, help='iou_threshold')
parser.add_argument('--levenshtein_threshold', default=0.8, type=float, help='Subtitles above <levenshtein_threshold> are consider the same')
parser.add_argument('--size', default=640, type=int, help='Video canvas size to perform recognition')
parser.add_argument('--skip_frame', default=0, type=int, help='Skip every <skip_frame> frames')
parser.add_argument('--verbose', default=False, action='store_true', help='Show process')
parser.add_argument('--draw_subtitle', default=False, action='store_true', help='Draw subtitle on frames, requires --verbose')
parser.add_argument('--subtitle_discard_threshold', default=200, type=int, help='Discard subtitle lasts less than <subtitle_discard_threshold> milliseconds')
args = parser.parse_args()

args.batch_size = 1

print(' -- Creating models')
model = CRAFT()

clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))

probs = {}
with open('bigram_probs.txt') as fp :
    for i, l in enumerate(fp) :
        l = l.strip()
        if l :
            try :
                [w, f] = l.split(' ')
                probs[w] = float(f)
            except :
                pass

alphabet = []
with open('alphabet.txt', 'r') as fp :
	alphabet = list(fp.read())
model_ocr = OCR(alphabet, 0, probs)

print(' -- Loading saved models')
d = torch.load('textdet_300k.pth')
model.load_state_dict(d['model'])

d = torch.load('ocr_640k.pth')
model_ocr.load_state_dict(d['model'])

model = model.cuda()
model_ocr = model_ocr.cuda()
model.eval()
model_ocr.eval()

print(' -- Loading video')

cap = cv2.VideoCapture(args.video)

text_threshold = args.text_threshold
link_threshold = args.link_threshold
low_text = args.low_text
use_poly = False

batch = None#np.empty((args.batch_size, args.size, args.size, 3), dtype = np.float32)

def can_merge(x1, y1, w1, h1, x2, y2, w2, h2) :
	char_size = min(h1, h2)
	if abs(h1 - h2) > char_size / 1 :
		return False
	if abs(y1 - y2) > char_size / 1 :
		return False
	if x1 < x2 :
		if abs(x1 + w1 - x2) > char_size * 2 :
			return False
		else :
			return True
	else :
		if abs(x2 + w2 - x1) > char_size * 2 :
			return False
		else :
			return True

def merge_bboxes(bboxes) :
	G = nx.Graph()
	for i, box in enumerate(bboxes) :
		G.add_node(i, box = box)
	for ((u, ubox), (v, vbox)) in itertools.combinations(enumerate(bboxes), 2) :
		if can_merge(ubox[0][0], ubox[0][1], ubox[1][0] - ubox[0][0], ubox[2][1] - ubox[1][1], vbox[0][0], vbox[0][1], vbox[1][0] - vbox[0][0], vbox[2][1] - vbox[1][1]) :
			G.add_edge(u, v)
	merged_boxes = []
	for node_set in nx.algorithms.components.connected_components(G) :
		kq = np.concatenate(bboxes[list(node_set)], axis = 0)
		max_coord = np.max(kq, axis = 0)
		min_coord = np.min(kq, axis = 0)
		extend = int(min(max_coord[1] - min_coord[1], max_coord[0] - min_coord[0]) * 0.1)
		if max_coord[0] + extend - max(min_coord[0] - extend, 0) > 0 and max_coord[1] + extend - max(min_coord[1] - extend, 0) > 0 :
			merged_boxes.append(np.array([
				np.array([max(min_coord[0] - extend, 0), max(min_coord[1] - extend, 0)]),
				np.array([max_coord[0] + extend, max(min_coord[1] - extend, 0)]),
				np.array([max_coord[0] + extend, max_coord[1] + extend]),
				np.array([max(min_coord[0] - extend, 0), max_coord[1] + extend])
				]))
	return merged_boxes

def get_bbox(image_batch, ratio_w, ratio_h) :
	images_torch = torch.from_numpy(image_batch).cuda().permute(0, 3, 1, 2)
	with torch.no_grad() :
		pred = model(images_torch)
		rs_tensor = pred[:, 0, :, :].cpu().numpy()
		as_tensor = pred[:, 1, :, :].cpu().numpy()
	render_img = rs_tensor[0].copy()
	render_img = np.hstack((render_img, as_tensor[0]))
	ret_score_text = imgproc.cvt2HeatmapImg(render_img)
	if args.verbose :
		cv2.imshow('score', ret_score_text)
	ret = []
	for (rs_img, as_img) in zip(rs_tensor, as_tensor) :
		# Post-processing
		boxes, polys = craft_utils.getDetBoxes(rs_img, as_img, text_threshold, link_threshold, low_text, use_poly)

		# coordinate adjustment
		boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
		polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
		for k in range(len(polys)):
			if polys[k] is None: polys[k] = boxes[k]
		frame_bboxes = []
		polys = merge_bboxes(polys)
		for [tl, tr, br, bl] in polys :
			frame_bboxes.append({'x': int(tl[0]), 'y': int(tl[1]), 'width': int(tr[0] - tl[0]), 'height': int(br[1] - tr[1])})
		ret.append({'bboxes': frame_bboxes})
	return ret[0]['bboxes']

from text_utils import DrawRegion
from googletrans import Translator
trans = Translator()

from PIL import ImageFont
caption_font = ImageFont.truetype('Arial-Unicode-Regular.ttf', 20)

cached_text = None
cached_translation = None

SPACE_SYMBOLS = ':.\n\r[] \t\v\f{}-_■=+`~!@#$%^&*();\'",<>/?\\|－＞＜。，《》【】　？！￥…（）、：；·「」『』〔〕［］｛｝｟｠〉〈〖〗〘〙〚〛゠＝‥※＊〽〓〇＂“”‘’＃＄％＆＇＋．／＠＼＾＿｀｜～｡｢｣､･ｰﾟ￠￡￢￣￤￨￩￪￫￬￭￮・◊→←↑↓↔—'

def render_frame(frame, bboxes, draw_subtitle, texts) :
	for bbox in bboxes :
		x, y = bbox['x'], bbox['y']
		width, height = bbox['width'], bbox['height']
		frame = cv2.rectangle(frame, (x, y), (x + width, y + height), color=(0, 0, 255), thickness=2)
	if draw_subtitle and texts :
		texts_accepted = []
		for txt in texts :
			txt_nospace = ''.join([ch for ch in txt if ch not in SPACE_SYMBOLS])
			if txt_nospace :
				texts_accepted.append(txt)
		texts = texts_accepted
		global cached_text
		global cached_translation
		if cached_text != texts :
			texts_translated = trans.translate('\n'.join(texts), dest = args.language).text.split('\n')
			cached_text = texts
			cached_translation = texts_translated
		else :
			texts_translated = cached_translation
		text_img_mask = [DrawRegion(txt, caption_font, (255, 255, 255), (255, 255, 255), 0, 5) for txt in texts_translated]
		if text_img_mask :
			total_height = sum([img.shape[0] for (img, _) in text_img_mask])
			single_height = total_height // len(text_img_mask)
			y = frame.shape[0] - total_height - single_height
			frame_width = frame.shape[1]
			for (img, mask) in text_img_mask :
				x = frame_width // 2 - img.shape[1] // 2
				mask = mask > 0
				frame[y: y + img.shape[0], x: x + img.shape[1]] = (frame[y: y + img.shape[0], x: x + img.shape[1]].astype(np.float32) * 0.2).astype(np.uint8)
				frame[y: y + img.shape[0], x: x + img.shape[1]][mask] = img[mask]
				y += img.shape[0]
	return frame

def ms2sec(ms) :
	sec = int(ms / 1000)
	remain_ms = ms - sec * 1000
	minutes = int(sec / 60)
	remain_sec = sec - minutes * 60
	return '%d:%02d.%03d' % (minutes, remain_sec, remain_ms)

def levenshtein_ratio_and_distance(s, t, ratio_calc = True):
	""" levenshtein_ratio_and_distance:
		Calculates levenshtein distance between two strings.
		If ratio_calc = True, the function computes the
		levenshtein distance ratio of similarity between two strings
		For all i and j, distance[i,j] will contain the Levenshtein
		distance between the first i characters of s and the
		first j characters of t
	"""
	# Initialize matrix of zeros
	rows = len(s)+1
	cols = len(t)+1
	distance = np.zeros((rows,cols),dtype = int)

	# Populate matrix of zeros with the indeces of each character of both strings
	for i in range(1, rows):
		for k in range(1,cols):
			distance[i][0] = i
			distance[0][k] = k

	# Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
	for col in range(1, cols):
		for row in range(1, rows):
			if s[row-1] == t[col-1]:
				cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
			else:
				# In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
				# the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
				if ratio_calc == True:
					cost = 2
				else:
					cost = 1
			distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
								 distance[row][col-1] + 1,          # Cost of insertions
								 distance[row-1][col-1] + cost)     # Cost of substitutions
	if ratio_calc == True:
		# Computation of the Levenshtein Distance Ratio
		Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
		return Ratio + 1e-7
	else:
		# print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
		# insertions and/or substitutions
		# This is the minimum number of edits needed to convert string a to string b
		return "The strings are {} edits away".format(distance[row][col])

def mark_keyframe_and_OCR(frame, last_frame_time_ms, frame_time_ms, bboxes, text_height = 32) :
	print('Change at %s' % ms2sec(frame_time_ms))
	if len(bboxes) == 0 :
		return
	max_text_width = 0
	for box in bboxes :
		ratio = float(box['width']) / float(box['height'])
		new_width = int(text_height * ratio)
		max_text_width = max(max_text_width, new_width)
	extracted_regions = np.zeros((len(bboxes), text_height, max_text_width, 3), dtype = np.uint8)
	for i, box in enumerate(bboxes) :
		region = frame[box['y']: box['y'] + box['height'], box['x']: box['x'] + box['width'], :]
		ratio = float(box['width']) / float(box['height'])
		new_width = int(text_height * ratio)
		new_height = text_height
		region_normailzed = cv2.resize(region, (new_width, new_height), interpolation = cv2.INTER_AREA)
		extracted_regions[i][:, :new_width, :] = region_normailzed
	region_img = np.concatenate(extracted_regions, axis = 0)
	if args.verbose :
		cv2.imshow('text region', cv2.cvtColor(region_img, cv2.COLOR_RGB2BGR))
	ocr_result = model_ocr.predict_topk(extracted_regions, use_cuda = True)
	if args.verbose and ocr_result :
		print('========================================')
		for txt in ocr_result :
			if txt :
				print(txt)
		# print('===========================')
		# txt_all = '\n'.join([s for s in ocr_result if s])
		# chs = trans.translate(txt_all, dest = 'zh-cn')
		# print(chs.text)
		print('========================================')
	return ocr_result

def iou(box1, box2) :
	def bb_intersection_over_union(boxA, boxB):
		# determine the (x, y)-coordinates of the intersection rectangle
		xA = max(boxA[0], boxB[0])
		yA = max(boxA[1], boxB[1])
		xB = min(boxA[2], boxB[2])
		yB = min(boxA[3], boxB[3])

		# compute the area of intersection rectangle
		interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
		if interArea == 0:
			return 0
		# compute the area of both the prediction and ground-truth
		# rectangles
		boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
		boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the interesection area
		iou = interArea / float(boxAArea + boxBArea - interArea)

		# return the intersection over union value
		return iou
	return bb_intersection_over_union(
		[box1['x'], box1['y'], box1['x'] + box1['width'], box1['y'] + box1['height']],
		[box2['x'], box2['y'], box2['x'] + box2['width'], box2['y'] + box2['height']])

def detect_changes(last_frame, cur_frame, last_frame_bboxes, cur_frame_bboxes, ssim_threshold, iou_threshold) :
	N = len(last_frame_bboxes)
	M = len(cur_frame_bboxes)
	if N != M :
		print('change due to N != M                       ')
		return True
	cost = np.zeros(shape = ( N, M ), dtype = np.float32)
	for i in range(N) :
		for j in range(M) :
			cost[i, j] = 1 - iou(last_frame_bboxes[i], cur_frame_bboxes[j])
	assignment = []
	for _ in range( N ) :
		assignment.append( -1 )
	row_ind, col_ind = linear_sum_assignment( cost )
	target_assignment = [ False ] * M
	for i in range( len( row_ind ) ) :
		assignment[row_ind[i]] = col_ind[i]
		target_assignment[col_ind[i]] = True
	for i in range( M ) :
		if not target_assignment[i] :
			# new region appear
			print('change due to new region')
			return True
	for i in range( N ) :
		if assignment[i] == -1 :
			# region disappear
			print('change due to region disappear')
			return True
		else :
			if 1 - cost[i, assignment[i]] < iou_threshold :
				# test IOU, although assigned, their difference indicates change
				print('change due to iou = %f                     ' % (1 - cost[i, assignment[i]]))
				return True
			else :
				# test SSIM
				last_frame_box = last_frame_bboxes[i]
				cur_frame_box = cur_frame_bboxes[assignment[i]]
				last_frame_box_img = last_frame[last_frame_box['y']: last_frame_box['y'] + last_frame_box['height'], last_frame_box['x']: last_frame_box['x'] + last_frame_box['width'], :]
				cur_frame_box_img = cur_frame[cur_frame_box['y']: cur_frame_box['y'] + cur_frame_box['height'], cur_frame_box['x']: cur_frame_box['x'] + cur_frame_box['width'], :]
				# resize to same shape//1
				# TODO: replace with SIFT keypoint matching
				cur_frame_box_img = cv2.resize(cur_frame_box_img, (last_frame_box_img.shape[1] // 1, last_frame_box_img.shape[0] // 1), interpolation = cv2.INTER_AREA)
				last_frame_box_img = cv2.resize(last_frame_box_img, (last_frame_box_img.shape[1] // 1, last_frame_box_img.shape[0] // 1), interpolation = cv2.INTER_AREA)
				# RGB to GRAY
				last_frame_box_img = clahe.apply(cv2.cvtColor(last_frame_box_img, cv2.COLOR_RGB2GRAY))
				cur_frame_box_img = clahe.apply(cv2.cvtColor(cur_frame_box_img, cv2.COLOR_RGB2GRAY))
				# compute SSIM
				try :
					v = metrics.structural_similarity(last_frame_box_img, cur_frame_box_img)
				except :
					continue
				if v < ssim_threshold :
					# cv2.imshow('img1', last_frame_box_img)
					# cv2.imshow('img2', cur_frame_box_img)
					print('change due to ssim = %f                          ' % v)
					return True
	return False

def detect_text_change_and_keep_longest(last, cur, levenshtein_threshold = 0.9) :
	if not cur :
		cur = []
	if not last :
		last = []
	last2 = last
	cur2 = cur
	N = len(last)
	M = len(cur)
	if N != M :
		return True, []
	# remove spaces
	last = [''.join([ch for ch in s if s not in SPACE_SYMBOLS]) for s in last]
	cur = [''.join([ch for ch in s if s not in SPACE_SYMBOLS]) for s in cur]
	cost = np.zeros(shape = ( N, M ), dtype = np.float32)
	for i in range(N) :
		for j in range(M) :
			cost[i, j] = -np.log(levenshtein_ratio_and_distance(last[i], cur[j]))
	assignment = []
	for _ in range( N ) :
		assignment.append( -1 )
	row_ind, col_ind = linear_sum_assignment( cost )
	target_assignment = [ False ] * M
	for i in range( len( row_ind ) ) :
		assignment[row_ind[i]] = col_ind[i]
		target_assignment[col_ind[i]] = True
	for i in range( M ) :
		if not target_assignment[i] :
			# new text appear
			return True, []
	for i in range( N ) :
		if assignment[i] == -1 :
			# text disappear
			return True, []
		else :
			if cost[i, assignment[i]] > -np.log(levenshtein_threshold) :
				# text change
				return True, []
	# keep longest
	new_last_text = []
	for i in range(N) :
		if len(last2[i]) > len(cur2[assignment[i]]) :
			new_last_text.append(last2[i])
		else :
			new_last_text.append(cur2[assignment[i]])
	return False, new_last_text

def ms2hrs(ms) :
	sec = int(ms / 1000)
	remain_ms = ms - sec * 1000
	minutes = int(sec / 60)
	remain_sec = sec - minutes * 60
	hrs = int(minutes / 60)
	remain_minutes = minutes - hrs * 60
	return '%02d:%02d:%02d,%03d' % (hrs, remain_minutes, remain_sec, remain_ms)

def emit_srt(counter, from_time, to_time, text) :
	if args.out :
		with open(args.out, 'a+') as fp :
			print(counter, file = fp)
			print('%s --> %s' % (ms2hrs(from_time), ms2hrs(to_time)), file = fp)
			print('\n'.join(text), file = fp)
			print('', file = fp)
	if args.out_translated :
		text_trans = trans.translate('\n'.join(text), dest = args.language).text.split('\n')
		with open(args.out_translated, 'a+') as fp :
			print(counter, file = fp)
			print('%s --> %s' % (ms2hrs(from_time), ms2hrs(to_time)), file = fp)
			print('\n'.join(text_trans), file = fp)
			print('', file = fp)

all_frame_bboxes = []
counter = 0
last_frame = None
last_frame_time_ms = 0
text_counter = 0
last_texts = []
last_change_frametime = 0
fps_last_time = None
fps_last_frame = None
last_frame_resized = None

while cap.isOpened() :
	ret, frame = cap.read()
	if not ret :
		break
	if counter % (args.skip_frame + 1) != 0 :
		counter += 1
		last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		continue

	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # RGB order

	frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
	frame_resized, target_ratio, _ = imgproc.resize_aspect_ratio(frame, args.size, cv2.INTER_AREA, mag_ratio = 1)
	if last_frame_resized is not None and not args.verbose :
		ssim = metrics.structural_similarity(cv2.cvtColor(last_frame_resized, cv2.COLOR_RGB2GRAY), cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY))
		if ssim > 0.9 :
			counter += 1
			last_frame_resized = frame_resized
			continue
	last_frame_resized = frame_resized
	frame_resized = cv2.bilateralFilter(frame_resized, 17, 80, 80)
	ratio_h = ratio_w = 1 / target_ratio
	frame_norm = imgproc.normalizeMeanVariance(frame_resized, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	#frame_norm = imgproc.normalizeMeanVariance(frame_resized)
	if batch is None :
		batch = np.empty((args.batch_size, frame_norm.shape[0], frame_norm.shape[1], 3), dtype = np.float32)
	batch[counter % args.batch_size, :, :, :] = frame_norm
	counter += 1
	bboxes = get_bbox(batch, ratio_w, ratio_h)
	all_frame_bboxes.append(bboxes)
	texts = last_texts
	if counter == 1 :
		# first frame is key frame
		texts = mark_keyframe_and_OCR(frame, last_frame_time_ms, frame_timestamp_ms, bboxes)
		last_texts = texts
		last_change_frametime = frame_timestamp_ms
	else :
		# if two consective frames have changes we mark it as key frame
		if detect_changes(last_frame, frame, all_frame_bboxes[-2], all_frame_bboxes[-1], args.ssim_threshold, args.iou_threshold) :
			texts = mark_keyframe_and_OCR(frame, last_frame_time_ms, frame_timestamp_ms, bboxes)
			# detect change in texts
			try :
				changed, new_last_texts = detect_text_change_and_keep_longest(last_texts, texts, args.levenshtein_threshold)
			except :
				changed = True
			if changed :
				if last_texts and frame_timestamp_ms - last_change_frametime > args.subtitle_discard_threshold :
					text_counter += 1
					emit_srt(text_counter, last_change_frametime, frame_timestamp_ms, last_texts)
				last_change_frametime = frame_timestamp_ms
				last_texts = texts
			else :
				last_texts = new_last_texts
	last_frame = frame
	last_frame_time_ms = frame_timestamp_ms
	rendered_frame = render_frame(frame, bboxes, args.draw_subtitle, texts)
	if args.verbose:
		cv2.imshow('video', cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))
		if counter % 4 == 1 :
			if fps_last_time is None :
				fps_last_time = time.time()
				fps_last_frame = counter
				fps = 0
			else :
				elapsed_time = time.time() - fps_last_time
				elapsed_frames = counter - fps_last_frame
				fps_last_frame = counter
				fps_last_time = time.time()
				fps = float(elapsed_frames) / float(elapsed_time)
			print('frame: %d - %s - %.1f FPS%s\r' % (counter, ms2hrs(frame_timestamp_ms), fps, ' ' * 50), end = '')
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()


