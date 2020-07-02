
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F

from collections import OrderedDict

from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

from modules.feature_extraction import ResNet_FeatureExtractor, ResNet_FeatureExtractor_Small
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

from ctc_utils import CTCLabelConverter

class OCR(nn.Module) :
	def __init__(self, alphabet, max_seq_len = 32, bigram_probs = None) :
		super(OCR, self).__init__()
		self.alphabet = alphabet
		self.max_seq_len = max_seq_len
		self.converter = CTCLabelConverter(alphabet, bigram_probs = bigram_probs)
		self.feature_extractor = ResNet_FeatureExtractor_Small(3, 256)
		print('Backbone parameters: %d' % count_parameters(self.feature_extractor))
		self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
		self.pred = nn.Linear(256, len(self.converter.character))
		print(' -- Pred shape %s' % repr(self.pred.weight.shape))

	def load_pretrained_model(self, filename) :
		d = torch.load(filename)
		dict_resnet = OrderedDict()
		dict_bilstm = OrderedDict()
		for (k, v) in d.items() :
			if 'FeatureExtraction' in k and 'ConvNet.conv0_1.weight' not in k :
				dict_resnet[k.replace('module.FeatureExtraction.', '')] = v
		self.feature_extractor.load_state_dict(dict_resnet, strict = False)

	def extract_feature(self, images) :
		"""
		inputs:
			images: a torch tensor of shape (N, 3, H, W)
		outputs:
			1) a totch tensor of shape (N, 512, W - 31)
		"""
		return self.feature_extractor(images)

	def forward(self, images, labels) :
		"""
		inputs:
			images: a torch tensor of shape (N, 3, H, W)
			labels: a list of texts
		outputs:
			1) N probability distribution at each step [batch_size x num_steps x num_classes]
			2) labels target
		"""
		labels, length = self.converter.encode(labels, self.max_seq_len)
		#with torch.no_grad() :
		feats = self.feature_extractor(images)
		if torch.isnan(feats).any() :
			breakpoint()
		feats = self.AdaptiveAvgPool(feats.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
		feats = feats.squeeze(3)
		return self.pred(feats), labels, length

	def predict_topk(self, images, use_cuda = False) :
		"""
		inputs:
			images: a numpy tensor of shape (N, H, W, 3) of type uint8, color order RGB
		outputs:
			1) list of N texts
		"""
		with torch.no_grad() :
			images = (torch.from_numpy(images).float() - 127.5) / 127.5
			images = images.permute(0, 3, 1, 2)
			batch_size = images.size(0)
			if use_cuda :
				images = images.cuda()
			feats = self.feature_extractor(images)
			feats = self.AdaptiveAvgPool(feats.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
			feats = feats.squeeze(3)
			preds = self.pred(feats)
			probs = preds.softmax(2)
			return self.converter.decode_top_k(probs)

	def predict(self, images, use_cuda = False) :
		"""
		inputs:
			images: a numpy tensor of shape (N, H, W, 3) of type uint8, color order RGB
		outputs:
			1) list of N texts
		"""
		with torch.no_grad() :
			images = (torch.from_numpy(images).float() - 127.5) / 127.5
			images = images.permute(0, 3, 1, 2)
			batch_size = images.size(0)
			if use_cuda :
				images = images.cuda()
			feats = self.feature_extractor(images)
			feats = self.AdaptiveAvgPool(feats.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
			feats = feats.squeeze(3)
			preds = self.pred(feats)
			_, preds_index = preds.max(2)
			preds_index = preds_index.view(-1)
			preds_size = torch.IntTensor([preds.size(1)] * batch_size)
			if use_cuda :
				preds_size = preds_size.cuda()
			return self.converter.decode(preds_index.data, preds_size.data, pred = False)

