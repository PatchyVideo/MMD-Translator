
import torch
from collections import Counter
import networkx as nx
import numpy as np

class CTCLabelConverter(object):
	""" Convert between text-label and text-index """
	SPACE_SYMBOLS = ':.\n\r[] \t\v\f{}-■_=+`~!@#$%^&*();\'",<>/?\\|－＞＜。，《》【】　？！￥…（）、：；·「」『』〔〕［］｛｝｟｠〉〈〖〗〘〙〚〛゠＝‥※＊〽〓〇＂“”‘’＃＄％＆＇＋．／＠＼＾＿｀｜～｡｢｣､･ｰﾟ￠￡￢￣￤￨￩￪￫￬￭￮・◊→←↑↓↔—'

	def __init__(self, character, bigram_probs = None):
		# character (str): set of the possible characters.
		dict_character = list(character)
		self.bigram_probs = bigram_probs

		self.dict = {}
		for i, char in enumerate(dict_character):
			# NOTE: 0 is reserved for 'blank' token required by CTCLoss
			self.dict[char] = i + 1

		self.character = ['[blank]'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)

	def encode(self, text, batch_max_length=25):
		"""convert text-label into text-index.
		input:
			text: text labels of each image. [batch_size]

		output:
			text: concatenated text index for CTCLoss.
					[sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
			length: length of each text. [batch_size]
		"""
		length = [len(s) for s in text]
		text = ''.join(text)
		text = [self.dict[char] for char in text]

		return (torch.IntTensor(text), torch.IntTensor(length))

	def decode_single_bigram_map(self, idx, val, k, n_chars, threshold, int_scale = 100) :
		"""
		Decode a single sentence by solving Maximum A Posteriori over P(sentence) using min cost flow
		idx: index into dictionary, shape (n, k), type uint32
		val: probabilities associated with each characters, shape (n, k), type float32
		k: num of candidates
		n_chars: n
		threshold: do not generate candidate list if OCR model output prob > threshold
		int_scale: scaling floating point edge cost to integer for nx.algorithms.flow.min_cost_flow to work
		"""
		assert self.bigram_probs
		last_ch = ''
		sentence_candidates = []
		for i in range(n_chars) :
			ch = self.character[idx[i, 0]]
			prob = val[i, 0]
			if ch == last_ch and ch != '[blank]' :
				last_ch = ch
				continue # ignore repeated and not [blank] chars
			last_ch = ch
			if prob >= threshold and ch == '[blank]' :
				continue # we are certrain this is [blank], skipping
			# if not certain
			if prob < threshold :
				total_prob = prob
				cur_candidates = {ch: prob}
				# find all candidate chars around it
				for k2 in range(1, k) :
					# for j in range(max(i - 1, 0), min(i + 2, n_chars)) :
					for j in range(max(i, 0), min(i + 1, n_chars)) :
						if self.character[idx[j, k2]] not in cur_candidates :
							cur_candidates[self.character[idx[j, k2]]] = 0
						cur_candidates[self.character[idx[j, k2]]] += val[j, k2]
						total_prob += val[j, k2]
				candidates = [(ch, -np.log(prob / total_prob)) for (ch, prob) in cur_candidates.items()]
			else :
				# OCR model is certain
				candidates = [(ch, -np.log(prob))]
			sentence_candidates.append(candidates)
		if not sentence_candidates :
			# no candidates, return empty string
			return ''
		# build DAG
		G = nx.DiGraph()
		# 1 unit of flow
		G.add_node('[start]', demand = -1)
		G.add_node('[end]', demand = 1)
		for i, (ch, neglogprob) in enumerate(sentence_candidates[0]) :
			if ch == '[blank]' :
				continue
			G.add_node((0, i, 'a'), ch = ch)
			G.add_node((0, i, 'b'), ch = ch)
			G.add_edge((0, i, 'a'), (0, i, 'b'), cap = 1, cost = int(int_scale * neglogprob))
			G.add_edge('[start]', (0, i, 'a'), cap = 1, cost = int(0))
		def list_source_nodes(j) :
			if j == 0 :
				return [('[start]', 0)]
			ret = []
			for i, (ch, neglogprob) in enumerate(sentence_candidates[j - 1]) :
				if ch == '[blank]' :
					recursive_nodes = list_source_nodes(j - 1)
					for i in range(len(recursive_nodes)) :
						# flow through [blank] node, add [blank] cost
						(node_name, old_neglogprob) = recursive_nodes[i]
						recursive_nodes[i] = (node_name, old_neglogprob + neglogprob)
					ret.extend(recursive_nodes)
				else :
					ret.append(((j - 1, i, 'b'), 0))
			return ret
		def get_bigram_neglogprob(u, v) :
			bigram = u + v
			if bigram in self.bigram_probs :
				return -np.log(self.bigram_probs[bigram]) # -logP(W[i]|w[i-1])
			else :
				return 100000000
		for j, candidates in enumerate(sentence_candidates[1: ], start = 1) :
			# j is position in sentence
			source_nodes = list_source_nodes(j)
			for i, (ch, neglogprob) in enumerate(candidates) :
				# i is position in candidates
				G.add_node((j, i, 'a'), ch = ch)
				G.add_node((j, i, 'b'), ch = ch)
				G.add_edge((j, i, 'a'), (j, i, 'b'), cap = 1, cost = int(int_scale * neglogprob))
				for (source, addtional_cost) in source_nodes :
					if isinstance(source, str) : # source is [start]
						G.add_edge('[start]', (j, i, 'a'), cap = 1, cost = int(int_scale * addtional_cost))
					else :
						source_ch = G.nodes[source]['ch']
						G.add_edge(source, (j, i, 'a'), cap = 1, cost = int(int_scale * (get_bigram_neglogprob(source_ch, ch) + addtional_cost)))
		# add edge from last candidates to [end]
		for (source, addtional_cost) in list_source_nodes(len(sentence_candidates)) :
			G.add_edge(source, '[end]', cap = 1, cost = int(int_scale * addtional_cost))
		# find min cost flow of one unit from [start] to [end]
		# using Goldberg-Tarjan
		FLOW = nx.algorithms.flow.min_cost_flow(G, capacity = 'cap', weight = 'cost')
		cur_node = '[start]'
		ans = []
		# flow dict walk
		while cur_node != '[end]' :
			next_nodes = FLOW[cur_node]
			next_node = None
			for (node, flow) in next_nodes.items() :
				# find non-zero flow arc
				if flow > 0 :
					next_node = node
					break
			if 'ch' in G.nodes[next_node] :
				_, _, ab = next_node
				if ab == 'a' :
					ans.append(G.nodes[next_node]['ch'])
			cur_node = next_node
		ans = ''.join(ans)
		return ans

	def decode_top_k(self, probs) :
		k = 5
		ret = torch.topk(probs, k = k, dim = 2, sorted = True)
		n_chars = probs.size(1)
		idx = ret.indices.cpu().numpy()
		val = ret.values.cpu().numpy()
		ret = []
		for (single_idx, single_val) in zip(idx, val) :
			text = self.decode_single_bigram_map(single_idx, single_val, k, n_chars, 0.75)
			text_cleaned = ''.join([ch for ch in text if ch not in self.SPACE_SYMBOLS])
			if text_cleaned :
				ret.append(text)
		return ret

	def decode(self, text_index, length, pred = False):
		""" convert text-index into text-label. """
		texts = []
		index = 0
		for l in length:
			t = text_index[index:index + l]

			char_list = []
			char_list2 = []
			for i in range(l):
				char_list2.append('' if t[i] == 0 else self.character[t[i]])
				if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
					char_list.append(self.character[t[i]])
			text = ''.join(char_list)
			text2 = ''.join(char_list2)

			if pred :
				texts.append((text, text2))
			else :
				texts.append(text)
			index += l
		return texts

