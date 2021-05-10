import codecs
import logging
import numpy as np

logger = logging.getLogger(__name__)

class W2VEmbReader:
	def __init__(self, emb_path, emb_dim=None):
		logger.info('Loading embeddings from: ' + emb_path)
		has_header = False
		with codecs.open(emb_path, 'r', encoding='utf8') as emb_file:
			tokens = next(emb_file).split()
			if len(tokens) == 2:
				try:
					int(tokens[0])
					int(tokens[1])
					has_header = True
				except ValueError:
					pass
		if has_header:
			with codecs.open(emb_path, 'r', encoding='utf8') as emb_file:
				tokens = next(emb_file).split()
				assert len(tokens) == 2, 'The first line in W2V embeddings must be the pair (vocab_size, emb_dim)'
				self.vocab_size = int(tokens[0])
				self.emb_dim = int(tokens[1])
				assert self.emb_dim == emb_dim, 'The embeddings dimension does not match with the requested dimension'
				self.embeddings = {}
				counter = 0
				for line in emb_file:
					tokens = line.split()
					# assert len(tokens) == self.emb_dim + 1, 'The number of dimensions does not match the header info'
					# modified by Shengjia Yan @2017-11-03 Friday   对源码进行修改
					word = tokens[0]
					vec = tokens[1:]
					# vec = tokens[1].split(',')
					assert len(vec) == self.emb_dim, 'The number of dimensions does not match the header info'
					self.embeddings[word] = vec
					counter += 1
				assert counter == self.vocab_size, 'Vocab size does not match the header info'
		else:
			with codecs.open(emb_path, 'r', encoding='utf8') as emb_file:
				self.vocab_size = 0
				self.emb_dim = -1
				self.embeddings = {}
				for line in emb_file:
					tokens = line.split()
					if len(tokens) != 301:
						continue
					# print(len(tokens))
					word = tokens[0]
					vec = tokens[1:]
					# vec = tokens[1].split(',')
					if self.emb_dim == -1:
						# self.emb_dim = len(tokens) - 1
						self.emb_dim = len(vec)  # 修改处
						assert self.emb_dim == emb_dim, 'The embeddings dimension does not match with the requested dimension'
					else:
						if len(vec) != self.emb_dim:
							continue
						# assert len(vec) == self.emb_dim, 'The number of dimensions does not match the header info'  # 修
					self.embeddings[word] = vec
					self.vocab_size += 1
		
		logger.info('  #vectors: %i, #dimensions: %i' % (self.vocab_size, self.emb_dim))
	
	def get_emb_given_word(self, word):
		try:
			return self.embeddings[word]
		except KeyError:
			return None
	
	def get_emb_matrix_given_vocab(self, vocab, emb_matrix=None):
		counter = 0.
		if emb_matrix == None:
			emb_matrix = []
			for word, index in vocab.items():
				try:
					emb_matrix.append(self.embeddings[word])
					# emb_matrix[0][index] = self.embeddings[word]
					counter += 1
				except KeyError:
					pass
		else:
			for word, index in vocab.items():
				try:
					# print(emb_matrix[0][index])
					# emb_matrix[index] = self.embeddings[word]
					emb_matrix[0][index] = self.embeddings[word]
					counter += 1
				except KeyError:
					# print(word)
					pass
			logger.info('%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(vocab), 100*counter/len(vocab)))
			return emb_matrix
	
	def get_emb_dim(self):
		return self.emb_dim
	
	
	
	
