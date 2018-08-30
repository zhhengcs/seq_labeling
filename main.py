#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import json
import numpy as np
import sys
from torch import nn
sys.path.append('../')
from torch.optim import Adam,SGD
from pykp.dataloader import BucketIterator
from torch.autograd import Variable

use_gpu=True
batch_size = 32
EMBEDDING_DIM = 100
HIDDEN_DIM = 150
tagset_size=2
max_epoch = 5
learning_rate = 0.001


def Dataset(datapath, word2id, id2word,func=None):
	f = open(datapath)
	for line in f:
		sample = json.loads(line.strip())
		if func:
			yield func(sample)
		else:
			yield sample
class DataLoader(BucketIterator):
	def __init__(self,datapath,word2id,id2word,
					length=None,batch_size=None,Data_type=None):
		
		super(DataLoader,self).__init__(datapath,word2id,id2word,
					batch_size=batch_size,sort=True,
					length=length,Data_type=Data_type)
		
	
	def process_batch(self,batch):
		
		src_batch = [b['src'] for b in batch]
		label_batch = [b['label'] for b in batch]
		if self.sort:
			src_len_order = np.argsort([len(x) for x in src_batch])[::-1]
			src_batch = [src_batch[i] for i in src_len_order]
			label_batch = [label_batch[i] for i in src_len_order]
		
		src, src_len, src_mask = self._pad(src_batch)
		label,label_len,label_mask = self._pad(label_batch,pad_id=-1)

		if use_gpu:
			src = src.cuda()
			src_mask = src_mask.cuda()
			label = label.cuda()
			label_mask = label_mask.cuda()

		return src,src_len,src_mask,label,label_mask


	
		
		
def load_data_and_vocab():
	word2id,id2word,vocab = torch.load('../data/small/vocab.pt')
	train_data_loader = DataLoader('../data/small/seq_labeling_train.json',word2id,id2word,
										batch_size=batch_size,
										Data_type=Dataset,length=16726)

	test_data_loader = DataLoader('../data/small/seq_labeling_train.json',word2id,id2word,
										batch_size=batch_size,
										Data_type=Dataset,length=16726)

	return train_data_loader,word2id,id2word,test_data_loader



class GRUTagger(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
		super(GRUTagger, self).__init__()
		self.hidden_dim = hidden_dim

		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

		# The LSTM takes word embeddings as inputs, and outputs hidden states
		# with dimensionality hidden_dim.
		self.gru = nn.GRU(embedding_dim, hidden_dim,bidirectional=True,batch_first=True)

		# The linear layer that maps from hidden state space to tag space
		self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
		
		#init embedding
	def init_embedding(self,embedding):
		self.word_embeddings.weight = nn.Parameter(embedding)
		self.word_embeddings.weight.requires_grad = False

	def init_hidden(self,batch_size):
		
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		h_encoder = Variable(torch.zeros(2, batch_size, self.hidden_dim),requires_grad=False)
		if torch.cuda.is_available() and use_gpu:
			return h_encoder.cuda()
		else:
			return h_encoder
    
	def forward(self, input_src,input_src_len,input_mask):
		batch_size = input_src.size()[0]
		self.hidden = self.init_hidden(batch_size)
		# src_mask = self.get_mask(input_src_len)
		src_mask = input_mask

		src_emb = self.word_embeddings(input_src)
		
		src_emb = nn.utils.rnn.pack_padded_sequence(src_emb, input_src_len, batch_first=True)
		self.gru.flatten_parameters()
		src_out, self.hidden = self.gru(src_emb, self.hidden)
		src_out, _ = nn.utils.rnn.pad_packed_sequence(src_out, batch_first=True)
		src_out = src_out * src_mask.view(src_mask.size() + (1,)).float()
		# print(src_out.size(),'src_out')

		tag_space = self.hidden2tag(src_out)
		
		tag_scores = nn.functional.log_softmax(tag_space, dim=-1).view(batch_size,-1,tagset_size)
		
		return tag_scores

def evaluate(pred,label,src_len,idx):
	# print(idx)
	TP = 0
	FP = 0

	FN = 0
	TN = 0
	
	p = pred
	t = label
	match_score = (pred == label)+0
	acc = match_score.cpu().numpy()[:src_len].sum()*1.0/src_len

	for j in range(src_len):
		if t[j] == 1 and p[j] == 1:
			TP += 1
		if t[j] == 1 and p[j] == 0:
			FN += 1
		if t[j] == 0 and p[j] == 1:
			FP += 1
		if t[j] == 0 and p[j] == 0:
			TN += 1

	if TP == 0:
		return 0,0,0
	
	p = TP*1.0/(TP+FP)
	r = TP*1.0/(TP+FN)

	

	return p,r,acc

def adjust_learning_rate(optimizer, epoch,lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def test(model,train_data_loader):
	sum_p=0
	sum_r=0
	sum_acc = 0
	example_cnt = 0

	for batch_idx,batch in enumerate(train_data_loader):
		src,src_len,src_mask,label,label_mask = batch

		tag_scores = model(src,src_len,src_mask)

		probs, words = tag_scores.data.topk(1, dim=-1)
		words = words.squeeze(2)
		
		for i in range(len(src_len)):
			p,r,acc = evaluate(words[i],label[i],src_len[i],i)
			exit(0)		
			sum_p += p
			sum_r += r

		example_cnt += len(src_len)
		
		print('batch %d/%d: avg_p=%f,avg_r=%f,acc=%f'%(batch_idx,total_batch_cnt,sum_p/example_cnt,sum_r/example_cnt,sum_acc/example_cnt))

if __name__ == '__main__':
	train_data_loader,word2id,id2word,test_data_loader = load_data_and_vocab()
	embedding = torch.load('glove_embedding.pt')

	loss_function = torch.nn.NLLLoss(ignore_index=-1)
	model = GRUTagger(EMBEDDING_DIM,HIDDEN_DIM,len(word2id),tagset_size)
	model.init_embedding(embedding)

	if use_gpu:
		model = model.cuda()

	# optimizer = SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

	total_batch_cnt = 16726/batch_size+1
	total_batch = 0

	for epoch in range(max_epoch):

		for batch_idx,batch in enumerate(train_data_loader):
			src,src_len,src_mask,label,label_mask = batch
			# print(src.size())
			total_batch += 1
			model.zero_grad()
			tag_scores = model(src,src_len,src_mask)
			
			loss = loss_function(tag_scores.view(-1,tagset_size), label.view(-1))
			loss.backward()
			optimizer.step()

			print('epoch %d,batch %d/%d: loss %f'%(epoch,batch_idx,total_batch_cnt,loss.data.item()))
			# if batch_idx == 50:
			# 	break
		adjust_learning_rate(optimizer,epoch,learning_rate)
		
		torch.save(model,'model.'+str(epoch))
		test(model,test_data_loader)
	
	# model = torch.load('model.0')
	# test(model,test_data_loader)