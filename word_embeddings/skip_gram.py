import time
import math
import random

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import pandas as pd
import os
import re
import spacy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import string
import torch.nn.functional as F
import sys
sys.path.insert(0,'NMT_Model')
from data_extractor import read_data
from data_extractor import process_data
from data_extractor import word_frequency
from data_extractor import word_indexer

data = read_data('NMT_Model/data/Lit/train.lit.en','NMT_Model/data/Lit/train.lit.ro',[1])
pp_en = ['ct','cn','gm','lw']
pp_tg = ['ct','cn','lw']
data.head()
processed_data = process_data(data,pp_en,pp_tg)
vocab_en, vocab_ro = word_frequency(processed_data)


w2i_en, i2w_en, w2i_tg, i2w_tg = word_indexer(vocab_en,vocab_ro)


window_size = 2
idx_pair = []

english_sentences = processed_data[0]
processed_data.head()



def create_training_pairs(window_length):
    training_pairs = []
    for sentence in english_sentences:
        indices = [w2i_en[word] for word in sentence]
        for center_word_pos in range(len(indices)):
            
            targets = center_word_pos + \
                np.array(list(range(-window_length,window_length)))
            for i in targets:
                if i < 0 or i >= len(indices) or \
                    i == center_word_pos:
                        pass
                else:
                    training_pairs.append([indices[center_word_pos],
                                          indices[i]])

    return training_pairs


def get_input_layer(word_idx,vocab_size):
    x = torch.zeros(vocab_size).float()
    x[word_idx] = 1.0
    return x

training_pairs = create_training_pairs(3)
embedding_dim = 300
vocab_size = len(w2i_en)

W1 = Variable(torch.randn(embedding_dim, vocab_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocab_size, embedding_dim).float(), requires_grad=True)
num_epochs = 100
learning_rate = 0.001

for epo in range(num_epochs):
    loss_val = 0
    for data, target in training_pairs:
        x = Variable(get_input_layer(data,vocab_size)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)
    
        log_softmax = F.log_softmax(z2, dim=0)

        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.data
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()
    if epo % 10 == 0:    
        print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')
        

print(training_pairs.shape)

