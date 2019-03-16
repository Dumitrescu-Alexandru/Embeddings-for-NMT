# %%

import time
import math
import random

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

import numpy as np
import pandas as pd
import os
import re
import spacy

import torch
import torch.nn as nn
from torch import optim
import string
import torch.nn.functional as F

# preprocessing to be applied. Small description for each of them
# 'ct' : removes punctuation from the end of the words
# 'cn' : makes makes numbers into '#'/'##'/.../'#####'
# 'gm' : english only! replaces short words with full ('ain't -> is not')
# 'lw' : lower case the words

pp_en = ['ct','cn','gm','lw']
pp_tg = ['ct','cn','lw']


# Define some constants


teacher_forcing_ratio = 0.5
hidden_size = 256
ps_epochs = 200
PATH = "NMT_Models/"
MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_data(file_en, file_translated_lang, flags):
    print("HELLO")
    print(os.getcwd())

    with open(file_translated_lang,mode='r') as f:
        content = f.readlines()
    content_ro = [x.strip() for x in content] 

    with open(file_en,mode='r') as f:
        content = f.readlines()
    content_en = [x.strip() for x in content] 
    if flags[0] == True:
        content_ro[0] = content_ro[0].replace('\ufeff',"")
        content_en[0] = content_en[0].replace('\ufeff',"")
    content = [[content_en[i],content_ro[i]] for i in range(len(content_ro))]
    data = pd.DataFrame(data=content)
    return data

# %%

# Transform the data into lists 
# of strings and apply some preprocessing 
# functions (see# the pp_en and pp_tg above)

def process_data(train_data, pp_en, pp_tg, apply_on='both'):
    
    # change "apply_on" parameter to apply preprocessing on only one
    # of the languages
    import os
    from pre_processor import PreProcess
    pp = PreProcess(train_data,pp_en,pp_tg)
    data_processed = pp.apply_preprocessing()
    
    # Split the sentences into words
    
    eng_sentences = [sent.split() for sent in data_processed[0]]
    target_sentences = [sent.split() for sent in data_processed[1]]
    processed_data = pd.DataFrame([[eng_sentences[i], target_sentences[i]] 
    for i in range(len(target_sentences))])
    return processed_data


#%%

# Get work frequencies and arrange them and 
# Form a vocabulary

def word_frequency(processed_data):

    eng_data = processed_data[0]
    trg_data = processed_data[1]

    eng_vocab = {}
    trg_vocab = {}

    # Form vocabulary for the english language
    for sent in eng_data:
        for word in sent:
            if word in eng_vocab: 
                eng_vocab[word] += 1
            else: 
                eng_vocab[word] = 1
        
    # Form vocabulary for the target language
    for sent in trg_data:
        for word in sent:
            if word in trg_vocab: 
                trg_vocab[word] += 1
            else:
                 trg_vocab[word] = 1

    return eng_vocab, trg_vocab


# %%
def word_indexer(vocab_en,vocab_tg,no_word=15000):

    word2index_en = {'SOS': 0, 'EOS':1}
    index2word_en = {0: 'SOS', 1: 'EOS'}

    word2index_tg = {'SOS' : 0, 'EOS':1}
    index2word_tg = {0: 'SOS', 1: 'EOS'}

    sorted_tg = sorted(vocab_tg,key=vocab_tg.__getitem__,reverse=True)
    sorted_en = sorted(vocab_en,key=vocab_en.__getitem__,reverse=True)


    for ind, key in enumerate(sorted_tg):

        if ind > no_word:
            break
        word2index_tg[key] = ind+2
        index2word_tg[ind+2] = key

    for ind, key in enumerate(sorted_en):

        if ind > no_word:
            break
        word2index_en[key] = ind+2
        index2word_en[ind+2] = key

    return word2index_en, index2word_en, word2index_tg, index2word_tg



# %%
# Define functions for training data

def indexesFromSentence(word2index, sentence):

    return [word2index[word] for word in sentence]


def tensorFromSentence(word2index, sentence):

    indexes = indexesFromSentence(word2index, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(en_sent,tg_sent,w2i_en,w2i_tg):

    input_tensor = tensorFromSentence(w2i_en, en_sent)
    target_tensor = tensorFromSentence(w2i_tg, tg_sent)
    return input_tensor, target_tensor

def pair_data(w2i_en, w2i_tg , processed_data):
    
    input_tensors = []
    output_tensors = []

    for ind, sent in enumerate(processed_data[0]):
        i, t = tensorsFromPair(sent,processed_data[1][ind],w2i_en,w2i_tg)
        if len(sent) < MAX_LENGTH-1 and len(processed_data[1][ind]) < MAX_LENGTH-1:
            input_tensors.append(i)
            output_tensors.append(t)
    print(len(input_tensors))
    print(len(output_tensors))
    return input_tensors, output_tensors

# %%
# Define the model

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# %%

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length




#%%
def trainIters(input_tensors, output_tensors,
                no_training_sentences,ps_epochs,encoder, decoder, n_iters, 
                print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    for _ in range(ps_epochs):
        batch = random.sample(range(0,no_training_sentences-1),n_iters)
        for iter in range(1, n_iters + 1):
            target_tensor = output_tensors[batch[iter-1]]
            input_tensor = input_tensors[batch[iter-1]]
            loss = train(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)

#%%



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def train_a_model():

    data = read_data('data/Lit/train.lit.en','data/Lit/train.lit.ro',[1])
    processed_data = process_data(data,pp_en,pp_tg)
    vocab_en, vocab_ro = word_frequency(processed_data)
    w2i_en, i2w_en, w2i_tg, i2w_tg = word_indexer(vocab_en,vocab_ro)
    input_tensors, output_tensors = pair_data(w2i_en,w2i_tg,processed_data)
    no_training_sentences = len(output_tensors)
    tg_language_words = len(w2i_tg)
    en_language_words = len(w2i_en)


    encoder1 = EncoderRNN(en_language_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size,tg_language_words, dropout_p=0.1).to(device)

    trainIters(input_tensors, output_tensors,no_training_sentences,ps_epochs,encoder1, attn_decoder1, 500, print_every=200)

    enc = "encoder.pt"
    dcd = "attn_decoder.pt"
    torch.save(encoder1, PATH+enc)
    torch.save(attn_decoder1, PATH+dcd)


#train_a_model()