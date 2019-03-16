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

import time
import math
import random
from data_extractor import read_data
from data_extractor import process_data
from data_extractor import word_frequency
from data_extractor import word_indexer
from data_extractor import indexesFromSentence
from data_extractor import tensorFromSentence
from data_extractor import tensorsFromPair
from data_extractor import pair_data
from data_extractor import asMinutes
from data_extractor import timeSince
from data_extractor import EncoderRNN
from data_extractor import DecoderRNN
from data_extractor import AttnDecoderRNN

# preprocessing to be applied. Small description for each of them
# 'ct' : removes punctuation from the end of the words
# 'cn' : makes makes numbers into '#'/'##'/.../'#####'
# 'gm' : english only! replaces short words with full ('ain't -> is not')
# 'lw' : lower case the words

pp_en = ['ct','cn','gm','lw']
pp_tg = ['ct','cn','lw']


# Define some constants
enc = "encoder.pt"
dcd = "attn_decoder.pt"
PATH = "NMT_Models/"

MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = read_data('data/Lit/train.lit.en','data/Lit/train.lit.ro',[1])
processed_data = process_data(data,pp_en,pp_tg)
vocab_en, vocab_ro = word_frequency(processed_data)
w2i_en, i2w_en, w2i_tg, i2w_tg = word_indexer(vocab_en,vocab_ro)
input_tensors, output_tensors = pair_data(w2i_en,w2i_tg,processed_data)
no_training_sentences = len(output_tensors)
tg_language_words = len(w2i_tg)
en_language_words = len(w2i_en)



teacher_forcing_ratio = 0.5
hidden_size = 256

encoder1 = torch.load(PATH+enc)
attn_decoder1 = torch.load(PATH+dcd)

def evaluate(w2i, encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(w2i, sentence.split())
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(i2w_tg[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

dec_w, _ = evaluate(w2i_en,encoder1,attn_decoder1,"i am the law")
print(dec_w)