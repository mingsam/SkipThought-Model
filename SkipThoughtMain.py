
from data_readin import data_readin
from freq_revise import freq_revise
from dataloader import WordEmbeddingDataset
from EmbeddingModel import  EmbeddingModel
import torch
import torch.nn as n
import torch.utils.data as tud
from torch.nn.parameter import Parameter

import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

ENABLE_GPU = torch.cuda.is_available()

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if ENABLE_GPU:
    torch.cuda.manual_seed(53113)
    device = torch.device("cuda:0")

# 设定一些超参数
K = 100         # number of negative samples
C = 3           # nearby words threshold
NUM_EPOCHS = 2  # The number of epochs of training
MAX_VOCAB_SIZE = 30000  # the vocabulary size
BATCH_SIZE = 128        # the batch size
LEARNING_RATE = 0.2     # the initial learning rate
EMBEDDING_SIZE = 100
LOG_FILE = "word-embedding.log"


text, vocab, idx_to_word, word_to_idx = data_readin("text8.train.txt", MAX_VOCAB_SIZE)
word_freqs, word_counts = freq_revise(vocab)
dataset = WordEmbeddingDataset(text, word_to_idx,idx_to_word, word_freqs, word_counts, len(idx_to_word), C, K)
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


model = EmbeddingModel(len(idx_to_word), EMBEDDING_SIZE)
if ENABLE_GPU:
    model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
for e in range(NUM_EPOCHS):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):

        # TODO
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        if ENABLE_GPU:
            input_labels = input_labels.to(device)
            pos_labels = pos_labels.to(device)
            neg_labels = neg_labels.to(device)
        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            with open(LOG_FILE, "a") as fout:
                fout.write("epoch: {}, iter: {}, loss: {}\n".format(e, i, loss.item()))
                print("epoch: {}, iter: {}, loss: {}".format(e, i, loss.item()))
