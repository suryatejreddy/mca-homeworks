#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import *
from nltk.corpus import stopwords
from collections import defaultdict
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pickle
from torch.utils.tensorboard import SummaryWriter
import os


# In[2]:


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.layer = nn.Linear(context_size * embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.layer(embeds))
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

# In[3]:


def load_essentials():
    f = open("misc.pkl","rb")
    to_pickle_data = pickle.load(f)
    f.close()
    return to_pickle_data

vocab_size, vocabulary, word_to_num, num_to_word, training_data = load_essentials()
words = random.sample(vocabulary, 1000)


# In[4]:


def process_file(filename):
    model = Word2Vec(vocab_size, 300, 4)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    checkpoint = torch.load("backup/epoch1iteration10000.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()
    word_vectors = []
    for word in words:
        word_vector = model.embeddings(torch.tensor(word_to_num[word])).detach().numpy()
        word_vector.resize((300,1))
        word_vectors.append(word_vector)
    word_vectors = np.concatenate(word_vectors, axis = 1).T
    return word_vectors
    

def magic():
    writer = SummaryWriter()
    all_files = os.listdir("backup/")
    for file_ in tqdm(all_files):
        pos = file_.find(".pth")
        if pos > 0:
            name_of_epoch = file_[:pos]
            word_vectors = process_file(file_)
            writer.add_embedding(word_vectors, metadata = words, tag=name_of_epoch)
    
    writer.close()

magic()

