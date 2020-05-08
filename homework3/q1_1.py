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


def load_data():
    abc = LazyCorpusLoader(
        "abc",
        PlaintextCorpusReader,
        r"(?!\.).*\.txt",
        encoding=[("science", "latin_1"), ("rural", "utf8")],
    )
    
    raw = abc.sents()
    sentences = []
    
    stopwords_ = list(stopwords.words('english'))
    final_stopwords = {w : 1 for w in stopwords_}
    
    for s in raw:
        words = []
        for w in s:
            if w.isalpha() and w not in final_stopwords:
                words.append(w.lower())
        sentences.append(words)
    
    word_counts = defaultdict(int)
    for sentence in sentences:
        for word in sentence:
            word_counts[word] += 1
            
    vocabulary = list(word_counts.keys())
    vocabulary.extend(["<START>","<END>"])
    vocab_size = len(vocabulary)
    word_to_num = {word : n for n, word in enumerate(vocabulary)}
    num_to_word = {n : word for n, word in enumerate(vocabulary)}
    
    sums = [-2,-1,1,2]
    training_data = []
    for sentence in tqdm(sentences):
        length = len(sentence)
        for cur_index in range(length):
            cur_word = sentence[cur_index]
            context_vector = []
            for diff in sums:
                index = cur_index + diff
                if index >= 0  and index < length:
                    context_word = sentence[index]
                    context_vector.append(context_word)
            if len(context_vector) == 4:
                training_data.append([context_vector, cur_word])
            
    return vocab_size, vocabulary, word_to_num, num_to_word, training_data

vocab_size, vocabulary, word_to_num, num_to_word, training_data = load_data()


# In[42]:


# to_pickle_data = [vocab_size, vocabulary, word_to_num, num_to_word, training_data]
# import pickle
# f = open("misc.pkl","wb")
# pickle.dump(to_pickle_data, f)
# f.close()


# In[ ]:


losses = []
loss_function = nn.NLLLoss()
model = Word2Vec(vocab_size, 300, 4)
optimizer = optim.SGD(model.parameters(), lr=0.001)


# In[ ]:


for epoch in range(1,4):
    total_loss = 0
    iterations = 0
    for context, target in tqdm(training_data):
        context_idxs = torch.tensor([word_to_num[w] for w in context], dtype=torch.long)
        model.zero_grad()
        log_probs = model(context_idxs)
        loss = loss_function(log_probs, torch.tensor([word_to_num[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        iterations += 1
        if (iterations % 10000 == 0):
            print("Epoch #" + str(epoch) +  ", Iteration #" + str(iterations) +   " Loss : " + str(loss.item()))
            torch.save({
                'epoch': epoch,
                'iteration' : iterations,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, "backup/epoch" + str(epoch) + "iteration" + str(iterations) + ".pth")
