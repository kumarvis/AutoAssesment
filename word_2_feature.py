import os
import unicodedata
import string
import re
import random
import bcolz  # to process the data from Glove File
import pickle  # to dump and load pretrained glove vectors
import copy  # to make deepcopy of python lists and dictionaries
import operator
import numpy as np
from pandas import DataFrame  # to visualize the glove word embeddings in form of DataFrame

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

Word2Vec_Base_Path = '/home/shunya/Word2VecDB/'
Word2Vec_GlovePath = os.path.join(Word2Vec_Base_Path, 'glove.6B/glove.6B.50d.txt')
Vectors_Db_Path = os.path.join(Word2Vec_Base_Path, 'glove.6B/6B.50.dat')
Words_Pkl_Path = os.path.join(Word2Vec_Base_Path, 'glove.6B/6B.50_words.pkl')
Words_Index_Pkl_Path = os.path.join(Word2Vec_Base_Path, 'glove.6B/6B.50_idx.pkl.pkl')

def dump_pkl():
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=Vectors_Db_Path, mode='w')

    word_no = 1
    with open(Word2Vec_GlovePath, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
            print('Word Number = ', word_no)
            word_no += 1

    vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=Vectors_Db_Path, mode='w')
    vectors.flush()
    pickle.dump(words, open(Words_Pkl_Path, 'wb'))
    pickle.dump(word2idx, open(Words_Index_Pkl_Path, 'wb'))

def get_glove_embedings(my_word):
    vectors = bcolz.open(Vectors_Db_Path)[:]
    words = pickle.load(open(Words_Pkl_Path, 'rb'))
    word2idx = pickle.load(open(Words_Index_Pkl_Path, 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}

    if my_word in glove.keys():
        embeding = glove[my_word]
    else:
        embeding = -1

    #word2idx = {k: v for k, v in sorted(word2idx.items(), key=operator.itemgetter(1))}
    return embeding
    print('ss')

#get_glove_embedings('fdf')
