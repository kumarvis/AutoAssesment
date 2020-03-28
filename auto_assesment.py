import os
import tokenize_file as tkf
import word_2_feature as w2f
import numpy as np
from tokenize_file import TokenizeFile

import bcolz  # to process the data from Glove File
import pickle  # to dump and load pretrained glove vectors
import copy  # to make deepcopy of python lists and dictionaries
import operator
import numpy as np

def auto_assesment():
    file_path = 'dummy_dataset.txt'
    llst_token, total_tokens = tkf.tokenize_file(file_path)
    #total_tokens = 2000
    GtTokens_Matrix = np.empty((w2f.Embedding_Length, total_tokens), dtype=np.float32)

    token_col_count = 0
    for l_token in llst_token:
        for token in l_token:
            embedding = w2f.get_glove_embedings(token)
            #embedding.resize(w2f.Embedding_Length, 1)
            GtTokens_Matrix[:, token_col_count] = embedding
            print(embedding.shape)
            print('ff')

class Auto_Assesment:
    def __init__(self):
        self.GT = []
        self.Test = []
        self.TokenizeObj = TokenizeFile()
        self.load_gt = False
        self.GT_EmbeddingMat = None
        self.Test_EmbeddingMat = None
        self.glove
    def tokenize_file(self, file_path):
        ll_tokens, no_tokens = TokenizeFile.tokenize_file(file_path)
        return ll_tokens, no_tokens

    def load_embedding_dictionary(self):
        Word2Vec_Base_Path = '/home/shunya/Word2VecDB/'
        Word2Vec_GlovePath = os.path.join(Word2Vec_Base_Path, 'glove.6B/glove.6B.50d.txt')
        Vectors_Db_Path = os.path.join(Word2Vec_Base_Path, 'glove.6B/6B.50.dat')
        Words_Pkl_Path = os.path.join(Word2Vec_Base_Path, 'glove.6B/6B.50_words.pkl')
        Words_Index_Pkl_Path = os.path.join(Word2Vec_Base_Path, 'glove.6B/6B.50_idx.pkl.pkl')

        vectors = bcolz.open(Vectors_Db_Path)[:]
        words = pickle.load(open(Words_Pkl_Path, 'rb'))
        word2idx = pickle.load(open(Words_Index_Pkl_Path, 'rb'))
        self.glove = {w: vectors[word2idx[w]] for w in words}


    def run(self):
        if self.load_gt == False:
            self.load_embedding_dictionary()




auto_assesment()