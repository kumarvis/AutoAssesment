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

class Auto_Assesment:
    def __init__(self):
        self.ll_GT_Tokens = []
        self.GT_NoTokens = None
        self.Test = []
        self.TokenizeObj = TokenizeFile()
        self.initialize_gt = False
        self.GT_EmbeddingMat = None
        self.Test_EmbeddingMat = None
        self.glove = None
    def tokenize_file(self, file_path):
        ll_tokens, no_tokens = self.TokenizeObj.tokenize_file(file_path)
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

    def initialize(self, gt_file_path):
        if self.initialize_gt != True:
            self.load_embedding_dictionary()
            self.ll_GT_Tokens, self.GT_NoTokens = self.tokenize_file(gt_file_path)
            self.GT_EmbeddingMat= np.empty((w2f.Embedding_Length, self.GT_NoTokens), dtype=np.float32)

            token_col_count = 0
            for l_gt_tokens in self.ll_GT_Tokens:
                print('Loading GT Embedding')
                for token in l_gt_tokens:
                    embedding = w2f.get_glove_embedings(token)
                    self.GT_EmbeddingMat[:, token_col_count] = embedding
                    token_col_count += 1
        else:
            print('Init already done')
            self.initialize_gt = True

    def cosine_similarity(self, vA, vB):
        if np.linalg.norm(vA) == 0 or np.linalg.norm(vB) == 0:
            print('dd')
        score = np.dot(vA, vB) / (np.linalg.norm(vA) * np.linalg.norm(vB))
        return score

    def normalize_score(self, score):
        if score > 1.0:
            score = 1.0
        score = score * 10
        score = round(score, 3)
        return score
    def evaluate(self, test_file_path):
        ll_Test_Tokens, Test_NoTokens = self.tokenize_file(test_file_path)
        Test_EmbeddingMat = np.empty((w2f.Embedding_Length, Test_NoTokens), dtype=np.float32)

        token_col_count = 0
        for l_test_tokens in ll_Test_Tokens:
            print('Loading Test Embedding')
            for token in l_test_tokens:
                embedding = w2f.get_glove_embedings(token)
                Test_EmbeddingMat[:, token_col_count] = embedding
                token_col_count += 1

        total_test_score = 0
        no_tokens_test = Test_EmbeddingMat.shape[1]
        no_tokens_gt = self.GT_EmbeddingMat.shape[1]

        for tt in range(no_tokens_test):
            print(tt)
            vT = Test_EmbeddingMat[:, tt]
            max_similarity = 0
            for gg in range(no_tokens_gt):
                print(gg)
                vG = self.GT_EmbeddingMat[:, gg]
                score = self.cosine_similarity(vG, vT)
                if score > max_similarity:
                    max_similarity = score
            total_test_score += max_similarity
        total_test_score = total_test_score / no_tokens_test
        total_test_score = self.normalize_score(total_test_score)
        return total_test_score




gt_file_path = 'dummy_dataset.txt'
obj = Auto_Assesment()
obj.initialize(gt_file_path)
print('Score = ', obj.evaluate(gt_file_path))
