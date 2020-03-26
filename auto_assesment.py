import tokenize_file as tkf
import word_2_feature as w2f
import numpy as np

def auto_assesment():
    file_path = 'dummy_dataset.txt'
    llst_token, total_tokens = tkf.tokenize_file(file_path)

    for l_token in llst_token:
        for token in l_token:
            embedding = w2f.get_glove_embedings(token)
            embedding.resize(1, w2f.Embedding_Length)
            print('ff')

auto_assesment()