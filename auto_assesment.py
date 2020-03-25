import project_utils as pu
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from spellchecker import SpellChecker

import tokenize_file as tkf
import word_2_feature as w2f


def auto_assesment():
    file_path = 'dummy_dataset.txt'
    llst_token, total_tokens = tkf.tokenize_file(file_path)

    for l_token in llst_token:
        for token in l_token:
            embedding = w2f.get_glove_embedings(token)
            print('ff')




auto_assesment()