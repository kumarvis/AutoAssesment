import project_utils as pu
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from spellchecker import SpellChecker

import tokenize_file as tkf



def auto_assesment():
    file_path = 'dummy_dataset.txt'
    llst_token = tkf.tokenize_file(file_path)
    for l_token in llst_token:
        print(l_token)


auto_assesment()