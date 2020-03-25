import project_utils as pu
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from spellchecker import SpellChecker

spell = SpellChecker()

def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    stopwords.words('english')
    filtered_words = [w for w in tokens if not w in stopwords.words('english')]
    return " ".join(filtered_words)

def tokenize_file(file_path):
    file = open(file_path, 'r')
    Lines = file.readlines()
    Lines_Processed = []
    total_tokens = 0
    for line in Lines:
        line = preprocess(line)
        tokens = word_tokenize(line)
        total_tokens += len(tokens)
        misspelled = spell.unknown(tokens)
        for word in misspelled:
            idx = tokens.index(word)
            correct_word = spell.correction(word)
            tokens[idx] = correct_word
        Lines_Processed.append(tokens)

    return Lines_Processed, total_tokens