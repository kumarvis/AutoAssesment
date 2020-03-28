from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from spellchecker import SpellChecker

class TokenizeFile:
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.spell = SpellChecker()

    def preprocess(self, sentence):
        sentence = sentence.lower()
        tokens = self.tokenizer.tokenize(sentence)
        stopwords.words('english')
        filtered_words = [w for w in tokens if not w in stopwords.words('english')]
        return " ".join(filtered_words)

    def tokenize_file(self, file_path):
        file = open(file_path, 'r')
        Lines = file.readlines()
        Lines_Processed = []
        total_tokens = 0
        for line in Lines:
            line = self.preprocess(line)
            tokens = word_tokenize(line)
            total_tokens += len(tokens)
            misspelled = self.spell.unknown(tokens)
            for word in misspelled:
                idx = tokens.index(word)
                correct_word = self.spell.correction(word)
                tokens[idx] = correct_word
            Lines_Processed.append(tokens)

        return Lines_Processed, total_tokens