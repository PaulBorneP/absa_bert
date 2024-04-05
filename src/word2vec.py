import io
import numpy as np
import spacy
import re
import string 
import inflect
from spellchecker import SpellChecker


class Word2Vec:
    def __init__(self, file_name: str, nmax=150000):
        self.word2vec = {}
        self.load_wordvec(file_name, nmax)
        self.word2id = {w: i for i, w in enumerate(self.word2vec.keys())}
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.embeddings = np.array(list(self.word2vec.values()))
        self.parser = spacy.load("en_core_web_sm")
        self.dimension = -1
        self.unknown_token = np.random.rand(300)

    def load_wordvec(self, fname: str, nmax: int):
        """Load Word2Vec and map words to ID"""
        self.word2vec = {}
        with io.open(fname, encoding='utf-8') as file:
            next(file)
            for i, line in enumerate(file):
                word, vec = line.split(' ', 1)
                self.word2vec[word] = np.fromstring(vec, sep=' ')
                if i == (nmax - 1):
                    break
        self.dimension = self.word2vec[word].shape[0]
        print('Loaded %s word vectors of dim %s' % (len(self.word2vec),  self.dimension))

    @staticmethod
    def format_sentence(sentence: str) -> str:
        """ Simplification of sentences  """

        def replace_numbers_with_words(text):
            p = inflect.engine()
            words = text.split()
            for i, word in enumerate(words):
                if word.isdigit():
                    words[i] = p.number_to_words(word)
            return ' '.join(words)
        
        def correct_spelling(text):
            spell = SpellChecker()
            words = text.split()
            corrected_words = []
            for word in words:
                corrected_word = spell.correction(word)
                corrected_words.append(corrected_word)
            corrected_text = ' '.join(word for word in corrected_words if word is not None)
            return corrected_text
        
        sentence = sentence.lower() # lowercase
        # Replace contractions with their expanded forms
        contractions = {
            "'s": ' is',
            "n't": ' not',
            "'re": ' are',
            "'m": ' am',
            "'ve": ' have',
            "'ll": ' will',
            "'d": ' had',
            "kinda": 'kind of',
            "wanna": 'want to',
            "gonna": 'go to',
        }
        sentence = re.sub(r"\b(?:{})\b".format("|".join(map(re.escape, contractions.keys()))),
                        lambda match: contractions[match.group(0)], sentence)
        sentence = re.sub(r"[-!.,:;(')/]", ' ', sentence) # remove punctuation
        sentence = re.sub(r"[$]", 'price', sentence) # change to meaning
        sentence = correct_spelling(sentence) # correct spelling errors 
        sentence = re.sub(r'\b(?:the|a)\b', '', sentence) # remove the, a 
        #sentence = re.sub(r"\d+", ' ', sentence) # remove numbers 
        sentence = replace_numbers_with_words(sentence) # change numerical numbers to words
        sentence = sentence.strip() # remove leading and ending spaces
        sentence = re.sub(r"\s+", ' ', sentence) # multiple spaces into a single space, last to do 

        return sentence
    

    def encode_parse(self, sentences: list, idf=False) -> np.array:
        # Encode sentences into word embeddings
        sentences_embedded = []
        for sent in sentences:
            sent = self.format_sentence(sent)
            words_weights = []
            words_embedded = []
            for word in self.parser(sent):
                if word.pos_ in ['ADJ', 'VERB', 'NOUN', 'PROPN', 'INTJ']:
                    str_word = str(word)
                    try:
                        words_embedded.append(self.word2vec[str_word])
                        words_weights.append(1 if idf is False else idf[str_word])
                    except KeyError:
                        words_embedded.append(self.unknown_token)  # Use unknown token for unknown words
                        words_weights.append(1)  # Use weight of 1 for unknown words
            if len(words_embedded) > 0:
                sentences_embedded.append(np.average(words_embedded, weights=words_weights, axis=0))
            else:
                sentences_embedded.append(0.2 * np.random.rand(300) - 0.1)
        return np.array(sentences_embedded)
