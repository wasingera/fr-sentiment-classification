import os, pickle
import spacy
import pandas as pd
from collections import Counter

class Vocab:
    def __init__(self, max_words=10000, min_freq=5, special_tokens=[], default_token='[UNK]'):
        self.max_words = max_words
        self.min_freq = min_freq
        self.special_tokens = special_tokens
        self.word2idx = {}
        self.idx2word = {}
        self.default_token = default_token

        self.nlp = spacy.blank('fr')
        self.nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
        self.nlp.initialize()

    def build_vocab_from_file(self, file, out_file):
        if os.path.exists(out_file):
            print('Found vocab file, loading...')
            self.load(out_file)
            return

        print("Building vocab...")

        df = pd.read_csv(file)
        vocab = Counter()
        
        for text in df['text']:
            vocab.update([token.lemma_ for token in self.nlp(text.lower())])

        words = self.special_tokens + [word for word, freq in vocab.items() if freq >= self.min_freq]
        words = words[:self.max_words]

        for i, word in enumerate(words):
            self.word2idx[word] = i
            self.idx2word[i] = word

        print("Vocab size:", len(self.word2idx))

        self.save(out_file)

    def __len__(self):
        return len(self.word2idx)

    def __getitem__(self, key):
        return self.word2idx.get(key, self.word2idx[self.default_token])

    def numericalize_text(self, text):
        text = [token.lemma_ for token in self.nlp(text.lower())]
        return [self[word] for word in text]

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

        print("Saved vocab to", file)

    def load(self, file):
        with open(file, 'rb') as f:
            vocab = pickle.load(f)
            self.__dict__.update(vocab.__dict__)

        print("Loaded vocab from", file)
