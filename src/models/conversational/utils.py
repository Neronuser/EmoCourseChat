import itertools

import numpy as np

EOS = "EOS"

SOS = "SOS"

PAD = "PAD"

APP_NAME = "training_app"
SOS_INDEX = 0
EOS_INDEX = 1
PAD_INDEX = 2


class Vocabulary:
    def __init__(self, name, start_end_tokens=False, unique=-1):
        self.name = name
        self.unique = unique
        if start_end_tokens:
            self.word2index = {SOS: SOS_INDEX, EOS: EOS_INDEX, PAD: PAD_INDEX}
            self.index2count = [0, 0, 0]
            if unique == -1:
                self.index2word = [SOS, EOS, PAD]
            else:
                self.index2word = np.empty(unique, dtype="U14")
                self.index2word[0] = SOS
                self.index2word[1] = EOS
                self.index2word[2] = PAD
            self.n_words = 3  # Count SOS and EOS
        else:
            if unique == -1:
                self.index2word = []
            else:
                self.index2word = np.empty(unique, dtype="U14")
            self.word2index = {}
            self.index2count = []
            self.n_words = 0

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2count.append(1)
            if self.unique == -1:
                self.index2word.append(word)
            else:
                self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.index2count[self.word2index[word]] += 1

    def prune(self, max_words):
        if self.unique == -1:
            self.index2word = np.array(self.index2word)
        sort_indices = np.argsort(self.index2count)
        top_words = self.index2word[sort_indices][-max_words + 3:]
        index2word = np.empty(max_words, dtype='U14')
        word2index = {"SOS": SOS_INDEX, "EOS": EOS_INDEX, "PAD": PAD_INDEX}
        index2word[SOS_INDEX] = 'SOS'
        index2word[EOS_INDEX] = 'EOS'
        index2word[PAD_INDEX] = 'PAD'
        for i, word in enumerate(top_words):
            word2index[word] = 3 + i
            index2word[3 + i] = word
        self.index2word = index2word
        self.word2index = word2index
        self.n_words = max_words
        del self.index2count

    def encode(self, word_list):
        return [self.word2index[word] for word in word_list if word in self.word2index]