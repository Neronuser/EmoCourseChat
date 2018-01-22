import itertools

import numpy as np

APP_NAME = "training_app"
SOS_INDEX = 0
EOS_INDEX = 1


class Vocabulary:
    def __init__(self, name, start_end_tokens=False):
        self.name = name
        if start_end_tokens:
            self.word2index = {"SOS": SOS_INDEX, "EOS": EOS_INDEX}
            self.index2count = [0, 0]
            self.index2word = ["SOS", "EOS"]
            self.n_words = 2  # Count SOS and EOS
        else:
            self.word2index = {}
            self.index2count = []
            self.index2word = []
            self.n_words = 0

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2count.append(1)
            self.index2word.append(word)
            self.n_words += 1
        else:
            self.index2count[self.word2index[word]] += 1

    def prune(self, max_words):
        self.index2word = np.array(self.index2word)
        sort_indices = np.argsort(self.index2count)
        top_words = self.index2word[sort_indices][-max_words+2:]
        index2word = np.empty(max_words, dtype='U14')
        word2index = {"SOS": SOS_INDEX, "EOS": EOS_INDEX}
        index2word[SOS_INDEX] = 'SOS'
        index2word[EOS_INDEX] = 'EOS'
        for i, word in enumerate(top_words):
            word2index[word] = 2 + i
            index2word[2 + i] = word
        self.index2word = index2word
        self.word2index = word2index
        self.n_words = max_words
        del self.index2count

    def encode(self, word_list):
        return [self.word2index[word] for word in word_list if word in self.word2index]


def pad(l, fill=EOS_INDEX):
    return list(itertools.zip_longest(*l, fillvalue=fill))


def binary_mask(l, value=EOS_INDEX):
    m = []
    for i in range(len(l)):
        m.append([])
        for j in range(len(l[i])):
            if l[i][j] == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m
