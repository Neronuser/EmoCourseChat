import numpy as np

EOS = "EOS"
SOS = "SOS"
PAD = "PAD"

APP_NAME = "training_app"
SOS_INDEX = 0
EOS_INDEX = 1
PAD_INDEX = 2


class Vocabulary:
    """Language/Category vocabulary: word->index and index->word maps."""

    def __init__(self, start_end_tokens=False, unique=-1):
        """Initialize an empty vocabulary.

        Args:
            start_end_tokens (bool): Add start, end and pad tokens into the vocabulary.
            unique (Optional[int]): The number of unique words(if known, else -1). Defaults to -1.

        """
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
            self.n_words = 3
        else:
            if unique == -1:
                self.index2word = []
            else:
                self.index2word = np.empty(unique, dtype="U14")
            self.word2index = {}
            self.index2count = []
            self.n_words = 0

    def add_word(self, word):
        """Try adding `word` to the vocabulary. Increase its count.

        Args:
            word (str): Target word.

        """
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
        """Leave most common `max_words` in the vocabulary.

        Args:
            max_words (int): Number of most common words to leave in the vocabulary.

        """
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
        """Encode a list of words according to the vocabulary.

        Args:
            word_list (list(str)): Source text sequence.

        Returns:
            list(int): Encoded sequence.

        """
        return [self.word2index[word] for word in word_list if word in self.word2index]
