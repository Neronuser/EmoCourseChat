import configparser
import pickle
import re
import string

import numpy as np
from numpy import dtype, float32 as REAL, fromstring
from scipy.spatial.distance import cosine, cdist
from textacy.preprocess import preprocess_text

CONFIG_PATH = 'config.ini'
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'

EMOJI_PATTERN = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
    "+", flags=re.UNICODE)
MENTIONS_PATTERN = re.compile(u"@[a-z]+")
HASHTAGS_PATTERN = re.compile(u"#[a-z]+")
MULTI_SPACES = re.compile(u" +")


def preprocess(text):
    new_text = preprocess_text(text, fix_unicode=True, lowercase=True, no_urls=True,
                               no_emails=True, no_phone_numbers=True, no_numbers=True,
                               no_currency_symbols=True, no_contractions=True,
                               no_accents=True)
    no_mentions_text = re.sub(MENTIONS_PATTERN, u"", new_text)
    no_hashtags_text = re.sub(HASHTAGS_PATTERN, u"", no_mentions_text)
    no_emojis_text = re.sub(EMOJI_PATTERN, u"", no_hashtags_text)
    separated_punctuation_text = no_emojis_text.translate(
        str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    no_multi_spaces_text = re.sub(MULTI_SPACES, u" ", separated_punctuation_text)
    return no_multi_spaces_text.strip()


def split_list_pairs(l):
    return [[l[i], l[i + 1]] for i in range(len(l) - 1)]


def parse_config(section):
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(CONFIG_PATH)
    return config[section]


def load_word2vec(w2v_path, max_words=None):
    with open(w2v_path, "rb") as fin:
        header = fin.readline()
        vocab_size, vector_size = map(int, header.split())
        if max_words:
            vocab_size = max_words
        i = 0
        word2id = {}
        id2word = np.empty(vocab_size, dtype="U12")
        word_embeddings = np.zeros((vocab_size, vector_size))
        binary_len = dtype(REAL).itemsize * vector_size
        for line_no in range(vocab_size):
            # mixed text and binary: read text first, then binary
            word = []
            while True:
                ch = fin.read(1)
                if ch == b' ':
                    break
                if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                    word.append(ch)
            word = b''.join(word).decode('utf-8')
            weights = fromstring(fin.read(binary_len), dtype=REAL)
            word2id[word] = i
            id2word[i] = word
            word_embeddings[i] = weights
            i += 1
    return word2id, id2word, word_embeddings


def closest_to_vector(vector, word_embeddings, k=10, threshold=0.5):
    distances = cdist(vector.reshape((1, vector.shape[0])), word_embeddings, metric='cosine')[0]
    top_similar = np.argsort(distances)[:k]
    if distances[top_similar[0]] > threshold:
        return None
    return top_similar


def save_object(obj, obj_path):
    with open(obj_path, 'wb') as obj_handle:
        pickle.dump(obj, obj_handle)


def load_object(obj_path):
    with open(obj_path, 'rb') as obj_handle:
        return pickle.load(obj_handle)
