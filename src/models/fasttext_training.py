import csv
import subprocess

import textacy
from textacy.text_utils import detect_language

from src.utils import preprocess

if __name__ == '__main__':
    EMOTION_DATAPATH = 'data/processed/emotions_full.csv'
    FASTTEXT_FULL_FILE = 'data/processed/fasttext_full.txt'
    MODEL_PATH = 'models/emotion_classification/fasttext/model'
    label_prefix = '__label__'
    texts = []
    labels = []
    with open(EMOTION_DATAPATH) as data_file:
        reader = csv.reader(data_file, quoting=csv.QUOTE_MINIMAL)
        reader.__next__()
        for i, line in enumerate(reader):
            preprocessed_line = preprocess(line[1])
            if detect_language(preprocessed_line) == 'en':
                doc = textacy.Doc(preprocessed_line, lang='en_core_web_lg')
                texts.append(doc)
                labels.append(line[2])

    with open(FASTTEXT_FULL_FILE, 'w') as input_file:
        for x, y in zip(texts, labels):
            input_file.write(' , '.join([label_prefix + str(y), x.text.replace('\n', '')]) + '\n')

    # Hypertuned by fasttext_hypertuning.py
    dim = 300
    lr = 0.1
    epoch = 10
    word_ngrams = 1
    min_count = 1
    thread = 6
    lr_update_rate = 100000
    neg = 50
    subprocess.call(['./fastText-0.1.0/fasttext', 'supervised', '-input', FASTTEXT_FULL_FILE,
                     '-output', MODEL_PATH, '-dim', str(dim), '-lr', str(lr), '-epoch', str(epoch),
                     '-label', label_prefix, '-wordNgrams', str(word_ngrams), '-minCount', str(min_count),
                     '-thread', str(thread), '-lrUpdateRate', str(lr_update_rate), '-neg', str(neg)])
