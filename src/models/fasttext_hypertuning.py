import csv
import subprocess
from itertools import product

import textacy
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from textacy.text_utils import detect_language

from src.utils import preprocess


if __name__ == '__main__':
    EMOTION_DATAPATH = 'data/processed/emotions_full.csv'
    raw_data = []
    with open(EMOTION_DATAPATH) as data_file:
        reader = csv.reader(data_file, quoting=csv.QUOTE_MINIMAL)
        reader.__next__()
        for i, line in enumerate(reader):
            preprocessed_line = preprocess(line[1])
            if detect_language(preprocessed_line) == 'en':
                doc = textacy.Doc(preprocessed_line, lang='en_core_web_lg')
                raw_data.append((doc, line[2]))

    texts, labels = zip(*raw_data)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    x_train, x_test, y_train, y_test = \
        train_test_split(texts, encoded_labels, shuffle=True, stratify=encoded_labels,
                         random_state=42, test_size=0.2)

    MODELS_TEST_RESULTS = 'reports/tune_test_scores.csv'

    FASTTEXT_INPUT_FILE = 'data/processed/fasttext_input.txt'
    FASTTEXT_TEST_FILE = 'data/processed/fasttext_test.txt'
    FASTTEXT_FULL_FILE = 'data/processed/fasttext_full.txt'
    MODEL_PATH = 'models/fasttext/model'
    label_prefix = '__label__'
    with open(FASTTEXT_INPUT_FILE, 'w') as input_file:
        for x, y in zip(x_train, y_train):
            input_file.write(' , '.join([label_prefix + str(y), x.text]) + '\n')

    with open(FASTTEXT_TEST_FILE, 'w') as input_file:
        for x, y in zip(x_test, y_test):
            input_file.write(x.text.replace('\n', '') + '\n')

    tested_dims = [200, 300, 500]
    tested_lrs = [0.1, 0.01, 0.01]
    tested_epochs = [10, 20, 50]
    tested_min_counts = [1]
    lr_update_rates = [100, 100000, 1000000]
    negs = [5, 50, 100]
    word_ngrams = 1
    combinations = product(tested_dims, tested_lrs, tested_epochs, tested_min_counts, lr_update_rates, negs)

    thread = str(12)

    best_params = None
    best_score = 0
    n_combinations = len(tested_dims) * len(tested_lrs) * len(tested_epochs) * len(tested_min_counts) * \
                     len(lr_update_rates) * len(negs)
    for i, (dim, lr, epoch, min_count, lr_update_rate, neg) in enumerate(combinations):
        print("%d / %d" % (i, n_combinations))
        subprocess.call(['./fastText-0.1.0/fasttext', 'supervised', '-input', FASTTEXT_INPUT_FILE,
                         '-output', MODEL_PATH, '-dim', str(dim), '-lr', str(lr), '-epoch', str(epoch),
                         '-label', label_prefix, '-wordNgrams', str(word_ngrams), '-minCount', str(min_count),
                         '-thread', thread, '-lrUpdateRate', str(lr_update_rate), '-neg', str(neg)])
        test_preds = subprocess.check_output(['./fastText-0.1.0/fasttext', 'predict', MODEL_PATH + '.bin',
                                              FASTTEXT_TEST_FILE])
        preds = [int(pred[-1]) for pred in test_preds.decode("utf-8").split('\n') if pred != '']
        score = f1_score(y_test, preds, average='micro')
        accuracy = accuracy_score(y_test, preds)
        if best_score < score:
            best_score = score
            best_params = {"dim": dim, "lr": lr, "epochs": epoch, "min_count": min_count,
                           "lr_update_rate": lr_update_rate, "neg": neg, "accuracy": accuracy}

    with open(MODELS_TEST_RESULTS, "a") as test_scores_table:
        writer = csv.writer(test_scores_table, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["FT", best_score, '', str(best_params)])
