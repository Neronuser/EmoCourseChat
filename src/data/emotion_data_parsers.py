import csv
import numpy as np

EMOTION_SCORE_THRESHOLD = 50

HASHTAG_EMOTION_CORPUS = 'data/raw/emotion/Jan9-2012-tweets-clean.txt'
CROWDFLOWER_EMOTION_CORPUS = 'data/raw/emotion/text_emotion.csv'
AFFECTIVETEXT_TRIAL_LABELS = 'data/raw/emotion/AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial' \
                             '.emotions.gold'
AFFECTIVETEXT_TRIAL_TEXT = 'data/raw/emotion/AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.xml'
AFFECTIVETEXT_TEST_LABELS = 'data/raw/emotion/AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test' \
                             '.emotions.gold'
AFFECTIVETEXT_TEST_TEXT = 'data/raw/emotion/AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.xml'


def parse_hashtag_emotion_corpus():
    """Extract tweets and emotions from http://saifmohammad.com/WebDocs/Jan9-2012-tweets-clean.txt.zip.

    Returns:
        ([(str, str)]): Tweets and their emotion labels zipped.
    """
    data = []
    with open(HASHTAG_EMOTION_CORPUS) as corpus:
        for line in corpus:
            # TODO filter non-English
            _, sentence_label_str = line.split(':', 1)
            preprocessed_sentence_label_str = sentence_label_str.replace('\t', '').replace('\n', '')
            sentence, label = preprocessed_sentence_label_str.rsplit('::', 1)
            data.append((sentence.strip(), label.strip()))
    return data


def parse_crowdflower_emotion_corpus():
    """Extract tweets and emotions from https://www.crowdflower.com/wp-content/uploads/2016/07/text_emotion.csv.

    Returns:
        ([(str, str)]): Tweets and their emotion labels zipped.
    """
    data = []
    with open(CROWDFLOWER_EMOTION_CORPUS) as data_file:
        reader = csv.reader(data_file)
        reader.__next__()
        for line in reader:
            emotion = line[1]
            text = line[3]
            data.append((text, emotion))
    return data


def parse_affective_text_corpuses():
    """ Extract and merge trial and test data downloaded from
        http://web.eecs.umich.edu/~mihalcea/downloads/AffectiveText.Semeval.2007.tar.gz.

    Returns:
        ([(str, str)]): Tweets and their emotion labels zipped.
    """
    data = parse_affective_text_corpus(AFFECTIVETEXT_TRIAL_TEXT, AFFECTIVETEXT_TRIAL_LABELS)
    data.extend(parse_affective_text_corpus(AFFECTIVETEXT_TEST_TEXT, AFFECTIVETEXT_TEST_LABELS))
    return data


def parse_affective_text_corpus(text_filepath, labels_filepath):
    """Extract news headlines and emotions from affectivetext dataset

    Args:
        text_filepath (str): Path to affectivetext xml file with ids and texts.
        labels_filepath (str): Path to affectivetext emotions file.
    Returns:
        ([(str, str)]): News headlines and their top scored emotion labels zipped.
    """
    data = []
    columns = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    with open(text_filepath) as text_file, open(labels_filepath) as labels_file:
        text_file.__next__()
        for text_line, label_line in zip(text_file, labels_file):
            emotion_scores = np.fromstring(label_line, dtype='int', sep=' ')[1:]
            max_score_index = np.argmax(emotion_scores)
            if emotion_scores[max_score_index] <= EMOTION_SCORE_THRESHOLD:
                continue
            label = columns[max_score_index]
            text = text_line.split('>', 1)[1].rsplit('<', 1)[0]
            data.append((text, label))
    return data