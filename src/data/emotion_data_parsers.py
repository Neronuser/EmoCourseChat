import csv
import os
from collections import defaultdict

import numpy as np
import spacy

COURTESY_SENTENCES_WORDS = 9

EMOTION_SCORE_THRESHOLD = 50
WASSA_THRESHOLD = 0.5

HASHTAG_EMOTION_CORPUS = 'data/raw/emotion/Jan9-2012-tweets-clean.txt'
CROWDFLOWER_EMOTION_CORPUS = 'data/raw/emotion/text_emotion.csv'
AFFECTIVETEXT_TRIAL_LABELS = 'data/raw/emotion/AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial' \
                             '.emotions.gold'
AFFECTIVETEXT_TRIAL_TEXT = 'data/raw/emotion/AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.xml'
AFFECTIVETEXT_TEST_LABELS = 'data/raw/emotion/AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test' \
                            '.emotions.gold'
AFFECTIVETEXT_TEST_TEXT = 'data/raw/emotion/AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.xml'
ELECTORAL_DATA_BATCH1 = 'data/raw/emotion/ElectoralTweetsData/Annotated-US2012-Election-Tweets/Questionnaire2/Batch1' \
                        '/AnnotatedTweets.txt'
ELECTORAL_DATA_BATCH2 = 'data/raw/emotion/ElectoralTweetsData/Annotated-US2012-Election-Tweets/Questionnaire2/Batch2' \
                        '/AnnotatedTweets.txt'
WASSA_FOLDER = 'data/raw/emotion/Wassa-2017'
LOVE_LETTERS = 'data/raw/emotion/LoveHateSuicide/love-letters.txt'
SPUDISK_TRAIN = 'data/raw/emotion/spudisc-emotion-classification-master/train.txt'
SPUDISK_TEST = 'data/raw/emotion/spudisc-emotion-classification-master/test.txt'
NRC_LEXICON = 'data/raw/emotion/NRC-Sentiment-Emotion-Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon' \
              '-Wordlevel-v0.92.txt'
FULL_DATASET = 'data/processed/emotions_initial.csv'


def parse_hashtag_emotion_corpus():
    """Extract tweets and emotions from http://saifmohammad.com/WebDocs/Jan9-2012-tweets-clean.txt.zip.

    Returns:
        ([(str, str)]): Tweets and their emotion labels zipped.
    """
    data = []
    with open(HASHTAG_EMOTION_CORPUS) as corpus:
        for line in corpus:
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
        ([(str, str)]): News headlines and their top scored emotion labels zipped.
    """
    data = parse_affective_text_corpus(AFFECTIVETEXT_TRIAL_TEXT, AFFECTIVETEXT_TRIAL_LABELS)
    data.extend(parse_affective_text_corpus(AFFECTIVETEXT_TEST_TEXT, AFFECTIVETEXT_TEST_LABELS))
    return data


def parse_affective_text_corpus(text_filepath, labels_filepath):
    """Extract news headlines and emotions from affectivetext dataset. http://web.eecs.umich.edu/~mihalcea/downloads.html

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


def join_electoral_tweets_data():
    """ Extract and join 2 batches of data from electoral tweets:
    http://saifmohammad.com/WebDocs/ElectoralTweetsData.zip

    Returns:
        ([(str, str)]): Tweets and their emotion labels zipped.
    """
    data = parse_electoral_tweets_data(ELECTORAL_DATA_BATCH1)
    data.extend(parse_electoral_tweets_data(ELECTORAL_DATA_BATCH2))
    return data


def parse_electoral_tweets_data(data_path):
    """Extract electoral tweets and emotions from the dataset.

    Args:
        data_path (str): Path to annotated tweets file.
    Returns:
        ([(str, str)]): Tweets and their emotion labels zipped.
    """
    data = []
    with open(data_path) as data_file:
        data_file.__next__()
        previous_row = ''
        for line in data_file:
            line = previous_row + ' ' + line
            row = line.rstrip('\t\n').split('\t')
            if len(row) < 29:
                # Some rows in the data file are improperly newlined
                previous_row = line
                continue
            else:
                previous_row = ''
            tweet = row[13]
            emotion = row[15]
            data.append((tweet, emotion))
    return data


def parse_wassa_data():
    """Extract tweets and their emotions from WASSA-2017 dataset:
    http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html
    Filter by WASSA_THRESHOLD intensity.

    Returns:
        ([(str, str)]): Tweets and emotion labels zipped.
    """
    data = []
    for filename in os.listdir(WASSA_FOLDER):
        if not filename.startswith('.'):
            with open(WASSA_FOLDER + '/' + filename) as data_file:
                reader = csv.reader(data_file, delimiter='\t')
                for row in reader:
                    if float(row[3]) > WASSA_THRESHOLD:
                        data.append((row[1], row[2]))
    return data


def parse_love_letters():
    """Extract love sentences from love letter collection from http://saifmohammad.com/WebDocs/LoveHateSuicide.tar.gz

    Returns:
        ([(str, str)]): Sentence, 'love' pairs.
    """
    data = []
    with open(LOVE_LETTERS) as letters_file:
        nlp = spacy.load('en')
        for line in letters_file:
            doc = nlp(line)
            if len(doc) <= COURTESY_SENTENCES_WORDS:
                continue
            for sentence in doc.sents:
                data.append((sentence.text, 'love'))
    return data


def join_spudisc_datasets():
    """Extract and join test/train imdb review parts annotated with emotions from
    https://github.com/NLeSC/spudisc-emotion-classification.

    Returns:
        ([(str, str)]): Review part, emotion.
    """
    data = parse_spudisc_dataset(SPUDISK_TRAIN)
    data.extend(parse_spudisc_dataset(SPUDISK_TEST))
    return data


def parse_spudisc_dataset(data_path):
    """Extract imdb review parts annotated with emotions

    Args:
        data_path (str): Path to annotated review file.
    Returns:
        ([(str, str)]): Review part, emotion.
    """
    data = []
    with open(data_path) as data_file:
        for line in data_file:
            sentence, label = line.rsplit(maxsplit=1)
            if label == 'None':
                continue
            new_label = label.split('_')[0]
            data.append((sentence, new_label))
    return data


def parse_nrc_lexicon():
    """Extract National Resource Council Canada emotion lexicon from http://saifmohammad.com/WebPages/lexicons.html

    Returns:
        {str: [str]} A defaultdict of emotion to list of associated words
    """
    emotion2words = defaultdict(list)
    with open(NRC_LEXICON) as lexicon_file:
        lexicon_file.__next__()
        for line in lexicon_file:
            word, emotion, associated = line.split()
            if associated == '1':
                emotion2words[emotion].append(word)
    return emotion2words


# TODO maybe add StanceDataset

def write_data(writer, data, name):
    """Write a particular dataset into the file

    Args:
        writer (csv.Writer): writer handle of output file
        data ([(str, str)]: text, label tuples
        name (str): dataset identification
    """
    for text, label in data:
        writer.writerow([text, label, name])


if __name__ == '__main__':
    extractors = [parse_hashtag_emotion_corpus, parse_crowdflower_emotion_corpus, parse_affective_text_corpuses,
                  join_electoral_tweets_data, parse_wassa_data, parse_love_letters, join_spudisc_datasets]
    names = ["hashtag_emotion", "crowdflower", "affective_text", "electoral_tweets", "wassa", "love_letters", "spudisc"]
    with open(FULL_DATASET, 'w') as full_data_file:
        writer = csv.writer(full_data_file, quoting=csv.QUOTE_MINIMAL)
        for extractor_function, name in zip(extractors, names):
            data = extractor_function()
            write_data(writer, data, name)
        emotion2words = parse_nrc_lexicon()
        for emotion, word_list in emotion2words.items():
            for word in word_list:
                writer.writerow((word, emotion, "nrc_lexicon"))
