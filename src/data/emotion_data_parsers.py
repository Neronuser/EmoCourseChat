HASHTAG_EMOTION_CORPUS = 'data/raw/emotion/Jan9-2012-tweets-clean.txt'


def parse_hashtag_emotion_corpus():
    """Extract tweets and emotions from http://saifmohammad.com/WebDocs/Jan9-2012-tweets-clean.txt.zip

    Returns:
        ([(str, str)]): tweets and their emotion labels zipped
    """
    data = []
    with open(HASHTAG_EMOTION_CORPUS) as corpus:
        for line in corpus:
            # TODO filter non-English
            _, sentence_label_str = line.split(':', 1)
            preprocessed_sentence_label_str = sentence_label_str.replace('\t', '').replace('\n', '')
            print(preprocessed_sentence_label_str)
            sentence, label = preprocessed_sentence_label_str.rsplit('::', 1)
            data.append((sentence.strip(), label.strip()))
    return data
