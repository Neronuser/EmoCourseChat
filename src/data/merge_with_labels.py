import csv

from textacy.text_utils import detect_language

from src.utils import preprocess

UBUNTU_CORPUS = 'data/processed/dialogue/ubuntu.csv'
MICROSOFT_CORPUS = 'data/processed/dialogue/microsoft.csv'
MOVIES_CORPUS = 'data/processed/dialogue/movies.csv'
REDDIT_CORPUS = 'data/processed/dialogue/reddit.csv'
LABELS_FILE = 'data/processed/dialogue/fasttext_classified.csv'
OUTPUT_FILE = 'data/processed/dialogue/full_dialogues_labeled.csv'


def extract_responses(filepath, writer, labels_reader):
    with open(filepath) as input_file:
        reader = csv.reader(input_file, quoting=csv.QUOTE_MINIMAL)
        deleted = "deleted"
        for line in reader:
            if (deleted not in line[0]) and (deleted not in line[1]):
                preprocessed_line = preprocess(line[1])
                try:
                    if detect_language(preprocessed_line) == 'en':
                        writer.writerow([line[0], line[1], labels_reader.__next__().strip()])
                except ValueError:
                    continue


if __name__ == '__main__':
    with open(OUTPUT_FILE, 'w') as output_file, open(LABELS_FILE) as labels_file:
        writer = csv.writer(output_file)
        extract_responses(UBUNTU_CORPUS, writer, labels_file)
        extract_responses(MICROSOFT_CORPUS, writer, labels_file)
        extract_responses(MOVIES_CORPUS, writer, labels_file)
        extract_responses(REDDIT_CORPUS, writer, labels_file)
