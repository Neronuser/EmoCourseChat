import csv

from textacy.text_utils import detect_language

from src.utils import preprocess

UBUNTU_CORPUS = 'data/processed/dialogue/ubuntu.csv'
MICROSOFT_CORPUS = 'data/processed/dialogue/microsoft.csv'
MOVIES_CORPUS = 'data/processed/dialogue/movies.csv'
REDDIT_CORPUS = 'data/processed/dialogue/reddit.csv'
OUTPUT_FILE = 'data/processed/dialogue/fasttext.csv'


def extract_responses(filepath, writer):
    with open(filepath) as input_file:
        reader = csv.reader(input_file, quoting=csv.QUOTE_MINIMAL)
        deleted = "deleted"
        for line in reader:
            if (deleted not in line[0]) and (deleted not in line[1]):
                preprocessed_line = preprocess(line[1])
                try:
                    if detect_language(preprocessed_line) == 'en':
                        writer.writerow([preprocessed_line])
                except ValueError:
                    continue


if __name__ == '__main__':
    with open(OUTPUT_FILE, 'w') as output_file:
        writer = csv.writer(output_file)
        extract_responses(UBUNTU_CORPUS, writer)
        extract_responses(MICROSOFT_CORPUS, writer)
        extract_responses(MOVIES_CORPUS, writer)
        extract_responses(REDDIT_CORPUS, writer)
