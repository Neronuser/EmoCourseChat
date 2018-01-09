import pandas as pd

from src.utils import split_list_pairs

TRAIN_FILE = 'data/raw/dialogue/ubuntu-ranking-dataset-creator/src/train.csv'
TEST_FILE = 'data/raw/dialogue/ubuntu-ranking-dataset-creator/src/test.csv'
VALIDATION_FILE = 'data/raw/dialogue/ubuntu-ranking-dataset-creator/src/valid.csv'
OUTPUT_FILE = 'data/processed/dialogue/ubuntu.csv'



def parse_ubuntu_file(filename, label_column='Utterance'):
    train_df = pd.read_csv(filename, sep=',')
    train_df['Context'] = train_df['Context'].str.replace('__eou__', '')
    train_df[label_column] = train_df[label_column].str.replace('__eou__', '')
    text_df = train_df['Context'].str.cat(train_df[label_column], sep=" __eot__ ")
    text_df = text_df.str.split('__eot__')
    text_df = text_df.apply(lambda x: [i for i in x if i != '  '])
    text_df = text_df.apply(split_list_pairs).apply(pd.Series).unstack().reset_index().dropna()
    return text_df[0].apply(pd.Series)


if __name__ == '__main__':
    """ https://github.com/rkadlec/ubuntu-ranking-dataset-creator """
    train = parse_ubuntu_file(TRAIN_FILE)
    validation = parse_ubuntu_file(VALIDATION_FILE, label_column='Ground Truth Utterance')
    test = parse_ubuntu_file(TEST_FILE, label_column='Ground Truth Utterance')
    final_df = pd.concat([train, validation, test])
    final_df.to_csv(OUTPUT_FILE, index=False, header=False)
