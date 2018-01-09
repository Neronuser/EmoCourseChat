import pandas as pd

from src.utils import split_list_pairs

CONVERSATIONS_FILE = 'data/raw/dialogue/cornell movie-dialogs corpus/movie_conversations.txt'
LINES_FILE = 'data/raw/dialogue/cornell movie-dialogs corpus/movie_lines.txt'
OUTPUT_FILE = 'data/processed/dialogue/movies.csv'

if __name__ == '__main__':
    """ https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html """
    conversations_df = pd.read_csv(CONVERSATIONS_FILE, sep=' \+\+\+\$\+\+\+ ',
                                   names=['character1ID', 'character2ID', 'movieID', 'utterances'])
    lines_df = pd.read_csv(LINES_FILE, sep=' \+\+\+\$\+\+\+ ',
                           names=['lineID', 'characterID', 'movieID', 'characterName', 'text'])
    dialogues_utterances = conversations_df['utterances'].apply(lambda x: eval(x))
    dialogues_two_lines = dialogues_utterances.apply(split_list_pairs).apply(pd.Series).unstack().reset_index().dropna()
    dialogue_ids = dialogues_two_lines[0].apply(pd.Series)
    first_unrolled = pd.merge(dialogue_ids, lines_df, left_on=[0], right_on=['lineID'])
    second_unrolled = pd.merge(first_unrolled, lines_df, left_on=[1], right_on=['lineID'], suffixes=['1', '2'])
    clean_df = second_unrolled[['text1', 'text2']]
    clean_df.to_csv(OUTPUT_FILE, index=False, header=False)
