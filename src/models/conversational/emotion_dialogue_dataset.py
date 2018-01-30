import csv
import logging
import os

import torch
import torchtext
from torchtext import data

from src.models.conversational.fields import EncodedSentenceField
from src.models.conversational.utils import Vocabulary, EOS_INDEX, APP_NAME, SOS_INDEX, PAD_INDEX
from src.utils import preprocess

EMOTION_FIELD_NAME = 'emo'
RESPONSE_FIELD_NAME = 'trg'
UTTERANCE_FIELD_NAME = 'src'


class EmotionDialogueDataset(data.Dataset):
    """Dialogue dataset, compatible with torch.data.Dataset and torchtext.data.Dataset."""

    def __init__(self, corpus_path, save_dir, max_sentence_length, max_vocab_words, transform=None, **kwargs):
        """Load, preprocess and cache data.

        Args:
            corpus_path (str): Path to the original unpreprocessed dataset.
            save_dir (str): Path to the cache directory.
            max_sentence_length (int): Maximum sentence length: number of words.
            max_vocab_words (int): Number of words to prune vocabulary to.
            transform (Optional[unc]): Per line preprocessing function. Defaults to None.
            **kwargs: Any additional Dataset parameters,
                see https://github.com/pytorch/text/blob/master/torchtext/data/dataset.py.

        """
        self.corpus_path = corpus_path
        self.save_dir = save_dir
        self.max_sentence_length = max_sentence_length
        self.max_vocab_words = max_vocab_words
        self.transform = transform
        self.corpus_name = corpus_path.split('/')[-1].split('.')[0]
        self.logger = logging.getLogger(APP_NAME + '.Dataset')
        try:
            logging.info("Start loading training data ...")
            self.vocabulary = torch.load(os.path.join(save_dir, 'training_data', self.corpus_name, 'vocabulary.tar'))
            self.emotion_vocabulary = torch.load(
                os.path.join(save_dir, 'training_data', self.corpus_name, 'emotion_vocabulary.tar'))
            self.triplets = torch.load(os.path.join(save_dir, 'training_data', self.corpus_name, 'triplets.tar'))
        except FileNotFoundError:
            self.logger.info("Saved data not found, start preparing training data ...")
            self.vocabulary, self.emotion_vocabulary, self.triplets = self.prepare_data()

        fields = [
            (UTTERANCE_FIELD_NAME,
             EncodedSentenceField(sequential=True, pad_token=PAD_INDEX, include_lengths=True, batch_first=True)),
            (RESPONSE_FIELD_NAME, EncodedSentenceField(sequential=True, pad_token=PAD_INDEX, batch_first=True)),
            (EMOTION_FIELD_NAME, EncodedSentenceField(sequential=False))]
        self.logger.info("Start converting to Examples")
        examples = []
        for src_line, trg_line, emotion in self.triplets:
            examples.append(data.Example.fromlist([src_line, trg_line, emotion], fields))
        del self.triplets
        super(EmotionDialogueDataset, self).__init__(examples, fields, **kwargs)

    def read_data(self):
        """Read raw data, preprocess it, build vocabularies.

        Returns:
            (Vocabulary, Vocabulary, ([int], [int], int)): Source, target vocabularies,
                encoded utterance, response and emotion triplets.

        """
        self.logger.info("Reading lines...")
        triplets = []
        vocabulary = Vocabulary(start_end_tokens=True, unique=1842343)
        emotion_vocabulary = Vocabulary()

        with open(self.corpus_path) as corpus_handle:
            reader = csv.reader(corpus_handle)
            for line_number, (utterance, response, emotion) in enumerate(reader):
                prep_utterance = preprocess(utterance)
                split_utterance = prep_utterance.split(' ')
                if len(split_utterance) >= self.max_sentence_length:
                    continue

                prep_response = preprocess(response)
                split_response = prep_response.split(' ')
                if len(split_response) >= self.max_sentence_length:
                    continue

                for word in split_utterance:
                    vocabulary.add_word(word)

                for word in split_response:
                    vocabulary.add_word(word)

                emotion_vocabulary.add_word(emotion)

                triplets.append((prep_utterance, prep_response, emotion))

                if line_number % 100000 == 0 and line_number != 0:
                    self.logger.info("Still reading, %d lines already read" % line_number)

        self.logger.info("Full vocabulary size: %d, pruning to %d" % (vocabulary.n_words, self.max_vocab_words))
        vocabulary.prune(self.max_vocab_words)
        self.logger.info("Encoding training data")
        encoded_triplets = []
        for utterance, response, emotion in triplets:
            encoded_utterance = vocabulary.encode(utterance.split())
            if not encoded_utterance:
                continue

            encoded_response = vocabulary.encode(response.split())
            if not encoded_response:
                continue
            encoded_response = [SOS_INDEX] + encoded_response
            encoded_response.append(EOS_INDEX)

            encoded_triplets.append((encoded_utterance, encoded_response, emotion_vocabulary.word2index[emotion]))
        return vocabulary, emotion_vocabulary, encoded_triplets

    def prepare_data(self):
        """Read, preprocess and cache raw dataset.

        Returns:
            (Vocabulary, Vocabulary, ([int], [int], int)): Source, target vocabularies,
                encoded utterance, response and emotion triplets.
        """
        vocabulary, emotion_vocabulary, triplets = self.read_data()
        self.logger.info("Read {!s} sentence triplets".format(len(triplets)))
        self.logger.info("Counted words: %d" % vocabulary.n_words)
        directory = os.path.join(self.save_dir, 'training_data', self.corpus_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(vocabulary, os.path.join(directory, '{!s}.tar'.format('vocabulary')))
        torch.save(emotion_vocabulary, os.path.join(directory, '{!s}.tar'.format('emotion_vocabulary')))
        torch.save(triplets, os.path.join(directory, '{!s}.tar'.format('triplets')))
        return vocabulary, emotion_vocabulary, triplets

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.examples)

    def __getitem__(self, idx):
        """Get an example from the dataset by index."""
        sample = self.examples[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
