import csv
import logging
import os

import torch
from torchtext import data

from src.models.conversational.fields import EncodedSentenceField
from src.models.conversational.utils import Vocabulary, EOS_INDEX, APP_NAME, SOS_INDEX
from src.utils import preprocess


class EmotionDialogueDataset(data.Dataset):

    def __init__(self, corpus, save_dir, max_length, max_words, transform=None, **kwargs):
        self.corpus = corpus
        self.save_dir = save_dir
        self.max_length = max_length
        self.max_words = max_words
        self.transform = transform
        self.corpus_name = corpus.split('/')[-1].split('.')[0]
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

        fields = [('src', EncodedSentenceField(sequential=True, eos_token=EOS_INDEX, include_lengths=True, batch_first=True)),
                  ('trg', EncodedSentenceField(sequential=True, eos_token=EOS_INDEX, batch_first=True)),
                  ('emo', data.RawField())]

        examples = []
        for src_line, trg_line, emotion in self.triplets:
            examples.append(data.Example.fromlist([src_line, trg_line, emotion], fields))
        super(EmotionDialogueDataset, self).__init__(examples, fields, **kwargs)

    def read_data(self):
        self.logger.info("Reading lines...")
        triplets = []
        vocabulary = Vocabulary(self.corpus_name, start_end_tokens=True)
        emotion_vocabulary = Vocabulary(self.corpus_name + '_emotion')

        with open(self.corpus) as corpus_handle:
            reader = csv.reader(corpus_handle)
            for line_number, (utterance, response, emotion) in enumerate(reader):
                prep_utterance = preprocess(utterance)
                split_utterance = prep_utterance.split(' ')
                if len(split_utterance) >= self.max_length:
                    continue

                prep_response = preprocess(response)
                split_response = prep_response.split(' ')
                if len(split_response) >= self.max_length:
                    continue

                for word in split_utterance:
                    vocabulary.add_word(word)

                for word in split_response:
                    vocabulary.add_word(word)

                emotion_vocabulary.add_word(emotion)

                triplets.append((split_utterance, split_response, emotion))

                # if line_number == 100000:
                #     break
        self.logger.info("Full vocabulary size: %d, pruning to %d" % (vocabulary.n_words, self.max_words))
        vocabulary.prune(self.max_words)
        self.logger.info("Encoding training data")
        encoded_triplets = []
        for utterance, response, emotion in triplets:
            encoded_utterance = vocabulary.encode(utterance)
            if not encoded_utterance:
                continue
            # encoded_utterance.append(EOS_INDEX)

            encoded_response = vocabulary.encode(response)
            if not encoded_response:
                continue
            encoded_response = [SOS_INDEX] + encoded_response
            encoded_response.append(EOS_INDEX)

            encoded_triplets.append((encoded_utterance, encoded_response, emotion_vocabulary.word2index[emotion]))
        return vocabulary, emotion_vocabulary, encoded_triplets

    def prepare_data(self):
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
        return len(self.examples)

    def __getitem__(self, idx):
        sample = self.examples[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
