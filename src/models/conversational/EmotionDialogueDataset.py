import csv
import os

import torch
from torch.utils.data import Dataset

from src.models.conversational.utils import Vocabulary, EOS_INDEX
from src.utils import preprocess


class EmotionDialogueDataset(Dataset):

    def __init__(self, corpus, save_dir, max_length, max_words, transform=None):
        self.corpus = corpus
        self.save_dir = save_dir
        self.max_length = max_length
        self.max_words = max_words
        self.transform = transform
        self.corpus_name = corpus.split('/')[-1].split('.')[0]
        try:
            print("Start loading training data ...")
            self.vocabulary = torch.load(os.path.join(save_dir, 'training_data', self.corpus_name, 'vocabulary.tar'))
            self.emotion_vocabulary = torch.load(
                os.path.join(save_dir, 'training_data', self.corpus_name, 'emotion_vocabulary.tar'))
            self.triplets = torch.load(os.path.join(save_dir, 'training_data', self.corpus_name, 'triplets.tar'))
        except FileNotFoundError:
            print("Saved data not found, start preparing training data ...")
            self.vocabulary, self.emotion_vocabulary, self.triplets = self.prepare_data()

    def read_data(self):
        print("Reading lines...")
        triplets = []
        vocabulary = Vocabulary(self.corpus_name)
        emotion_vocabulary = Vocabulary(self.corpus_name + '_emotion')

        with open(self.corpus) as corpus_handle:
            reader = csv.reader(corpus_handle)
            for utterance, response, emotion in reader:
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
        vocabulary.prune(self.max_words)
        encoded_triplets = []
        for utterance, response, emotion in triplets:
            encoded_utterance = vocabulary.encode(utterance)
            if not encoded_utterance:
                continue
            encoded_utterance.append(EOS_INDEX)

            encoded_response = vocabulary.encode(response)
            if not encoded_response:
                continue
            encoded_response.append(EOS_INDEX)

            encoded_triplets.append((encoded_utterance, encoded_response, emotion_vocabulary.encode(emotion)))
        return vocabulary, emotion_vocabulary, encoded_triplets

    def prepare_data(self):
        vocabulary, emotion_vocabulary, triplets = self.read_data()
        print("Read {!s} sentence triplets".format(len(triplets)))
        print("Counted words:", vocabulary.n_words)
        directory = os.path.join(self.save_dir, 'training_data', self.corpus_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(vocabulary, os.path.join(directory, '{!s}.tar'.format('vocabulary')))
        torch.save(emotion_vocabulary, os.path.join(directory, '{!s}.tar'.format('emotion_vocabulary')))
        torch.save(triplets, os.path.join(directory, '{!s}.tar'.format('triplets')))
        return vocabulary, emotion_vocabulary, triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        sample = self.triplets[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class PrepareForSeq2Seq(object):
    def __init__(self, max_length, reverse):
        self.max_length = max_length
        self.reverse = reverse

    def __call__(self, sample):
        if self.reverse:
            sample = (sample[1], sample[0])
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        input_batch, output_batch = [], []
        for i in range(len(pair_batch)):
            input_batch.append(pair_batch[i][0])
            output_batch.append(pair_batch[i][1])
        input, lengths = input_var(input_batch, voc)
        output, mask, max_target_len = output_var(output_batch, voc)
        return input, lengths, output, mask, max_target_len