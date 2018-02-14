import logging

import torch

from src.models.conversational.checkpoint import Checkpoint
from src.models.conversational.emotion_model import EmotionSeq2seq, EmotionTopKDecoder
from src.models.conversational.predictor import Predictor
from src.models.conversational.utils import APP_NAME
from src.models.courses.recommender import Recommender


class EmoryChatBot(object):

    def __init__(self, checkpoint_path, vocabulary_path, emotion_vocabulary_path, beam_size=20):
        self.logger = logging.getLogger(APP_NAME + ".EmoryChatBot")
        self.recommender = Recommender()
        self.logger.info("Loading checkpoint from {}".format(checkpoint_path))
        vocabulary = torch.load(vocabulary_path)
        emotion_vocabulary = torch.load(emotion_vocabulary_path)
        checkpoint = Checkpoint.load(checkpoint_path)
        seq2seq = checkpoint.model
        beam_search = EmotionSeq2seq(seq2seq.encoder, EmotionTopKDecoder(seq2seq.decoder, beam_size))
        predictor = Predictor(beam_search, vocabulary, emotion_vocabulary=emotion_vocabulary)

        seq = "how are you ?".split()
        self.logger.info("Happy: " + " ".join(predictor.predict(seq, 'happiness')))
        self.logger.info("Angry: " + " ".join(predictor.predict(seq, 'anger')))
        self.logger.info("Sad: " + " ".join(predictor.predict(seq, 'sadness')))
        self.logger.info("Neutral: " + " ".join(predictor.predict(seq, 'neutral')))
        self.logger.info("Love: " + " ".join(predictor.predict(seq, 'love')))
