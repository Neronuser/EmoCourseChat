import logging

import torch

from src.models.conversational.checkpoint import Checkpoint
from src.models.conversational.emotion_model import EmotionSeq2seq, EmotionTopKDecoder
from src.models.conversational.predictor import Predictor
from src.models.conversational.utils import APP_NAME
from src.models.courses.recommender import Recommender
from src.utils import preprocess


class EmoCourseChat(object):

    def __init__(self, checkpoint_path, vocabulary_path, emotion_vocabulary_path, word2vec_path, beam_size=20, threshold=0.5):
        self.threshold = threshold
        self.logger = logging.getLogger(APP_NAME + ".EmoryChatBot")
        self.recommender = Recommender(word2vec_path)
        self.logger.info("Loading checkpoint from {}".format(checkpoint_path))
        vocabulary = torch.load(vocabulary_path)
        emotion_vocabulary = torch.load(emotion_vocabulary_path)
        checkpoint = Checkpoint.load(checkpoint_path)
        seq2seq = checkpoint.model
        beam_search = EmotionSeq2seq(seq2seq.encoder, EmotionTopKDecoder(seq2seq.decoder, beam_size))
        self.predictor = Predictor(beam_search, vocabulary, emotion_vocabulary=emotion_vocabulary)

    def respond(self, text):
        text, emotion = text.rsplit("#")
        recommended = self.recommender.recommend(text, threshold=self.threshold)
        if recommended is None:
            text = preprocess(text)
            return " ".join(self.predictor.predict(text.split(), emotion.lower())[:-1])
        return "Try " + recommended.title + " " + recommended.link
