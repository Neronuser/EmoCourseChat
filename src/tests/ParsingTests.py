from src.data.emotion_data_parsers import parse_hashtag_emotion_corpus
import unittest

class ParsingTests(unittest.TestCase):

    def test_parse_hashtag_emotion_corpus(self):
        first_tweet = "Thinks that @melbahughes had a great 50th birthday party :)"
        first_label = "surprise"
        desired_length = 21051
        data = parse_hashtag_emotion_corpus()
        self.assertEqual(desired_length, len(data))
        self.assertEqual(first_tweet, data[0][0])
        self.assertEqual(first_label, data[0][1])
