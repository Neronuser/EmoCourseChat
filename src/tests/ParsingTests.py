from src.data.emotion_data_parsers import parse_hashtag_emotion_corpus, parse_crowdflower_emotion_corpus, \
    parse_affective_text_corpuses
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

    def test_parse_text_emotion_corpus(self):
        last_tweet = "@mopedronin bullet train from tokyo    the gf and i have been visiting japan since thursday  " \
                     "vacation/sightseeing    gaijin godzilla"
        last_label = "love"
        desired_length = 40000
        data = parse_crowdflower_emotion_corpus()
        self.assertEqual(desired_length, len(data))
        self.assertEqual(last_tweet, data[-1][0])
        self.assertEqual(last_label, data[-1][1])

    def test_parse_affective_text_corpus(self):
        trial_text = "Mortar assault leaves at least 18 dead"
        trial_label = 'sadness'
        test_text = "Baby born on turnpike after dad misses exit"
        test_label = 'surprise'
        data = parse_affective_text_corpuses()
        self.assertEqual(trial_text, data[0][0])
        self.assertEqual(trial_label, data[0][1])
        self.assertEqual(test_text, data[-1][0])
        self.assertEqual(test_label, data[-1][1])
