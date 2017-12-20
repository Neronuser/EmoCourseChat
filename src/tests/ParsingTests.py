from src.data.emotion_data_parsers import parse_hashtag_emotion_corpus, parse_crowdflower_emotion_corpus, \
    parse_affective_text_corpuses, join_electoral_tweets_data, parse_wassa_data, parse_love_letters, \
    join_spudisc_datasets, parse_nrc_lexicon
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

    def test_parse_electoral_tweets_data(self):
        batch1_text = "I'm a #Republican, but if I have to hear my mom talk about #Romney one more time, I'm going to " \
                     "stab myself with my bayonet."
        batch1_label = 'disgust'
        batch2_text = " @GovGaryJohnson Will recognize marriage equality. #election2012 Vote Gary Johnson #tcot #tiot " \
                      "#tlot #Obama #Romney"
        batch2_label = 'like'
        data = join_electoral_tweets_data()
        self.assertEqual(batch1_text, data[0][0])
        self.assertEqual(batch1_label, data[0][1])
        self.assertEqual(batch2_text, data[-1][0])
        self.assertEqual(batch2_label, data[-1][1])

    def test_parse_wassa_data(self):
        data = parse_wassa_data()
        self.assertTrue(data)

    def test_love_letters(self):
        first_sentence = "If only I could have come up with the right words to describe the depth of this beautiful " \
                         "feeling that I have for you, I would have whispered them to you the first time we met."
        last_sentence = ". I will be eternally grateful for everything, Babe.\n"
        data = parse_love_letters()
        self.assertEqual(first_sentence, data[0][0])
        self.assertEqual(last_sentence, data[-1][0])

    def test_parse_spudisc_data(self):
        train_text = "Kevin Spacey truly plays a bone chilling character, almost just as legendary and chilling as " \
                      "Hannibal Lecter."
        train_label = 'Fear'
        test_text = "It is such an emotional movie that you can't keep from being caught up in what is happening."
        test_label = 'Interest'
        data = join_spudisc_datasets()
        self.assertEqual(train_text, data[0][0])
        self.assertEqual(train_label, data[0][1])
        self.assertEqual(test_text, data[-1][0])
        self.assertEqual(test_label, data[-1][1])

    def test_load_nrc_lexicon(self):
        emotion2words = parse_nrc_lexicon()
        self.assertIn('abduction', emotion2words['fear'])
        self.assertIn('wonderful', emotion2words['joy'])
