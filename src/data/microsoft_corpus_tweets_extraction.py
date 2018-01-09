import configparser
import csv
import time

import tweepy

MICROSOFT_TUNING_DATA = 'data/raw/dialogue/MSRSocialMediaConversationCorpus/twitter_ids.tuning.txt'
MICROSOFT_VALIDATION_DATA = 'data/raw/dialogue/MSRSocialMediaConversationCorpus/twitter_ids.validation.txt'
OUTPUT_FILE = 'data/processed/dialogue/microsoft.csv'


def extract_tweets(datapath, writer):
    with open(datapath) as tuning_file:
        for line in tuning_file:
            ids = line.split('\t')
            try:
                statuses = api.statuses_lookup(ids)
            except tweepy.RateLimitError:
                print("Rate limit exceeded")
                time.sleep(15 * 60)
                print("Resuming")
                statuses = api.statuses_lookup(ids)
            texts = [status.text.replace('\n', '') for status in statuses]
            if len(texts) > 1:
                writer.writerow(texts)


if __name__ == '__main__':
    """ https://www.microsoft.com/en-us/download/details.aspx?id=52375&from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2F6096d3da-0c3b-42fa-a480-646929aa06f1%2F """
    config = configparser.ConfigParser()
    config.read('config.ini')
    twitter_config = config['twitter']
    auth = tweepy.OAuthHandler(twitter_config['ConsumerToken'], twitter_config['ConsumerSecret'])
    auth.secure = True
    auth.set_access_token(twitter_config['AccessToken'], twitter_config['AccessSecret'])
    api = tweepy.API(auth)

    with open(OUTPUT_FILE, 'w') as output_file:
        writer = csv.writer(output_file, quoting=csv.QUOTE_MINIMAL)
        extract_tweets(MICROSOFT_TUNING_DATA, writer)
        extract_tweets(MICROSOFT_VALIDATION_DATA, writer)
