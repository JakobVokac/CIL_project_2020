import sys
import os
import json
# regex library
import re


class TweetProcessor(object):
    """
    pre-process and clean the tweets (works for training and test data1)
    source: https://towardsdatascience.com/basic-tweet-preprocessing-in-python-efd8360d529ehttps://towardsdatascience.com/basic-tweet-preprocessing-in-python-efd8360d529e
    """

    def __init__(self):
        """
        case 1: source = train_pos, source2= train_neg
        case 2: source = test_data
        """
        #self.stopwords = [str(line).replace("\n", "") for line in open(os.getcwd()+"/src/helpers/stopwords.txt").readlines()]
        self.dictionary = json.load(open(os.getcwd() + "/src/helpers/data1.json"))

    def handle_emojis(self, tweet):
        for positive in self.dictionary["POS_EMOJI"]:
            regex = positive["regex"]
            tweet = re.sub(regex, positive["replacement"], tweet)
        for negative in self.dictionary["NEG_EMOJI"]:
            regex = negative["regex"]
            tweet = re.sub(regex, negative["replacement"], tweet)
        return tweet

    def preprocess_word(self, word):
        # remove punctuation
        word = word.strip('\'"?!,.():;')
        for entry in self.dictionary["WORD_CLEANING"]:
            word = re.sub(entry["regex"], entry["replacement"], word)
        return word

    def preprocess_tweet(self, tweet):
        for entry in self.dictionary["TWEET_CLEANING"]:
            tweet = re.sub(entry["regex"], entry["replacement"], tweet)
        return tweet

    def is_valid_word(self, word):
        for validity in self.dictionary["WORD_VALIDITY"]:
            if re.search(validity["regex"], word) is not None:
                return True
        return False

    def clean(self, tweet):
        processed_tweet = []
        # preprocess tweet as a whole
        tweet = self.preprocess_tweet(tweet)
        # strip space
        tweet = tweet.strip(' "\'')
        # replace emojis with either POS_EMOJI or NEG_EMOJI
        tweet = self.handle_emojis(tweet)
        words = tweet.split()
        for word in words:
            # pre-process word
            word = self.preprocess_word(word)
            if self.is_valid_word(word): # and word not in self.stopwords:
                processed_tweet.append(word)
        # return tweet as string
        return ' '.join(processed_tweet)

    def process(self, tweet):
        tweet = self.clean(tweet)
        return tweet


if __name__ == '__main__':
    pass

