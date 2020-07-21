import sys
import os
import json
# regex library
import re


class TweetProcessor(object):
    """
    pre-process and clean the tweets (works for training and test data)
    source: https://towardsdatascience.com/basic-tweet-preprocessing-in-python-efd8360d529ehttps://towardsdatascience.com/basic-tweet-preprocessing-in-python-efd8360d529e
    """

    def __init__(self, source, dest, source2=None):
        """
        case 1: source = train_pos, source2= train_neg
        case 2: source = test_data
        """
        #self.stopwords = [str(line).replace("\n", "") for line in open(os.getcwd()+"/src/helpers/stopwords.txt").readlines()]
        self.dictionary = json.load(open(os.getcwd() + "/src/helpers/data.json"))
        self.source = open(source, "r")
        if source2 is not None:
            self.neg = open(source2, "r")
        else:
            self.neg = None
        self.dest = open(dest, "w+")

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

    def preprocess(self):
        # distinguish case
        if self.neg is None:
            # Test
            files = [self.source]
        else:
            # Train
            files = [self.source, self.neg]
        tweet_id = 1
        for file in files:
            for tweet in file.readlines():
                tweet = self.clean(tweet)

                # write to dest
                if self.neg is not None:  # write labels
                    if file is self.source:  # label pos = 1
                        self.dest.write("{},{}\n".format(tweet.replace('\n', ''), str(1)))
                    else:  # label neg = 0
                        self.dest.write("{},{}\n".format(tweet.replace('\n', ''), str(0)))
                else:  # do not write labels
                    self.dest.write(tweet.replace('\n', '') + '\n')
                # increment id
                tweet_id += 1
            # close file
            file.close()
        # close dest file
        print('\nSaved processed tweets to: %s' % str(self.dest.name))
        self.dest.close()


if __name__ == '__main__':
    # pre-process training tweets
    train_pos_full = os.getcwd() + "/src/data/train_pos_full.txt"
    train_neg_full = os.getcwd() + "/src/data/train_neg_full.txt"
    train_processed_file_name = os.getcwd() + "/src/data/train_preprocessed_full.txt"
    train_cleaned = TweetProcessor(train_pos_full, train_processed_file_name, train_neg_full)
    train_cleaned.preprocess()

    # pre-process test tweets
    test_file_name = os.getcwd() + "/src/data/test_data.txt"
    test_processed_file_name = os.getcwd() + "/src/data/test_preprocessed.txt"
    test_cleaned = TweetProcessor(test_file_name, test_processed_file_name)
    test_cleaned.preprocess()

    exit(1)
