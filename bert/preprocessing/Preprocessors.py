import sys
import os
import json
# regex library
import re

from nltk.stem import PorterStemmer
import preprocessor as p

CONTRACTiONS = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "dunno": "do not know",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "ill" : "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    "dot": "do not",
    "lil": "little",
    "tmr": "tomorrow",
    "ur": "you are",
    "im": "i am",
    "idk": "i do not know",
    "rt": "retweet",
    "cuz": "because",
    "jk": "just kidding",
    "asap": "as soon as possible",
    "btw": "by the way",
    "plz": "please",
    "kinda": "kind of",
    "u": "you",
    "r": "are",
    "<3": "$LOVE$",
    "<": "",
    ">": "",
    "cnt": "cannot",
    "till": "until",
    "outta": "out of",
    "omg": "oh my god",
    "gr8": "great",
    "hv": "have",
    "thx": "thanks",
    "bout": "about",
    "dont": "do not",
    "bday": "birthday",
    "&": "and",
    "wanna": "want to",
    "m": "i am",
    "xd": "$SMILEY$",
    ":-d": "$SMILEY$",
    "bk": "back"
}

class TweetProcessor(object):
    """
    pre-process and clean the tweets (works for training and test data)
    source: https://towardsdatascience.com/basic-tweet-preprocessing-in-python-efd8360d529ehttps://towardsdatascience.com/basic-tweet-preprocessing-in-python-efd8360d529e
    https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908
    """

    def __init__(self):
        """
        case 1: source = train_pos, source2= train_neg
        case 2: source = test_data
        """
        #self.stopwords = [str(line).replace("\n", "") for line in open(os.getcwd()+"/src/helpers/stopwords.txt").readlines()]
        self.dictionary = json.load(open(os.getcwd() + "/src/helpers/data.json"))

    def handle_emojis(self, tweet):
        for positive in self.dictionary["POS_EMOJi"]:
            regex = positive["regex"]
            tweet = re.sub(regex, positive["replacement"], tweet)
        for negative in self.dictionary["NEG_EMOJi"]:
            regex = negative["regex"]
            tweet = re.sub(regex, negative["replacement"], tweet)
        return tweet

    def preprocess_word(self, word):
        # remove punctuation
        word = word.strip('\'"?!,.():;')
        for entry in self.dictionary["WORD_CLEANiNG"]:
            word = re.sub(entry["regex"], entry["replacement"], word)
        return word

    def preprocess_tweet(self, tweet):
        for entry in self.dictionary["TWEET_CLEANiNG"]:
            tweet = re.sub(entry["regex"], entry["replacement"], tweet)
        return tweet

    def is_valid_word(self, word):
        for validity in self.dictionary["WORD_VALiDiTY"]:
            if re.search(validity["regex"], word) is not None:
                return True
        return False

    def stem_word(self, tweet):
        output = []
        stemmer = PorterStemmer()
        for word in tweet.split():
            output.append(stemmer.stem(word))
        return ' '.join(output)

    def remove_contractions(self, tweet):
        output = []
        for word in tweet.split():
            if word in CONTRACTiONS:
                word = CONTRACTiONS[word]
            output.append(word)
        return ' '.join(output)



    def clean(self, tweet):
        output = []
        # preprocess tweet as a whole
        # tweet = self.preprocess_tweet(tweet)
        # strip space
        tweet = tweet.strip(' "\'')
        # replace emojis with either POS_EMOJi or NEG_EMOJi
        # tweet = self.handle_emojis(tweet)
        tweet = tweet.strip()
        # tweet = p.tokenize(tweet)
        tweet = self.remove_contractions(tweet)
        for word in tweet.split():
            word = re.sub("(?:(h|a)*(?:aha)+(h|a)?|(?:l+o+)+l+)", "$LAUGH$", word)
            word = re.sub(r"(.)\1\1", r"\1", word)
            output.append(word)
        # tweet = self.stem_word(tweet)
        return " ".join(output)
        # words = tweet.split()
        # for word in words:
        #     # pre-process word
        #     word = self.preprocess_word(word)
        #     if self.is_valid_word(word): # and word not in self.stopwords:
        #         processed_tweet.append(word)
        # # return tweet as string
        # return ' '.join(processed_tweet)

    def process(self, tweet):
        tweet = self.clean(tweet)
        return tweet


if __name__ == '__main__':
    pass

