import json
import os

from torchtext.data import Field, TabularDataset, BucketIterator


class TweetDataset(object):

    def __init__(self):
        self.pos = open(os.getcwd() + "/src/data/train_pos.txt", "r")
        self.neg = open(os.getcwd() + "/src/data/train_neg.txt", "r")
        self.output = open(os.getcwd() + "/bert/train.json", "w+")
        self.output_test = open(os.getcwd() + "/bert/test.json", "w+")

    def create_json(self):
        for i, line in enumerate(self.pos.readlines()):
            tweet = json.dumps({"tweet": line, "prediction": 1})
            self.output.write(tweet + "\n")
            if i < 1000:
                self.output_test.write(tweet + "\n")

        for i, line in enumerate(self.neg.readlines()):
            tweet = json.dumps({"tweet": line, "prediction": 0})
            self.output.write(tweet + "\n")
            if i < 1000:
                self.output_test.write(tweet + "\n")


        self.output_test.close()
        self.output.close()
        self.pos.close()
        self.neg.close()

    def torchtext(self):
        tokenize = lambda x: x.split()
        tweet = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
        prediction = Field(sequential=False, use_vocab=False)

        fields = {'tweet': ('t', tweet), 'prediction': ('p', prediction)}
        train_data, test_data = TabularDataset.splits(
            path=os.getcwd() + "/bert",
            train='train.json',
            test='test.json',
            format='json',
            fields=fields
        )
        return train_data, test_data


if __name__ == '__main__':
    dataset = TweetDataset()
    # dataset.create_json()
    train_data, test_data = dataset.torchtext()