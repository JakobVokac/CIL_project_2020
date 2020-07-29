import os
import glob
import io
import json
import sys

from torchtext import data

from preprocessing.Preprocessors import TweetProcessor

config = json.load(open("configurations/full.json", "r"))


class Twitter(data.Dataset):
    """
    torchtext datastructure for our tweets
    """
    name = 'twitter'
    urls = ""
    dirname = ""

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
        """Create an IMDB dataset instance given a path and fields.

        Arguments:
            path: Path to the dataset's highest level directory
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        tweets = open(os.getcwd()+"/data/"+config["train_file"], "r+")

        for i, tweet in enumerate(tweets.readlines()):
            json_obj = json.loads(tweet)
            examples.append(data.Example.fromlist([json_obj["tweet"], json_obj["prediction"]], fields))

        super(Twitter, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, root='',
               train='train', test='test', **kwargs):
        """Create dataset objects for splits of the IMDB dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: Root dataset storage directory. Default is '.data'.
            train: The directory that contains the training examples
            test: The directory that contains the test examples
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        return super(Twitter, cls).splits(
            root=root, text_field=text_field, label_field=label_field,
            train=train, validation=None, test=test, **kwargs)

    @classmethod
    def iters(cls, batch_size=1024, device=0, root='.data', vectors=None, **kwargs):
        """Create iterator objects for splits of the IMDB dataset.

        Arguments:
            batch_size: Batch_size
            device: Device to create batches on. Use - 1 for CPU and None for
                the currently active GPU device.
            root: The root directory that contains the imdb dataset subdirectory
            vectors: one of the available pretrained vectors or a list with each
                element one of the available pretrained vectors (see Vocab.load_vectors)

            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        train, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, test), batch_size=batch_size, device=device)



