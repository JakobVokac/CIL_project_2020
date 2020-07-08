import json
import os

# pos = open(os.getcwd()+"/src/data/train_pos.txt", "r")
# neg = open(os.getcwd()+"/src/data/train_neg.txt", "r")
# output = open(os.getcwd()+"/src/data/train.json", "w+")
#
# for i, line in enumerate(pos.readlines()):
#     tweet = json.dumps({"tweet": line, "prediction": 1})
#     output.write(tweet+"\n")
#
# for i, line in enumerate(neg.readlines()):
#     tweet = json.dumps({"tweet": line, "prediction": 0})
#     output.write(tweet+"\n")

from torchtext.data import Field, TabularDataset, BucketIterator

tokenize = lambda x: x.split()

tweet = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
prediction = Field(sequential=False, use_vocab=False)

fields = {'tweet': ('t', tweet), 'prediction': ('p', prediction)}

train_data = TabularDataset.splits(
    path=os.getcwd()+"/src/data",
    train='train.json',
    format='json',
    fields=fields
)