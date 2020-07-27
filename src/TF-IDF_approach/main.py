from autocorrect import Speller
from collections import Counter
import numpy as np
from itertools import chain
from tqdm import tqdm


spell = Speller()

with open('train_neg.txt') as neg:
  neg_list = neg.readlines()

with open('train_pos.txt') as pos:
  pos_list = pos.readlines()

with open('vocab_cut.txt') as voc:
  voc_list = voc.readlines()

neg_list_merged = [x.split() for x in neg_list]
pos_list_merged = [x.split() for x in pos_list]

vocab = dict()
with open('vocab_cut.txt') as f:
  for idx, line in enumerate(f):
    vocab[line.strip()] = idx

tokenized_pos = []
token_list_pos = []
tokenized_neg = []
token_list_neg = []

for line in pos_list:
    tokens = [vocab.get(t, -1) for t in line.strip().split()]
    tokens = [t for t in tokens if t >= 0]
    tokenized_pos.append(tokens)
    for t in tokens:
      token_list_pos.append(t)

for line in neg_list:
    tokens = [vocab.get(t, -1) for t in line.strip().split()]
    tokens = [t for t in tokens if t >= 0]
    tokenized_neg.append(tokens)
    for t in tokens:
      token_list_neg.append(t)


def term_frequency(term, doc):
  return doc.count(term)

def inv_doc_frequency(term, corpus):
  return np.log(len(corpus)/(1 + len([x for x in corpus if x.count(term) > 0])))

def tf_idf(term, doc, corpus):
  return term_frequency(term, doc) * inv_doc_frequency(term, corpus)


#Simple word count per positive and negative ( corpus1 + , corpus2 - )
def create_freq_dict(vocab, corpus1, corpus2):
  freq_dict = dict()
  freq_dict_pos = dict()
  freq_dict_neg = dict()
  corpus1 = chain.from_iterable(corpus1)
  corpus2 = chain.from_iterable(corpus2)  
  
  for item in tqdm(vocab.values()):
    freq_dict[item] = 0
  
  freq_dict_pos = freq_dict.copy()
  freq_dict_neg = freq_dict.copy()
    
  for token in tqdm(corpus1):
    freq_dict[token] += 1
    freq_dict_pos[token] += 1
  
  for token in tqdm(corpus2):
    freq_dict[token] += 1
    freq_dict_neg[token] += 1

  
  for key in freq_dict.keys():

    pos = freq_dict_pos[key]
    neg = freq_dict_neg[key]
    if pos - neg == 0:
      freq_dict[key] = 0
    else:
      freq_dict[key] /= pos - neg
    
  return freq_dict

#TODO: implement tfidf, but more efficiently
#TF-IDF dict for all words (in: vocab, corpus1 +, corpus2 -; out: dict{ word: (pos_tf-idf, neg_tf-idf)}
def create_tfidf_dict(vocab, corpus1, corpus2):
  tfidf_dict = dict()
  
  corpus1 = list(chain.from_iterable(corpus1))
  corpus2 = list(chain.from_iterable(corpus2))
  
  for item in tqdm(vocab.values()):
    pos_score = tf_idf(item, corpus1, [corpus1, corpus2])
    neg_score = tf_idf(item, corpus2, [corpus1, corpus2])

    tfidf_dict[item] = (pos_score, neg_score)
  
  return tfidf_dict

#90/10 split test with simple frequency count
size_pos = len(tokenized_pos)
size_neg = len(tokenized_neg)
eval_size_pos = size_pos // 10
train_size_pos = size_pos - eval_size_pos
eval_size_neg = size_neg // 10
train_size_neg = size_neg - eval_size_neg

freq_dict = create_freq_dict(vocab, tokenized_pos[:train_size_pos], tokenized_neg[:train_size_neg])

#tfidf_dict = create_tfidf_dict(vocab, tokenized_pos[:train_size_pos], tokenized_neg[:train_size_neg])

num_test_total = len(tokenized_pos[train_size_neg:]) + len(tokenized_neg[train_size_neg:]) 
num_test_correct = 0
print("Pos examples:")
count = 0
for i, doc in enumerate(tokenized_pos[train_size_pos:]):
  val = sum([freq_dict[word] for word in doc])
  if val > 0:
    if count < 10:
      count += 1
      print(pos_list[train_size_neg + i])
    num_test_correct += 1
print('Neg examples:')  
count = 0
for i, doc in enumerate(tokenized_neg[train_size_neg:]):
  val = sum([freq_dict[word] for word in doc])
  if val < 0:
    if count < 10:
      count += 1
      print(pos_list[train_size_neg + i])
    num_test_correct += 1
    
print("Total tested:", num_test_total)
print("Correctly classified:", num_test_correct)
print("Simple count acc: %.3f"%(num_test_correct/num_test_total))


#End results:
#
# TF-IDF doesn't make much sense since the corpus only consists of 2 documents
#
# A simple + - word count (multiplied times 2 if word only appears in positive or negative part) gives a low 0.668 accuracy with sample data.
#
# Taking the log of the absolute and multiplying by the sign function imporves only to 0.671.
#
# Taking the total occurence and dividing it by the difference gets you 0.581.
