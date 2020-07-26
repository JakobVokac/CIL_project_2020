import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def word_to_vector(context,vocab,vocab_size):

    out = np.zeros(vocab_size)
    out[vocab[context]] = 1.0

    return torch.tensor(out, dtype=torch.float)

class cbow(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super(cbow,self).__init__()

        self.context_mat = nn.Linear(vocab_size,embed_dim)
        self.word_mat = nn.Linear(embed_dim,vocab_size)
        self.softmax = nn.Softmax()

    def forward(self, context):
        h = self.context_mat(context)
        target = self.word_mat(h)
        output = self.softmax(target)
        return output

def main():

    text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit,
    sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
    nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in
    reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla
    pariatur. Excepteur sint occaecat cupidatat non proident, sunt in
    culpa qui officia deserunt mollit anim id est laborum.""".split()

    vocab_set = set(text)
    vocab = dict()
    for idx, word in enumerate(vocab_set):
        vocab[word] = idx
    # vocab = dict()
    # with open('vocab_cut.txt') as f:
    #     for idx, line in enumerate(f):
    #         vocab[line.strip()] = idx
    vocab_size = len(vocab)
    model = cbow(vocab_size, 10)

    optimizer = torch.optim.Adagrad(model.parameters(), lr = 0.1)

    loss_fn = nn.CrossEntropyLoss(reduction='none')

    loss = 100
    for epoch in range(50):
        print(loss)
        for context, target in list(zip(text[:-1],text[1:])):
            context_vector = word_to_vector(context,vocab,vocab_size)

            log_probs = model(context_vector).unsqueeze(0)
            # print(log_probs.size())
            # print(torch.tensor([vocab[target]],dtype=torch.long))
            loss = loss_fn(log_probs,torch.tensor([vocab[target]],dtype=torch.long))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for context, target in list(zip(text[:-1],text[1:])):
        context_vector = word_to_vector(context,vocab,vocab_size)

        preds = model(context_vector).unsqueeze(0)
        idx = torch.argmax(preds).numpy()
        for key, value in vocab.items():
            if value == idx:
                pred = key

        print(context," ", pred," (",target,")")

if __name__ == "__main__":
    main()
