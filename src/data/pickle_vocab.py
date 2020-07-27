#!/usr/bin/env python3
import pickle


def main():
    """
    save vocab in a pickle file
    pickle files are a serializer format (object) to save data in python
    :return:
    """
    vocab = dict()
    with open('dataset/vocab_cut.txt') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open('dataset/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
