from scipy.sparse import *
from scipy.sparse.linalg import svds
import numpy as np
import pickle
import random
from tqdm import tqdm
import json


def main():

    print("loading cooccurrence matrix")
    with open('cooc.pkl', 'rb') as f:
        cooc = pickle.load(f)

    print("{} nonzero entries".format(cooc.nnz))

    cooc_upcasted = cooc.asfptype()
    u, s, vt = svds(cooc_upcasted, k=20)

    print(u)
    print(s)


if __name__ == '__main__':
    main()
