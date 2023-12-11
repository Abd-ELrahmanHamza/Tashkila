import numpy as np
import torch
from constants import *

# Define the vocabulary of harakat and taskeel and tanween

NON_HARAKA = '$'

all_harakat = [
    FATHA,
    DAMMA,
    KASRA,
    SUKUN,
    FATHATAN,
    DAMMATAN,
    KASRATAN,
    SHADDA,
    SHADDA + FATHA,
    SHADDA + DAMMA,
    SHADDA + KASRA,
    SHADDA + FATHATAN,
    SHADDA + DAMMATAN,
    SHADDA + KASRATAN,
    NON_HARAKA
]

# harakat to index and vice versa
harakat2idx = {u:i for i, u in enumerate(all_harakat)}
idx2harakat = np.array(all_harakat)

# create one hot encoding for each harakat
hot_encoding = torch.eye(len(all_harakat))

if __name__ == '__main__':
    print(idx2harakat.shape)
    print("All harakat:", all_harakat)
    print("Harakat2idx:", harakat2idx)
    print("Idx2harakat:", idx2harakat)
    print("Hot encoding:", hot_encoding)
    