import numpy as np
import torch

from constants import *

# add start and end of sequence characters and space

vocab: list[str] = ARABIC_LETTERS + [' '] + ARABIC_PUNCTUATIONS + ['.'] + ENGLISH_PUNCTUATIONS + ['\u200f'] + ['']

# characters to index and vice versa
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# create one hot encoding for each character
hot_encoding = torch.eye(len(vocab))

if __name__ == '__main__':
    print("Vocab:", vocab)
    print("Char2idx:", char2idx)
    print("Idx2char:", idx2char)
    print("Hot encoding:", hot_encoding)