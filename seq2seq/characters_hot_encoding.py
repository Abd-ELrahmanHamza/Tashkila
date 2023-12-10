import numpy as np
import pyarabic.araby as araby
import re

regex = re.compile(r'.')

# Define the vocabulary of characters
vocab = regex.findall(araby.LETTERS)

# add start and end of sequence characters and space
vocab.append('<s>')
vocab.append(' ')
vocab.append('</s>')

# characters to index and vice versa
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# create one hot encoding for each character
hot_encoding = np.eye(len(vocab), dtype='int32')

if __name__ == '__main__':
    print("Vocab:", vocab)
    print("Char2idx:", char2idx)
    print("Idx2char:", idx2char)
    print("Hot encoding:", hot_encoding)