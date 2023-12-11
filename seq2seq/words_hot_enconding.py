import numpy as np


def create_word_index(corpus):
    words = set(corpus)
    word_index = {word: idx for idx, word in enumerate(words)}
    return word_index


def one_hot_encode(word, word_index):
    encoded_word = np.zeros(len(word_index))
    encoded_word[word_index[word]] = 1
    return encoded_word
