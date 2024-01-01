"""
    This file is used to create a dataset of letters from the dataset of words in the dataset folder.
"""

from re import T
from uu import encode
from numpy import mean
import torch
from torch.utils.data import Dataset, DataLoader
from out_encoder import OutputEncoder
from text_encoder import TextEncoder
import numpy as np
from train_collections import *
from nltk.stem.isri import ISRIStemmer

input_file = 'clean_out/X.csv'
output_file = 'clean_out/Y.csv'
input_val = 'clean_out/X_val.csv'
output_val = 'clean_out/Y_val.csv'
stemmer = ISRIStemmer()


def read_data(input_file, output_file, verbose=False):
    X = []
    Y = []
    # read csv files
    with open(input_file, 'r', encoding="utf8") as f:
        X = f.readlines()
    with open(output_file, 'r', encoding="utf8") as f:
        Y = f.readlines()

    # remove the \n from the end of each line
    # make sure the number of lines in the two files are the same
    assert len(X) == len(Y)

    # split each line into a list of characters
    X = [x.strip().split('s') for x in X]
    Y = [y.strip().split('s') for y in Y]

    if verbose:
        print('Data read successfully.')
        print(f'Number of lines: {len(X)}')
        print(f'X: {X[0]}')
        print(f'Y: {Y[0]}')

    return X, Y


embedding_weights = np.load("embedding/embedding_weights.npy")


def get_word_embedding(word, word2idx):
    # print(word)
    if word in word2idx:
        return embedding_weights[word2idx[word]]
    else:
        return embedding_weights[word2idx["<UNK>"]]


# Average word embedding
def get_sentence_embedding(sentence, word2idx):
    return np.mean([get_word_embedding(stemmer.stem(word), word2idx) for word in sentence.split()], axis=0)


def find_width_99_percentile(data):
    # Flatten the list to get the lengths of all inner lists
    lengths = np.array([len(inner_list) for inner_list in data])

    # Calculate the 99th percentile
    return int(np.percentile(lengths, 99))


class LettersDataset(Dataset):
    """
        This class is used to create a dataset of letters from the dataset of words in the dataset folder.
    """

    def __init__(self, input_data_file='clean_out/X.csv', output_data_file='clean_out/Y.csv', val_mode=False, return_sent_emb=False, word2idx=None, device=torch.device('cpu'), special_tokens=[PAD_TOKEN, UNK_TOKEN], verbose=False):
        """
            This method is used to initialize the class.
            :param words_dataset: The dataset of words.
        """
        self.return_sent_emb = return_sent_emb
        self.word2idx = word2idx
        # input encoder
        self.char_encoder = TextEncoder(DS_ARABIC_LETTERS, special_tokens)
        # output encoder
        self.harakat_encoder = OutputEncoder()
        X, Y = read_data(input_data_file, output_data_file, verbose=verbose)
        self.X = X
        if return_sent_emb:
            self.emb = [get_sentence_embedding(
                "".join(sent), self.word2idx) for sent in self.X]

        self.encoded_X = []
        self.encoded_Y = []

        for ind, (x, y) in enumerate(zip(X, Y)):

            xx = self.char_encoder.encode(x)
            yy = self.harakat_encoder.encode(y)
            self.encoded_X.append(xx)
            self.encoded_Y.append(yy)
            # make sure the number of characters in each line is the same
            assert len(xx) == len(
                yy), f'There is aline with different length in input and outsputs at ind={ind} len(x) = {len(xx)} and len(y) = {len(yy)}'

        if verbose:
            max_len = max([len(x) for x in self.encoded_X])
            print(f'max_len = {max_len}')
            min_len = min([len(x) for x in self.encoded_X])
            print(f'min_len = {min_len}')
            mean_len = mean([len(x) for x in self.encoded_X])
            print(f'mean_len = {mean_len}')
            all_sent_count = len(self.encoded_X)
            num_sent_ = len([x for x in self.encoded_X if len(x) < 400])
            print(f'num_sent = {num_sent_}')
            print(
                f'percent = {num_sent_/all_sent_count}')

        # choose fixed length that perseves 99% of the data
        if val_mode:
            # we can't use the 99 percentile in validation mode
            # because we shouldn't cut the validation data
            w = max([len(x) for x in self.encoded_X])
        else:
            w = find_width_99_percentile(self.encoded_X)
        print(f'w = {w}')
        # pad the data
        self.encoded_X = [x + [self.char_encoder.get_pad_id()] * (w - len(x))
                          for x in self.encoded_X]

        self.encoded_Y = [
            y + [self.harakat_encoder.get_pad_id()] * (w - len(y)) for y in self.encoded_Y]

        # clip the data to w
        self.encoded_X = [x[:w] for x in self.encoded_X]
        self.encoded_Y = [y[:w] for y in self.encoded_Y]

        # print(self.encoded_X[0])
        self.encoded_X = torch.tensor(self.encoded_X, device=device)
        self.encoded_Y = torch.tensor(self.encoded_Y, device=device)

    def __getitem__(self, index):
        """
            This method is used to get an item from the dataset.

            :param index: The index of the item to get.
            :return: The item at the given index.
        """
        if self.return_sent_emb:
            return self.encoded_X[index], self.encoded_Y[index], self.emb[index]
        return self.encoded_X[index], self.encoded_Y[index]

    def __len__(self):
        """
            This method is used to get the length of the dataset.

            :return: The length of the dataset.
        """
        return len(self.encoded_X)

    def get_input_vocab_size(self):
        return self.char_encoder.get_vocab_size()

    def get_output_vocab_size(self):
        return self.harakat_encoder.get_vocab_size()


if __name__ == '__main__':
    dataset = LettersDataset(return_sent_emb=True)
    print(dataset[0])
    print("returned successfully without errors")
    print(len(dataset))

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    sample = next(iter(data_loader))
    print(sample[0].shape)
    print(sample[1].shape)
    print(sample[0])
    print(sample[1])
