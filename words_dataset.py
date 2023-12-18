"""
    This file is used to create a dataset of letters from the dataset of words in the dataset folder.
"""

from numpy import mean
import torch
from torch.utils.data import Dataset, DataLoader
from text_encoder import TextEncoder
import numpy as np
from train_collections import DS_ARABIC_LETTERS, DS_HARAKAT
from seq2seq.byte_pair_encoding import Byte_Pair_Encoding

input_file = 'clean_out/X_words.txt'
input_val = 'clean_out/X_words_val.txt'


def read_data_words(input_file, verbose=False):
    X = []
    # read csv files
    with open(input_file, 'r', encoding="utf8") as f:
        X = f.readlines()

    # split each line into a list of characters
    X = [x.strip() for x in X]

    if verbose:
        print('Data read successfully.')
        print(f'Number of lines read from {input_file}: {len(X)}')
        print(f'X[0]: {X[0]}')
    return X


def find_width_99_percentile(data):
    # Flatten the list to get the lengths of all inner lists
    lengths = np.array([len(inner_list) for inner_list in data])

    # Calculate the 99th percentile
    return int(np.percentile(lengths, 99))


class WordsDataset(Dataset):
    """
        This class is used to create a dataset of letters from the dataset of words in the dataset folder.
    """

    def __init__(self, input_data_file='clean_out/X_words.txt', device=torch.device('cpu'), max_sentence_length=400,tokenizer, verbose=False):
        """
            This method is used to initialize the class.
            :param words_dataset: The dataset of words.
        """
        self.bpe = Byte_Pair_Encoding(max_sentence_length)
        self.bpe.train("./clean_out/merged.txt")

        # returns [['ahmed','atta'][''][]]
        X = read_data_words(input_data_file,  verbose=verbose)
        self.X_tokenized = [self.bpe.encode(xx) for xx in X]

        vocab = self.bpe.tokenizer.get_vocab()
        if verbose:
            print('vocab', type(vocab))
            print('vocab', vocab)

        self.encoded_X = []

        for x in self.X_tokenized:
            xx = [self.bpe.tokenizer.token_to_id(xxx) for xxx in x]
            self.encoded_X.append(xx)
        if verbose:
            print(self.encoded_X[:4])

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
        # w = find_width_99_percentile(self.encoded_X)
        # print(f'w = {w}')
        # pad the data
        # self.encoded_X = [x + [self.bpe.tokenizer.token_to_id('[PAD]')] * (w - len(x))
        #                   for x in self.encoded_X]

        # clip the data to w
        # self.encoded_X = [x[:w] for x in self.encoded_X]
        # print(self.encoded_X[0])
        self.encoded_X = torch.tensor(self.encoded_X, device=device)

    def __getitem__(self, index):
        """
            This method is used to get an item from the dataset.

            :param index: The index of the item to get.
            :return: The item at the given index.
        """

        return self.encoded_X[index]

    def __len__(self):
        """
            This method is used to get the length of the dataset.

            :return: The length of the dataset.
        """
        return len(self.encoded_X)


if __name__ == '__main__':
    dataset = WordsDataset()
    print(dataset[0])
    print("returned successfully without errors")
    print(len(dataset))

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    sample = next(iter(data_loader))
    print(sample.shape)
    print(sample[:4])

