"""
    This file is used to create a dataset of letters from the dataset of words in the dataset folder.
"""

from re import T
from uu import encode
from numpy import mean
import torch
from torch.utils.data import Dataset, DataLoader
from text_encoder import TextEncoder
import numpy as np
from train_collections import DS_ARABIC_LETTERS, DS_HARAKAT
input_file = 'clean_out/X.csv'
output_file = 'clean_out/Y.csv'
input_val = 'clean_out/X_val.csv'
output_val = 'clean_out/Y_val.csv'


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


def find_width_99_percentile(data):
    # Flatten the list to get the lengths of all inner lists
    lengths = np.array([len(inner_list) for inner_list in data])

    # Calculate the 99th percentile
    return int(np.percentile(lengths, 99))


class LettersDataset(Dataset):
    """
        This class is used to create a dataset of letters from the dataset of words in the dataset folder.
    """

    def __init__(self, input_data_file='clean_out/X.csv', output_data_file='clean_out/Y.csv', device=torch.device('cpu'), verbose=False):
        """
            This method is used to initialize the class.
            :param words_dataset: The dataset of words.
        """
        self.char_encoder = TextEncoder(DS_ARABIC_LETTERS)
        self.harakat_encoder = TextEncoder(DS_HARAKAT)
        X, Y = read_data(input_data_file, output_data_file, verbose=verbose)

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
        w = find_width_99_percentile(self.encoded_X)
        print(f'w = {w}')
        # pad the data
        self.encoded_X = [x + [self.char_encoder.get_pad_token()] * (w - len(x))
                          for x in self.encoded_X]

        self.encoded_Y = [
            y + [self.harakat_encoder.get_pad_token()] * (w - len(y)) for y in self.encoded_Y]

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

        return self.encoded_X[index], self.encoded_Y[index]

    def __len__(self):
        """
            This method is used to get the length of the dataset.

            :return: The length of the dataset.
        """
        return len(self.encoded_X)


if __name__ == '__main__':
    read_data(input_file, output_file, verbose=True)
    dataset = LettersDataset()
    print(dataset[0])
    print("returned successfully without errors")
    print(len(dataset))

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    sample = next(iter(data_loader))
    print(sample['input'].shape)
    print(sample['output'].shape)
