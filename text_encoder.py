
from tkinter import NONE
from constants import ARABIC_LETTERS
from train_collections import *
import torch


class TextEncoder:

    def __init__(self, vocab, spectial_tokens=[PAD_TOKEN, UNK_TOKEN]):
        self.vocab = vocab
        self.vocab_size = len(vocab) + len(spectial_tokens)

        self.special_tokens = spectial_tokens

        self.word2idx = {word: i for i, word in enumerate(vocab)}

        for token in self.special_tokens:
            self.word2idx[token] = len(self.word2idx)

        self.idx2word = {i: word for i, word in enumerate(vocab)}

        for token in self.special_tokens:
            self.idx2word[len(self.idx2word)] = token

    def encode(self, seq):
        return [self.word2idx.get(word, self.word2idx[UNK_TOKEN]) for word in seq]

    def decode(self, idxs):
        return ''.join([self.idx2word.get(idx, UNK_TOKEN) for idx in idxs])

    def is_arabic_letter(self, id):
        letter = self.idx2word.get(id, UNK_TOKEN)
        if letter in ARABIC_LETTERS:
            return letter
        return None

    def is_arabic_letter_batch(self, ids,device):
        # Convert ARABIC_LETTERS to a tensor for fast comparison
        arabic_letters_ids = torch.tensor(
            [self.word2idx[letter] for letter in ARABIC_LETTERS if letter in self.word2idx], dtype=torch.long).to(device)

        # Expand ids to compare against all Arabic letters
        ids_expanded = ids.unsqueeze(1)
        comparison = ids_expanded == arabic_letters_ids

        # Any row with a True value is an Arabic letter
        is_arabic = comparison.any(dim=1)

        return is_arabic

    def get_pad_id(self):
        return self.get_id_by_token(PAD_TOKEN)

    def get_id_by_token(self, token):
        return self.word2idx[token]

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.vocab


# if main script
if __name__ == '__main__':
    enc = TextEncoder(ARABIC_LETTERS)
    l = enc.is_arabic_letter(1)
    print(l)
