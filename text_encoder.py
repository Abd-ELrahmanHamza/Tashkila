from train_collections import *


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

    def get_pad_id(self):
        return self.get_id_by_token(PAD_TOKEN)

    def get_id_by_token(self, token):
        return self.word2idx[token]

    def get_token_by_id(self, idx):
        return self.idx2word[idx]

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.vocab


# if main script
if __name__ == '__main__':
    enc = TextEncoder(ARABIC_LETTERS)
    l = enc.is_arabic_letter(1)
    print(l)
