
from constants import ARABIC_LETTERS


class TextEncoder:

    def __init__(self, vocab, pad_token='<pad>', unk_token='<unk>'):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx = {word: i for i, word in enumerate(vocab)}
        self.word2idx[unk_token] = len(self.word2idx)
        self.word2idx[pad_token] = len(self.word2idx)
        self.idx2word = {i: word for i, word in enumerate(vocab)}
        self.idx2word[len(self.idx2word)] = unk_token
        self.idx2word[len(self.idx2word)] = pad_token

    def encode(self, seq):
        return [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in seq]

    def decode(self, idxs):
        return ''.join([self.idx2word.get(idx, '$') for idx in idxs])

    def get_pad_token(self):
        return self.word2idx[self.pad_token]


# if main script
if __name__ == '__main__':
    enc = TextEncoder(ARABIC_LETTERS)
