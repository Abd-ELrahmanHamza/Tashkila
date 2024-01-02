from train_collections import *


class OutputEncoder:

    def __init__(self, vocab=DS_HARAKAT):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.word2idx = harakat2id
        self.idx2word = {i: word for word, i in harakat2id.items()}

    def encode(self, seq):
        return [self.word2idx.get(word, self.word2idx['']) for word in seq]

    def decode(self, idxs):
        return ''.join([self.idx2word.get(idx, '') for idx in idxs])

    def get_pad_id(self):
        return self.get_id_by_token("")

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
    enc = OutputEncoder(ARABIC_LETTERS)
    out = enc.encode('ًُُ')
    print(out)
