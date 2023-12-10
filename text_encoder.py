
from constants import ARABIC_LETTERS


class TextEncoder:

    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.word2idx = {word: i for i, word in enumerate(vocab)}
        self.word2idx['<unk>'] = len(self.word2idx)
        self.idx2word = {i: word for i, word in enumerate(vocab)}
        self.idx2word[len(self.idx2word)] = '<unk>'
    
    def encode(self, seq):
        return [self.word2idx.get(word, self.word2idx['<unk>']) for word in seq]
    def decode(self, idxs):
        return ' '.join([self.idx2word.get(idx, '<unk>') for idx in idxs])


# if main script
if __name__ == '__main__':
    enc = TextEncoder(ARABIC_LETTERS)
    