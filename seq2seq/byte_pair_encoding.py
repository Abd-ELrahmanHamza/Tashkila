from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


class Byte_Pair_Encoding:
    def __init__(self, max_length=100, padding_token="[PAD]"):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.enable_padding(
            length=max_length, direction="right", pad_token=padding_token)
        self.trainer = BpeTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    def train(self, path: str):
        self.tokenizer.train(files=[path], trainer=self.trainer)

    def encode(self, text: str):
        return self.tokenizer.encode(text).tokens

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()
