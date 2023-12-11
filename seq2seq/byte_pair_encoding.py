from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


class Byte_Pair_Encoding:
    def __init__(self, max_length=100):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.enable_padding(length=max_length, direction="left")
        self.trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    def train(self, path: str):
        self.tokenizer.train(files=[path], trainer=self.trainer)

    def encode(self, text: str):
        return self.tokenizer.encode(text).tokens
