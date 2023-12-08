# %%
import nltk
import tkseem as tk

# %%
txt_file = open("clean_out/X_words.txt", "r")

txt = txt_file.read()
txt.replace("\n", " ")
# print(txt)
toks = nltk.word_tokenize(txt)
print(toks[0:100])
print(len(toks))

# %%


class MegaTokenizer:
    """
    A wrapper class for the tkseem tokenizer.
    """

    def __init__(self, text, tokenizer='morph'):
        self.text = text

        self.tok = tk.SentencePieceTokenizer()
        self.tok.train('clean_out/X_words.txt')

    def tokenize(self,txt):
        """
        Tokenize the text.
        """
        return self.tok.tokenize(txt)

    def tokenize_to_file(self, filename):
        """
        Tokenize the text and write to a file.
        """
        # toks = self.tokenize()
        # with open(filename, "w") as f:
        #     for tok in toks:
        #         f.write(tok + "\n")
        pass


C = MegaTokenizer(txt)
out = C.tokenize(txt)

# %%
ll = txt.split()[:30]
print(ll)
print(len(txt.split()))
print(len(out))
print(out[:30])

# print num of unique tokens in the output
print(len(set(out)))