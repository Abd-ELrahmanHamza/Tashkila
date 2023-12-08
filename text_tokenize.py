# %%
import nltk
import tkseem as tk

# %%
txt_file = open("clean_out/merged.txt", "r")

txt = txt_file.read()
txt = txt.replace("\n", " ")
# print(txt)
toks = nltk.word_tokenize(txt)
print(toks[0:100])
print(len(toks))
print(len(set(toks)) , "unique words")
# %%


class MegaTokenizer:
    """
    A wrapper class for the tkseem tokenizer.
    """

    def __init__(self, text):
        self.text = text

        self.tok = tk.SentencePieceTokenizer(vocab_size=50000)
        self.tok.train('clean_out/merged.txt')
        # -----------------
        # self.tok = tk.WordTokenizer(vocab_size=20000)
        # self.tok.train('clean_out/merged.txt')
        # ------------------
        # self.tok = tk.MorphologicalTokenizer()
        # self.tok.train()

    def tokenize(self, txt):
        """
        Tokenize the text.
        """
        return self.tok.tokenize(txt)

    def get_tokenizer(self):
        """
        Return the tokenizer.
        """
        return self.tok


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

#% 

co = C.tokenize('السلام عليكم ورحمة الله وبركاته')
print(co)