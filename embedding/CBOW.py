#!/usr/bin/env python
# coding: utf-8

# In[70]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from nltk.stem.isri import ISRIStemmer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using CUDA")
else:
    print("Using CPU")


# In[71]:


stemmer = ISRIStemmer()
w = "انتزاعا"
print(stemmer.stem(w))


# In[72]:


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        embeds = torch.sum(self.embeddings(inputs), dim=1)
        out = self.linear(embeds)
        out = self.activation(out)
        return out


# In[73]:


class CBOWDataset(Dataset):
    def __init__(self, text, word2idx, window_size, lengths):
        super(CBOWDataset, self).__init__()
        # keep text_encoded 2d
        self.text_encoded = [[word2idx[word] if word in word2idx else word2idx['<UNK>'] for word in sentence] for sentence in text]
        self.text_encoded = [torch.tensor(sentence, device=device) for sentence in self.text_encoded]
        self.window_size = window_size
        self.lengths = lengths

    def __getitem__(self, idx):
        sentence_idx = np.searchsorted(self.lengths, idx, side='right')
        idx = idx - self.lengths[sentence_idx - 1] if sentence_idx > 0 else idx
        center_word = self.text_encoded[sentence_idx][idx]
        start_idx = idx - self.window_size if (idx - self.window_size) > 0 else 0
        end_idx = idx + self.window_size
        before_context = self.text_encoded[sentence_idx][start_idx:idx]
        after_context = self.text_encoded[sentence_idx][idx + 1:end_idx + 1]
        if len(before_context) < self.window_size:
            before_context = torch.cat((torch.tensor([word2idx['<S>']] * (self.window_size - len(before_context)), device=device), before_context))
        if len(after_context) < self.window_size:
            after_context = torch.cat((after_context, torch.tensor([word2idx['</S>']] * (self.window_size - len(after_context)), device=device)))
        context = torch.cat((before_context, after_context))
        return context, center_word

    def __len__(self):
        return self.lengths[-1]


# In[74]:


# A function to get the max length that will cover 99% of the data
def get_max_len(text):
    lengths = [len(sentence.split()) for sentence in text]
    return np.percentile(lengths, 99)


# In[75]:


# text = """We are about to study the idea of a computational process. Computational processes are abstract beings that inhabit computers.
# As they evolve, processes manipulate other abstract things called data. The evolution of a process is directed by a pattern of rules called a program.
# People create programs to direct processes. In effect, we conjure the spirits of the computer with our spells."""
with open("../clean_out/merged_unsplited.txt", "r", encoding="utf8") as f:
    text = f.read()

# replace , and - with space
text = text.replace("،", "")
text = text.replace("-", "")
# Split into sentences
text = text.split("\n")
# remove sentences with length more than 99% of the data
# max_len = get_max_len(text)
# text = [sentence for sentence in text if len(sentence.split()) <= max_len]
# make all sentences with same length by padding with <PAD>
# max_len = max([len(sentence.split()) for sentence in text])
# text = [sentence + " <PAD>" * (max_len - len(sentence.split())) for sentence in text]
# Split into words
text = [sentence.split() for sentence in text]
# get array of length of all sentences
lengths = [len(sentence) for sentence in text]
# prefix sum of lengths
lengths = np.cumsum(lengths)
# stem words
text = [[stemmer.stem(word) for word in sentence] for sentence in text]
# # Flatten list of lists
# text = [word for sentence in text for word in sentence]
# # Stem words
# text = [stemmer.stem(word) for word in text]

# print(text)
# Hyperparameters
vocab = set([word for sentence in text for word in sentence] + ["<S>", "</S>", "<UNK>"])
vocab_size = len(vocab)
embedding_size = 256
window_size = 4
batch_size = 64
num_epochs = 5
print("Vocab size: ", vocab_size)


# In[76]:


print(text[:5])


# In[77]:


word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for i, word in enumerate(vocab)}

dataset = CBOWDataset(text, word2idx, window_size, lengths)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# In[78]:


model = CBOW(vocab_size, embedding_size).to(device)
# criterion = nn.NLLLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# In[79]:


for epoch in range(num_epochs):
    torch.cuda.empty_cache()  # Clear CUDA cache
    for i, (context, target) in enumerate(dataloader):
        log_probs = model(context)
        loss = criterion(log_probs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("Epoch: %d, Iteration: %d, Loss: %0.4f out of %d" % (epoch, i, loss, len(dataloader)))


# In[80]:


embedding_weights = model.embeddings.weight.data.cpu().numpy()
np.save("../embedding/embedding_weights.npy", embedding_weights)


# In[81]:


embedding_weights = np.load("../embedding/embedding_weights.npy")


# In[82]:


def get_word(word):
    return embedding_weights[word2idx[word]]


def get_closest_word(word, n=5):
    word_distance = []
    if word not in word2idx:
        word = "<UNK>"
    word_vec = get_word(word)
    for i, vec in enumerate(embedding_weights):
        distance = np.linalg.norm(vec - word_vec)
        word_distance.append((idx2word[i], distance))
    word_distance = sorted(word_distance, key=lambda k: k[1])[1:n + 1]
    return word_distance


# In[83]:


print(get_closest_word(stemmer.stem("محمد")))
print(embedding_weights[word2idx[stemmer.stem("قال")]])
print(len(embedding_weights[word2idx[stemmer.stem("قال")]]))

