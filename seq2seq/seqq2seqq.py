# %%
from letters_dataset import LettersDataset
from seq2seq.byte_pair_encoding import Byte_Pair_Encoding
from words_dataset import WordsDataset
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import random
import math
import time

from train_collections import DS_ARABIC_LETTERS, DS_HARAKAT

print("all imports done")

# %% [markdown]
# ## Define the device

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %% [markdown]
# ## Encoder

# %%


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=128, hidden_dim=256, num_layers=1, dropout_probability=0.1):
        super().__init__()
        # TODO: replace with one hot encoding
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm_layer = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, dropout=dropout_probability, batch_first=True)

        # Dropout layer to prevent over fitting (regularization)
        # it randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, inputs):
        # inputs = [inputs len, batch size]
        embeddings = self.dropout(self.embedding(inputs))

        # embedded = [inputs len, batch size, emb dim]
        outputs, (hidden, cell) = self.lstm_layer(embeddings)

        # outputs = [inputs len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # outputs are always from the top hidden layer
        return outputs,(hidden, cell)


# %% [markdown]
# ## Decoder

# %%


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, device='cuda'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size+hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, context, h0, c0):
        # print("from decoder forward")
        # print(x.shape)
        embeddings = self.embedding(x)
        # print("from decoder forward after embedding")
        # print(embeddings.shape)
        lstm_input = torch.cat((embeddings, context), dim=2)
        outs, (h1,c1) = self.lstm(lstm_input, (h0, c0))
        # h is the output of the RNN
        # hn is the hidden state of the last timestep
        # cn is the cell state of the last timestep
        scores = self.fc(outs)
        return scores,(h1,c1)


# %%
# Seq2Seq

# %%
print("seq2seq")


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_inputs, decoder_inputs):
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(encoder_inputs)
        decoder_output = self.decoder(
            decoder_inputs, encoder_hidden, encoder_cell)
        return decoder_output


# %%Attention
class AttentionSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.softmax = nn.Softmax(dim=2)

    def forward(self, encoder_inputs, decoder_inputs):
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(encoder_inputs)
        # print("hello")
        # add attention
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        sequence_length = decoder_inputs.size(1)
        batch_size = decoder_inputs.size(0)
        # final output of the decoder
        # batch size * sequence length * output size
        final_output = torch.zeros(
            batch_size, sequence_length, self.decoder.output_size, device=device)
        for i in range(sequence_length):
            attention_weights = self.calculate_attention_weights(encoder_output, decoder_hidden)
            attention_vectors = attention_weights * encoder_output
            context_vector = torch.sum(attention_vectors, dim=1, keepdim=True)
            scores ,(decoder_hidden, decoder_cell) = self.decoder(decoder_inputs[:, i:i+1], context_vector, decoder_hidden, decoder_cell)
            final_output[:, i:i+1, :] = scores
        return final_output

    def calculate_attention_weights(self, encoder_output, decoder_hidden):
        # encoder output: [batch size, seq len, hidden size]
        # decoder hidden: [1, batch size, hidden size]
        # attention weights: [batch size, seq len, 1]
        decoder_hidden_permuted = decoder_hidden.permute(1, 2, 0)
        attention_weights = torch.bmm(encoder_output, decoder_hidden_permuted)
        attention_weights = self.softmax(attention_weights)
        return attention_weights



# %%


decoder_dim_vocab = len(DS_ARABIC_LETTERS)
decoder_dim_out = len(DS_HARAKAT) + 2  # harakat

# encoder_dim_vocab = #tokens

embedding_dim = 64
n_epochs = 5
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

bpe = Byte_Pair_Encoding(450)
bpe.train("../clean_out/merged.txt")
# %%
decodor_dataset = LettersDataset(
    "../clean_out/X.csv", "../clean_out/Y.csv", device=device)
encoder_dataset = WordsDataset(
    "../clean_out/X_words.txt", device=device, tokenizer=bpe)
print("adham")


# %%

class CombinedDataset(Dataset):
    def __init__(self, words_dataset, letters_dataset):
        self.words_dataset = words_dataset
        self.letters_dataset = letters_dataset

        # Ensure both datasets are of the same size
        assert len(words_dataset) == len(
            letters_dataset), "Datasets must be of the same size"

    def __len__(self):
        return len(self.words_dataset)

    def __getitem__(self, idx):
        word = self.words_dataset[idx]
        letters, letter_tashkeel = self.letters_dataset[idx]

        # Combine or process the features as needed for your model
        # This can vary depending on how your seq2seq model is set up

        return word, letters, letter_tashkeel


merged_set = CombinedDataset(encoder_dataset, decodor_dataset)

seq2seq_loader = DataLoader(merged_set, shuffle=True, batch_size=batch_size)

sample = next(iter(seq2seq_loader))
print(sample[0].shape)
print(sample[1].shape)
print(sample[2].shape)
# %%
enc_model = Encoder(
    encoder_dataset.bpe.tokenizer.get_vocab_size(), hidden_dim=128, num_layers=1, dropout_probability=0)

dec_model = Decoder(decoder_dim_vocab, embedding_size=128,
                    hidden_size=128, output_size=decoder_dim_out, device=device.type)


model = AttentionSeq2Seq(encoder=enc_model, decoder=dec_model).to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
num_batches = len(seq2seq_loader)
print("Number of batches:", num_batches)
best_model = None
best_loss = np.inf
for epoch in range(n_epochs):
    model.train()
    for i, (X_encoder, X_decoder, Y_batch) in enumerate(seq2seq_loader):
        y_pred = ''
        y_pred = model(X_encoder, X_decoder)
        y_pred = y_pred.transpose(1, 2)
        # print(y_pred.shape)
        # print(y_batch.shape)
        loss = loss_fn(y_pred, Y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("Epoch %d, batch %d: Loss = %.4f" % (epoch, i, loss))

    # Validation
    model.eval()
    loss = 0
    with torch.no_grad():
        for (X_encoder, X_decoder, Y_batch) in seq2seq_loader:
            y_pred = model(X_encoder, X_decoder)
            y_pred = y_pred.transpose(1, 2)

            loss += loss_fn(y_pred, Y_batch)
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))

# %%

val_dataset = LettersDataset(
    '../clean_out/X_val.csv', '../clean_out/y_val.csv', device=device)
val_words_dataset = WordsDataset(
    '../clean_out/X_words_val.txt', device=device, tokenizer=bpe)
val_merged = CombinedDataset(val_words_dataset, val_dataset)
seq2seq_loader = DataLoader(merged_set, shuffle=True, batch_size=batch_size)

# evaluaate accuracy on validation set


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for (X_encoder, X_decoder, Y_batch) in seq2seq_loader:
        is_padding = (X_decoder == val_dataset.char_encoder.get_pad_token())
        y_pred = model(X_encoder, X_decoder)
        y_pred = y_pred.transpose(1, 2)
        _, predicted = torch.max(y_pred.data, 1)
        # Count only non-padding characters
        total += torch.sum(~is_padding).item()

        # Count correct predictions
        correct += torch.sum((predicted == Y_batch) & (~is_padding)).item()
print("Accuracy: %.2f%%" % (100 * correct / total))

# %%
print("adham")
# %%
