from torch import nn
import torch


class BiLSTM(nn.Module):
    def __init__(self, n_chars, n_harakat, embedding_dim=128, n_hidden=512):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(n_chars, embedding_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=n_hidden, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2*n_hidden, n_harakat)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # pass thru embedding layer
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.linear(self.dropout(x))
        return x
