import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, dropout_probability):
        super().__init__()
        # TODO: replace with one hot encoding
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm_layer = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout_probability)

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
        return hidden, cell
