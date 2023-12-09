import torch # torch will allow us to create tensors.
import torch.nn as nn # torch.nn allows us to create a neural network.
import torch.nn.functional as F # nn.functional give us access to the activation and loss functions.
from torch.optim import Adam # optim contains many optimizers. This time we're using Adam

import lightning as L # lightning has tons of cool tools that make neural networks easier
from torch.utils.data import TensorDataset, DataLoader # these are needed for the training data

# batch size, sequence length, input size

class Decoder(L.LightningModule):
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, h0, c0):
        x = self.embedding(x)
        h, (hn, cn) = self.rnn(x, (h0, c0)) 
        # h is the output of the RNN
        # hn is the hidden state of the last timestep
        # cn is the cell state of the last timestep
        out = self.fc(h)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        h0 = torch.zeros(1, x.shape[0], self.hidden_size)
        c0 = torch.zeros(1, x.shape[0], self.hidden_size)
        y_hat = self.forward(x, h0, c0)
        # y_hat is the output of the model of shape (batch_size, sequence_length, output_size)
        # y is the target of shape (batch_size, sequence_length)
        # y contains the index of the correct word in the vocabulary
        loss = self.loss(y_hat.view(-1, self.output_size), y.view(-1))
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)
    