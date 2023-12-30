from torch import nn
import torch

class Accio(nn.Module):
    def __init__(self, input_size,output_size, embedding_size=128, hidden_size=64, device=torch.device('cpu')):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.bidirectional = 2
        self.num_layers = 3
        self.device = device
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True, num_layers=3)
        self.fc = nn.Linear(2*hidden_size, output_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.bidirectional*self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.bidirectional*self.num_layers, batch_size, self.hidden_size).to(self.device)
        embeddings = self.embedding(x).to(self.device)
        h, (hn, cn) = self.rnn(embeddings, (h0, c0))
        # h is the output of the RNN
        # hn is the hidden state of the last timestep
        # cn is the cell state of the last timestep
        out = self.fc(h)
        return out