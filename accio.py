from torch import nn
import torch

class Accio(nn.Module):
    def __init__(self, input_size,output_size, embedding_size=128, hidden_size=512, device=torch.device('cpu')):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.bidirectional = 2
        self.device = device
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.layer1 = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer2 = nn.LSTM(hidden_size*self.bidirectional, hidden_size, batch_first=True, bidirectional=True)
        self.layer3 = nn.LSTM(hidden_size*self.bidirectional, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.bidirectional*hidden_size, output_size)
        self.loss = nn.CrossEntropyLoss()

    # hn is the hidden state of the last timestep
    # cn is the cell state of the last timestep
    def forward(self, x):
        embeddings = self.embedding(x).to(self.device)
        # pass the input through the first layer
        h, (hn, cn) = self.layer1(embeddings)
        # pass the output of the first layer to the second layer
        h, (hn, cn) = self.layer2(h, (hn, cn))
        # pass the output of the second layer to the third layer
        h, (hn, cn) = self.layer3(h, (hn, cn))
        # pass the output of the third layer to the fully connected layer
        out = self.fc(h)
        return out