import torch  # torch will allow us to create tensors.
import torch.nn as nn  # torch.nn allows us to create a neural network.
from torch.optim import Adam  # optim contains many optimizers. This time we're using Adam
from torchmetrics import Accuracy
import lightning as L  # lightning has tons of cool tools that make neural networks easier


# batch size, sequence length, input size

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, device='cuda'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = torch.device(device)

    def forward(self, x, h0, c0):
        # print("from decoder forward")
        # print(x.shape)
        embeddings = self.embedding(x).to(self.device)
        # print("from decoder forward after embedding")
        # print(embeddings.shape)
        h, (hn, cn) = self.rnn(embeddings, (h0, c0))
        # h is the output of the RNN
        # hn is the hidden state of the last timestep
        # cn is the cell state of the last timestep
        out = self.fc(h)
        return out

    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     batch_size = x.shape[0]
    #     # print("from decoder training step")
    #     # print(x.shape)
    #     h0 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
    #     c0 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
    #     y_hat = self.forward(x, h0, c0).to(self.device)
    #     # y_hat is the output of the model of shape (batch_size, sequence_length, output_size)
    #     # y is the target of shape (batch_size, sequence_length)
    #     # y contains the index of the correct word in the vocabulary
    #     loss = self.loss(y_hat.view(-1, self.output_size), y.view(-1)).to(self.device)
    #     self.log('train_loss', loss)
    #     return loss
    #
    # def configure_optimizers(self):
    #     return Adam(self.parameters(), lr=0.1)
    #
    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     batch_size = x.shape[0]
    #     h0 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
    #     c0 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
    #     y_hat = self.forward(x, h0, c0).to(self.device)
    #     loss = self.loss(y_hat.view(-1, self.output_size), y.view(-1)).to(self.device)
    #     accuracy = Accuracy().to(self.device)
    #     acc = accuracy(y_hat.view(-1, self.output_size), y.view(-1)).to(self.device)
    #     self.log('val_acc', acc,on_epoch=True)
    #     return loss
    #
    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     batch_size = x.shape[0]
    #     h0 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
    #     c0 = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
    #     y_hat = self.forward(x, h0, c0).to(self.device)
    #     loss = self.loss(y_hat.view(-1, self.output_size), y.view(-1)).to(self.device)
    #     accuracy = Accuracy().to(self.device)
    #     acc = accuracy(y_hat.view(-1, self.output_size), y.view(-1)).to(self.device)
    #     self.log('test_acc', acc,on_epoch=True)
    #     return loss
