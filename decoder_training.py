#!/usr/bin/env python
# coding: utf-8

# In[1]:



import torch
import torch.optim as optim
import torch.utils.data as data
from letters_dataset import LettersDataset
import torch.nn as nn
from train_collections import *
import numpy as np
from tqdm import tqdm
import lightning as pl


# In[2]:


# model and training parameters
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 1


# In[3]:


# load train data
dataset = LettersDataset(device=device, input_data_file="./clean_out/X.csv", output_data_file="./clean_out/Y.csv")
loader = data.DataLoader(dataset, shuffle=True, batch_size=batch_size)
n_chars = dataset.get_input_vocab_size()
n_harakat = dataset.get_output_vocab_size()
print("n_chars: ", n_chars)
print("n_harakat: ", n_harakat)


# In[4]:


import torch # torch will allow us to create tensors.
import torch.nn as nn # torch.nn allows us to create a neural network.
from torch.optim import Adam # optim contains many optimizers. This time we're using Adam
from torchmetrics import Accuracy


# batch size, sequence length, input size

class Decoder(pl.LightningModule):
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
        print("from decoder init")
        print("adham")

    def forward(self, x, h0, c0):
        # print("from decoder forward")
        # print(x.shape)
        embeddings = self.embedding(x).cuda()
        # print("from decoder forward after embedding")
        # print(embeddings.shape)
        h, (hn, cn) = self.rnn(embeddings, (h0, c0))
        # h is the output of the RNN
        # hn is the hidden state of the last timestep
        # cn is the cell state of the last timestep
        out = self.fc(h)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.shape[0]
        # print("from decoder training step")
        # print(x.shape)
        h0 = torch.zeros(1, batch_size, self.hidden_size).cuda()
        c0 = torch.zeros(1, batch_size, self.hidden_size).cuda()
        y_hat = self.forward(x, h0, c0).cuda()
        # y_hat is the output of the model of shape (batch_size, sequence_length, output_size)
        # y is the target of shape (batch_size, sequence_length)
        # y contains the index of the correct word in the vocabulary
        loss = self.loss(y_hat.view(-1, self.output_size), y.view(-1)).cuda()
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters())
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.shape[0]
        h0 = torch.zeros(1, batch_size, self.hidden_size).cuda()
        c0 = torch.zeros(1, batch_size, self.hidden_size).cuda()
        y_hat = self.forward(x, h0, c0).cuda()
        loss = self.loss(y_hat.view(-1, self.output_size), y.view(-1)).cuda()
        accuracy = Accuracy().cuda()
        acc = accuracy(y_hat.view(-1, self.output_size), y.view(-1)).cuda()
        self.log('val_acc', acc,on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.shape[0]
        h0 = torch.zeros(1, batch_size, self.hidden_size).cuda()
        c0 = torch.zeros(1, batch_size, self.hidden_size).cuda()
        y_hat = self.forward(x, h0, c0).cuda()
        loss = self.loss(y_hat.view(-1, self.output_size), y.view(-1)).cuda()
        accuracy = Accuracy().cuda()
        acc = accuracy(y_hat.view(-1, self.output_size), y.view(-1)).cuda()
        self.log('test_acc', acc,on_epoch=True)
        return loss


# In[6]:


model = Decoder(input_size=n_chars, output_size=n_harakat, embedding_size=512, hidden_size=256).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.char_encoder.get_pad_id())


# In[7]:


trainer = pl.Trainer(max_epochs=10, accelerator="auto", devices="auto",log_every_n_steps=10)
# trainer = pl.Trainer(max_epochs=3)
trainer.fit(model, train_dataloaders=loader)


# In[8]:


# print training loss
print(trainer.logged_metrics)


# In[9]:


# save the model 
torch.save(model.state_dict(), "./models/decoder.pth")


# In[ ]:




