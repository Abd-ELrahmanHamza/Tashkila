#!/usr/bin/env python
# coding: utf-8

# In[31]:


import torch
import torch.optim as optim
import torch.utils.data as data
from letters_dataset import LettersDataset
import torch.nn as nn
from train_collections import *
from tqdm import tqdm
import pandas as pd
import numpy as np
from nltk.stem.isri import ISRIStemmer


# In[32]:


# model and training parameters
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[25]:


# load train data
dataset =LettersDataset(device=device) 
loader = data.DataLoader(dataset, shuffle=True, batch_size=1)
n_chars = dataset.get_input_vocab_size()
n_harakat = dataset.get_output_vocab_size()


# In[26]:


def save_checkpoint(model, optimizer, epoch, loss, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, filename)
    
    
def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    return epoch, loss


# In[27]:


from accio import Accio
model = Accio(input_size=n_chars, output_size=n_harakat,device=device).to(device)
# model.load_state_dict(torch.load("models/accio_test_epoch_19.pth"))
optimizer = optim.Adam(model.parameters())

_ = load_checkpoint(model, optimizer, "models/accio_epoch_19.pth")
loss_fn = nn.CrossEntropyLoss()


# In[28]:


test_dataset = LettersDataset('clean_out/X_test_no_diacritics.csv', 'clean_out/Y_test_no_diacritics.csv',val_mode=True, device=device)   
val_loader = data.DataLoader(test_dataset,  batch_size=batch_size)
print(test_dataset.char_encoder.word2idx)
# evaluaate accuracy on validation set

model.eval()
letter_haraka = []
with torch.no_grad():
    for (X_batch,y_batch) in tqdm(val_loader):
        # y_pred = model(X_batch)['diacritics']
        y_pred = model(X_batch)
        # we transpose because the loss function expects the second dimension to be the classes
        # y_pred is now (batch_size, n_classes, seq_len)
        y_pred = y_pred.transpose(1, 2) 
        _, predicted = torch.max(y_pred.data, 1)
        # Count only non-padding characters
        for x,y in zip(X_batch,predicted):
            for xx,yy in zip(x,y):
                # we reached the end of the sentence
                # print(xx.item())
                # print(test_dataset.char_encoder.get_pad_id())
                # print(test_dataset.char_encoder.get_id_by_token(UNK_TOKEN))
                if xx.item() == test_dataset.char_encoder.get_pad_id():
                    break
                ll = test_dataset.char_encoder.is_arabic_letter(xx.item())
                if ll:
                    letter_haraka.append(yy.item())

# save ID,Label pairs in a csv file
import pandas as pd
df = pd.DataFrame(letter_haraka, columns=['label'])
df.to_csv('./results/out.csv', index=True, index_label='ID')


