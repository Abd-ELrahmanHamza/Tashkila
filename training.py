#!/usr/bin/env python
# coding: utf-8

# In[1]:


# to reload modules automatically without having to restart the kernel
import torch
import torch.optim as optim
import torch.utils.data as data
from letters_dataset import LettersDataset
import torch.nn as nn
from train_collections import *
import numpy as np
from tqdm import tqdm


# In[2]:


# model and training parameters
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 10


# In[3]:


# load train data
dataset = LettersDataset(device=device)
loader = data.DataLoader(dataset, shuffle=True, batch_size=batch_size)
n_chars = dataset.get_input_vocab_size()
n_harakat = dataset.get_output_vocab_size()
print("n_chars: ", n_chars)
print("n_harakat: ", n_harakat)


# In[4]:


from accio import Accio
model = Accio(input_size=n_chars, output_size=n_harakat,device=device).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.char_encoder.get_pad_id())


# In[5]:


num_batches = len(loader)
print("Number of batches:", num_batches)
best_model = None
best_loss = np.inf
for epoch in range(n_epochs):
    torch.cuda.empty_cache()  # Clear CUDA cache to avoid memory error
    model.train()
    for i, (X_batch,y_batch) in tqdm(enumerate(loader)):
        y_pred = ''
        y_pred = model(X_batch)
        # we transpose because the loss function expects the second dimension to be the classes
        # y_pred is now (batch_size, n_classes, seq_len)
        y_pred = y_pred.transpose(1, 2)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("Epoch %d, batch %d: Loss = %.4f" % (epoch, i, loss))

    # save model after each epoch
    torch.save(model.state_dict(), f'models/accio_epoch_{epoch}.pth')
        
    # Validation
    model.eval()
    loss = 0
    with torch.no_grad():
        for (X_batch,y_batch) in loader:
            y_pred = model(X_batch)
            y_pred = y_pred.transpose(1, 2) 
            loss += loss_fn(y_pred, y_batch)
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))


# In[ ]:


# load validation data
val_dataset = LettersDataset('clean_out/X_val.csv', 'clean_out/y_val.csv',val_mode=True, device=device)   
val_loader = data.DataLoader(val_dataset,  batch_size=batch_size)
print(val_dataset.char_encoder.word2idx)


# In[ ]:


# evaluaate accuracy on validation set
model.eval()
letter_haraka = []
with torch.no_grad():
    for (X_batch,y_batch) in val_loader:
        y_pred = model(X_batch)
        # we transpose because the loss function expects the second dimension to be the classes
        # y_pred is now (batch_size, n_classes, seq_len)
        y_pred = y_pred.transpose(1, 2) 
        _, predicted = torch.max(y_pred.data, 1)
        # Count only non-padding characters
        for x,y in zip(X_batch,predicted):
            for xx,yy in zip(x,y):
                # we reached the end of the sentence
                if xx.item() == val_dataset.char_encoder.get_pad_id():
                    break
                ll = val_dataset.char_encoder.is_arabic_letter(xx.item())
                if ll:
                    letter_haraka.append([ll,yy.item()])

# save ID,Label pairs in a csv file
import pandas as pd
df = pd.DataFrame(letter_haraka, columns=['letter','label'])
df.to_csv('./results/letter_haraka.csv', index=True, index_label='ID')


# In[ ]:


gold_val = pd.read_csv('clean_out/val_gold.csv',index_col=0)
sys_val = pd.read_csv('results/letter_haraka.csv',index_col=0)
# Accuracy per letter
correct = 0
total = len(gold_val)
for i in range(total):
    # print(gold_val[i][0], sys_val[i][0])
    correct +=( gold_val.iloc[i]['label'] == sys_val.iloc[i]['label'])
    
print("Accuracy: %.2f%%" % (100.0 * correct / total))


# In[ ]:


print('DER of the network on the validation set: %d %%' % (100 * (1 - correct / total)))

