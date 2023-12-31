#!/usr/bin/env python
# coding: utf-8

# In[1]:


from letters_dataset import LettersDataset
from accio import Accio
from torch.utils import data
import torch


# In[ ]:


batch_size = 64
device = torch.device("cpu")
model = Accio(input_size=41,output_size=15,device=device)
model.load_state_dict(torch.load('./models/base_99.pth'))


# In[2]:


test_dataset = LettersDataset('clean_out/X_test.csv', 'clean_out/Y_test.csv',val_mode=True, device=device)  
test_loader = data.DataLoader(test_dataset,  batch_size=batch_size)


# In[3]:


# evaluaate accuracy on validation set
model.eval()
letter_haraka = []
with torch.no_grad():
    for (X_batch,y_batch) in test_loader:
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
                # print(val_dataset.char_encoder.get_pad_id())
                # print(val_dataset.char_encoder.get_id_by_token(UNK_TOKEN))
                if xx.item() == test_dataset.char_encoder.get_pad_id():
                    break
                ll = test_dataset.char_encoder.is_arabic_letter(xx.item())
                if ll:
                    letter_haraka.append([ll,yy.item()])

# save ID,Label pairs in a csv file
import pandas as pd
df = pd.DataFrame(letter_haraka, columns=['letter','label'])
df.to_csv('./results/letter_haraka2.csv', index=True, index_label='ID')

gold_test = pd.read_csv('clean_out/test_gold.csv',index_col=0)
sys_test = pd.read_csv('results/letter_haraka2.csv',index_col=0)
# Accuracy per letter

print("start evaluation:")
correct = 0
total = len(gold_test)
for i in range(total):
    correct +=( gold_test.iloc[i]['label'] == sys_test.iloc[i]['label'])
    
print("Accuracy: %.2f%%" % (100.0 * correct / total))

