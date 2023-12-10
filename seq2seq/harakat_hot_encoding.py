import numpy as np
import pyarabic.araby as araby

# Define the vocabulary of harakat and taskeel and tanween

harakat = araby.HARAKAT
tashkeel = araby.TASHKEEL 
tanwin = araby.TANWIN
shortharakat = araby.SHORTHARAKAT

all_harakat = list(harakat + tashkeel + tanwin + shortharakat)

# add start and end of sequence characters and space
all_harakat.append('<s>')
all_harakat.append(' ')
all_harakat.append('</s>')

# harakat to index and vice versa
harakat2idx = {u:i for i, u in enumerate(all_harakat)}
idx2harakat = np.array(all_harakat)

# create one hot encoding for each harakat
hot_encoding = np.eye(len(all_harakat), dtype='int32')

if __name__ == '__main__':
    print("All harakat:", all_harakat)
    print("Harakat2idx:", harakat2idx)
    print("Idx2harakat:", idx2harakat)
    print("Hot encoding:", hot_encoding)
    