import torch
from torch.utils.data import TensorDataset, DataLoader  # these are needed for the training data

# create x of shape (batch_size, seq_len, input_size)
# create y of shape (batch_size, seq_len, output_size)
input_size = 10
output_size = 20
seq_len = 5
batch_size = 2
x = torch.rand(batch_size, seq_len, input_size)
y = torch.rand(batch_size, seq_len, output_size)

print(x.shape)
print(y.shape)

# create a dataset
dataset = TensorDataset(x, y)
print(dataset)