import torch

# x is 3D tensor
x = torch.randn(1, 3, 4)

print(x[:, 0:1, :].shape)
