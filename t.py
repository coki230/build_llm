import torch

a = torch.arange(1, 16)
a = a.reshape(1, 3, 5)
print(a)
b = a[0, [0, 1], [2, 3]]
print(b)