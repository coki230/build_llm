import torch

a = torch.arange(0, 1, 0.01)
b = torch.multinomial(a, 1)
c = torch.topk(a, 2)
print(b)
print(c)