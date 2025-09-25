import torch

a = torch.randint(0, 10, (1,  4))
print(a)

e = torch.nn.Embedding(10, 4)
print(e(a))
