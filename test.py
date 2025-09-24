import tiktoken
import torch

torch.manual_seed(123)
input = torch.randn(2, 5)
print(input)
layer = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.ReLU())

output = layer(input)
print(output)

mean = output.mean(dim=-1, keepdim=True)
print(mean)
var = output.var(dim=-1, keepdim=True)
print(var)

out_norm = (output - mean) / torch.sqrt(var)
print(out_norm)
print(out_norm.mean(dim=-1, keepdim=True))
print(out_norm.var(dim=-1, keepdim=True))