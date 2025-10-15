import tiktoken
import torch
import SelfAttention_v1
import SelfAttention_v2
import numpy as np

# tokenizer = tiktoken.encoding_for_model("gpt2")
#
# s = "Akwirw ier"
# a = tokenizer.encode(s)
# print(a)
#
# b = tokenizer.decode(a)
# print(b)

#
# for i in a:R
#     print(tokenizer.decode([i]))

#-----------------------------------
# set the linear param from the parameter mode
#-----------------------------------
a = torch.arange(0, 20)
print(a)

sa1 = SelfAttention_v1.SelfAttention_v1(3, 2)
sa2 = SelfAttention_v2.SelfAttention_v2(3, 2)
a1 = torch.arange(1, 22).reshape(7, 3).type(torch.float)
a2 = torch.arange(1, 22).reshape(7, 3).type(torch.float)
q_p, k_p, v_p = sa1.get_parameter()
print("first param ", sa2.get_parameter())
sa2.set_parameters(k_p, q_p, v_p)
print("----",q_p)
print(sa2.get_parameter())

print(sa1(a1))
print(sa2(a2))
#
# w = torch.tensor([1, 2, 3])
# v = torch.tensor([[1, 2], [3, 4], [5, 6]])
#
# print(w.unsqueeze(0))
# print(w @ v)