import tiktoken
import torch
import SelfAttention_v1
import SelfAttention_v2
import numpy as np
import CausalAttention as ca
import MultiHeadAttentionWrapper as mw
import GPTModel as gpt

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
# a = torch.arange(0, 20)
# print(a)
#
# sa1 = SelfAttention_v1.SelfAttention_v1(3, 2)
# sa2 = SelfAttention_v2.SelfAttention_v2(3, 2)
# a1 = torch.arange(1, 22).reshape(7, 3).type(torch.float)
# a2 = torch.arange(1, 22).reshape(7, 3).type(torch.float)
# q_p, k_p, v_p = sa1.get_parameter()
# print("first param ", sa2.get_parameter())
# sa2.set_parameters(k_p, q_p, v_p)
# print("----",q_p)
# print(sa2.get_parameter())
#
# print(sa1(a1))
# print(sa2(a2))


#
# w = torch.tensor([1, 2, 3])
# v = torch.tensor([[1, 2], [3, 4], [5, 6]])
#
# print(w.unsqueeze(0))
# print(w @ v)

#------------------------------------
# demo for masked the data
#------------------------------------
# b = torch.tril(torch.ones(5, 5))
# print(b.bool())
# print(b)
# mask = b == 0
# print(mask)
# b.masked_fill_(mask, -torch.inf)
# print(b)
#
# b = torch.softmax(b, dim=-1)
# print(b)


#------------------------------------
# demo for masked the data
#------------------------------------
# drop = torch.nn.Dropout(0.7)
# data = torch.ones((5, 5))
# print(drop(data))


#------------------------------------
# demo for torch cat
#------------------------------------
# a = torch.arange(1, 7).reshape((2, 3))
# b = torch.arange(1, 7).reshape((2, 3))
# c = torch.cat((a, b), dim=-1)
# print(c)


#------------------------------------
# demo for CasalAttention test
#------------------------------------
# input = torch.rand((2, 6, 3 )).type(torch.float)
# print(input)
# c_a = ca.CausalAttention(3, 2, 6, 0.0)
# print(input.shape)
# context_vector = c_a(input)
# print(context_vector)


#------------------------------------
# demo for MultiHeadAttentionWrapper test
#------------------------------------
#
# input = torch.rand((2, 6, 3 )).type(torch.float)
# model = mw.MultiHeadAttentionWrapper(3, 2, 6, 0.0, 6)
# output = model(input)
# print(output.shape, output)


#------------------------------------
# demo for GPTModel test
#------------------------------------

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_layer": 12,
    "n_head": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
model = gpt.GPTModel(GPT_CONFIG_124M)

# input = torch.randint(0, 10, (2, 1024), dtype=torch.long)
# out = model(input)
# print(input.shape)
# print(out.shape)
# p_sum = sum(p.numel() for p in model.parameters())
# print(p_sum)

start_context = "Hello, I am"
tokenizer = tiktoken.encoding_for_model("gpt2")
encoded = tokenizer.encode(start_context)
print(encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print(encoded_tensor)
model.eval()
out = gpt.generate_text_simple(model, encoded_tensor, 6, 1024)
print(out)
print(tokenizer.decode(out.squeeze(0).tolist()))
print(len(out[0]))