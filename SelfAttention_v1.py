import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        querys = x @ self.W_query
        values = x @ self.W_value

        attention_scores = querys @ keys.T
        attention_weights = torch.softmax(attention_scores ** 0.5, dim=-1)
        context_vector = attention_weights @ values
        return context_vector