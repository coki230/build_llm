import torch
import torch.nn as nn

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        keys = self.W_key(x)
        querys = self.W_query(x)
        values = self.W_value(x)

        attention_scores = querys @ keys.T
        attention_weights = torch.softmax(attention_scores ** 0.5, dim=-1)
        context_vector = attention_weights @ values
        return context_vector

    def set_parameters(self, k_p, q_p, v_p):
        self.W_key.weight.data = k_p.T
        self.W_query.weight.data = q_p.T
        self.W_value.weight.data = v_p.T

    def get_parameter(self):
        return self.W_query.weight