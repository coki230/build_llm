import torch
import CausalAttention as ca
class MultiHeadAttentionWrapper(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, head_num):
        super().__init__()
        self.heads = torch.nn.ModuleList([ca.CausalAttention(d_in, d_out, context_length, dropout) for _ in range(head_num)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)