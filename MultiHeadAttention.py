import torch
import tiktoken
import gptDatasetV1



class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super(MultiHeadAttention, self).__init__()
        assert  d_out % num_heads == 0, "Embedding size must be divisible by number of heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.last_Layer = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('mask',
                             torch.tril(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, inputs):
        b, num_tokens, d_in = inputs.shape
        query = self.W_query(inputs)
        key = self.W_key(inputs)
        value = self.W_value(inputs)

        key = key.view(b, num_tokens, self.num_heads, self.head_dim)
        value = value.view(b, num_tokens, self.num_heads, self.head_dim)
        query = query.view(b, num_tokens, self.num_heads, self.head_dim)

        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query = query.transpose(1, 2)

        attn_scores = query @ key.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -1e9)


        attn_weights = torch.nn.functional.softmax(attn_scores / key.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)


        context_vec = (attn_weights @ value).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        output = self.last_Layer(context_vec)
        return output

class FeedForward(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            torch.nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)


class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.erf(x / 1.41421)) * (x + 0.044715 * torch.pow(x, 3.0))

# text = "Hello lily, just a test."
# en_model = tiktoken.encoding_for_model("gpt2")
# ids = en_model.encode(text, allowed_special={"<|endOfText|>"})
# print(ids)
# print(len(ids))
# att = MultiHeadAttention(len(ids), len(ids))
# print(att(torch.tensor([ids], dtype=torch.float32)))

# data = gptDatasetV1.GptDatasetV1("verdict.txt", 8, 8)
# inputs, targets = next(iter(data))
# print(inputs)





