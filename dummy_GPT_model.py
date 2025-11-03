import torch
import torch.nn as nn
import tiktoken

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_layers": 12,
    "n_head": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

class DummyGPTModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])

    def forward(self, inputs):
        batch_size, seq_len = inputs.shape
        tok_embeds = self.tok_emb(inputs)
        pos_embeds = self.pos_emb(torch.arange(seq_len))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, inputs):
        return inputs

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, inputs):
        mean = inputs.mean(dim=-1, keepdim=True)
        var = inputs.var(dim=-1, keepdim=True, unbiased=False)
        inputs = (inputs - mean) / torch.sqrt(var + self.eps)
        inputs = self.scale * inputs + self.bias
        return inputs


# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"
# token_util = tiktoken.encoding_for_model("gpt2")
# # print(token_util.encode(txt1))
# # print(token_util.encode(txt2))
# stack = []
#
# stack.append(torch.tensor(token_util.encode(txt1)))
# stack.append(torch.tensor(token_util.encode(txt2)))
# batch = torch.stack(stack, dim=0)
# # print(batch)
#
# model = DummyGPTModel(GPT_CONFIG_124M)
# logits = model(batch)
# print(logits.shape)
# print(logits)

nor_layer = DummyLayerNorm(4)

nor_result = nor_layer(torch.randint(0, 10, (4, 4), dtype=torch.float32))
print(nor_result)
print(nor_result.mean(dim=-1, keepdim=True))
print(nor_result.var(dim=-1, keepdim=True, unbiased=False))
