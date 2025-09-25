import torch.nn as nn
from TransformerBlock import TransformerBlock
import torch
import tiktoken

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layer"])])
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
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

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


torch.manual_seed(123)
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_layer": 12,
    "n_head": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
model = GPTModel(GPT_CONFIG_124M)
input = torch.randint(0, 10, (2, 1024), dtype=torch.long)
out = model(input)

print(input.shape)
print(out.shape)

p_sum = sum(p.numel() for p in model.parameters())
print(p_sum)
start_context = "Hello, I am"
tokenizer = tiktoken.encoding_for_model("gpt2")
encoded = tokenizer.encode(start_context)
print(encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print(encoded_tensor)
model.eval()
out = generate_text_simple(model, encoded_tensor, 6, 1024)
print(out)
print(len(out[0]))