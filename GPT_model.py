import torch.nn as nn
from transformer_block import TransformerBlock
import torch
from layer_norm import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])

    def forward(self, inputs):
        batch_size, seq_len = inputs.shape
        print(inputs.shape)
        tok_embeds = self.tok_emb(inputs)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=inputs.device))

        tok_and_ops = tok_embeds + pos_embeds
        tok_and_ops = self.drop_emb(tok_and_ops)
        tok_and_ops = self.trf_blocks(tok_and_ops)
        tok_and_ops = self.final_norm(tok_and_ops)

        logits = self.out_head(tok_and_ops)

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

def calculate_loss(result, target):
    return torch.nn.functional.cross_entropy(result.flatten(0, 1), target.flatten())
