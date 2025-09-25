import tiktoken
from GPTModel import generate_text_simple
import torch

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allow_special={"<|endOfText|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def test():
    model =