from gpt_download import download_and_load_gpt2
import util as u
import GPT_model as g
import tiktoken
import torch
from GPT_model import generate_text_simple

model_configs = {
    "gpt2-small(124M)": {"emb_size": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium(355M)": {"emb_size": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large(774M)": {"emb_size": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl(1558M)": {"emb_size": 1600, "n_layers": 48, "n_heads": 25},
}
model_name = "gpt2-small(124M)"
NEW_CONFIG = u.GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})
gpt_model = g.GPTModel(NEW_CONFIG)

from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

u.load_weights_into_gpt(gpt_model, params)

start_context = "Hello, what is "
tokenizer = tiktoken.encoding_for_model("gpt2")
encoded = tokenizer.encode(start_context)
print(encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print(encoded_tensor)
gpt_model.eval()
out = generate_text_simple(gpt_model, encoded_tensor, 6, 1024)
print(out)
print(tokenizer.decode(out.squeeze(0).tolist()))
print(len(out[0]))