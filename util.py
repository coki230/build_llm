import tiktoken
from GPT_model import generate_text_simple
import torch
import GPT_model
import numpy as np

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_layers": 1,
    "n_head": 12,
    "drop_rate": 0.1,
    "stride": 1,
    "qkv_bias": False
}

class GPTDatasetV1(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, max_len, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_len, stride):
            input_chunk = token_ids[i:i+max_len]
            target_chunk = token_ids[i+stride:i+max_len+stride]
            if len(input_chunk) == max_len and len(target_chunk) == max_len:
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endOfText|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def wrap_as_dataloader(dataset, batch_size, max_len, stride, drop_last, shuffle):
    tokenizer = tiktoken.encoding_for_model("gpt2")
    dataset = GPTDatasetV1(dataset, tokenizer, max_len, stride)
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
    return dl

def calc_loss_batch(model, inputs, targets):

    logits = model(inputs)

    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
    return loss

def calc_loss_loader(model, loader, num_batches=None):
    batch = 0
    if num_batches is None:
        batch = len(loader)
    else:
        batch = min(num_batches, len(loader))
    total_loss = 0
    for i, (inputs, targets) in enumerate(loader):
        if i < batch:
            loss = calc_loss_batch(model, inputs, targets)
            total_loss += loss
        else:
            break
    return total_loss / batch

def train_model_simple(model, train_loader, valid_loader, optimizer, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(model, input_batch, target_batch)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, valid_loader, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch: {epoch+1}/{num_epochs}, Step: {global_step}, Loss: {loss:.4f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Tokens Seen: {tokens_seen}")
        generate_and_print_sample(model, start_context, tokenizer)

    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, valid_loader, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(model, train_loader, eval_iter)
        val_loss = calc_loss_loader(model, valid_loader, eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, start_context, tokenizer):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer)
    with torch.no_grad():
        token_ids = generate_text_simple(model, encoded, 50, context_size)
        decoded_text = token_ids_to_text(token_ids, tokenizer)
    model.train()

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError("Shape of left and right must be equal ", left.shape, "--", right.shape)
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(model, params):
    model.pos_emb.weight = assign(model.pos_emb.weight, params["wpe"])
    model.tok_emb.weight = assign(model.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        model.trf_blocks[b].att.W_query.weight = assign(model.trf_blocks[b].att.W_query.weight, q_w.T)
        model.trf_blocks[b].att.W_key.weight = assign(model.trf_blocks[b].att.W_key.weight, k_w.T)
        model.trf_blocks[b].att.W_value.weight = assign(model.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, k_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        model.trf_blocks[b].att.W_query.bias = assign(model.trf_blocks[b].att.W_query.bias, q_b.T)
        model.trf_blocks[b].att.W_key.bias = assign(model.trf_blocks[b].att.W_key.bias, k_b.T)
        model.trf_blocks[b].att.W_value.bias = assign(model.trf_blocks[b].att.W_value.bias, k_b.T)

        model.trf_blocks[b].att.last_Layer.weight = assign(model.trf_blocks[b].att.last_Layer.weight,
                                                         params["blocks"][b]["attn"]["c_proj"]["w"].T)
        model.trf_blocks[b].att.last_Layer.bias = assign(model.trf_blocks[b].att.last_Layer.bias,
                                                         params["blocks"][b]["attn"]["c_proj"]["b"])

        model.trf_blocks[b].ff.layers[0].weight = assign(model.trf_blocks[b].ff.layers[0].weight,
                                                         params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        model.trf_blocks[b].ff.layers[0].bias = assign(model.trf_blocks[b].ff.layers[0].bias,
                                                         params["blocks"][b]["mlp"]["c_fc"]["b"])

        model.trf_blocks[b].ff.layers[2].weight = assign(model.trf_blocks[b].ff.layers[2].weight,
                                                         params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        model.trf_blocks[b].ff.layers[2].bias = assign(model.trf_blocks[b].ff.layers[2].bias,
                                                         params["blocks"][b]["mlp"]["c_proj"]["b"])


        model.trf_blocks[b].norm1.scale = assign(model.trf_blocks[b].norm1.scale,
                                                         params["blocks"][b]["ln_1"]["g"])
        model.trf_blocks[b].norm1.shift = assign(model.trf_blocks[b].norm1.shift,
                                                         params["blocks"][b]["ln_1"]["b"])
        model.trf_blocks[b].norm2.scale = assign(model.trf_blocks[b].norm2.scale,
                                                         params["blocks"][b]["ln_2"]["g"])
        model.trf_blocks[b].norm2.shift = assign(model.trf_blocks[b].norm2.shift,
                                                         params["blocks"][b]["ln_2"]["b"])

        model.final_norm.scale = assign(model.final_norm.scale, params["g"])
        model.final_norm.shift = assign(model.final_norm.scale, params["b"])
        model.out_head.weight = assign(model.out_head.weight, params["wte"])


    return model

def test():
    model = GPTModel.GPTModel(GPT_CONFIG_124M)
    idx = text_to_token_ids("Hello, I am", tiktoken.encoding_for_model("gpt2"))
    max_new_tokens = 10
    context_size = 1024
    out = generate_text_simple(model, idx, max_new_tokens, context_size)
    print(token_ids_to_text(out, tiktoken.encoding_for_model("gpt2")))

# test()