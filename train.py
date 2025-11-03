import gpt_dataset_v1 as gd
import torch
import util as ut
import GPT_model as gpt
import tiktoken


#------------------------------------
# train the model of GPTModel
#------------------------------------
# f = open("verdict.txt", "r")
# all_text = f.read()
# train_ratio = 0.9
# train_len = int(len(all_text) * train_ratio)
# train_data_set, val_data_set = all_text[:train_len], all_text[train_len:]
#
# train_data = gd.GptDatasetV1(train_data_set, ut.GPT_CONFIG_124M.get("context_length"), ut.GPT_CONFIG_124M.get("stride"))
# val_data = gd.GptDatasetV1(val_data_set, ut.GPT_CONFIG_124M.get("context_length"), ut.GPT_CONFIG_124M.get("stride"))
# # print(len(train_data))
# # print(len(val_data))
# t_data = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=True)
# v_data = torch.utils.data.DataLoader(val_data, batch_size=5, shuffle=True)
# # print(len(t_data))
# # print(len(v_data))
# # print(next(iter(t_data)))
#
# GPT_CONFIG_124M = {
#     "vocab_size": 50257,
#     "context_length": 1024,
#     "emb_dim": 768,
#     "n_layers": 12,
#     "n_head": 12,
#     "drop_rate": 0.1,
#     "qkv_bias": False
# }
# model = gpt.GPTModel(GPT_CONFIG_124M)
# device = "mps"
# model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# for epoch in range(3):
#     model.train()
#     i = 0
#     for inputs, targets in t_data:
#         i = i + 1
#         optimizer.zero_grad()
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = model(inputs)
#         loss = gpt.calculate_loss(outputs, targets)
#         loss.backward()
#         optimizer.step()
#         if i % 10 == 0:
#             print(epoch, loss.item())
#     # check the validation
#     model.eval()
#     with torch.no_grad():
#         for inputs, targets in v_data:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             loss = gpt.calculate_loss(outputs, targets)
#             print("validation mode epoch ", epoch, " loss is " ,loss.item())
# torch.save(model.state_dict(), "model.pt")


#------------------------------------
# test the model of GPTModel
#------------------------------------

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_layers": 12,
    "n_head": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
model = gpt.GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pt"))
model.eval()
text = "how it would be"
tokenize = tiktoken.encoding_for_model("gpt2")
tokens = tokenize.encode(text)
# print(tokens)
# output = model(torch.tensor([tokens]))
# print(output.shape)
# output_ids = torch.argmax(output, dim=-1).flatten()
# print(output_ids)
# print(tokenize.decode(output_ids.squeeze(0).tolist()))
out_ids = gpt.generate_text_simple(model, torch.tensor(tokens).unsqueeze(0), max_new_tokens=10, context_size=5)
print(out_ids)
print(tokenize.decode(out_ids.squeeze(0).tolist()))

