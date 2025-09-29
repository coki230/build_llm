import torch
import tiktoken
import util
import GPTModel

tn = tiktoken.encoding_for_model("gpt2")
file = open("verdict.txt", "r").read()
total_char = len(file)
total_tokens = len(tn.encode(file))

print(total_char, total_tokens)

train_ratio = 0.9
train_data = file[:int(total_char*train_ratio)]
test_data = file[int(total_char*train_ratio):]

train_loader = util.wrap_as_dataloader(train_data, 2, util.GPT_CONFIG_124M["context_length"],
                                       util.GPT_CONFIG_124M["stride"], True, True)
test_loader = util.wrap_as_dataloader(test_data, 2, util.GPT_CONFIG_124M["context_length"],
                                       util.GPT_CONFIG_124M["stride"], False, False)


# for x,y in test_loader:
#     print(x.shape, y.shape)

torch.autograd.set_detect_anomaly(True)

model = GPTModel.GPTModel(util.GPT_CONFIG_124M)
# with torch.no_grad():
#     train_loss = util.calc_loss_loader(model, train_loader)
#     test_loss = util.calc_loss_loader(model, test_loader)
#
# print(train_loss, test_loss)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
num_epochs = 3
train_losses, val_losses, track_tokens_seen = util.train_model_simple(model, train_loader,
                                                                      test_loader, optimizer, num_epochs,
                                                                      5, 1, "Hello, I am", tn)