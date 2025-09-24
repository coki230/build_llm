import torch
from torch.utils.data import Dataset
import tiktoken

class GptDatasetV1(Dataset):
    def __init__(self, file_path, max_len, stride_len):
        f = open(file_path, "r")
        all_text = f.read()
        self.token_encoder = tiktoken.get_encoding("gpt2")
        self.token_ids = self.token_encoder.encode(all_text)
        self.input_ids = []
        self.target_ids = []

        # print(self.token_ids[:10])
        for i in range(0, len(self.token_ids) - max_len, stride_len):
            self.input_ids.append(torch.tensor(self.token_ids[i:i+max_len]))
            self.target_ids.append(torch.tensor(self.token_ids[i+stride_len:i+max_len+stride_len]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

    def get_token_len(self):
        return len(self.token_ids)

    def get_max_id(self):
        return max(self.token_ids)

# g_data_set = GptDatasetV1("verdict.txt", 4, 4)
#
# # it = iter(g_data_set)
# # print(next(it))
# #
# data_loader = torch.utils.data.DataLoader(g_data_set, batch_size=8, shuffle=True)
# inputs, targets = next(iter(data_loader))
# # print(inputs, targets)
# print(inputs.shape, targets.shape)
#
# torch.manual_seed(123)
# in_len = len(g_data_set)
# emb_layer = torch.nn.Embedding(g_data_set.get_max_id(), 256)
# # print(emb_layer.weight)
# # print(emb_layer(torch.tensor([2])))
# print(inputs)
# print(g_data_set.get_token_len())
# token_emb = emb_layer(inputs)
#
# pos_layer = torch.nn.Embedding(g_data_set.get_max_id(), 256)
# pos_emb = pos_layer(torch.arange(4))
# print(token_emb.shape, pos_emb.shape)
# print(token_emb[0], pos_emb[0])
# input_emb = token_emb + pos_emb
# print(input_emb.shape)