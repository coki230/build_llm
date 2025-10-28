import gptDatasetV1 as gd
data = gd.GptDatasetV1("verdict.txt", 6, 1)
print(len(data))
train_ratio = 0.9
train_len = int(len(data) * train_ratio)
train_data, val_data = data[:train_len], data[train_len:]