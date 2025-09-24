import torch
import tiktoken
import gptDatasetV1



class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super(MultiHeadAttention, self).__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.last_Layer = torch.nn.Linear(d_out, d_out)


    def forward(self, inputs):
        query = self.W_query(inputs)
        key = self.W_key(inputs)
        value = self.W_value(inputs)

        attn_scores = query @ key.mT
        attn_scores = attn_scores / (key.shape[-1] ** 0.5)
        attn_scores = torch.nn.functional.softmax(attn_scores, dim=-1)

        output = attn_scores @ value
        output = self.last_Layer(output)
        return output

# text = "Hello lily, just a test."
# en_model = tiktoken.encoding_for_model("gpt2")
# ids = en_model.encode(text, allowed_special={"<|endOfText|>"})
# print(ids)
# print(len(ids))
# att = MultiHeadAttention(len(ids), len(ids))
# print(att(torch.tensor([ids], dtype=torch.float32)))

data = gptDatasetV1.GptDatasetV1("verdict.txt", 8, 8)
inputs, targets = next(iter(data))
print(inputs)





