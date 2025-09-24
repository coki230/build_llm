import re

class Tokenizer:
    def __init__(self, all_text):
        self.all_text = all_text
        self.key_value, self.value_key = self.parse_text()
        pass

    def split_text(self, text):
        split_text = re.split("([.,?!()\"\';]|--|\\s)", text)
        return split_text

    def parse_text(self):
        split_text_list = self.split_text(self.all_text)
        split_text_list.extend(["<|undefined|>", "<|endOfText|>"])
        index_text = enumerate(split_text_list)
        key_value = {index:val for index, val in index_text}
        value_key = {val:index for index, val in key_value.items()}
        return key_value, value_key

    def get_ids(self, text):
        return [self.value_key[token] if token in self.value_key else self.value_key["<|undefined|>"] for token in self.split_text(text)]

    def get_text(self, ids):
        return "".join([self.key_value[index] for index in ids])


f = open("verdict.txt", "r")
all_text = f.read()
tokenizer = Tokenizer(all_text)
text1 = "I like to fancy"
text2 = "Hello lily, just a test."
text = " <|endOfText|> ".join([text1, text2])
print(text)
ids = tokenizer.get_ids(text)
print(ids)
print(tokenizer.get_text(ids))