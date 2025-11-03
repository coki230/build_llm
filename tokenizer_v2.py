import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text1 = "I like to fancy"
text2 = "Hello lily, just a test."
text = " <|endOfText|> ".join([text1, text2])
print(text)
ids = tokenizer.encode(text, allowed_special={"<|endOfText|>"})
print(ids)
print(tokenizer.decode(ids))