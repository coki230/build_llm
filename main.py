import re
# read all characters in a file
f = open("verdict.txt", "r")
all_text = f.read()
print(len(all_text))
split_text = re.split("([.,?!()\"\';]|--|\\s)", all_text)
print(len(split_text))
order_list = sorted(set(split_text))
print(len(order_list))
text_map = {token:index for index, token in enumerate(order_list)}
for token, index in text_map.items():
    if index > 100:
        break
    print(token, index)