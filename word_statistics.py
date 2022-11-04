
from util.pertrel_oss_helper import init_clients
from collections import Counter
from tqdm import tqdm

path = "s3://minedojo/feats/test/"
client = init_clients(1)[0]
contents = list(client.list(path))
contents = [name for name in contents]

start = -4
end = 4
lens = Counter()
total_lens = 0
words = Counter()

for name in tqdm(contents):
    data = client.load_npz(path + name).item()
    masks = (data["starts"] > start) & (data["starts"] < end)
    text = " ".join(data["words"][masks])
    for word in data["words"][masks]:
        words[word.item()] += 1
    text_len = masks.sum().item()
    lens[text_len] += 1
    total_lens += text_len

print(f"[{start}, {end}] mean length: {total_lens / len(contents)}")


print('1')
