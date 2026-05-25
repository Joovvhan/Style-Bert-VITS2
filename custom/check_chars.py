import re

with open("Data/kss/train.list", encoding="utf-8") as f:
    lines = f.readlines()

pattern = re.compile(r"[^가-힣㄰-㆏\s\.,!?\-'\"\(\)\[\]~…·:;%]")
samples = []
for line in lines:
    parts = line.split("|")
    text = parts[3].strip() if len(parts) > 3 else ""
    m = pattern.findall(text)
    if m:
        samples.append((text, set(m)))

for text, chars in samples[:30]:
    print(sorted(chars), " | ", text)
