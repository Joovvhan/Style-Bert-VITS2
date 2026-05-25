import re

for fname in ["Data/kss/train.list", "Data/kss/esd.list"]:
    with open(fname, encoding="utf-8") as f:
        lines = f.readlines()
    matches = []
    for line in lines:
        parts = line.split("|")
        text = parts[3].strip() if len(parts) > 3 else ""
        if re.search(r"[0-9]", text):
            matches.append(text)
    print(f"{fname}: 숫자 포함 {len(matches)}개")
    for t in matches[:10]:
        print(f"  {t}")
