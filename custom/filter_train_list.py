"""
train/val 리스트에서 알파벳·숫자가 포함된 문장을 제거하는 스크립트.

필터링 결과를 custom/output/filtered/ 에 먼저 저장하여 사용자 검토 후 적용.

Usage:
    uv run python custom/filter_train_list.py
    uv run python custom/filter_train_list.py --preview-removed   # 제거 대상 문장 출력
"""

import argparse
import re
from pathlib import Path


def has_non_korean(text: str) -> bool:
    return bool(re.search(r"[a-zA-Z0-9]", text))


def filter_list(src: Path, dst: Path, preview_removed: bool) -> tuple[int, int]:
    with open(src, encoding="utf-8") as f:
        lines = f.readlines()

    kept, removed = [], []
    for line in lines:
        parts = line.split("|")
        text = parts[3] if len(parts) > 3 else ""
        if has_non_korean(text):
            removed.append(line)
        else:
            kept.append(line)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(kept)

    if preview_removed:
        print(f"\n[{src.name}] 제거 대상 {len(removed)}문장:")
        for line in removed:
            parts = line.split("|")
            text = parts[3].strip() if len(parts) > 3 else line.strip()
            print(f"  {parts[0]}  |  {text}")

    return len(kept), len(removed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-removed", action="store_true",
                        help="제거 대상 문장을 화면에 출력")
    args = parser.parse_args()

    targets = [
        (Path("Data/kss/train.list"),       Path("custom/output/filtered/train.list")),
        (Path("Data/kss/train_g2pk2.list"), Path("custom/output/filtered/train_g2pk2.list")),
    ]

    print(f"{'파일':<25} {'유지':>6} {'제거':>6} {'저장 경로'}")
    print("-" * 70)
    for src, dst in targets:
        kept, removed = filter_list(src, dst, args.preview_removed)
        print(f"{src.name:<25} {kept:>6} {removed:>6}   → {dst}")

    print("\n검토 후 Data/kss/ 에 복사하려면:")
    print("  copy custom\\output\\filtered\\train.list Data\\kss\\train.list")
    print("  copy custom\\output\\filtered\\train_g2pk2.list Data\\kss\\train_g2pk2.list")


if __name__ == "__main__":
    main()
