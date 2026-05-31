"""
custom/filter_dataset_ko.py

esd.list에서 영어/숫자가 포함된 라인을 검사하고, 선택적으로 제거한다.

사용법:
    # 검사만 (수정 없음)
    uv run python custom/filter_dataset_ko.py --esd Data/jinx_ko/esd.list

    # 검사 + 필터링 (esd.list를 in-place로 수정)
    uv run python custom/filter_dataset_ko.py --esd Data/jinx_ko/esd.list --filter

    # 두 데이터셋 한꺼번에
    uv run python custom/filter_dataset_ko.py \
        --esd Data/jinx_ko/esd.list Data/YSOYA21/esd.list --filter
"""

import argparse
import re
from pathlib import Path


# 문제 패턴별 분류
_HAS_LATIN = re.compile(r"[A-Za-z]")
_HAS_DIGIT = re.compile(r"[0-9０-９]")
# 한글·공백·일반 문장부호·영숫자 외의 문자
_HAS_OTHER = re.compile(
    r"[^가-힣ㄱ-ㆎ"   # 한글 음절·자모
    r"\s.,!?~…·\-"          # 공백, 문장부호
    r"A-Za-z0-9０-９]"        # 영숫자
)


def classify(text: str) -> list[str]:
    flags = []
    if _HAS_LATIN.search(text):
        flags.append("영어")
    if _HAS_DIGIT.search(text):
        flags.append("숫자")
    if _HAS_OTHER.search(text):
        flags.append("기타특수문자")
    return flags


def check_esd(esd_path: Path, do_filter: bool) -> None:
    lines = esd_path.read_text(encoding="utf-8").splitlines(keepends=True)
    total = len(lines)

    problem_lines: list[tuple[int, str, list[str]]] = []
    clean_lines:   list[str] = []

    for i, line in enumerate(lines, 1):
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        text = parts[3]
        flags = classify(text)
        if flags:
            problem_lines.append((i, line.rstrip(), flags))
        else:
            clean_lines.append(line)

    print(f"\n{'='*60}")
    print(f"{esd_path}  (전체 {total}줄)")
    print(f"{'='*60}")

    if not problem_lines:
        print("  문제 없음 — 영어/숫자 포함 라인 없음")
        return

    print(f"  문제 라인: {len(problem_lines)} / {total}\n")
    for lineno, raw, flags in problem_lines:
        text = raw.split("|")[3] if "|" in raw else raw
        print(f"  L{lineno:4d} [{', '.join(flags)}]  {text[:80]}")

    if do_filter:
        backup = esd_path.with_suffix(".esd.list.bak")
        esd_path.rename(backup)
        esd_path.write_text("".join(clean_lines), encoding="utf-8")
        print(f"\n  → {len(problem_lines)}줄 제거 완료")
        print(f"  → 원본 백업: {backup.name}")
        print(f"  → 남은 라인: {len(clean_lines)}")
    else:
        print(f"\n  (--filter 옵션을 추가하면 위 {len(problem_lines)}줄을 제거합니다)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esd", nargs="+", required=True,
                        help="검사할 esd.list 경로 (여러 개 가능)")
    parser.add_argument("--filter", action="store_true",
                        help="문제 라인을 실제로 제거 (원본은 .bak으로 백업)")
    args = parser.parse_args()

    for path_str in args.esd:
        path = Path(path_str)
        if not path.exists():
            print(f"[ERROR] 파일 없음: {path}")
            continue
        check_esd(path, do_filter=args.filter)


if __name__ == "__main__":
    main()
