"""
val 리스트 후보 문장 선정 스크립트.

jamo G2P와 g2pk2 G2P의 phoneme 시퀀스 차이가 가장 큰 문장을 찾아냄.
겹받침 수를 별도 컬럼으로 출력한다.

Usage:
    uv run python custom/find_val_candidates.py
    uv run python custom/find_val_candidates.py --top 50
    uv run python custom/find_val_candidates.py --top 50 --workers 4
    uv run python custom/find_val_candidates.py --top 50 --out custom/output/candidates.tsv
"""

import argparse
import multiprocessing
from pathlib import Path

from tqdm import tqdm


# 겹받침 종성 인덱스 (유니코드 한글 음절 기준)
# 종성 = (code - 0xAC00) % 28
# 3=ㄳ 5=ㄵ 6=ㄶ 9=ㄺ 10=ㄻ 11=ㄼ 12=ㄽ 13=ㄾ 14=ㄿ 15=ㅀ 18=ㅄ
_DOUBLE_FINAL_IDX = {3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 18}
_KO_SYL_START = 0xAC00
_KO_SYL_END = 0xD7A3


def diff_count(a: list[str], b: list[str]) -> int:
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[lb]


def count_double_finals(text: str) -> int:
    count = 0
    for ch in text:
        code = ord(ch)
        if _KO_SYL_START <= code <= _KO_SYL_END:
            if (code - _KO_SYL_START) % 28 in _DOUBLE_FINAL_IDX:
                count += 1
    return count


def _worker(line: str) -> dict | None:
    from style_bert_vits2.nlp.korean.g2p import g2p as g2p_jamo
    from style_bert_vits2.nlp.korean.g2p_g2pk2 import g2p as g2p_g2pk2_fn
    from style_bert_vits2.nlp.korean.normalizer import normalize_text

    parts = line.strip().split("|")
    if len(parts) < 4:
        return None
    wav_path, _, lang, text = parts[0], parts[1], parts[2], parts[3]
    if lang != "KO":
        return None
    try:
        norm = normalize_text(text)
        phones_jamo, _, _ = g2p_jamo(norm)
        phones_g2pk2, _, _ = g2p_g2pk2_fn(norm)
    except Exception:
        return None
    d = diff_count(phones_jamo, phones_g2pk2)
    if d == 0:
        return None
    return {
        "wav": wav_path,
        "text": text,
        "diff": d,
        "double_finals": count_double_finals(text),
        "jamo": phones_jamo,
        "g2pk2": phones_g2pk2,
    }


def analyze(esd_path: Path, workers: int) -> list[dict]:
    with open(esd_path, encoding="utf-8") as f:
        lines = f.readlines()

    results = []
    if workers == 1:
        for line in tqdm(lines, desc="분석 중", unit="문장"):
            r = _worker(line)
            if r is not None:
                results.append(r)
    else:
        with multiprocessing.Pool(workers) as pool:
            for r in tqdm(
                pool.imap(_worker, lines, chunksize=64),
                total=len(lines),
                desc=f"분석 중 (workers={workers})",
                unit="문장",
            ):
                if r is not None:
                    results.append(r)

    print(f"diff>0: {len(results)} / {len(lines)} 문장")
    return results


def write_tsv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True) if not out_path.parent.is_file() else None
    if out_path.is_dir():
        raise ValueError(f"출력 경로가 디렉토리입니다: {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("rank\tdiff\tdouble_finals\twav\ttext\tjamo\tg2pk2\n")
        for rank, c in enumerate(rows, 1):
            f.write(
                f"{rank}\t{c['diff']}\t{c['double_finals']}\t{c['wav']}\t{c['text']}\t"
                f"{' '.join(c['jamo'])}\t{' '.join(c['g2pk2'])}\n"
            )
    print(f"→ {out_path} 저장 완료")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--esd", default="Data/kss/esd.list")
    parser.add_argument("--workers", type=int, default=8,
                        help="병렬 프로세스 수 (0=cpu_count)")
    parser.add_argument("--out-dir", default="custom/output", help="출력 디렉토리")
    args = parser.parse_args()

    workers = multiprocessing.cpu_count() if args.workers == 0 else args.workers
    results = analyze(Path(args.esd), workers)

    out_dir = Path(args.out_dir)

    by_diff = sorted(results, key=lambda x: x["diff"], reverse=True)
    write_tsv(by_diff, out_dir / "candidates_by_diff.tsv")

    by_double = sorted(results, key=lambda x: (x["double_finals"], x["diff"]), reverse=True)
    write_tsv(by_double, out_dir / "candidates_by_double_finals.tsv")


if __name__ == "__main__":
    main()
