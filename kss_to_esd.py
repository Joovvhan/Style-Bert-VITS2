"""
KSS transcript.v.1.x.txt → esd.list 변환 스크립트

KSS transcript 형식:
  {wav}|{원문}|{정제문}|{정제문2}|{길이}|{영문}

출력 esd.list 형식 (preprocess_text_ko.py 입력):
  {wav_path}|kss|KO|{text}

Usage:
    uv run python kss_to_esd.py
    uv run python kss_to_esd.py --transcript Data/transcript.v.1.4.txt \
                                 --kss-dir    Data/kss \
                                 --output     Data/kss/esd.list \
                                 --text-field 2
"""

import argparse
from pathlib import Path


def convert(transcript_path: Path, kss_dir: Path, output_path: Path, text_field: int) -> None:
    lines_ok = 0
    lines_skip = 0

    with (
        transcript_path.open("r", encoding="utf-8") as fin,
        output_path.open("w", encoding="utf-8") as fout,
    ):
        for lineno, raw in enumerate(fin, 1):
            raw = raw.rstrip("\n")
            if not raw:
                continue

            parts = raw.split("|")
            if len(parts) < text_field:
                print(f"[skip] line {lineno}: 필드 수 부족 ({len(parts)}개): {raw[:60]}")
                lines_skip += 1
                continue

            rel_wav = parts[0].strip()          # e.g. "1/1_0000.wav"
            text    = parts[text_field - 1].strip()  # 1-indexed

            wav_path = kss_dir / rel_wav
            if not wav_path.is_file():
                print(f"[skip] line {lineno}: 파일 없음: {wav_path}")
                lines_skip += 1
                continue

            if not text:
                print(f"[skip] line {lineno}: 텍스트 비어있음")
                lines_skip += 1
                continue

            fout.write(f"{wav_path}|kss|KO|{text}\n")
            lines_ok += 1

    print(f"\n완료: {lines_ok}개 변환, {lines_skip}개 건너뜀")
    print(f"출력: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transcript",
        default="Data/transcript.v.1.4.txt",
        help="KSS transcript 파일 경로",
    )
    parser.add_argument(
        "--kss-dir",
        default="Data/kss",
        help="KSS wav 파일이 있는 폴더",
    )
    parser.add_argument(
        "--output",
        default="Data/kss/esd.list",
        help="출력 esd.list 경로",
    )
    parser.add_argument(
        "--text-field",
        type=int,
        default=2,
        help="사용할 텍스트 필드 번호 (1-indexed, 기본값: 2=원문)",
    )
    args = parser.parse_args()

    convert(
        transcript_path=Path(args.transcript),
        kss_dir=Path(args.kss_dir),
        output_path=Path(args.output),
        text_field=args.text_field,
    )
