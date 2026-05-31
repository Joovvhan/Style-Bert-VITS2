"""
custom/prepare_dataset_ko.py

jinx_ko, YSOYA21 한국어 데이터셋을 Style-Bert-VITS2 학습 형식으로 변환한다.

출력 구조:
  Data/<name>/wavs/*.wav   -- 44100 Hz 리샘플링된 WAV
  Data/<name>/esd.list     -- wav_path|speaker|KO|text  (4필드)

이후 실행 순서:
  1. preprocess_text_ko.py  → train.list / val.list
  2. style_gen.py           → .npy  스타일 벡터
  3. bert_gen.py            → .bert.pt
  4. train_ms_ko.py
"""

import argparse
import json
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
TARGET_SR = 44100

JINX_SRC  = Path(r"C:\Users\joovv\Documents\Qwen3-TTS-Custom\custom_data\jinx")
YSOYA_SRC = Path(r"C:\Users\joovv\Documents\Qwen3-TTS-Custom\custom_data\YSOYA21")


# ---------------------------------------------------------------------------

def _resample_and_save(src: Path, dst: Path, target_sr: int) -> None:
    data, sr = sf.read(str(src), always_2d=True)
    data = data.mean(axis=1)  # stereo → mono (mono는 그대로)
    if sr != target_sr:
        from scipy.signal import resample_poly
        g = gcd(sr, target_sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
    sf.write(str(dst), data.astype(np.float32), target_sr)


def _rel(path: Path) -> str:
    """프로젝트 루트 기준 상대 경로 (백슬래시 → 슬래시)."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------

def prepare_jinx(src_dir: Path, out_dir: Path, speaker: str, target_sr: int) -> list[str]:
    """transcription.json + segment/*.wav → esd.list"""
    meta = json.loads((src_dir / "transcription.json").read_text(encoding="utf-8"))
    wav_src = src_dir / "segment"
    wav_dst = out_dir / "wavs"
    wav_dst.mkdir(parents=True, exist_ok=True)

    lines = []
    skipped = 0
    for fname, text in tqdm(sorted(meta.items()), desc="jinx_ko"):
        src_wav = wav_src / fname
        if not src_wav.exists():
            skipped += 1
            continue
        dst_wav = wav_dst / fname
        _resample_and_save(src_wav, dst_wav, target_sr)
        lines.append(f"{_rel(dst_wav)}|{speaker}|KO|{text}\n")

    if skipped:
        print(f"  [jinx_ko] skipped {skipped} missing WAV files")
    return lines


def prepare_ysoya21(src_dir: Path, out_dir: Path, speaker: str, target_sr: int) -> list[str]:
    """YSOYA21.jsonl + YSOYA21_*.wav → esd.list"""
    raw = (src_dir / "YSOYA21.jsonl").read_text(encoding="utf-8")
    records = [json.loads(l) for l in raw.splitlines() if l.strip()]
    wav_dst = out_dir / "wavs"
    wav_dst.mkdir(parents=True, exist_ok=True)

    lines = []
    skipped = 0
    for rec in tqdm(records, desc="YSOYA21"):
        fname = rec["file"]
        text  = rec["text"]
        src_wav = src_dir / fname
        if not src_wav.exists():
            skipped += 1
            continue
        dst_wav = wav_dst / fname
        _resample_and_save(src_wav, dst_wav, target_sr)
        lines.append(f"{_rel(dst_wav)}|{speaker}|KO|{text}\n")

    if skipped:
        print(f"  [YSOYA21] skipped {skipped} missing WAV files")
    return lines


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["jinx_ko", "YSOYA21"],
                        choices=["jinx_ko", "YSOYA21"],
                        help="변환할 데이터셋 (기본: 둘 다)")
    parser.add_argument("--target-sr", type=int, default=TARGET_SR)
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "Data",
                        help="출력 루트 디렉토리 (기본: Data/)")
    args = parser.parse_args()

    if "jinx_ko" in args.datasets:
        print("=== jinx_ko ===")
        out_dir = args.out_root / "jinx_ko"
        lines = prepare_jinx(JINX_SRC, out_dir, "jinx_ko", args.target_sr)
        esd = out_dir / "esd.list"
        esd.write_text("".join(lines), encoding="utf-8")
        print(f"  {len(lines)} entries → {esd.relative_to(PROJECT_ROOT)}")

    if "YSOYA21" in args.datasets:
        print("=== YSOYA21 ===")
        out_dir = args.out_root / "YSOYA21"
        lines = prepare_ysoya21(YSOYA_SRC, out_dir, "YSOYA21", args.target_sr)
        esd = out_dir / "esd.list"
        esd.write_text("".join(lines), encoding="utf-8")
        print(f"  {len(lines)} entries → {esd.relative_to(PROJECT_ROOT)}")

    print("\n다음 단계:")
    for ds in args.datasets:
        print(f"\n  [{ds}]")
        print(f"  uv run python preprocess_text_ko.py \\")
        print(f"      --transcription-path Data/{ds}/esd.list \\")
        print(f"      --train-path Data/{ds}/train.list \\")
        print(f"      --val-path   Data/{ds}/val.list \\")
        print(f"      --config-path Data/{ds}/config.json \\")
        print(f"      --use-g2pk2")
        print(f"  uv run python style_gen.py -c Data/{ds}/config.json")
        print(f"  uv run python bert_gen.py  -c Data/{ds}/config.json")


if __name__ == "__main__":
    main()
