"""
실험용 디렉토리 셋업 스크립트.

Data/kss_exp_jamo/ 및 Data/kss_exp_g2pk2/ 를 생성하고
train.list, val_exp.list, config.json 을 배치한다.

Usage:
    uv run python custom/setup_exp_dirs.py
    uv run python custom/setup_exp_dirs.py --dry-run   # 실제 파일 생성 없이 결과만 출력
"""

import argparse
import json
import shutil
from pathlib import Path

from style_bert_vits2.nlp.korean.g2p import g2p as g2p_jamo
from style_bert_vits2.nlp.korean.g2p_g2pk2 import g2p as g2p_g2pk2
from style_bert_vits2.nlp.korean.normalizer import normalize_text

# ── 실험 설정 ──────────────────────────────────────────────
VAL_WAV_PATHS = [
    "Data\\kss\\3\\3_0881.wav",
    "Data\\kss\\2\\2_0199.wav",
    "Data\\kss\\4\\4_3551.wav",
    "Data\\kss\\3\\3_0041.wav",
    "Data\\kss\\1\\1_0336.wav",
    "Data\\kss\\3\\3_1733.wav",
    "Data\\kss\\3\\3_3145.wav",
    "Data\\kss\\2\\2_0288.wav",
    "Data\\kss\\4\\4_3091.wav",
    "Data\\kss\\3\\3_1975.wav",
    "Data\\kss\\3\\3_3490.wav",
    "Data\\kss\\3\\3_4500.wav",
    "Data\\kss\\3\\3_1105.wav",
    "Data\\kss\\3\\3_1499.wav",
    "Data\\kss\\3\\3_2499.wav",
    "Data\\kss\\3\\3_2566.wav",
    "Data\\kss\\3\\3_3667.wav",
    "Data\\kss\\3\\3_4204.wav",
    "Data\\kss\\4\\4_0201.wav",
    "Data\\kss\\4\\4_3179.wav",
]

EXPERIMENTS = {
    "kss_exp_jamo":  {"filtered_train": "custom/output/filtered/train.list",      "g2p": "jamo"},
    "kss_exp_g2pk2": {"filtered_train": "custom/output/filtered/train_g2pk2.list", "g2p": "g2pk2"},
}

BASE_CONFIG_PATH = Path("Data/kss/config.json")
TARGET_STEPS = 65_000
BATCH_SIZE = 8


def make_list_line(wav: str, spk: str, text: str, use_g2pk2: bool) -> str:
    norm = normalize_text(text)
    fn = g2p_g2pk2 if use_g2pk2 else g2p_jamo
    phones, tones, word2ph = fn(norm)
    return "|".join([
        wav, spk, "KO", norm,
        " ".join(phones),
        " ".join(str(t) for t in tones),
        " ".join(str(w) for w in word2ph),
    ]) + "\n"


def load_esd_index(esd_path: Path) -> dict[str, tuple[str, str]]:
    """wav_path → (speaker, text) 인덱스"""
    index = {}
    with open(esd_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 4:
                index[parts[0]] = (parts[1], parts[3])
    return index


def build_val_list(esd_index: dict, use_g2pk2: bool, dry_run: bool) -> list[str]:
    lines = []
    missing = []
    for wav in VAL_WAV_PATHS:
        if wav not in esd_index:
            missing.append(wav)
            continue
        spk, text = esd_index[wav]
        line = make_list_line(wav, spk, text, use_g2pk2)
        lines.append(line)
        if dry_run:
            print(f"  VAL: {wav.split(chr(92))[-1]}  |  {text}")
    if missing:
        print(f"  [경고] esd.list에서 찾지 못한 파일: {missing}")
    return lines


def compute_epochs(train_count: int) -> int:
    import math
    steps_per_epoch = train_count / BATCH_SIZE
    return math.ceil(TARGET_STEPS / steps_per_epoch)


def setup(exp_name: str, cfg: dict, esd_index: dict, base_cfg: dict,
          dry_run: bool) -> None:
    out_dir = Path("Data") / exp_name
    use_g2pk2 = cfg["g2p"] == "g2pk2"
    filtered_train = Path(cfg["filtered_train"])

    print(f"\n{'[DRY-RUN] ' if dry_run else ''}=== {exp_name} ===")

    # ── train list: val 문장 제외 ────────────────────────────
    with open(filtered_train, encoding="utf-8") as f:
        train_lines = f.readlines()

    val_set = set(VAL_WAV_PATHS)
    train_filtered = [l for l in train_lines if l.split("|")[0] not in val_set]
    removed = len(train_lines) - len(train_filtered)
    print(f"  train: {len(train_lines)} → val 제외 {removed}개 → {len(train_filtered)}개")

    epochs = compute_epochs(len(train_filtered))
    print(f"  epoch: {epochs}  ({len(train_filtered)} / {BATCH_SIZE} x {epochs} ~= {len(train_filtered)//BATCH_SIZE*epochs:,} steps)")

    # ── val list ─────────────────────────────────────────────
    val_lines = build_val_list(esd_index, use_g2pk2, dry_run)
    print(f"  val: {len(val_lines)}개")

    if dry_run:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.list"
    train_path.write_text("".join(train_filtered), encoding="utf-8")

    val_path = out_dir / "val_exp.list"
    val_path.write_text("".join(val_lines), encoding="utf-8")

    # ── config.json ──────────────────────────────────────────
    new_cfg = json.loads(json.dumps(base_cfg))
    new_cfg["model_name"] = exp_name
    new_cfg["data"]["training_files"] = str(train_path).replace("\\", "/")
    new_cfg["data"]["validation_files"] = str(val_path).replace("\\", "/")
    new_cfg["train"]["epochs"] = epochs
    new_cfg["train"]["save_interval"] = 5000
    new_cfg["train"]["eval_interval"] = 1000
    new_cfg["train"]["batch_size"] = BATCH_SIZE
    new_cfg["train"]["keep_ckpts"] = 0

    config_path = out_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(new_cfg, f, indent=2, ensure_ascii=False)

    print(f"  → {out_dir}/  생성 완료")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    esd_index = load_esd_index(Path("Data/kss/esd.list"))
    with open(BASE_CONFIG_PATH, encoding="utf-8") as f:
        base_cfg = json.load(f)

    for exp_name, cfg in EXPERIMENTS.items():
        setup(exp_name, cfg, esd_index, base_cfg, args.dry_run)

    if not args.dry_run:
        print("\n완료. 생성된 폴더:")
        for name in EXPERIMENTS:
            print(f"  Data/{name}/")


if __name__ == "__main__":
    main()
