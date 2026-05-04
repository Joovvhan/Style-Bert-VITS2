"""
Korean-specific text preprocessing script (Phase 1).

Input  esd.list format : {wav_path}|{speaker}|KO|{text}
Output train/val .list : {wav_path}|{speaker}|KO|{norm_text}|{phones}|{tones}|{word2ph}

Usage:
    uv run python preprocess_text_ko.py \\
        --transcription-path Data/{model_name}/esd.list \\
        --train-path        Data/{model_name}/train.list \\
        --val-path          Data/{model_name}/val.list \\
        --config-path       Data/{model_name}/config.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from random import sample
from typing import Optional

from tqdm import tqdm

from config import get_config
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp.korean.g2p import g2p
from style_bert_vits2.nlp.korean.normalizer import normalize_text
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


preprocess_text_config = get_config().preprocess_text_config


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _write_error(log_path: Path, line: str, error: Exception) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{line.strip()}\n{error}\n\n")


def process_line(line: str, transcription_path: Path, correct_path: bool) -> str:
    parts = line.strip().split("|")
    if len(parts) != 4:
        raise ValueError(f"Invalid format (expected 4 fields): {line.strip()}")
    utt, spk, language, text = parts
    if language != "KO":
        raise ValueError(f"Expected language KO, got {language!r}")

    norm_text = normalize_text(text)
    phones, tones, word2ph = g2p(norm_text)

    if correct_path:
        utt = str(transcription_path.parent / "wavs" / utt)

    return "{}|{}|{}|{}|{}|{}|{}\n".format(
        utt,
        spk,
        language,
        norm_text,
        " ".join(phones),
        " ".join(str(t) for t in tones),
        " ".join(str(w) for w in word2ph),
    )


def preprocess(
    transcription_path: Path,
    cleaned_path: Optional[Path],
    train_path: Path,
    val_path: Path,
    config_path: Path,
    val_per_spk: int,
    max_val_total: int,
    correct_path: bool,
) -> None:
    if not cleaned_path:
        cleaned_path = transcription_path.with_name(transcription_path.name + ".cleaned")

    error_log = transcription_path.parent / "text_error.log"
    if error_log.exists():
        error_log.unlink()
    error_count = 0

    total = _count_lines(transcription_path)
    with (
        transcription_path.open("r", encoding="utf-8") as fin,
        cleaned_path.open("w", encoding="utf-8") as fout,
    ):
        for line in tqdm(fin, file=SAFE_STDOUT, total=total, dynamic_ncols=True):
            try:
                fout.write(process_line(line, transcription_path, correct_path))
            except Exception as e:
                logger.error(f"Error at line:\n{line.strip()}\n{e}")
                _write_error(error_log, line, e)
                error_count += 1

    if error_count:
        logger.warning(
            f"{error_count} lines failed. See {error_log} for details."
        )

    # Build speaker map and split train/val
    spk_utt_map: dict[str, list[str]] = defaultdict(list)
    spk_id_map: dict[str, int] = {}
    current_sid = 0

    with cleaned_path.open("r", encoding="utf-8") as f:
        audio_paths: set[str] = set()
        for line in f:
            utt, spk = line.strip().split("|")[:2]
            if utt in audio_paths:
                logger.warning(f"Duplicate audio: {utt}")
                continue
            if not Path(utt).is_file():
                logger.warning(f"Audio not found: {utt}")
                continue
            audio_paths.add(utt)
            spk_utt_map[spk].append(line)
            if spk not in spk_id_map:
                spk_id_map[spk] = current_sid
                current_sid += 1

    train_list: list[str] = []
    val_list: list[str] = []

    for spk, utts in spk_utt_map.items():
        if val_per_spk == 0:
            train_list.extend(utts)
            continue
        val_idx = set(sample(range(len(utts)), min(val_per_spk, len(utts))))
        for i, utt in enumerate(utts):
            (val_list if i in val_idx else train_list).append(utt)

    if len(val_list) > max_val_total:
        train_list.extend(val_list[max_val_total:])
        val_list = val_list[:max_val_total]

    train_path.write_text("".join(train_list), encoding="utf-8")
    val_path.write_text("".join(val_list), encoding="utf-8")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["data"]["spk2id"] = spk_id_map
    cfg["data"]["n_speakers"] = len(spk_id_map)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    logger.info("Korean preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcription-path", default=preprocess_text_config.transcription_path)
    parser.add_argument("--cleaned-path", default=None)
    parser.add_argument("--train-path", default=preprocess_text_config.train_path)
    parser.add_argument("--val-path", default=preprocess_text_config.val_path)
    parser.add_argument("--config-path", default=preprocess_text_config.config_path)
    parser.add_argument("--val-per-spk", type=int, default=preprocess_text_config.val_per_lang)
    parser.add_argument("--max-val-total", type=int, default=preprocess_text_config.max_val_total)
    parser.add_argument("--correct_path", action="store_true")
    args = parser.parse_args()

    preprocess(
        transcription_path=Path(args.transcription_path),
        cleaned_path=Path(args.cleaned_path) if args.cleaned_path else None,
        train_path=Path(args.train_path),
        val_path=Path(args.val_path),
        config_path=Path(args.config_path),
        val_per_spk=args.val_per_spk,
        max_val_total=args.max_val_total,
        correct_path=args.correct_path,
    )
