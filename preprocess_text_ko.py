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
import multiprocessing
from collections import defaultdict
from functools import partial
from pathlib import Path
from random import sample
from typing import Optional

from tqdm import tqdm

from config import get_config
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp.korean.g2p import g2p as g2p_jamo
from style_bert_vits2.nlp.korean.g2p_g2pk2 import g2p as g2p_g2pk2
from style_bert_vits2.nlp.korean.normalizer import normalize_text
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


preprocess_text_config = get_config().preprocess_text_config


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _write_error(log_path: Path, line: str, error: Exception) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{line.strip()}\n{error}\n\n")


def _process_one(args: tuple) -> tuple[str | None, str | None]:
    """Worker function for multiprocessing. Returns (result, error_str)."""
    line, transcription_path_str, correct_path, use_g2pk2 = args
    try:
        return process_line(line, Path(transcription_path_str), correct_path, use_g2pk2), None
    except Exception as e:
        return None, f"{line.strip()}\n{e}\n\n"


def process_line(
    line: str, transcription_path: Path, correct_path: bool, use_g2pk2: bool = False
) -> str:
    parts = line.strip().split("|")
    if len(parts) != 4:
        raise ValueError(f"Invalid format (expected 4 fields): {line.strip()}")
    utt, spk, language, text = parts
    if language != "KO":
        raise ValueError(f"Expected language KO, got {language!r}")

    norm_text = normalize_text(text)
    g2p_fn = g2p_g2pk2 if use_g2pk2 else g2p_jamo
    phones, tones, word2ph = g2p_fn(norm_text)

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
    use_g2pk2: bool = False,
    workers: int = 1,
) -> None:
    if not cleaned_path:
        cleaned_path = transcription_path.with_name(transcription_path.name + ".cleaned")

    error_log = transcription_path.parent / "text_error.log"
    if error_log.exists():
        error_log.unlink()
    error_count = 0

    lines = transcription_path.read_text(encoding="utf-8").splitlines(keepends=True)
    total = len(lines)

    with cleaned_path.open("w", encoding="utf-8") as fout:
        if workers > 1:
            tasks = [(line, str(transcription_path), correct_path, use_g2pk2) for line in lines]
            with multiprocessing.Pool(workers) as pool:
                for result, err in tqdm(
                    pool.imap(_process_one, tasks, chunksize=32),
                    file=SAFE_STDOUT, total=total, dynamic_ncols=True,
                ):
                    if err:
                        logger.error(f"Error:\n{err.strip()}")
                        with error_log.open("a", encoding="utf-8") as ef:
                            ef.write(err)
                        error_count += 1
                    else:
                        fout.write(result)
        else:
            for line in tqdm(lines, file=SAFE_STDOUT, total=total, dynamic_ncols=True):
                try:
                    fout.write(process_line(line, transcription_path, correct_path, use_g2pk2))
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
    parser.add_argument(
        "--use-g2pk2",
        action="store_true",
        help="g2pk2로 음운 변동 규칙 적용 후 자모 분해 (기본값: 단순 자모 분해)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="병렬 처리 프로세스 수 (기본값: 1, 0=CPU 코어 수)",
    )
    args = parser.parse_args()

    n_workers = args.workers if args.workers > 0 else multiprocessing.cpu_count()

    preprocess(
        transcription_path=Path(args.transcription_path),
        cleaned_path=Path(args.cleaned_path) if args.cleaned_path else None,
        train_path=Path(args.train_path),
        val_path=Path(args.val_path),
        config_path=Path(args.config_path),
        val_per_spk=args.val_per_spk,
        max_val_total=args.max_val_total,
        correct_path=args.correct_path,
        use_g2pk2=args.use_g2pk2,
        workers=n_workers,
    )
