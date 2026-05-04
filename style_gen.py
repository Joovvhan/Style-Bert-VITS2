import argparse
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from numpy.typing import NDArray

# ── torchaudio 2.11 compatibility shim ───────────────────────────────────────
# torchaudio 2.11 removed AudioMetaData, info(), list_audio_backends(), and
# replaced load() with a torchcodec-only implementation.
# pyannote.audio 3.x uses these APIs, so we restore them via soundfile.

if not hasattr(torchaudio, "AudioMetaData"):
    @dataclasses.dataclass
    class _AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str
    torchaudio.AudioMetaData = _AudioMetaData  # type: ignore[attr-defined]

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]

if not hasattr(torchaudio, "info"):
    def _info(path, backend=None):
        i = sf.info(path)
        return torchaudio.AudioMetaData(
            sample_rate=i.samplerate,
            num_frames=i.frames,
            num_channels=i.channels,
            bits_per_sample=16,
            encoding="PCM_S",
        )
    torchaudio.info = _info  # type: ignore[attr-defined]

_orig_load = torchaudio.load
def _load_soundfile(uri, frame_offset=0, num_frames=-1, normalize=True,
                    channels_first=True, format=None, buffer_size=4096, backend=None):
    data, sr = sf.read(uri, start=frame_offset, frames=num_frames,
                       always_2d=True, dtype="float32")
    waveform = torch.from_numpy(data.T if channels_first else data)
    return waveform, sr
torchaudio.load = _load_soundfile  # type: ignore[attr-defined]
# ─────────────────────────────────────────────────────────────────────────────


# PyTorch 2.6 changed weights_only default to True; pyannote checkpoints
# contain arbitrary classes that are not in the safe-globals list.
# Patch lightning_fabric's loader (used internally by pyannote) to allow this.
import lightning_fabric.utilities.cloud_io as _cloud_io
_orig_lf_load = _cloud_io._load
def _lf_load_compat(path_or_url, map_location=None, weights_only=None):
    return torch.load(path_or_url, map_location=map_location, weights_only=False)
_cloud_io._load = _lf_load_compat

from pyannote.audio import Inference, Model
from tqdm import tqdm

from config import get_config
from style_bert_vits2.logging import logger
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


config = get_config()

model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
inference = Inference(model, window="whole")
device = torch.device(config.style_gen_config.device)
inference.to(device)


class NaNValueError(ValueError):
    """カスタム例外クラス。NaN値が見つかった場合に使用されます。"""


# 推論時にインポートするために短いが関数を書く
def get_style_vector(wav_path: str) -> NDArray[Any]:
    return inference(wav_path)  # type: ignore


def save_style_vector(wav_path: str):
    try:
        style_vec = get_style_vector(wav_path)
    except Exception as e:
        print("\n")
        logger.error(f"Error occurred with file: {wav_path}, Details:\n{e}\n")
        raise
    # 値にNaNが含まれていると悪影響なのでチェックする
    if np.isnan(style_vec).any():
        print("\n")
        logger.warning(f"NaN value found in style vector: {wav_path}")
        raise NaNValueError(f"NaN value found in style vector: {wav_path}")
    np.save(f"{wav_path}.npy", style_vec)  # `test.wav` -> `test.wav.npy`


def process_line(line: str):
    wav_path = line.split("|")[0]
    try:
        save_style_vector(wav_path)
        return line, None
    except NaNValueError:
        return line, "nan_error"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=config.style_gen_config.config_path
    )
    parser.add_argument(
        "--num_processes", type=int, default=config.style_gen_config.num_processes
    )
    args, _ = parser.parse_known_args()
    config_path: str = args.config
    num_processes: int = args.num_processes

    hps = HyperParameters.load_from_json(config_path)

    device = config.style_gen_config.device

    training_lines: list[str] = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        training_lines.extend(f.readlines())
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        training_results = list(
            tqdm(
                executor.map(process_line, training_lines),
                total=len(training_lines),
                file=SAFE_STDOUT,
                dynamic_ncols=True,
            )
        )
    ok_training_lines = [line for line, error in training_results if error is None]
    nan_training_lines = [
        line for line, error in training_results if error == "nan_error"
    ]
    if nan_training_lines:
        nan_files = [line.split("|")[0] for line in nan_training_lines]
        logger.warning(
            f"Found NaN value in {len(nan_training_lines)} files: {nan_files}, so they will be deleted from training data."
        )

    val_lines: list[str] = []
    with open(hps.data.validation_files, encoding="utf-8") as f:
        val_lines.extend(f.readlines())

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        val_results = list(
            tqdm(
                executor.map(process_line, val_lines),
                total=len(val_lines),
                file=SAFE_STDOUT,
                dynamic_ncols=True,
            )
        )
    ok_val_lines = [line for line, error in val_results if error is None]
    nan_val_lines = [line for line, error in val_results if error == "nan_error"]
    if nan_val_lines:
        nan_files = [line.split("|")[0] for line in nan_val_lines]
        logger.warning(
            f"Found NaN value in {len(nan_val_lines)} files: {nan_files}, so they will be deleted from validation data."
        )

    with open(hps.data.training_files, "w", encoding="utf-8") as f:
        f.writelines(ok_training_lines)

    with open(hps.data.validation_files, "w", encoding="utf-8") as f:
        f.writelines(ok_val_lines)

    ok_num = len(ok_training_lines) + len(ok_val_lines)

    logger.info(f"Finished generating style vectors! total: {ok_num} npy files.")
