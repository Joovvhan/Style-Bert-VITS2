"""
DataLoader 속도 벤치마크: STFT 경로 vs spec.pt 캐시 경로 비교.

Usage:
    uv run python bench_dataloader.py -c Data/kss/config.json
    uv run python bench_dataloader.py -c Data/kss/config.json --workers 4 8 16
    uv run python bench_dataloader.py -c Data/kss/config.json --single-only
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader

from data_utils_ko import TextAudioSpeakerCollateKO, TextAudioSpeakerLoaderKO
from mel_processing import spectrogram_torch
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.utils import load_wav_to_torch


# ---------------------------------------------------------------------------
# Dataset variants
# ---------------------------------------------------------------------------

class ForceSTFTDataset(TextAudioSpeakerLoaderKO):
    """get_audio에서 캐시를 무시하고 항상 STFT 계산."""

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if audio.ndim == 2:
            audio = audio.mean(dim=-1)
        audio_norm = (audio / self.max_wav_value).unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.filter_length,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        return spec, audio_norm


class ForceCacheDataset(TextAudioSpeakerLoaderKO):
    """get_audio에서 항상 spec.pt를 로드 (캐시 없으면 에러)."""

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if audio.ndim == 2:
            audio = audio.mean(dim=-1)
        audio_norm = (audio / self.max_wav_value).unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        spec = torch.load(spec_filename, weights_only=True)
        return spec, audio_norm


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def make_loader(dataset_cls, hps, num_workers, batch_size):
    ds = dataset_cls(hps.data.training_files, hps.data)
    return DataLoader(
        ds,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        collate_fn=TextAudioSpeakerCollateKO(),
        batch_size=batch_size,
        drop_last=True,
    )


def bench_loader(loader, n_batches: int, warmup: int = 2) -> list[float]:
    it = iter(loader)
    for _ in range(warmup):
        next(it)
    times = []
    for _ in range(n_batches):
        t0 = time.perf_counter()
        next(it)
        times.append(time.perf_counter() - t0)
    return times


def bench_single(dataset_cls, hps, n_samples: int = 100):
    """DataLoader 없이 단일 샘플 __getitem__ 속도만 측정."""
    ds = dataset_cls(hps.data.training_files, hps.data)
    indices = list(range(min(n_samples, len(ds))))
    times = []
    for i in indices:
        t0 = time.perf_counter()
        ds[i]
        times.append(time.perf_counter() - t0)
    return times


def stats(times: list[float]) -> str:
    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)
    return f"avg={avg*1000:.1f}ms  min={mn*1000:.1f}ms  max={mx*1000:.1f}ms  (n={len(times)})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="Data/kss/config.json")
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--batches", type=int, default=20,
                        help="DataLoader 벤치마크에서 측정할 배치 수")
    parser.add_argument("--single-samples", type=int, default=100,
                        help="단일 샘플 벤치마크 샘플 수")
    parser.add_argument("--single-only", action="store_true",
                        help="단일 샘플 측정만 실행 (DataLoader 생략)")
    args = parser.parse_args()

    hps = HyperParameters.load_from_json(args.config)
    sep = "=" * 60

    # ------------------------------------------------------------------
    # 1. 단일 샘플: DataLoader 오버헤드 없이 순수 처리 시간
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("  [1] 단일 샘플 처리 시간 (DataLoader 없음, 메인 프로세스)")
    print(sep)

    print("  STFT 경로 (WAV read + STFT 계산) ...")
    stft_times = bench_single(ForceSTFTDataset, hps, args.single_samples)
    print(f"    {stats(stft_times)}")

    print("  캐시 경로 (WAV read + spec.pt 로드) ...")
    try:
        cache_times = bench_single(ForceCacheDataset, hps, args.single_samples)
        print(f"    {stats(cache_times)}")
    except Exception as e:
        print(f"    SKIP: spec.pt 없음 — 먼저 epoch 1을 돌려야 합니다 ({e})")
        cache_times = None

    if cache_times:
        ratio = (sum(cache_times) / len(cache_times)) / (sum(stft_times) / len(stft_times))
        print(f"\n  캐시 / STFT 비율: {ratio:.2f}x  ({'캐시가 빠름' if ratio < 1 else 'STFT가 빠름'})")

    if args.single_only:
        return

    # ------------------------------------------------------------------
    # 2. DataLoader 처리량: workers 수별 비교
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print(f"  [2] DataLoader 처리량 (batch_size={args.batch_size}, "
          f"batches={args.batches}+{2} warmup)")
    print(sep)

    for nw in args.workers:
        print(f"\n  --- num_workers={nw} ---")

        print("  STFT 경로 ...")
        loader = make_loader(ForceSTFTDataset, hps, nw, args.batch_size)
        t = bench_loader(loader, args.batches)
        print(f"    {stats(t)}")
        del loader

        print("  캐시 경로 ...")
        try:
            loader = make_loader(ForceCacheDataset, hps, nw, args.batch_size)
            t2 = bench_loader(loader, args.batches)
            print(f"    {stats(t2)}")
            del loader
            r = (sum(t2) / len(t2)) / (sum(t) / len(t))
            print(f"    → 비율 {r:.2f}x  ({'캐시가 빠름' if r < 1 else 'STFT가 빠름'})")
        except Exception as e:
            print(f"    SKIP: {e}")

    print(f"\n{sep}")
    print("  완료")
    print(sep)


if __name__ == "__main__":
    main()
