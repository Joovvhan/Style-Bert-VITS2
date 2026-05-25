"""
TensorBoard 이벤트 파일에서 오디오를 추출하는 스크립트.

Usage:
    uv run python extract_tb_audio.py --logdir Data/kss/models/20260505_124239/eval
    uv run python extract_tb_audio.py --logdir Data/kss/models/20260505_124239/eval --out output/tb_audio
"""

import argparse
import glob
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tqdm import tqdm


def extract_audio(logdir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    event_files = sorted(logdir.glob("events.out.tfevents.*"))
    if not event_files:
        print(f"이벤트 파일을 찾을 수 없습니다: {logdir}")
        return

    saved = 0
    for event_file in event_files:
        total_bytes = event_file.stat().st_size
        print(f"스트리밍: {event_file.name} ({total_bytes / 1024 / 1024:.1f} MB)")
        loader = EventFileLoader(str(event_file))
        for event in tqdm(loader.Load(), desc="reading events", unit="event"):
            if not event.HasField("summary"):
                continue
            for value in event.summary.value:
                if value.metadata.plugin_data.plugin_name != "audio":
                    continue
                if value.tag.startswith("gt/"):
                    continue
                wav_bytes = value.tensor.string_val[0]
                tag_safe = value.tag.replace("/", "_").replace(" ", "_")
                filename = out_dir / f"{tag_safe}_step{event.step:06d}.wav"
                filename.write_bytes(wav_bytes)
                saved += 1

    print(f"\n총 {saved}개 파일 → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="TensorBoard eval 로그 디렉토리")
    parser.add_argument("--out", default=None, help="출력 디렉토리 (기본값: logdir/extracted_audio)")
    args = parser.parse_args()

    logdir = Path(args.logdir)
    out_dir = Path(args.out) if args.out else logdir / "extracted_audio"

    extract_audio(logdir, out_dir)
