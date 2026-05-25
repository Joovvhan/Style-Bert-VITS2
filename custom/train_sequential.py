"""
Jamo → G2PK2 순차 학습 스크립트.

Usage:
    uv run python custom/train_sequential.py
    uv run python custom/train_sequential.py --skip-jamo    # g2pk2만 실행
    uv run python custom/train_sequential.py --skip-g2pk2   # jamo만 실행
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


EXPERIMENTS = [
    {
        "name": "kss_exp_jamo",
        "config": "Data/kss_exp_jamo/config.json",
        "model":  "Data/kss_exp_jamo",
    },
    {
        "name": "kss_exp_g2pk2",
        "config": "Data/kss_exp_g2pk2/config.json",
        "model":  "Data/kss_exp_g2pk2",
    },
]


def run_training(exp: dict, epochs_override: int | None = None) -> int:
    cmd = [
        sys.executable, "train_ms_ko.py",
        "--config", exp["config"],
        "--model",  exp["model"],
        "--skip_default_style",
    ]
    if epochs_override is not None:
        cmd += ["--epochs", str(epochs_override)]
    print(f"\n{'='*60}")
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 학습 시작: {exp['name']}")
    print(f"  명령: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    status = "완료" if result.returncode == 0 else f"오류 (code={result.returncode})"
    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] {exp['name']} {status}  ({h:02d}:{m:02d}:{s:02d})")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-jamo",  action="store_true", help="jamo 학습 건너뜀")
    parser.add_argument("--skip-g2pk2", action="store_true", help="g2pk2 학습 건너뜀")
    parser.add_argument("--test", action="store_true", help="1 epoch씩 테스트 실행")
    args = parser.parse_args()

    skip = {"kss_exp_jamo": args.skip_jamo, "kss_exp_g2pk2": args.skip_g2pk2}
    epochs_override = 1 if args.test else None

    for exp in EXPERIMENTS:
        if skip[exp["name"]]:
            print(f"[skip] {exp['name']}")
            continue
        rc = run_training(exp, epochs_override)
        if rc != 0:
            print(f"\n학습 실패로 중단: {exp['name']}")
            sys.exit(rc)

    print("\n모든 학습 완료.")


if __name__ == "__main__":
    main()
