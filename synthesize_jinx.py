"""
jinx 모델로 음성 합성하는 스크립트.

Usage:
    uv run python synthesize_jinx.py
"""
from pathlib import Path

import soundfile as sf

from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel

MODEL_DIR = Path("model_assets/jinx")
MODEL = MODEL_DIR / "jinx_e250_s7500.safetensors"
CONFIG = MODEL_DIR / "config.json"
STYLE_VEC = MODEL_DIR / "style_vectors.npy"
OUTPUT_DIR = Path("output/jinx_s7500")

TEXTS = [
    "デイジーにその程度しか見せられないの？ パク所長。",
    "デイジーの時間を無駄にしないで。",
    "ふーん、やっぱりそんなものね。",
    "パク所長、デイジーに逆らうつもり？",
    "デイジーを相手にした時点で、もう勝負は決まってたの。",
    "デイジーはここよ。ちゃんと見てなさい。",
    "別に恨みはないけど、パク所長の態度は気に入らないわ。",
    "デイジーって、悪役みたいなことまで似合っちゃうのよね。",
    "やっぱりデイジーがいると、空気が違うでしょう？",
    "パク所長、次はもっとデイジーを楽しませてみせて。"
]

SDP_RATIO = 0.2
LENGTH = 1.0
DEVICE = "cuda"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = TTSModel(
        model_path=MODEL,
        config_path=CONFIG,
        style_vec_path=STYLE_VEC,
        device=DEVICE,
    )
    model.load()

    for i, text in enumerate(TEXTS):
        sr, audio = model.infer(
            text=text,
            language=Languages.JP,
            sdp_ratio=SDP_RATIO,
            length=LENGTH,
        )
        out_path = OUTPUT_DIR / f"{i+1:02d}.wav"
        sf.write(out_path, audio, sr)
        print(f"[{i+1:02d}/{len(TEXTS)}] {out_path}  — {text}")


if __name__ == "__main__":
    main()
