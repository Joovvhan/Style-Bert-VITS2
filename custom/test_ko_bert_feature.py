"""
L2: Korean BERT bert_feature.py 파이프라인 단위 테스트

실행 전 필요:
  uv run huggingface-cli download klue/roberta-large --local-dir bert/klue-roberta-large

실행:
  uv run python custom/test_ko_bert_feature.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.korean.bert_feature import extract_bert_feature
from style_bert_vits2.nlp.korean.g2p_g2pk2 import g2p as g2p_g2pk2
from style_bert_vits2.nlp.korean.normalizer import normalize_text


def get_word2ph(text: str) -> list[int]:
    _, _, word2ph = g2p_g2pk2(normalize_text(text))
    return word2ph


# ── 테스트 함수 ────────────────────────────────────────────────────────────────

def test_bert_feature_shape():
    text = "안녕하세요"
    word2ph = get_word2ph(text)
    feat = extract_bert_feature(text, word2ph, "cpu")
    expected_L = sum(word2ph)
    assert feat.shape == (1024, expected_L), f"shape {feat.shape} != (1024, {expected_L})"
    print(f"[PASS] shape: {feat.shape}")


def test_no_nan():
    text = "저는 학생입니다."
    word2ph = get_word2ph(text)
    feat = extract_bert_feature(text, word2ph, "cpu")
    assert not feat.isnan().any(), "NaN found in features"
    print("[PASS] no NaN")


def test_not_zero():
    text = "저는 학생입니다."
    word2ph = get_word2ph(text)
    feat = extract_bert_feature(text, word2ph, "cpu")
    assert not (feat == 0).all(), "All zeros in features"
    print("[PASS] not all zeros")


def test_assist_text():
    text = "안녕하세요"
    word2ph = get_word2ph(text)
    feat = extract_bert_feature(text, word2ph, "cpu", assist_text="반갑습니다")
    assert feat.shape == (1024, sum(word2ph))
    print(f"[PASS] assist_text shape: {feat.shape}")


def test_various_sentences():
    cases = [
        "대한민국",
        "나는 밥을 먹었다.",
        "읽었어요",
        "닭볶음",
    ]
    for text in cases:
        word2ph = get_word2ph(text)
        feat = extract_bert_feature(text, word2ph, "cpu")
        assert feat.shape == (1024, sum(word2ph)), f"{text!r}: {feat.shape}"
        assert not feat.isnan().any(), f"{text!r}: NaN"
        print(f"[PASS] {text!r}: shape={feat.shape}")


# ── 진입점 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading KO BERT model (klue/roberta-large)...")
    bert_models.load_model(Languages.KO)
    bert_models.load_tokenizer(Languages.KO)
    print("Model loaded.\n")

    test_bert_feature_shape()
    test_no_nan()
    test_not_zero()
    test_assist_text()
    test_various_sentences()

    print("\nAll L2 tests passed!")
