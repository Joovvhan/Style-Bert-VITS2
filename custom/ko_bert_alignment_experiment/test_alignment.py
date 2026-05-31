"""
L1: klue/roberta-large 토크나이저 정렬 알고리즘 검증.

실행:
  cd custom/ko_bert_alignment_experiment
  uv sync
  uv run pytest test_alignment.py -v
"""

import pytest
from transformers import AutoTokenizer

from decompose import text_to_jamo_counts
from alignment import apply_add_blank, compute_word2ph_tok


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("klue/roberta-large")


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def run(text: str, tokenizer, use_g2pk2: bool = True) -> dict:
    jamo = text_to_jamo_counts(text, use_g2pk2=use_g2pk2)
    w2ph = compute_word2ph_tok(text, tokenizer, jamo)
    w2ph_blank = apply_add_blank(w2ph)
    return {"jamo": jamo, "w2ph": w2ph, "w2ph_blank": w2ph_blank}


# ── sum 불변식 ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "안녕하세요",
    "읽었어요",
    "닭볶음",
    "좋아요",
    "학교에 갑니다.",
    "대한민국",
    "서울특별시",
    "나는 밥을 먹었다.",
])
def test_sum_invariant(text, tokenizer):
    r = run(text, tokenizer)
    assert sum(r["w2ph"]) == sum(r["jamo"]) + 2, (
        f"{text!r}: sum(w2ph)={sum(r['w2ph'])} != sum(jamo)+2={sum(r['jamo'])+2}"
    )


@pytest.mark.parametrize("text", [
    "안녕하세요",
    "읽었어요",
    "닭볶음",
])
def test_add_blank_sum_invariant(text, tokenizer):
    r = run(text, tokenizer)
    expected = 2 * (sum(r["jamo"]) + 2) + 1
    assert sum(r["w2ph_blank"]) == expected, (
        f"{text!r}: sum(w2ph_blank)={sum(r['w2ph_blank'])} != {expected}"
    )


# ── 최솟값 보장 ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "안녕하세요",
    "읽었어요",
    "닭볶음",
    "좋아요",
    "학교에 갑니다.",
])
def test_minimum_one(text, tokenizer):
    r = run(text, tokenizer)
    assert all(v >= 1 for v in r["w2ph"]), (
        f"{text!r}: w2ph has zero value: {r['w2ph']}"
    )


# ── KO_BERT_SPEC.md 구체적 기댓값 검증 ───────────────────────────────────────

def test_annyeonghaseyo_body(tokenizer):
    """안녕하세요 → 토큰 body [안녕, ##하, ##세요] → [5, 2, 3]"""
    r = run("안녕하세요", tokenizer)
    assert r["w2ph"][1:-1] == [5, 2, 3], r["w2ph"]
    assert sum(r["w2ph"]) == 12


def test_ilgeosseoyo_sum(tokenizer):
    """읽었어요 → sum(jamo)=7, sum(w2ph)=9"""
    r = run("읽었어요", tokenizer)
    assert sum(r["jamo"]) == 7
    assert sum(r["w2ph"]) == 9


def test_dakbokkeum_sum(tokenizer):
    """닭볶음 → sum(jamo)=8, sum(w2ph)=10"""
    r = run("닭볶음", tokenizer)
    assert sum(r["jamo"]) == 8
    assert sum(r["w2ph"]) == 10


def test_johayo_body(tokenizer):
    """좋아요 → sum(jamo)=4, sum(w2ph)=6"""
    r = run("좋아요", tokenizer)
    assert sum(r["jamo"]) == 4
    assert sum(r["w2ph"]) == 6


# ── g2pk2 발음 변환 음절 수 보존 ──────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "읽었어요",
    "닭볶음",
    "좋아요",
    "국민",
    "밥이",
])
def test_g2pk2_syllable_count_preserved(text):
    from decompose import _is_korean_or_punct
    try:
        from g2pk2 import G2p
    except ImportError:
        pytest.skip("g2pk2 not installed")
    pronounced = G2p()(text)
    orig_syls = [ch for ch in text if _is_korean_or_punct(ch)]
    pron_syls = [ch for ch in pronounced if _is_korean_or_punct(ch)]
    assert len(orig_syls) == len(pron_syls), (
        f"{text!r}: {len(orig_syls)} orig vs {len(pron_syls)} pron"
    )


# ── 비한국어·혼합 텍스트 fallback ─────────────────────────────────────────────

def test_mixed_korean_english(tokenizer):
    """영어 혼합 텍스트: 예외 없이 처리, 한국어 부분만 jamo 계산"""
    text = "hello 안녕"
    r = run(text, tokenizer)
    assert all(v >= 1 for v in r["w2ph"])


def test_punctuation_only_korean(tokenizer):
    """구두점 포함 텍스트"""
    text = "안녕하세요."
    r = run(text, tokenizer)
    assert all(v >= 1 for v in r["w2ph"])


def test_no_korean(tokenizer):
    """한국어 없는 텍스트: jamo_counts 빈 리스트, w2ph=[1,1]"""
    jamo = text_to_jamo_counts("hello world", use_g2pk2=False)
    assert jamo == []
    # tokenizer 에 한국어 없는 텍스트를 넣으면 토큰이 존재하지만 syllable_spans=[]
    # → assert len(syllable_spans)==len(jamo_counts) 통과 (둘 다 0)
    w2ph = compute_word2ph_tok("hello world", tokenizer, jamo)
    assert w2ph[0] == 1 and w2ph[-1] == 1
    assert all(v >= 1 for v in w2ph)


# ── add_blank 후 w2ph[0] 값 ───────────────────────────────────────────────────

def test_add_blank_first_element(tokenizer):
    """apply_add_blank: w2ph[0]=1 → 3 (1*2+1)"""
    r = run("안녕", tokenizer)
    assert r["w2ph"][0] == 1
    assert r["w2ph_blank"][0] == 3
