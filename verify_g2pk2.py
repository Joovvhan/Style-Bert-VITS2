"""
g2pk2 G2P 모듈 검증 스크립트.

Usage:
    uv run python verify_g2pk2.py

출력: 각 문장의 jamo 분해 결과(기존)와 g2pk2 변환 결과(신규) 비교
"""

from style_bert_vits2.nlp.korean.g2p import g2p as g2p_jamo
from style_bert_vits2.nlp.korean.g2p_g2pk2 import g2p as g2p_g2pk2
from style_bert_vits2.nlp.korean.normalizer import normalize_text

SENTENCES = [
    "국민의 건강을 위해 노력합니다.",   # 연음: 국민 → 궁민
    "밥이 맛있어요.",                   # 연음: 밥이 → 바비
    "한국어 발음 규칙이 복잡합니다.",
    "꽃이 피었습니다.",                 # 연음: 꽃이 → 꼬치
    "읽기 어렵습니다.",                 # 자음군 단순화: 읽 → 익
    "신라면을 먹었습니다.",             # 비음화: 신라 → 실라
    "학교에 갑니다.",
    "오늘 날씨가 좋네요.",
    "감사합니다.",
    "안녕하세요, 반갑습니다.",
]


def fmt_phones(phones: list[str]) -> str:
    return " ".join(phones)


def main() -> None:
    print("=" * 70)
    print(f"{'문장':<30} {'방식':<8} {'phones'}")
    print("=" * 70)

    for sent in SENTENCES:
        norm = normalize_text(sent)

        phones_jamo, _, _ = g2p_jamo(norm)
        phones_g2pk2, _, _ = g2p_g2pk2(norm)

        print(f"{sent}")
        print(f"  norm   : {norm}")
        print(f"  jamo   : {fmt_phones(phones_jamo)}")
        print(f"  g2pk2  : {fmt_phones(phones_g2pk2)}")
        print()


if __name__ == "__main__":
    main()
