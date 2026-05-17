"""
g2pk2 기반 G2P: 음운 변동 규칙 적용 후 자모 분해.

g2pk2가 텍스트를 실제 발음 표기(한글 음절)로 변환한 뒤,
기존 _decompose()로 자모 분해하여 KO_* phone 심볼을 생성한다.

예)
  "국민" → g2pk2 → "궁민" → _decompose → KO_kh KO_u KO_NG / KO_m KO_i KO_N
  "밥이" → g2pk2 → "바비" → _decompose → KO_b KO_a / KO_b KO_i
"""

from style_bert_vits2.nlp.korean.g2p import _KO_SYL_END, _KO_SYL_START, _decompose
from style_bert_vits2.nlp.symbols import PUNCTUATIONS


_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        try:
            from g2pk2 import G2p
        except ImportError:
            raise ImportError("g2pk2 패키지가 필요합니다. 설치: uv add g2pk2")
        _converter = G2p()
    return _converter


def g2p(text: str) -> tuple[list[str], list[int], list[int]]:
    """
    Convert normalized Korean text to (phones, tones, word2ph) using g2pk2.

    Raises:
        ImportError: g2pk2 패키지가 없을 때.
    """
    converter = _get_converter()
    pronounced: str = converter(text)

    phones: list[str] = []
    tones: list[int] = []
    word2ph: list[int] = []

    for char in pronounced:
        code = ord(char)
        if _KO_SYL_START <= code <= _KO_SYL_END:
            char_phones = _decompose(char)
        elif char in PUNCTUATIONS:
            char_phones = [char]
        else:
            # 공백, 미지원 문자 건너뜀
            continue

        if char_phones:
            phones.extend(char_phones)
            tones.extend([0] * len(char_phones))
            word2ph.append(len(char_phones))

    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]

    return phones, tones, word2ph
