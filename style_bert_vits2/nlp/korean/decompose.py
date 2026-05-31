"""자모 분해 + g2pk2 발음 변환 유틸리티."""

from style_bert_vits2.nlp.korean.g2p import _KO_SYL_END, _KO_SYL_START, _decompose
from style_bert_vits2.nlp.symbols import PUNCTUATIONS


def _is_korean(char: str) -> bool:
    return _KO_SYL_START <= ord(char) <= _KO_SYL_END


def _is_korean_or_punct(char: str) -> bool:
    return _is_korean(char) or char in PUNCTUATIONS


def text_to_jamo_counts(text: str, use_g2pk2: bool = True) -> list[int]:
    """
    원문 음절 위치 기준 자모 수 리스트 (len = M).

    use_g2pk2=True: g2pk2 발음 기준으로 자모 수 계산.
    음절 수 불일치 시 표기 기준 fallback.
    비한국어·공백은 skip.
    """
    if use_g2pk2:
        try:
            from style_bert_vits2.nlp.korean.g2p_g2pk2 import _get_converter
            pronounced = _get_converter()(text)
            orig_syls = [ch for ch in text if _is_korean_or_punct(ch)]
            pron_syls = [ch for ch in pronounced if _is_korean_or_punct(ch)]
            if len(orig_syls) == len(pron_syls):
                return [
                    len(_decompose(ch)) if _is_korean(ch) else 1
                    for ch in pron_syls
                ]
        except ImportError:
            pass
    return [
        len(_decompose(ch)) if _is_korean(ch) else 1
        for ch in text if _is_korean_or_punct(ch)
    ]
