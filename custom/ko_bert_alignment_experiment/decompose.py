"""자모 분해 + g2pk2 발음 변환 유틸리티 (독립 버전 — Style-Bert-VITS2 의존성 없음)."""

_KO_SYL_START = 0xAC00
_KO_SYL_END = 0xD7A3

PUNCTUATIONS = ["!", "?", "…", ",", ".", "'", "-"]

_INITIAL = [
    "KO_g", "KO_gg", "KO_n", "KO_d", "KO_dd", "KO_r", "KO_m", "KO_b", "KO_bb",
    "KO_s", "KO_ss", None, "KO_j", "KO_jj", "KO_ch", "KO_kh", "KO_th", "KO_ph", "KO_hh",
]

_MEDIAL = [
    "KO_a", "KO_ae", "KO_ya", "KO_yae", "KO_eo", "KO_e", "KO_yeo", "KO_ye",
    "KO_o", "KO_wa", "KO_wae", "KO_oe", "KO_yo", "KO_u", "KO_wo", "KO_we",
    "KO_wi", "KO_yu", "KO_eu", "KO_eui", "KO_i",
]

_FINAL = [
    None, "KO_G", "KO_GG", "KO_GS", "KO_N", "KO_NJ", "KO_NH", "KO_D",
    "KO_L", "KO_LG", "KO_LM", "KO_LB", "KO_LS", "KO_LT", "KO_LP", "KO_LH",
    "KO_M", "KO_B", "KO_BS", "KO_S", "KO_SS", "KO_NG", "KO_J", "KO_C",
    "KO_K", "KO_T", "KO_P", "KO_H",
]


def _is_korean(char: str) -> bool:
    return _KO_SYL_START <= ord(char) <= _KO_SYL_END


def _is_korean_or_punct(char: str) -> bool:
    return _is_korean(char) or char in PUNCTUATIONS


def decompose_syllable(char: str) -> list[str]:
    """한국어 음절 1자를 KO_* phone 리스트로 분해. ㅇ 초성 생략."""
    code = ord(char) - _KO_SYL_START
    final_idx = code % 28
    code //= 28
    medial_idx = code % 21
    initial_idx = code // 21

    phones: list[str] = []
    onset = _INITIAL[initial_idx]
    if onset is not None:
        phones.append(onset)
    phones.append(_MEDIAL[medial_idx])
    coda = _FINAL[final_idx]
    if coda is not None:
        phones.append(coda)
    return phones


def text_to_jamo_counts(text: str, use_g2pk2: bool = True) -> list[int]:
    """
    원문 음절 위치 기준 자모 수 리스트 (len = M).

    use_g2pk2=True: g2pk2 발음 기준으로 자모 수 계산.
    음절 수 불일치 시 표기 기준 fallback.
    비한국어·공백은 skip.
    """
    if use_g2pk2:
        try:
            from g2pk2 import G2p
            pronounced = G2p()(text)
            orig_syls = [ch for ch in text if _is_korean_or_punct(ch)]
            pron_syls = [ch for ch in pronounced if _is_korean_or_punct(ch)]
            if len(orig_syls) == len(pron_syls):
                return [
                    len(decompose_syllable(ch)) if _is_korean(ch) else 1
                    for ch in pron_syls
                ]
        except ImportError:
            pass
    return [
        len(decompose_syllable(ch)) if _is_korean(ch) else 1
        for ch in text if _is_korean_or_punct(ch)
    ]
