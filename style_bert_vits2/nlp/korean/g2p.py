"""
Phase 1: literal jamo decomposition (no phonological rules).
Each Korean syllable is split into initial / medial / final jamo.
Initial ㅇ (null onset) emits no phone symbol.
Tones are all 0 (Korean has no lexical tone).
"""

from style_bert_vits2.nlp.symbols import PUNCTUATIONS


_KO_SYL_START = 0xAC00
_KO_SYL_END = 0xD7A3

# Initial consonant symbols (index 0-18; ㅇ = None = silent onset)
_INITIAL = [
    "KO_g",   # 0  ㄱ
    "KO_gg",  # 1  ㄲ
    "KO_n",   # 2  ㄴ
    "KO_d",   # 3  ㄷ
    "KO_dd",  # 4  ㄸ
    "KO_r",   # 5  ㄹ
    "KO_m",   # 6  ㅁ
    "KO_b",   # 7  ㅂ
    "KO_bb",  # 8  ㅃ
    "KO_s",   # 9  ㅅ
    "KO_ss",  # 10 ㅆ
    None,     # 11 ㅇ  (silent onset)
    "KO_j",   # 12 ㅈ
    "KO_jj",  # 13 ㅉ
    "KO_ch",  # 14 ㅊ
    "KO_kh",  # 15 ㅋ
    "KO_th",  # 16 ㅌ
    "KO_ph",  # 17 ㅍ
    "KO_hh",  # 18 ㅎ
]

# Medial vowel symbols (index 0-20)
_MEDIAL = [
    "KO_a",    # 0  ㅏ
    "KO_ae",   # 1  ㅐ
    "KO_ya",   # 2  ㅑ
    "KO_yae",  # 3  ㅒ
    "KO_eo",   # 4  ㅓ
    "KO_e",    # 5  ㅔ
    "KO_yeo",  # 6  ㅕ
    "KO_ye",   # 7  ㅖ
    "KO_o",    # 8  ㅗ
    "KO_wa",   # 9  ㅘ
    "KO_wae",  # 10 ㅙ
    "KO_oe",   # 11 ㅚ
    "KO_yo",   # 12 ㅛ
    "KO_u",    # 13 ㅜ
    "KO_wo",   # 14 ㅝ
    "KO_we",   # 15 ㅞ
    "KO_wi",   # 16 ㅟ
    "KO_yu",   # 17 ㅠ
    "KO_eu",   # 18 ㅡ
    "KO_eui",  # 19 ㅢ
    "KO_i",    # 20 ㅣ
]

# Final consonant symbols (index 0-27; 0 = no coda)
_FINAL = [
    None,      # 0  (no coda)
    "KO_G",    # 1  ㄱ
    "KO_GG",   # 2  ㄲ
    "KO_GS",   # 3  ㄳ
    "KO_N",    # 4  ㄴ
    "KO_NJ",   # 5  ㄵ
    "KO_NH",   # 6  ㄶ
    "KO_D",    # 7  ㄷ
    "KO_L",    # 8  ㄹ
    "KO_LG",   # 9  ㄺ
    "KO_LM",   # 10 ㄻ
    "KO_LB",   # 11 ㄼ
    "KO_LS",   # 12 ㄽ
    "KO_LT",   # 13 ㄾ
    "KO_LP",   # 14 ㄿ
    "KO_LH",   # 15 ㅀ
    "KO_M",    # 16 ㅁ
    "KO_B",    # 17 ㅂ
    "KO_BS",   # 18 ㅄ
    "KO_S",    # 19 ㅅ
    "KO_SS",   # 20 ㅆ
    "KO_NG",   # 21 ㅇ
    "KO_J",    # 22 ㅈ
    "KO_C",    # 23 ㅊ
    "KO_K",    # 24 ㅋ
    "KO_T",    # 25 ㅌ
    "KO_P",    # 26 ㅍ
    "KO_H",    # 27 ㅎ
]


def _decompose(char: str) -> list[str]:
    """Return phone symbols for one Korean syllable."""
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


def g2p(text: str) -> tuple[list[str], list[int], list[int]]:
    """
    Convert normalized Korean text to (phones, tones, word2ph).

    - phones: KO_* symbols + punctuation symbols, wrapped with '_' padding
    - tones:  all 0 (Korean has no lexical tone)
    - word2ph: phone count per character (including padding entries)
    """
    phones: list[str] = []
    tones: list[int] = []
    word2ph: list[int] = []

    for char in text:
        code = ord(char)
        if _KO_SYL_START <= code <= _KO_SYL_END:
            char_phones = _decompose(char)
        elif char in PUNCTUATIONS:
            char_phones = [char]
        else:
            # non-Korean, non-punctuation (spaces, unknown): skip
            continue

        if char_phones:
            phones.extend(char_phones)
            tones.extend([0] * len(char_phones))
            word2ph.append(len(char_phones))

    # wrap with padding token
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]

    return phones, tones, word2ph
