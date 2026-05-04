"""
KO-extended symbol table.
Imports all existing symbols from symbols.py and appends Korean-specific ones.
Existing models trained with the original symbols.py are NOT affected.
"""

from style_bert_vits2.nlp.symbols import (
    PAD,
    PUNCTUATIONS,
    PUNCTUATION_SYMBOLS,
    ZH_SYMBOLS,
    JP_SYMBOLS,
    EN_SYMBOLS,
    NUM_ZH_TONES,
    NUM_JP_TONES,
    NUM_EN_TONES,
)


# ── Korean phoneme symbols ────────────────────────────────────────────────────
# Initial consonants (18 symbols; ㅇ null onset emits no phone)
KO_INITIAL_SYMBOLS = [
    "KO_g",   # ㄱ
    "KO_gg",  # ㄲ
    "KO_n",   # ㄴ
    "KO_d",   # ㄷ
    "KO_dd",  # ㄸ
    "KO_r",   # ㄹ
    "KO_m",   # ㅁ
    "KO_b",   # ㅂ
    "KO_bb",  # ㅃ
    "KO_s",   # ㅅ
    "KO_ss",  # ㅆ
    "KO_j",   # ㅈ
    "KO_jj",  # ㅉ
    "KO_ch",  # ㅊ
    "KO_kh",  # ㅋ
    "KO_th",  # ㅌ
    "KO_ph",  # ㅍ
    "KO_hh",  # ㅎ
]

# Medial vowels (21 symbols)
KO_MEDIAL_SYMBOLS = [
    "KO_a",    # ㅏ
    "KO_ae",   # ㅐ
    "KO_ya",   # ㅑ
    "KO_yae",  # ㅒ
    "KO_eo",   # ㅓ
    "KO_e",    # ㅔ
    "KO_yeo",  # ㅕ
    "KO_ye",   # ㅖ
    "KO_o",    # ㅗ
    "KO_wa",   # ㅘ
    "KO_wae",  # ㅙ
    "KO_oe",   # ㅚ
    "KO_yo",   # ㅛ
    "KO_u",    # ㅜ
    "KO_wo",   # ㅝ
    "KO_we",   # ㅞ
    "KO_wi",   # ㅟ
    "KO_yu",   # ㅠ
    "KO_eu",   # ㅡ
    "KO_eui",  # ㅢ
    "KO_i",    # ㅣ
]

# Final consonants / codas (27 symbols; null coda emits no phone)
KO_FINAL_SYMBOLS = [
    "KO_G",    # ㄱ  coda
    "KO_GG",   # ㄲ  coda
    "KO_GS",   # ㄳ  coda
    "KO_N",    # ㄴ  coda
    "KO_NJ",   # ㄵ  coda
    "KO_NH",   # ㄶ  coda
    "KO_D",    # ㄷ  coda
    "KO_L",    # ㄹ  coda
    "KO_LG",   # ㄺ  coda
    "KO_LM",   # ㄻ  coda
    "KO_LB",   # ㄼ  coda
    "KO_LS",   # ㄽ  coda
    "KO_LT",   # ㄾ  coda
    "KO_LP",   # ㄿ  coda
    "KO_LH",   # ㅀ  coda
    "KO_M",    # ㅁ  coda
    "KO_B",    # ㅂ  coda
    "KO_BS",   # ㅄ  coda
    "KO_S",    # ㅅ  coda
    "KO_SS",   # ㅆ  coda
    "KO_NG",   # ㅇ  coda  (ng sound)
    "KO_J",    # ㅈ  coda
    "KO_C",    # ㅊ  coda
    "KO_K",    # ㅋ  coda
    "KO_T",    # ㅌ  coda
    "KO_P",    # ㅍ  coda
    "KO_H",    # ㅎ  coda
]

KO_SYMBOLS = KO_INITIAL_SYMBOLS + KO_MEDIAL_SYMBOLS + KO_FINAL_SYMBOLS  # 66 symbols
NUM_KO_TONES = 1  # Korean has no lexical tone


# ── Extended combined tables ──────────────────────────────────────────────────
NORMAL_SYMBOLS_KO = sorted(
    set(ZH_SYMBOLS + JP_SYMBOLS + EN_SYMBOLS + KO_SYMBOLS)
)
SYMBOLS_KO = [PAD] + NORMAL_SYMBOLS_KO + PUNCTUATION_SYMBOLS
SIL_PHONEMES_IDS_KO = [SYMBOLS_KO.index(i) for i in PUNCTUATION_SYMBOLS]

NUM_TONES_KO = NUM_ZH_TONES + NUM_JP_TONES + NUM_EN_TONES + NUM_KO_TONES  # 13

LANGUAGE_ID_MAP_KO = {"ZH": 0, "JP": 1, "EN": 2, "KO": 3}
NUM_LANGUAGES_KO = len(LANGUAGE_ID_MAP_KO)  # 4

LANGUAGE_TONE_START_MAP_KO = {
    "ZH": 0,
    "JP": NUM_ZH_TONES,
    "EN": NUM_ZH_TONES + NUM_JP_TONES,
    "KO": NUM_ZH_TONES + NUM_JP_TONES + NUM_EN_TONES,
}


def cleaned_text_to_sequence_ko(
    cleaned_phones: list[str],
    tones: list[int],
    language: str,
) -> tuple[list[int], list[int], list[int]]:
    """Convert phones/tones/language to integer IDs using the KO-extended tables."""
    symbol_to_id = {s: i for i, s in enumerate(SYMBOLS_KO)}
    phones_ids = [symbol_to_id[p] for p in cleaned_phones]
    tone_start = LANGUAGE_TONE_START_MAP_KO[language]
    tones_ids = [t + tone_start for t in tones]
    lang_id = LANGUAGE_ID_MAP_KO[language]
    lang_ids = [lang_id] * len(phones_ids)
    return phones_ids, tones_ids, lang_ids
