import re
import unicodedata


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _convert_numbers(text)
    text = replace_punctuation(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def replace_punctuation(text: str) -> str:
    REPLACE_MAP = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "…": "...",
        "“": "'",
        "”": "'",
        "‘": "'",
        "’": "'",
        "「": "'",
        "」": "'",
        "（": "'",
        "）": "'",
        "(": "'",
        ")": "'",
        "—": "-",
        "−": "-",
        "～": "-",
        "~": "-",
    }
    pattern = re.compile("|".join(re.escape(k) for k in REPLACE_MAP))
    text = pattern.sub(lambda m: REPLACE_MAP[m.group()], text)
    # 한글, ASCII 영숫자, 허용 구두점만 유지
    text = re.sub(
        r"[^가-힣ᄀ-ᇿ㄰-㆏a-zA-Z0-9!?,.'`\-\s]",
        "",
        text,
    )
    return text


def _convert_numbers(text: str) -> str:
    try:
        from num2words import num2words

        def _replace(m: re.Match) -> str:
            try:
                return num2words(int(m.group()), lang="ko")
            except Exception:
                return m.group()

        return re.sub(r"\d+", _replace, text)
    except ImportError:
        return text
