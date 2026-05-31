"""offset_mapping 기반 word2ph_tok 계산."""

from style_bert_vits2.nlp.korean.decompose import _is_korean_or_punct


def _build_syllable_spans(text: str) -> list[tuple[int, int]]:
    """원문에서 한국어 음절·구두점의 char offset 범위 목록."""
    return [(i, i + 1) for i, ch in enumerate(text) if _is_korean_or_punct(ch)]


def _distribute(total: int, n: int) -> list[int]:
    """total을 n개로 균등 분배. 나머지는 앞쪽에 배분."""
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def compute_word2ph_tok(
    text: str,
    tokenizer,
    jamo_counts: list[int],
) -> list[int]:
    """
    Returns [1(CLS), tok_1_jamo, ..., tok_k_jamo, 1(SEP)]

    sum == sum(jamo_counts) + 2  (순수 한국어 기준)
    비한국어·공백 토큰은 fallback=1.
    1음절이 여러 토큰으로 분할된 경우 자모 수를 균등 배분.
    """
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    syllable_spans = _build_syllable_spans(text)
    assert len(syllable_spans) == len(jamo_counts), (
        f"syllable_spans {len(syllable_spans)} != jamo_counts {len(jamo_counts)}"
    )

    body_offsets = offsets[1:-1]

    tok_to_syl: list[list[int]] = []
    for tok_start, tok_end in body_offsets:
        if tok_start == tok_end:
            tok_to_syl.append([])
        else:
            covered = [
                i for i, (s, e) in enumerate(syllable_spans)
                if s >= tok_start and e <= tok_end
            ]
            tok_to_syl.append(covered)

    # 1음절 → 여러 토큰 분할 처리
    syl_to_toks: dict[int, list[int]] = {}
    for tok_i, syls in enumerate(tok_to_syl):
        for syl_i in syls:
            syl_to_toks.setdefault(syl_i, []).append(tok_i)

    split_alloc: dict[tuple[int, int], int] = {}
    for syl_i, tok_list in syl_to_toks.items():
        if len(tok_list) > 1:
            allocated = _distribute(jamo_counts[syl_i], len(tok_list))
            for tok_i, val in zip(tok_list, allocated):
                split_alloc[(tok_i, syl_i)] = val

    word2ph = [1]  # CLS
    for tok_i, syls in enumerate(tok_to_syl):
        if not syls:
            word2ph.append(1)
            continue
        total = sum(
            split_alloc.get((tok_i, syl_i), jamo_counts[syl_i])
            for syl_i in syls
        )
        word2ph.append(total if total > 0 else 1)
    word2ph.append(1)  # SEP
    return word2ph


def apply_add_blank(word2ph: list[int]) -> list[int]:
    """word2ph[i] *= 2, word2ph[0] += 1"""
    result = [v * 2 for v in word2ph]
    result[0] += 1
    return result
