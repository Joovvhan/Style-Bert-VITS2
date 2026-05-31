from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.korean.alignment import apply_add_blank, compute_word2ph_tok
from style_bert_vits2.nlp.korean.decompose import text_to_jamo_counts

if TYPE_CHECKING:
    import torch


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> torch.Tensor:
    """
    한국어 텍스트에서 BERT 피처를 추출한다.

    Args:
        text: normalize 완료된 한국어 텍스트
        word2ph: g2p()가 반환한 add_blank 적용 후 word2ph. sum = L.
        device: 추론 디바이스
        assist_text: 스타일 보조 텍스트 (선택)
        assist_text_weight: 보조 텍스트 혼합 비율

    Returns:
        Tensor shape (1024, L), L = sum(word2ph)
    """
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = bert_models.load_model(Languages.KO, device_map=device)
    bert_models.transfer_model(Languages.KO, device)
    tokenizer = bert_models.load_tokenizer(Languages.KO)

    jamo_counts = text_to_jamo_counts(text, use_g2pk2=True)
    word2ph_tok = apply_add_blank(compute_word2ph_tok(text, tokenizer, jamo_counts))

    assert sum(word2ph_tok) == sum(word2ph), (
        f"word2ph_tok sum {sum(word2ph_tok)} != word2ph sum {sum(word2ph)}\n"
        f"text={text!r}  word2ph_tok={word2ph_tok}  word2ph={word2ph}"
    )

    style_res_mean = None
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

        if assist_text:
            style_inputs = tokenizer(assist_text, return_tensors="pt")
            for k in style_inputs:
                style_inputs[k] = style_inputs[k].to(device)
            style_res = model(**style_inputs, output_hidden_states=True)
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            style_res_mean = style_res.mean(0)

    phone_level_feature = []
    for i, n in enumerate(word2ph_tok):
        if assist_text and style_res_mean is not None:
            feat = (
                res[i].repeat(n, 1) * (1 - assist_text_weight)
                + style_res_mean.repeat(n, 1) * assist_text_weight
            )
        else:
            feat = res[i].repeat(n, 1)
        phone_level_feature.append(feat)

    return torch.cat(phone_level_feature, dim=0).T
