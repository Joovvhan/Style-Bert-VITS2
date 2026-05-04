"""
KO-specific model file (Phase 1).
Architecture is identical to models_jp_extra, but TextEncoder uses the
KO-extended symbol/tone/language counts from symbols_ko.py.
Only TextEncoder and SynthesizerTrn are redefined here; all other classes
are re-exported from models_jp_extra unchanged.
"""

import math
from typing import Any, Optional

import torch
from torch import nn

from style_bert_vits2.models import attentions, commons, modules
from style_bert_vits2.models.models_jp_extra import (
    DurationDiscriminator,
    DurationPredictor,
    MultiPeriodDiscriminator,
    PosteriorEncoder,
    Generator,
    ResidualCouplingBlock,
    ReferenceEncoder,
    StochasticDurationPredictor,
    TransformerCouplingBlock,
    WavLMDiscriminator,
)
from style_bert_vits2.nlp.symbols_ko import (
    NUM_LANGUAGES_KO as NUM_LANGUAGES,
    NUM_TONES_KO as NUM_TONES,
    SYMBOLS_KO as SYMBOLS,
)


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        self.emb = nn.Embedding(len(SYMBOLS), hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        self.tone_emb = nn.Embedding(NUM_TONES, hidden_channels)
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels**-0.5)
        self.language_emb = nn.Embedding(NUM_LANGUAGES, hidden_channels)
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels**-0.5)
        # Phase 1: bert_proj receives zero tensors, so its contribution is zero.
        # Kept so the checkpoint format matches Phase 3 (BERT-enabled).
        self.bert_proj = nn.Conv1d(1024, hidden_channels, 1)
        self.style_proj = nn.Linear(256, hidden_channels)
        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        tone: torch.Tensor,
        language: torch.Tensor,
        bert: torch.Tensor,
        style_vec: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bert_emb = self.bert_proj(bert).transpose(1, 2)
        style_emb = self.style_proj(style_vec.unsqueeze(1))
        x = (
            self.emb(x)
            + self.tone_emb(tone)
            + self.language_emb(language)
            + bert_emb
            + style_emb
        ) * math.sqrt(self.hidden_channels)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.encoder(x * x_mask, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class SynthesizerTrn(nn.Module):
    """Synthesizer for Training (KO version)."""

    def __init__(
        self,
        n_vocab: int,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        n_speakers: int = 256,
        gin_channels: int = 256,
        use_sdp: bool = True,
        n_flow_layer: int = 4,
        n_layers_trans_flow: int = 6,
        flow_share_parameter: bool = False,
        use_transformer_flow: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.n_layers_trans_flow = n_layers_trans_flow
        self.use_spk_conditioned_encoder = kwargs.get("use_spk_conditioned_encoder", True)
        self.use_sdp = use_sdp
        self.use_noise_scaled_mas = kwargs.get("use_noise_scaled_mas", False)
        self.mas_noise_scale_initial = kwargs.get("mas_noise_scale_initial", 0.01)
        self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)
        self.current_mas_noise_scale = self.mas_noise_scale_initial
        if self.use_spk_conditioned_encoder and gin_channels > 0:
            self.enc_gin_channels = gin_channels
        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.enc_gin_channels,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels, 5, 1, 16,
            gin_channels=gin_channels,
        )
        if use_transformer_flow:
            self.flow = TransformerCouplingBlock(
                inter_channels, hidden_channels, filter_channels, n_heads,
                n_layers_trans_flow, 5, p_dropout, n_flow_layer,
                gin_channels=gin_channels, share_parameter=flow_share_parameter,
            )
        else:
            self.flow = ResidualCouplingBlock(
                inter_channels, hidden_channels, 5, 1, n_flow_layer,
                gin_channels=gin_channels,
            )
        self.sdp = StochasticDurationPredictor(
            hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
        )
        self.dp = DurationPredictor(
            hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
        )
        if n_speakers >= 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
        else:
            self.ref_enc = ReferenceEncoder(spec_channels, gin_channels)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        sid: torch.Tensor,
        tone: torch.Tensor,
        language: torch.Tensor,
        bert: torch.Tensor,
        style_vec: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
        tuple[torch.Tensor, ...], tuple[torch.Tensor, ...],
    ]:
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)
        else:
            g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, style_vec, g=g
        )
        from style_bert_vits2.models import monotonic_alignment

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            import math as _math
            s_p_sq_r = torch.exp(-2 * logs_p)
            neg_cent1 = torch.sum(-0.5 * _math.log(2 * _math.pi) - logs_p, [1], keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), m_p * s_p_sq_r)
            neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            if self.use_noise_scaled_mas:
                epsilon = (
                    torch.std(neg_cent)
                    * torch.randn_like(neg_cent)
                    * self.current_mas_noise_scale
                )
                neg_cent = neg_cent + epsilon
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                monotonic_alignment.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

        w = attn.sum(2)
        l_length_sdp = self.sdp(x, x_mask, w, g=g) / torch.sum(x_mask)
        logw_ = torch.log(w + 1e-6) * x_mask
        logw = self.dp(x, x_mask, g=g)
        l_length_dp = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)
        l_length = l_length_dp + l_length_sdp

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        return (
            o, l_length, attn, ids_slice, x_mask, y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (x, logw, logw_),
            g,
        )

    def infer(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: torch.Tensor,
        tone: torch.Tensor,
        language: torch.Tensor,
        bert: torch.Tensor,
        style_vec: torch.Tensor,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_scale_w: float = 0.8,
        max_len: Optional[int] = None,
        sdp_ratio: float = 0.0,
        y: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)
        else:
            assert y is not None
            g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, style_vec, g=g
        )
        logw = (
            self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * sdp_ratio
            + self.dp(x, x_mask, g=g) * (1 - sdp_ratio)
        )
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)
