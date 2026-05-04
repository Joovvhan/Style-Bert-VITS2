"""
KO Phase 1 모델 구조 확인 스크립트.

- 서브모듈별 파라미터 수 표
- forward hook으로 실제 입출력 tensor shape + 이름 캡처
- torchinfo로 레이어 트리 + 파라미터 수

Usage:
    uv run python inspect_model_ko.py
    uv run python inspect_model_ko.py -c Data/kss/config.json --depth 4
    uv run python inspect_model_ko.py --no-forward   # shape 캡처 없이 트리만
"""

import argparse
import inspect
from typing import Any

import torch
from torchinfo import summary

from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.models_ko import SynthesizerTrn
from style_bert_vits2.nlp.symbols_ko import SYMBOLS_KO


def build_model(hps) -> SynthesizerTrn:
    m = hps.model
    return SynthesizerTrn(
        n_vocab=len(SYMBOLS_KO),
        spec_channels=hps.data.filter_length // 2 + 1,
        segment_size=hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=0.01 if m.use_noise_scaled_mas else 0.0,
        noise_scale_delta=2e-6 if m.use_noise_scaled_mas else 0.0,
        use_spk_conditioned_encoder=m.use_spk_conditioned_encoder,
        use_noise_scaled_mas=m.use_noise_scaled_mas,
        use_mel_posterior_encoder=m.use_mel_posterior_encoder,
        use_duration_discriminator=m.use_duration_discriminator,
        inter_channels=m.inter_channels,
        hidden_channels=m.hidden_channels,
        filter_channels=m.filter_channels,
        n_heads=m.n_heads,
        n_layers=m.n_layers,
        kernel_size=m.kernel_size,
        p_dropout=m.p_dropout,
        resblock=m.resblock,
        resblock_kernel_sizes=m.resblock_kernel_sizes,
        resblock_dilation_sizes=m.resblock_dilation_sizes,
        upsample_rates=m.upsample_rates,
        upsample_initial_channel=m.upsample_initial_channel,
        upsample_kernel_sizes=m.upsample_kernel_sizes,
        n_layers_q=m.n_layers_q,
        use_spectral_norm=m.use_spectral_norm,
        gin_channels=m.gin_channels,
    )


def fmt_shape(x: Any) -> str:
    if isinstance(x, torch.Tensor):
        return str(list(x.shape))
    if isinstance(x, (tuple, list)):
        parts = [fmt_shape(v) for v in x if isinstance(v, torch.Tensor)]
        return "[" + ", ".join(parts) + "]" if parts else "-"
    return "-"


def capture_shapes(model: SynthesizerTrn, hps) -> dict:
    """
    with_kwargs=True 훅으로 positional + keyword 인자를 모두 캡처.
    inspect.signature로 파라미터 이름 매핑.
    """
    B, T_text, T_spec = 1, 20, 200
    spec_ch = hps.data.filter_length // 2 + 1

    dummy = dict(
        x=torch.randint(0, len(SYMBOLS_KO), (B, T_text)),
        x_lengths=torch.LongTensor([T_text]),
        y=torch.randn(B, spec_ch, T_spec),
        y_lengths=torch.LongTensor([T_spec]),
        sid=torch.LongTensor([0]),
        tone=torch.zeros(B, T_text, dtype=torch.long),
        language=torch.zeros(B, T_text, dtype=torch.long),
        bert=torch.zeros(B, 1024, T_text),
        style_vec=torch.zeros(B, 256),
    )

    captured: dict = {}
    hooks = []

    for name, module in model.named_children():
        param_names = list(inspect.signature(module.forward).parameters.keys())

        def make_hook(n, pnames):
            def hook(m, args, kwargs, out):
                named_in = []
                for i, val in enumerate(args):
                    if isinstance(val, torch.Tensor):
                        label = pnames[i] if i < len(pnames) else f"arg{i}"
                        named_in.append((label, fmt_shape(val)))
                for k, val in kwargs.items():
                    if isinstance(val, torch.Tensor):
                        named_in.append((f"{k} (kw)", fmt_shape(val)))

                named_out = []
                if isinstance(out, torch.Tensor):
                    named_out.append(("output", fmt_shape(out)))
                elif isinstance(out, (tuple, list)):
                    for i, v in enumerate(out):
                        if isinstance(v, torch.Tensor):
                            named_out.append((f"out[{i}]", fmt_shape(v)))

                captured[n] = {
                    "class": type(m).__name__,
                    "params": sum(p.numel() for p in m.parameters()),
                    "in": named_in,
                    "out": named_out,
                }
            return hook

        hooks.append(module.register_forward_hook(
            make_hook(name, param_names), with_kwargs=True))

    model.eval()
    with torch.no_grad():
        model(**dummy)

    for h in hooks:
        h.remove()

    return captured


def print_shapes(captured: dict):
    sep = "=" * 66
    for mod_name, data in captured.items():
        print(f"\n{sep}")
        print(f"  {mod_name}  ({data['class']})  --  {data['params']:,} params")
        print(sep)

        if data["in"]:
            print("  [ 입력 ]")
            for pname, shape in data["in"]:
                print(f"    {pname:<18}: {shape}")
        else:
            print("  [ 입력 ] (없음)")

        if data["out"]:
            print("  [ 출력 ]")
            for pname, shape in data["out"]:
                print(f"    {pname:<18}: {shape}")
        else:
            print("  [ 출력 ] (텐서 없음)")

    print(f"\n{sep}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="Data/kss/config.json")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--no-forward", action="store_true",
                        help="dummy forward 생략 (shape 없이 트리만)")
    parser.add_argument("--shapes-only", action="store_true",
                        help="shape 상세 출력만 (torchinfo 생략)")
    args = parser.parse_args()

    hps = HyperParameters.load_from_json(args.config)
    net_g = build_model(hps)

    total = sum(p.numel() for p in net_g.parameters())
    print(f"\n{'='*66}")
    print(f"  config  : {args.config}")
    print(f"  n_vocab : {len(SYMBOLS_KO)}  spec_ch : {hps.data.filter_length // 2 + 1}")
    print(f"  hidden  : {hps.model.hidden_channels}   filter  : {hps.model.filter_channels}")
    print(f"  n_layer : {hps.model.n_layers}   n_heads : {hps.model.n_heads}")
    print(f"  total   : {total/1e6:.2f}M params")
    print(f"{'='*66}\n")

    if not args.no_forward:
        print("[ 서브모듈 입출력 shapes  (B=1, T_text=20, T_spec=200) ]")
        captured = capture_shapes(net_g, hps)
        print_shapes(captured)

    if not args.shapes_only:
        print(f"\n[ torchinfo summary (depth={args.depth}) ]\n")
        summary(
            net_g,
            verbose=1,
            depth=args.depth,
            col_names=["num_params", "trainable"],
            row_settings=["var_names"],
        )


if __name__ == "__main__":
    main()
