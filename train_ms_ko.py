"""
Korean TTS training script (Phase 1: no BERT, single GPU).
Based on train_ms_jp_extra.py; distributed training and WavLM removed.
"""

import argparse
import datetime
import gc
import os

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import default_style
from config import get_config
from data_utils_ko import (
    TextAudioSpeakerCollateKO,
    TextAudioSpeakerLoaderKO,
)
from losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons, utils
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.models_ko import (
    DurationDiscriminator,
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from style_bert_vits2.nlp.symbols_ko import SYMBOLS_KO as SYMBOLS
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

config = get_config()
global_step = 0


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        default=config.train_ms_config.config_path)
    parser.add_argument("-m", "--model", type=str,
                        default=config.dataset_path)
    parser.add_argument("--assets_root", type=str,
                        default=config.assets_root)
    parser.add_argument("--skip_default_style", action="store_true")
    parser.add_argument("--style-source", type=str, default=None,
                        help="스타일 벡터 생성용 .npy 탐색 경로 (기본값: --model 경로와 동일)")
    parser.add_argument("--no_progress_bar", action="store_true")
    parser.add_argument("--speedup", action="store_true")
    parser.add_argument("--no-spec-cache", action="store_true",
                        help="spec.pt 저장/로드 없이 매번 STFT 계산 (epoch 간 속도 비교용)")
    parser.add_argument("--resume", type=str, default=None,
                        help="이어서 학습할 체크포인트 디렉토리 경로 "
                             "(예: Data/kss/models/20260505_124239).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="총 학습 epoch 수. 지정하면 config.json의 epochs를 덮어씀.")
    args = parser.parse_args()

    if args.resume:
        model_dir = args.resume
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(args.model, config.train_ms_config.model_dir, timestamp)
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.add(os.path.join(model_dir, f"train_{timestamp}.log"))

    hps = HyperParameters.load_from_json(args.config)
    hps.model_dir = model_dir
    hps.speedup = args.speedup
    if args.epochs is not None:
        hps.train.epochs = args.epochs

    # WARNING: 아래 블록은 --config로 지정한 파일을 config.yml의 dataset_path 기준 config.json에
    # 덮어쓰는 동작을 한다. --config와 dataset_path가 다른 모델을 가리킬 경우 엉뚱한 config.json이
    # 오염되므로 비활성화함.
    # if os.path.realpath(args.config) != os.path.realpath(config.train_ms_config.config_path):
    #     with open(args.config, encoding="utf-8") as f:
    #         data = f.read()
    #     os.makedirs(os.path.dirname(config.train_ms_config.config_path), exist_ok=True)
    #     with open(config.train_ms_config.config_path, "w", encoding="utf-8") as f:
    #         f.write(data)

    os.makedirs(config.out_dir, exist_ok=True)

    if not args.skip_default_style:
        style_source = args.style_source if args.style_source else args.model
        default_style.save_styles_by_dirs(
            style_source, config.out_dir,
            config_path=args.config,
            config_output_path=os.path.join(config.out_dir, "config.json"),
        )

    torch.manual_seed(hps.train.seed)

    global global_step
    writer = writer_eval = None
    if not args.speedup:
        utils.check_git_hash(model_dir)
        writer = SummaryWriter(log_dir=model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(model_dir, "eval"))

    train_dataset = TextAudioSpeakerLoaderKO(hps.data.training_files, hps.data,
                                              no_spec_cache=args.no_spec_cache)
    collate_fn = TextAudioSpeakerCollateKO()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn,
        batch_size=hps.train.batch_size,
        drop_last=True,
    )

    eval_dataset = TextAudioSpeakerLoaderKO(hps.data.validation_files, hps.data,
                                             no_spec_cache=args.no_spec_cache)
    eval_loader = DataLoader(
        eval_dataset,
        num_workers=0,
        shuffle=False,
        batch_size=1,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    mas_noise_scale_initial = 0.01 if hps.model.use_noise_scaled_mas else 0.0
    noise_scale_delta = 2e-6 if hps.model.use_noise_scaled_mas else 0.0

    net_dur_disc = None
    if hps.model.use_duration_discriminator:
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels, hps.model.hidden_channels, 3, 0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).cuda()

    net_g = SynthesizerTrn(
        len(SYMBOLS),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
        use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
        use_duration_discriminator=hps.model.use_duration_discriminator,
        inter_channels=hps.model.inter_channels,
        hidden_channels=hps.model.hidden_channels,
        filter_channels=hps.model.filter_channels,
        n_heads=hps.model.n_heads,
        n_layers=hps.model.n_layers,
        kernel_size=hps.model.kernel_size,
        p_dropout=hps.model.p_dropout,
        resblock=hps.model.resblock,
        resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
        resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
        upsample_rates=hps.model.upsample_rates,
        upsample_initial_channel=hps.model.upsample_initial_channel,
        upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
        gin_channels=hps.model.gin_channels,
    ).cuda()

    if getattr(hps.train, "freeze_KO_bert", False):
        for param in net_g.enc_p.bert_proj.parameters():
            param.requires_grad = False
    if getattr(hps.train, "freeze_style", False):
        for param in net_g.enc_p.style_proj.parameters():
            param.requires_grad = False
    if getattr(hps.train, "freeze_decoder", False):
        for param in net_g.dec.parameters():
            param.requires_grad = False

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda()

    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_dur_disc = None
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

    if utils.is_resuming(model_dir):
        if net_dur_disc is not None:
            try:
                _, _, dur_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
                    utils.checkpoints.get_latest_checkpoint_path(model_dir, "DUR_*.pth"),
                    net_dur_disc, optim_dur_disc,
                    skip_optimizer=hps.train.skip_optimizer)
                if not optim_dur_disc.param_groups[0].get("initial_lr"):
                    optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr
            except Exception:
                pass
        try:
            _, optim_g, g_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
                utils.checkpoints.get_latest_checkpoint_path(model_dir, "G_*.pth"),
                net_g, optim_g, skip_optimizer=hps.train.skip_optimizer)
            _, optim_d, d_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
                utils.checkpoints.get_latest_checkpoint_path(model_dir, "D_*.pth"),
                net_d, optim_d, skip_optimizer=hps.train.skip_optimizer)
            if not optim_g.param_groups[0].get("initial_lr"):
                optim_g.param_groups[0]["initial_lr"] = g_resume_lr
            if not optim_d.param_groups[0].get("initial_lr"):
                optim_d.param_groups[0]["initial_lr"] = d_resume_lr
            epoch_str = max(epoch_str, 1)
            global_step = int(utils.get_steps(
                utils.checkpoints.get_latest_checkpoint_path(model_dir, "G_*.pth")))
            logger.info(f"Resuming from epoch {epoch_str}, step {global_step}")
        except Exception as e:
            logger.warning(f"Training from scratch: {e}")
            epoch_str = 1
            global_step = 0
    else:
        try:
            utils.safetensors.load_safetensors(
                os.path.join(model_dir, "G_0.safetensors"), net_g)
            utils.safetensors.load_safetensors(
                os.path.join(model_dir, "D_0.safetensors"), net_d)
            if net_dur_disc is not None:
                utils.safetensors.load_safetensors(
                    os.path.join(model_dir, "DUR_0.safetensors"), net_dur_disc)
            logger.info("Loaded pretrained models.")
        except Exception as e:
            logger.warning(f"Training from scratch: {e}")
        finally:
            epoch_str = 1
            global_step = 0

    def lr_lambda(epoch):
        if epoch < hps.train.warmup_epochs:
            return float(epoch) / float(max(1, hps.train.warmup_epochs))
        return hps.train.lr_decay ** (epoch - hps.train.warmup_epochs)

    scheduler_last_epoch = epoch_str - 2
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        optim_g, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(
        optim_d, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch)
    scheduler_dur_disc = None
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.LambdaLR(
            optim_dur_disc, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch)

    scaler = GradScaler(enabled=hps.train.bf16_run)
    logger.info("Start training (KO Phase 1, single GPU).")

    diff = abs(epoch_str * len(train_loader) - (hps.train.epochs + 1) * len(train_loader))
    pbar = None
    if not args.no_progress_bar:
        pbar = tqdm(total=global_step + diff, initial=global_step,
                    smoothing=0.05, file=SAFE_STDOUT, dynamic_ncols=True)
    initial_step = global_step

    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_and_evaluate(
            epoch, hps,
            [net_g, net_d, net_dur_disc],
            [optim_g, optim_d, optim_dur_disc],
            [scheduler_g, scheduler_d, scheduler_dur_disc],
            scaler,
            [train_loader, eval_loader],
            [writer, writer_eval],
            pbar, initial_step,
        )
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()

        if epoch == hps.train.epochs:
            utils.checkpoints.save_checkpoint(
                net_g, optim_g, hps.train.learning_rate, epoch,
                os.path.join(model_dir, f"G_{global_step}.pth"))
            utils.checkpoints.save_checkpoint(
                net_d, optim_d, hps.train.learning_rate, epoch,
                os.path.join(model_dir, f"D_{global_step}.pth"))
            if net_dur_disc is not None:
                utils.checkpoints.save_checkpoint(
                    net_dur_disc, optim_dur_disc, hps.train.learning_rate, epoch,
                    os.path.join(model_dir, f"DUR_{global_step}.pth"))
            utils.safetensors.save_safetensors(
                net_g, epoch,
                os.path.join(config.out_dir,
                             f"{config.model_name}_e{epoch}_s{global_step}.safetensors"),
                for_infer=True)

    if pbar is not None:
        pbar.close()


def train_and_evaluate(
    epoch, hps,
    nets, optims, schedulers,
    scaler, loaders, writers,
    pbar, initial_step,
):
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    train_loader, eval_loader = loaders
    writer, writer_eval = writers

    global global_step
    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()

    for batch_idx, (
        x, x_lengths, spec, spec_lengths,
        y, y_lengths, speakers, tone, language, bert, style_vec,
    ) in enumerate(train_loader):
        if net_g.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.mas_noise_scale_initial
                - net_g.noise_scale_delta * global_step
            )
            net_g.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)

        x           = x.cuda(non_blocking=True)
        x_lengths   = x_lengths.cuda(non_blocking=True)
        spec        = spec.cuda(non_blocking=True)
        spec_lengths = spec_lengths.cuda(non_blocking=True)
        y           = y.cuda(non_blocking=True)
        y_lengths   = y_lengths.cuda(non_blocking=True)
        speakers    = speakers.cuda(non_blocking=True)
        tone        = tone.cuda(non_blocking=True)
        language    = language.cuda(non_blocking=True)
        bert        = bert.cuda(non_blocking=True)
        style_vec   = style_vec.cuda(non_blocking=True)

        with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
            (y_hat, l_length, attn, ids_slice, x_mask, z_mask,
             (z, z_p, m_p, logs_p, m_q, logs_q),
             (hidden_x, logw, logw_), g,
             ) = net_g(x, x_lengths, spec, spec_lengths,
                       speakers, tone, language, bert, style_vec)

            mel = spec_to_mel_torch(spec, hps.data.filter_length,
                                    hps.data.n_mel_channels, hps.data.sampling_rate,
                                    hps.data.mel_fmin, hps.data.mel_fmax)
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(), hps.data.filter_length,
                hps.data.n_mel_channels, hps.data.sampling_rate,
                hps.data.hop_length, hps.data.win_length,
                hps.data.mel_fmin, hps.data.mel_fmax)
            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size)

            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc

            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(), x_mask.detach(),
                    logw_.detach(), logw.detach(), g.detach())
                with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
                    loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g = discriminator_loss(
                        y_dur_hat_r, y_dur_hat_g)
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc).backward()
                scaler.unscale_(optim_dur_disc)
                commons.clip_grad_value_(net_dur_disc.parameters(), None)
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                _, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw, g)
            with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=500)
        commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if global_step % hps.train.log_interval == 0 and not hps.speedup:
            lr = optim_g.param_groups[0]["lr"]
            scalar_dict = {
                "loss/g/total": loss_gen_all,
                "loss/d/total": loss_disc_all,
                "learning_rate": lr,
                "grad_norm_d": grad_norm_d,
                "loss/g/fm": loss_fm,
                "loss/g/mel": loss_mel,
                "loss/g/dur": loss_dur,
                "loss/g/kl": loss_kl,
            }
            scalar_dict.update({f"loss/g/{i}": v for i, v in enumerate(losses_gen)})
            scalar_dict.update({f"loss/d_r/{i}": v for i, v in enumerate(losses_disc_r)})
            scalar_dict.update({f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)})
            if net_dur_disc is not None:
                scalar_dict.update({"loss/dur_disc/total": loss_dur_disc})
                scalar_dict.update({"loss/g/dur_gen": loss_dur_gen})
            utils.summarize(writer=writer, global_step=global_step, scalars=scalar_dict)

        not_initial = global_step != 0 and initial_step != global_step
        if (global_step % hps.train.eval_interval == 0 and not_initial):
            if not hps.speedup:
                evaluate(hps, net_g, eval_loader, writer_eval)

        save_interval = hps.train.save_interval or hps.train.eval_interval
        if (global_step % save_interval == 0 and not_initial):
            utils.checkpoints.save_checkpoint(
                net_g, optim_g, hps.train.learning_rate, epoch,
                os.path.join(hps.model_dir, f"G_{global_step}.pth"))
            utils.checkpoints.save_checkpoint(
                net_d, optim_d, hps.train.learning_rate, epoch,
                os.path.join(hps.model_dir, f"D_{global_step}.pth"))
            if net_dur_disc is not None:
                utils.checkpoints.save_checkpoint(
                    net_dur_disc, optim_dur_disc, hps.train.learning_rate, epoch,
                    os.path.join(hps.model_dir, f"DUR_{global_step}.pth"))
            keep_ckpts = config.train_ms_config.keep_ckpts
            if keep_ckpts > 0:
                utils.checkpoints.clean_checkpoints(
                    model_dir_path=hps.model_dir, n_ckpts_to_keep=keep_ckpts,
                    sort_by_time=True)
            utils.safetensors.save_safetensors(
                net_g, epoch,
                os.path.join(config.out_dir,
                             f"{config.model_name}_e{epoch}_s{global_step}.safetensors"),
                for_infer=True)

        global_step += 1
        if pbar is not None:
            pbar.set_description(
                f"Epoch {epoch}({100.0 * batch_idx / len(train_loader):.0f}%)/{hps.train.epochs}")
            pbar.update()

    # gc.collect()
    # torch.cuda.empty_cache()
    if pbar is None:
        logger.info(f"====> Epoch: {epoch}, step: {global_step}")


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    audio_dict = {}
    logger.info("Evaluating ...")
    with torch.no_grad():
        for batch_idx, (
            x, x_lengths, spec, spec_lengths,
            y, y_lengths, speakers, tone, language, bert, style_vec,
        ) in enumerate(eval_loader):
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()
            speakers = speakers.cuda()
            bert = bert.cuda()
            tone = tone.cuda()
            language = language.cuda()
            style_vec = style_vec.cuda()
            for use_sdp in [True, False]:
                y_hat, attn, mask, *_ = generator.infer(
                    x, x_lengths, speakers, tone, language, bert, style_vec,
                    y=spec, max_len=1000, sdp_ratio=0.0 if not use_sdp else 1.0)
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
                audio_dict[f"gen/audio_{batch_idx}_{use_sdp}"] = y_hat[0, :, :y_hat_lengths[0]]
                audio_dict[f"gt/audio_{batch_idx}"] = y[0, :, :y_lengths[0]]
    utils.summarize(writer=writer_eval, global_step=global_step,
                    audios=audio_dict, audio_sampling_rate=hps.data.sampling_rate)
    generator.train()


if __name__ == "__main__":
    run()
