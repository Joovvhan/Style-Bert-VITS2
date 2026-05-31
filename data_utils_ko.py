"""
KO-specific dataset utilities (Phase 1, no BERT).
Subclasses the standard TextAudioSpeakerLoader/Collate, overriding only
the language-specific parts so the rest of the pipeline is unchanged.
"""

import numpy as np
import torch

from data_utils import (
    DistributedBucketSampler,  # re-exported for train_ms_ko.py convenience
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from mel_processing import spectrogram_torch
from style_bert_vits2.models import commons
from style_bert_vits2.models.utils import load_wav_to_torch
from style_bert_vits2.nlp.symbols_ko import cleaned_text_to_sequence_ko


class TextAudioSpeakerLoaderKO(TextAudioSpeakerLoader):
    """
    Drop-in replacement for TextAudioSpeakerLoader for Korean training.
    - Uses the KO-extended symbol table (symbols_ko.py)
    - Returns zero BERT tensors (Phase 1: no language model)
    - Batch format matches JP-Extra single-bert layout
    """

    def __init__(self, audiopaths_sid_text, hparams, no_spec_cache: bool = False):
        super().__init__(audiopaths_sid_text, hparams)
        self._no_spec_cache = no_spec_cache
        self.use_ko_bert = getattr(hparams, "use_ko_bert", False)

    def get_audio(self, filename):
        if not self._no_spec_cache:
            return super().get_audio(filename)
        # --no-spec-cache: WAV read + STFT 계산만, 저장/로드 없음
        audio, sampling_rate = load_wav_to_torch(filename)
        if audio.ndim == 2:
            audio = audio.mean(dim=-1)
        audio_norm = (audio / self.max_wav_value).unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.filter_length,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            center=False,
        )
        return torch.squeeze(spec, 0), audio_norm

    def get_text(self, text, word2ph, phone, tone, language_str, wav_path):
        phone, tone, language = cleaned_text_to_sequence_ko(phone, tone, language_str)
        if self.add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1

        if self.use_ko_bert:
            bert_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".bert.pt")
            ko_bert = torch.load(bert_path, weights_only=True)
            assert ko_bert.shape[-1] == len(phone), (
                f"bert.pt shape {ko_bert.shape} != phone len {len(phone)}: {wav_path}"
            )
        else:
            ko_bert = torch.zeros(1024, len(phone))
        return (
            ko_bert,
            torch.LongTensor(phone),
            torch.LongTensor(tone),
            torch.LongTensor(language),
        )

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        audiopath, sid, language, text, phones, tone, word2ph = audiopath_sid_text
        ko_bert, phones, tone, language = self.get_text(
            text, word2ph, phones, tone, language, audiopath
        )
        spec, wav = self.get_audio(audiopath)
        sid = torch.LongTensor([int(self.spk_map[sid])])
        style_vec = torch.FloatTensor(np.load(f"{audiopath}.npy"))
        # (phones, spec, wav, sid, tone, language, bert, style_vec)
        # — same shape as JP-Extra single-bert batch
        return phones, spec, wav, sid, tone, language, ko_bert, style_vec


class TextAudioSpeakerCollateKO(TextAudioSpeakerCollate):
    """Collate for KO: single bert slot, same layout as JP-Extra."""

    def __init__(self, return_ids: bool = False):
        # use_jp_extra=True selects the single-bert branch in the parent collate
        super().__init__(return_ids=return_ids, use_jp_extra=True)
