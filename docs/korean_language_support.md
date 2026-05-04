# 한국어(KO) 언어 지원

Style-Bert-VITS2에 한국어 학습 파이프라인을 추가하기 위한 설계 및 구현 기록입니다.

## 언어별 처리 스택 비교

| 레이어 | 중국어 (ZH) | 일본어 (JP) | 영어 (EN) | 한국어 (KO) |
|--------|------------|------------|----------|--------------------|
| 정규화 | 숫자→한자, 구두점 통일 | NFKC + 숫자→일본어 | 구두점 통일 | 숫자→한국어, 구두점 통일 |
| G2P | pypinyin + jieba | pyopenjtalk | g2p_en + CMU Dict | Phase 1: 유니코드 자모 분해 / Phase 2: g2pk |
| 음소 수 | 77개 | 45개 | 39개 | 66개 (초성 18 + 중성 21 + 종성 27) |
| 음색(tone) 수 | 6 (성조) | 2 (고/저) | 4 (강세) | 1 (성조 없음) |
| BERT 모델 | chinese-roberta-wwm-ext-large | deberta-v2-large-japanese-char-wwm | deberta-v3-large | Phase 3에서 도입 |
| word2ph 방식 | 한자 1자 → N개 음소 | 토크나이저 기반 | 토크나이저 기반 | 음절 단위 자모 수 |

## 구현 전략: 독립 스크립트 신규 생성

JP-Extra 패턴과 동일하게 KO 전용 스크립트를 별도로 생성합니다. **기존 ZH/JP/EN 코드는 수정하지 않습니다.**

기존 파일 중 최소한으로 수정한 것:
- `style_bert_vits2/constants.py` — `Languages.KO = "KO"` 1줄 추가
- `style_bert_vits2/nlp/__init__.py` — `clean_text()`, `extract_bert_feature()` KO 분기 추가

`data_utils.py`, `train_ms.py`, `models.py`, `symbols.py`는 **무수정**입니다.

---

## 구현 단계

### Phase 1 — BERT 없음 / 자모 분해 입력 ✅ 완료

**G2P 방식**: 유니코드 산술로 한글 음절 → 초성/중성/종성 자모 분해 (외부 라이브러리 없음)

```
한글 음절 코드포인트 = U+AC00 + (초성 × 21 + 중성) × 28 + 종성
```

- 초성 ㅇ(null onset)은 음소를 발행하지 않음
- 연음·음운 변동 처리 없음, 표기 그대로 분해
- 예: `"안녕"` → `["KO_a", "KO_N", "KO_n", "KO_yeo", "KO_NG"]`

**심볼 테이블**: `symbols_ko.py`에 KO 전용 66개 심볼 추가 (총 178개)

```
SYMBOLS_KO = [PAD] + sorted(ZH + JP + EN + KO_SYMBOLS) + PUNCTUATION_SYMBOLS
NUM_TONES_KO = 13  (ZH 6 + JP 2 + EN 4 + KO 1)
NUM_LANGUAGES_KO = 4
```

**BERT 자리 유지**: Phase 3 호환을 위해 `bert_proj` 레이어는 모델에 포함, Phase 1에서는 zero 텐서 전달

**구현 파일**:

```
style_bert_vits2/nlp/korean/
├── __init__.py
├── g2p.py           # 유니코드 자모 분해
└── normalizer.py    # 정규화 (숫자→한글, 구두점 통일)

style_bert_vits2/nlp/symbols_ko.py      # KO 확장 심볼 테이블
style_bert_vits2/models/models_ko.py    # KO TextEncoder + SynthesizerTrn
data_utils_ko.py                        # KO 데이터로더 (JP-Extra 배치 포맷 호환)
preprocess_text_ko.py                   # 학습 데이터 전처리
train_ms_ko.py                          # 학습 진입점 (WavLM 제거, 3-net 구성)
```

---

### Phase 2 — BERT 없음 / G2PK 적용 (예정)

**G2P 방식**: `g2pk` 라이브러리로 음운 변동 반영

- 연음: 입력 → 임녁
- 격음화: 좋고 → 조코
- 비음화: 국민 → 궁민

Phase 1과 차이점: `g2p.py`만 교체, 나머지 구조 동일

---

### Phase 3 — BERT 도입 (예정)

**BERT 모델 후보**: `klue/roberta-large`, `snunlp/KR-ELECTRA-discriminator`

Phase 1/2 결과 비교 후 더 나은 G2P 방식을 채택하고 BERT 임베딩을 추가합니다.

---

## KSS 데이터셋으로 학습하기 (단계별 절차)

KSS(Korean Single Speaker) 데이터셋을 이용한 Phase 1 학습 전 과정입니다.

### 0. 데이터셋 배치

KSS 데이터셋을 다음 구조로 배치합니다.

```
Style-Bert-VITS2/
├── Data/
│   ├── transcript.v.1.4.txt   ← KSS 전사 파일 (루트에 위치)
│   └── kss/
│       ├── 1/  (wav 파일 1,040개)
│       ├── 2/  (wav 파일 1,157개)
│       ├── 3/  (wav 파일 5,025개)
│       └── 4/  (wav 파일 5,632개)
```

### 1. config.json 생성

`Data/kss/config.json`을 직접 생성합니다. `configs/config_jp_extra.json`을 기반으로 아래 항목을 변경합니다.

- `training_files` / `validation_files` → `Data/kss/train.list` / `Data/kss/val.list`
- `n_speakers` → `1`
- `use_wavlm_discriminator` → `false` (WavLM 미사용)
- `slm` 섹션 제거
- `freeze_ZH/JP/EN_bert` → 제거, `freeze_KO_bert: false` 추가
- `version` → `"2.7.0-KO-Phase1"`

### 2. KSS 전사 파일 → esd.list 변환

KSS transcript 형식을 `preprocess_text_ko.py` 입력 형식으로 변환합니다.

```
uv run python kss_to_esd.py
```

기본값으로 `Data/transcript.v.1.4.txt` → `Data/kss/esd.list` 변환 (12,854줄).

커스텀 경로 지정 시:
```
uv run python kss_to_esd.py --transcript Data/transcript.v.1.4.txt --kss-dir Data/kss --output Data/kss/esd.list
```

**출력 형식** (`esd.list`):
```
Data\kss\1\1_0000.wav|kss|KO|그는 괜찮은 척하려고 애쓰는 것 같았다.
Data\kss\1\1_0001.wav|kss|KO|그녀의 사랑을 얻기 위해 애썼지만 헛수고였다.
```

### 3. 텍스트 전처리

`esd.list`의 한국어 텍스트를 자모 분해하여 `train.list`, `val.list`를 생성하고 `config.json`의 `spk2id`와 `n_speakers`를 업데이트합니다.

```
uv run python preprocess_text_ko.py --transcription-path Data/kss/esd.list --train-path Data/kss/train.list --val-path Data/kss/val.list --config-path Data/kss/config.json --val-per-spk 10
```

- `--val-per-spk 10`: 화자당 10개를 검증 데이터로 분리
- 결과: train 12,844개 / val 10개
- `config.json` 자동 갱신: `"spk2id": {"kss": 0}`, `"n_speakers": 1`

**출력 형식** (`train.list`):
```
Data\kss\1\1_0000.wav|kss|KO|그는 ...|_ KO_g KO_eu KO_n ...|0 0 0 ...|1 2 3 ...
```

### 4. 스타일 벡터 생성

각 wav 파일에 대해 256차원 화자 임베딩 `.npy` 파일을 생성합니다. `pyannote/wespeaker-voxceleb-resnet34-LM` 모델을 사용합니다 (최초 실행 시 HuggingFace에서 다운로드).

```
uv run python style_gen.py -c Data/kss/config.json
```

- 12,854개 wav → 12,854개 `.npy` 파일 생성 (동일 경로에 `{wav}.npy` 형태로 저장)
- 처리 속도: GPU 기준 약 150~170 파일/초, 총 소요 시간 약 1~2분

> **참고**: torchaudio 2.x와 pyannote.audio 3.x의 API 호환성 문제로 `style_gen.py`에 아래 패치가 적용되어 있습니다.
> - torchaudio 2.11에서 제거된 `AudioMetaData`, `info()`, `list_audio_backends()`, `load()`를 soundfile로 대체
> - PyTorch 2.6의 `weights_only=True` 기본값 변경으로 인한 모델 로드 실패를 `lightning_fabric` 패치로 우회

### 5. 학습 실행

```
uv run python train_ms_ko.py -c Data/kss/config.json -m Data/kss
```

- 실행할 때마다 타임스탬프 서브디렉토리가 생성됩니다:
  ```
  Data/kss/models/
    20260505_011000/        ← 1차 학습
      train_20260505_011000.log
      G_4000.pth / D_4000.pth
      eval/
    20260505_120000/        ← 2차 학습
      ...
  ```
- `eval_interval` (config): evaluation + TensorBoard 기록 주기 (기본 1000 steps)
- `save_interval` (config): 체크포인트 저장 주기 (기본 eval_interval과 동일, 별도 지정 가능)
- TensorBoard: `uv run tensorboard --logdir Data/kss/models`

---

## 기능 테스트

### 1. G2P 단위 테스트

정규화 → 자모 분해 → 심볼 ID 변환의 전체 흐름을 확인합니다.

```bash
uv run python -c "
from style_bert_vits2.nlp.korean.normalizer import normalize_text
from style_bert_vits2.nlp.korean.g2p import g2p
from style_bert_vits2.nlp.symbols_ko import cleaned_text_to_sequence_ko

text = '안녕하세요, 반갑습니다.'
norm = normalize_text(text)
phones, tones, word2ph = g2p(norm)
ids, tone_ids, lang_ids = cleaned_text_to_sequence_ko(phones, tones, 'KO')

print('정규화:', norm)
print('음소:', phones)
print('word2ph 합계 == 음소 수:', sum(word2ph) == len(phones))
print('phone IDs:', ids[:8], '...')
"
```

기대 출력:
```
정규화: 안녕하세요, 반갑습니다.
음소: ['_', 'KO_a', 'KO_N', 'KO_n', 'KO_yeo', 'KO_NG', ...]
word2ph 합계 == 음소 수: True
```

### 2. clean_text 인터페이스 테스트

`nlp/__init__.py`의 공통 인터페이스를 통해 KO 분기가 올바르게 동작하는지 확인합니다.

```bash
uv run python -c "
from style_bert_vits2.nlp import clean_text, extract_bert_feature
from style_bert_vits2.constants import Languages

norm_text, phones, tones, word2ph = clean_text('오늘 날씨가 좋네요.', Languages.KO)
print('norm_text:', norm_text)
print('phones:', phones)

# Phase 1: zero tensor 반환 확인
bert = extract_bert_feature('오늘 날씨', word2ph, Languages.KO, 'cpu')
print('bert shape:', bert.shape)   # 기대: torch.Size([1024, N])
print('bert all zeros:', bert.sum().item() == 0.0)
"
```

### 3. 전처리 스크립트 테스트

`esd.list` 형식 파일을 만들어 전처리 결과를 확인합니다.

**입력 파일 형식** (`Data/test_ko/esd.list`):
```
wavs/001.wav|speaker1|KO|안녕하세요 반갑습니다
wavs/002.wav|speaker1|KO|오늘 날씨가 참 좋네요
wavs/003.wav|speaker2|KO|한국어 TTS 테스트입니다
```

**실행**:
```bash
uv run python preprocess_text_ko.py \
    --transcription-path Data/test_ko/esd.list \
    --train-path Data/test_ko/train.list \
    --val-path   Data/test_ko/val.list \
    --config-path Data/test_ko/config.json
```

**출력 파일 형식** (`train.list`):
```
wavs/001.wav|speaker1|KO|안녕하세요 반갑습니다|KO_a KO_N KO_n KO_yeo ...|0 0 0 ...|2 3 2 ...
```

### 4. 데이터로더 단위 테스트

배치 텐서의 shape와 dtype을 확인합니다.

```bash
uv run python -c "
from data_utils_ko import TextAudioSpeakerLoaderKO, TextAudioSpeakerCollateKO
from style_bert_vits2.models.hyper_parameters import HyperParameters

hps = HyperParameters.load_from_json('Data/test_ko/config.json')
dataset = TextAudioSpeakerLoaderKO(hps.data.training_files, hps.data)
collate = TextAudioSpeakerCollateKO()

sample = dataset[0]
# (phones, spec, wav, sid, tone, language, bert, style_vec)
phones, spec, wav, sid, tone, language, bert, style_vec = sample
print('phones shape:', phones.shape)
print('bert shape  :', bert.shape)    # 기대: [1024, T]
print('bert zeros  :', bert.sum().item() == 0.0)
"
```

### 5. 모델 인스턴스화 테스트

`models_ko.py`의 SynthesizerTrn이 올바르게 생성되는지 확인합니다.

```bash
uv run python -c "
import torch
from style_bert_vits2.models.models_ko import SynthesizerTrn
from style_bert_vits2.nlp.symbols_ko import SYMBOLS_KO

net_g = SynthesizerTrn(
    n_vocab=len(SYMBOLS_KO),
    spec_channels=513,
    segment_size=32,
    inter_channels=192,
    hidden_channels=192,
    filter_channels=768,
    n_heads=2,
    n_layers=6,
    kernel_size=3,
    p_dropout=0.1,
    resblock='1',
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]],
    upsample_rates=[8, 8, 2, 2],
    upsample_initial_channel=512,
    upsample_kernel_sizes=[16, 16, 4, 4],
    n_speakers=1,
    gin_channels=256,
)
total = sum(p.numel() for p in net_g.parameters()) / 1e6
print(f'파라미터 수: {total:.1f}M')
print('enc_p.emb.weight shape:', net_g.enc_p.emb.weight.shape)
# 기대: [178, 192] (len(SYMBOLS_KO) x hidden_channels)
"
```

---

## 음색(tone) 오프셋 구조

```python
LANGUAGE_TONE_START_MAP_KO = {
    "ZH":  0,   # 0~5  (6가지 성조)
    "JP":  6,   # 6~7  (고/저)
    "EN":  8,   # 8~11 (강세 4단계)
    "KO": 12,   # 12   (단일값)
}
NUM_TONES_KO = 13
```

## 모델 구조

KO 모델은 JP-Extra 아키텍처를 그대로 사용하며, 심볼/tone/language 수만 KO 기준으로 교체합니다.

### 구조 파악 시 볼 파일

| 파일 | 내용 |
|------|------|
| `style_bert_vits2/models/models_ko.py` | KO 전용 `TextEncoder`, `SynthesizerTrn` (진입점) |
| `style_bert_vits2/models/models_jp_extra.py` | 나머지 전체 컴포넌트 원본 |

### 컴포넌트 출처

| 컴포넌트 | 출처 |
|----------|------|
| `TextEncoder` | `models_ko.py` — KO 심볼 수로 재정의 |
| `SynthesizerTrn` | `models_ko.py` — KO TextEncoder를 조립 |
| `PosteriorEncoder` | `models_jp_extra.py` 그대로 사용 |
| `Generator` (HiFi-GAN decoder) | `models_jp_extra.py` 그대로 사용 |
| `ResidualCouplingBlock` | `models_jp_extra.py` 그대로 사용 |
| `StochasticDurationPredictor` | `models_jp_extra.py` 그대로 사용 |
| `MultiPeriodDiscriminator` | `models_jp_extra.py` 그대로 사용 |
| `ReferenceEncoder` (style) | `models_jp_extra.py` 그대로 사용 |

`TextEncoder`는 `bert_proj` 레이어(1024→hidden_channels)를 포함합니다. Phase 1에서는 zero 텐서를 입력하고, Phase 3에서 실제 KO BERT 임베딩으로 교체됩니다.

---

## 관련 파일

| 구분 | 파일 |
|------|------|
| 언어 상수 | `style_bert_vits2/constants.py` |
| KO 심볼 테이블 | `style_bert_vits2/nlp/symbols_ko.py` |
| KO 정규화 | `style_bert_vits2/nlp/korean/normalizer.py` |
| KO G2P | `style_bert_vits2/nlp/korean/g2p.py` |
| NLP 공통 인터페이스 | `style_bert_vits2/nlp/__init__.py` |
| KO 모델 (진입점) | `style_bert_vits2/models/models_ko.py` |
| 전체 아키텍처 원본 | `style_bert_vits2/models/models_jp_extra.py` |
| KO 데이터로더 | `data_utils_ko.py` |
| KO 전처리 | `preprocess_text_ko.py` |
| KO 학습 | `train_ms_ko.py` |
