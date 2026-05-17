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

**G2P 방식**: 규칙 기반 G2PK 라이브러리로 음운 변동 반영 (사용 패키지 미결정, 하단 참고)

#### 음운 변동 규칙

| 규칙 | 예시 (표기 → 발음) |
|------|-------------------|
| 연음 | 입력 → 임녁 |
| 격음화 | 좋고 → 조코 |
| 경음화 | 학교 → 학꾜 |
| 비음화 | 국민 → 궁민 |
| ㄴ삽입 | 솜이불 → 솜니불 |
| 구개음화 | 같이 → 가치 |

#### 데이터 흐름

G2PK는 전처리 시점(`preprocess_text_ko.py`)에만 적용. `g2p.py`(자모 분해)는 수정하지 않음.

```
Phase 1: 텍스트 ─────────────────────→ 자모 분해 → train.list
Phase 2: 텍스트 → g2pk(음운 변동) → 발음형 → 자모 분해 → train_g2pk.list
```

Phase 1과 Phase 2는 별도 `.list` 파일을 사용하므로 동시에 유지 가능.

#### 구현 방식: `--g2p` 플래그 (방안 C)

전처리와 학습 모두 `--g2p {jamo,g2pk}` 플래그 하나로 모드 전환.

**파일 경로 규칙**: `--g2p g2pk` 지정 시 `.list` → `_g2pk.list` 자동 치환.

| 모드 | 전처리 출력 | 학습 입력 |
|------|------------|----------|
| `--g2p jamo` (기본) | `train.list` | `train.list` |
| `--g2p g2pk` | `train_g2pk.list` | `train_g2pk.list` |

**전처리 실행**:
```bash
# Phase 1 (기존 그대로)
uv run python preprocess_text_ko.py --transcription-path Data/kss/esd.list ...

# Phase 2
uv run python preprocess_text_ko.py --transcription-path Data/kss/esd.list ... --g2p g2pk
# → Data/kss/train_g2pk.list, Data/kss/val_g2pk.list 생성
```

**학습 실행**:
```bash
# Phase 1 (기존 그대로)
uv run python train_ms_ko.py -c Data/kss/config.json -m Data/kss

# Phase 2
uv run python train_ms_ko.py -c Data/kss/config.json -m Data/kss --g2p g2pk
# → config의 training_files 경로에서 .list → _g2pk.list 자동 치환
```

#### 변경 파일 범위

| 파일 | 변경 내용 |
|------|----------|
| `preprocess_text_ko.py` | `--g2p` 플래그 추가, g2pk 호출 분기, 출력 경로 자동 변경 |
| `train_ms_ko.py` | `--g2p` 플래그 추가, `hps.data.training/validation_files` 경로 치환 |
| `style_bert_vits2/nlp/korean/g2p.py` | **수정 없음** (자모 분해만 담당) |
| `pyproject.toml` | G2PK 패키지 의존성 추가 (패키지명 미결정) |

#### G2PK 패키지 후보 (미결정)

| 패키지 | 특징 |
|--------|------|
| `g2pk` | 원본 (Kyubyong Park), v0.9.4 |
| `g2pk2` | 업데이트 포크, 일부 규칙 개선 |
| `KoG2P` | KAIST 기반, 규칙셋 상이 |

> 사용자가 패키지를 검토 후 결정. 결정되면 `pyproject.toml` 의존성 추가 및 `preprocess_text_ko.py` import 확정.

#### 제한사항

- 순수 규칙 기반 — 문맥 의존 발음 변이 처리 불완전
- 특수어(외래어, 신조어) 처리 약함
- Phase 3 BERT 도입 시 음운 오류가 운율에 미치는 영향 감소 기대

---

### Phase 3 — BERT 도입 (예정)

Phase 1/2 학습 결과 비교 후 더 나은 G2P 방식을 채택하고 BERT 임베딩을 추가합니다.

#### 아키텍처 요구사항

JP-Extra 모델은 `ku-nlp/deberta-v2-large-japanese-char-wwm`(은닉층 1024차원)을 사용.
`bert_proj = nn.Conv1d(1024, hidden_channels, 1)` — 입력 차원이 1024로 고정.

한국어 BERT도 **1024차원 출력**이어야 모델 수정 없이 바로 교체 가능.

#### 한국어 BERT 모델 후보

| 모델 | 은닉 차원 | 비고 |
|------|----------|------|
| `klue/roberta-large` | **1024** | RoBERTa-large 아키텍처, KLUE 벤치마크 기준 최상위 |
| `snunlp/KR-ELECTRA-discriminator` | 768 | 34GB 한국어 코퍼스, 형태소 토크나이저 |
| `monologg/koelectra-base-v3-discriminator` | 768 | KoELECTRA 최신판 |
| `kakaobank/kf-deberta-base` | 768 | DeBERTa 기반, 금융 특화 |

> **결론**: `klue/roberta-large`가 유일한 1024차원 후보. `bert_proj` 입력 차원 수정 없이 사용 가능.
> 768차원 모델 사용 시 `bert_proj = nn.Conv1d(768, hidden_channels, 1)`로 변경 필요.

#### Phase 3 구현 변경 범위

| 파일 | 변경 내용 |
|------|----------|
| `style_bert_vits2/nlp/korean/bert_ko.py` | 신규: KO BERT 특징 추출 (`klue/roberta-large` 로드, `[1024, T]` 텐서 반환) |
| `style_bert_vits2/nlp/__init__.py` | `extract_bert_feature()` KO 분기에서 zero 텐서 → 실제 BERT 출력으로 교체 |
| `data_utils_ko.py` | `get_text()` — zero 텐서 대신 `.bert.pt` 캐시 파일 로드 |
| `preprocess_text_ko.py` | BERT 특징 사전 계산 및 `.bert.pt` 저장 스텝 추가 |
| `models_ko.py` | `bert_proj` 입력 차원 확인 (1024 모델이면 무수정) |

#### 데이터 흐름 변화

```
Phase 1/2: wav → .spec.pt  +  zero bert [1024, T]
Phase 3:   wav → .spec.pt  +  .bert.pt [1024, T]  (klue/roberta-large 사전 계산)
```

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

### 학습 최적화: Bucket Sampler (미적용, 권장)

현재 `train_ms_ko.py`는 `DataLoader(shuffle=True, ...)`만 사용하며 **버킷 샘플러를 적용하지 않음**.

#### 문제

KSS는 문장 길이 편차가 크다 (짧은 단문 ~ 긴 복문). 무작위 배치 구성 시:

- 한 배치에 짧은 문장 + 긴 문장이 섞임
- 짧은 문장이 긴 문장 길이에 맞춰 zero-패딩
- 배치 내 실제 연산 대비 패딩 비율 증가 → GPU 효율 20~30% 손실

#### 해결: DistributedBucketSampler

`data_utils.py`에 이미 `DistributedBucketSampler`가 구현되어 있고, `data_utils_ko.py`에도 import 되어 있음.

```python
# train_ms_ko.py 변경 예시 (현재 미적용)
from data_utils_ko import DistributedBucketSampler, ...

train_sampler = DistributedBucketSampler(
    train_dataset,
    batch_size=hps.train.batch_size,
    boundaries=[32, 64, 128, 256, 512, 1024],  # spec 길이 기준 버킷
    num_replicas=1,
    rank=0,
    shuffle=True,
)
train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,   # shuffle=True 대신
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    collate_fn=collate_fn,
)
```

`batch_sampler` 사용 시 `batch_size`와 `shuffle` 파라미터는 `DataLoader`에서 제거해야 함 (`batch_sampler`가 우선).

#### OOM 방지 효과

버킷 내 시퀀스 길이 분포가 고름 → 긴 시퀀스 집중 배치 방지 → OOM 리스크 감소.

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

## Epoch 2 속도 저하 현상 분석

### 현상

```
Epoch 1 (step 1000): ~1.25s/it
Epoch 2 (step 1123): ~2.72s/it  (약 2x 느림)
```

`--no-spec-cache` 옵션(spec.pt 캐시 완전 비활성화)으로 실험한 결과 동일한 패턴이 재현됨 → DataLoader의 spec.pt I/O는 원인이 아님.

### 원인 추정

`train_and_evaluate()` 마지막 두 줄([train_ms_ko.py:453-454](../train_ms_ko.py)):

```python
gc.collect()
torch.cuda.empty_cache()
```

`torch.cuda.empty_cache()`는 PyTorch가 예약해둔 CUDA 메모리 블록을 전부 OS에 반납한다. epoch 2 첫 스텝부터 모든 텐서 할당을 새로 시작해야 하는데, VRAM이 거의 꽉 찬 상태라면 allocator가 연속 블록을 찾는 데 비용이 든다.

**batch_size=12 → 8 실험으로 검증**: batch_size를 줄였더니 epoch 2 속도 저하가 사라짐.

| batch_size | VRAM 점유 | epoch 2 저하 |
|------------|----------|--------------|
| 12 | ~95% (15,517/16,303 MiB) | 약 2x 느림 |
| 8 | 여유 있음 | 저하 없음 |

VRAM 여유가 충분하면 `empty_cache()` 후 재할당 비용이 작아 문제가 없다. 반대로 VRAM이 한계치에 가까울수록 재할당 단편화 비용이 커진다.

### 부작용 없는 완화 방법

`gc.collect() + torch.cuda.empty_cache()`를 주석 처리하면 PyTorch가 메모리 풀을 epoch 간에 유지하므로 재할당 비용이 사라진다. 단일 GPU·단일 프로세스 학습에서는 다른 프로세스가 VRAM을 필요로 하지 않으므로 부작용 없음.

---

## 일본어 학습 코드와의 차이점 (`train_ms_jp_extra.py`)

그라디언트 누적(gradient accumulation) 같은 특수 기법은 양쪽 모두 사용하지 않는다.

---

### 1. CUDA 백엔드 최적화 (JP Extra에만 존재)

```python
# train_ms_jp_extra.py
torch.set_num_threads(1)
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.sdp_kernel("flash")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
```

KO에는 `set_float32_matmul_precision("medium")`과 TF32만 있고 Flash Attention 설정이 없다.

- **`torch.set_num_threads(1)`**: 메인 프로세스의 CPU 스레드를 1개로 제한. DataLoader worker들이 CPU(STFT 계산)를 써야 하는데, 메인 스레드도 기본적으로 여러 CPU 코어를 사용하면 경쟁이 발생한다. 1로 제한하면 worker들이 CPU를 더 많이 가져갈 수 있다.

- **Flash Attention**: Attention 연산에서 메모리와 속도를 동시에 절약. TextEncoder의 multi-head attention에 적용됨.

**KO에 추가 가능**: `set_num_threads(1)`과 Flash Attention 설정은 단일 GPU에서도 효과적이다.

---

### 2. DataLoader 설정

```python
# JP Extra
num_workers=1,          # 의도적으로 1 (메모리 절약, 주석에 명시)
# prefetch_factor=6,    # 주석 처리됨 (메모리 절약)
persistent_workers=True,

# KO (현재)
num_workers=16,
prefetch_factor=2,
persistent_workers=True,
```

JP Extra는 원래 `num_workers=config.train_ms_config.num_workers`와 `prefetch_factor=6`을 쓰다가 메모리 문제로 줄였다. WavLM Discriminator로 VRAM이 더 빡빡하기 때문이다.

`prefetch_factor=2`는 worker 1개당 2배치를 미리 준비한다는 의미이므로, `num_workers=16`이면 최대 32배치가 pinned memory에 상주할 수 있다. 배치 1개의 크기가 수백 MB 수준이면 이 자체가 VRAM/RAM 압박이 된다.

---

### 3. Batch Sampler

```python
# JP Extra
train_sampler = DistributedBucketSampler(
    train_dataset,
    hps.train.batch_size,
    [32, 300, 400, 500, 600, 700, 800, 900, 1000],  # spec 길이 기준 버킷
    num_replicas=n_gpus,
    rank=rank,
    shuffle=True,
)
# KO (현재)
DataLoader(train_dataset, shuffle=True, ...)
```

JP Extra의 버킷 경계는 spec 길이(프레임 수) 기준이다. KSS 적용 시 적절한 버킷 경계를 새로 추정해야 한다 (문장 길이 분포에 따라 다름). Bucket Sampler 적용 시 배치 내 패딩 낭비가 줄어 GPU 효율이 올라가고 OOM 리스크도 감소한다 (상단 참조).

`--not_use_custom_batch_sampler` 플래그로 대안인 `DistributedLengthGroupedSampler`도 선택 가능하다. 이는 HuggingFace Transformers의 sampler로 버킷 개수 없이 단순히 길이 기준으로 그룹화한다.

---

### 4. 체크포인트 저장 구조

```python
# JP Extra: eval과 save가 항상 같은 스텝에 발생
if global_step % hps.train.eval_interval == 0:
    evaluate(...)
    save_checkpoint(net_g, ...)
    save_checkpoint(net_d, ...)

# KO: eval_interval과 save_interval 분리
if global_step % hps.train.eval_interval == 0:
    evaluate(...)
if global_step % save_interval == 0:
    save_checkpoint(net_g, ...)
    save_checkpoint(net_d, ...)
```

KO는 eval과 save 주기를 독립적으로 설정할 수 있다. JP Extra는 save_interval 개념이 없고 eval마다 항상 저장한다.

---

### 5. global_step 재계산 방식

```python
# JP Extra: 체크포인트 파일명에서 직접 추출
global_step = int(utils.get_steps(
    utils.checkpoints.get_latest_checkpoint_path(model_dir, "G_*.pth")
))

# KO: epoch_str 기반 추정
global_step = (epoch_str - 1) * len(train_loader)
```

JP Extra 방식이 더 정확하다. KO 방식은 epoch당 step 수(`len(train_loader)`)를 곱하므로, 도중에 batch_size나 데이터셋이 바뀌면 틀릴 수 있다. 재개 시 `global_step`이 잘못 계산되면 MAS noise scale, eval/save 타이밍이 어긋난다.

---

### 6. grad_norm_g TensorBoard 기록 누락

```python
# JP Extra
scalar_dict = {
    "grad_norm_d": grad_norm_d,
    "grad_norm_g": grad_norm_g,   # ← Generator gradient norm도 기록
    ...
}

# KO (현재)
scalar_dict = {
    "grad_norm_d": grad_norm_d,   # grad_norm_g 없음
    ...
}
```

Generator의 gradient norm은 학습 안정성 모니터링에 중요하다. `clip_grad_norm_`으로 max_norm=500을 적용하고 있지만 실제 norm 값이 TensorBoard에 찍히지 않아서 발산 여부를 사후에만 확인할 수 있다.

---

### 7. Discriminator gradient clipping 조건

```python
# JP Extra
optim_d.zero_grad()
scaler.unscale_(optim_d)
if getattr(hps.train, "bf16_run", False):
    torch.nn.utils.clip_grad_norm_(net_d.parameters(), max_norm=200)  # bf16일 때만
commons.clip_grad_value_(net_d.parameters(), None)
scaler.step(optim_d)

# KO
optim_d.zero_grad()
scaler.unscale_(optim_d)
commons.clip_grad_value_(net_d.parameters(), None)   # clip_grad_norm_ 없음
scaler.step(optim_d)
```

현재 `bf16_run=False`이므로 결과는 동일하지만, 나중에 bf16으로 전환할 경우 KO 코드에는 Discriminator용 `clip_grad_norm_`이 없다.

---

### 개선 우선순위 정리

| 항목 | 난이도 | 기대 효과 |
|------|--------|----------|
| `torch.set_num_threads(1)` 추가 | 1줄 | DataLoader worker CPU 경쟁 완화 |
| Flash Attention 활성화 | 3줄 | Attention 속도/메모리 개선 |
| `grad_norm_g` TensorBoard 기록 | 2줄 | 학습 모니터링 개선 |
| `global_step` 재계산 방식 교체 | 5줄 | 재개 정확도 향상 |
| BucketSampler 적용 | 별도 섹션 참조 | 패딩 낭비 제거, OOM 리스크 감소 |

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
