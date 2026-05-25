# 실험: 한국어 BERT 통합 (Phase 3)

## 개요

Style-Bert-VITS2의 일본어/중국어 모델은 사전학습 BERT 모델에서 추출한 문맥 임베딩을 TextEncoder 입력에 더한다. 한국어 모델(`models_ko.py`)은 이미 `bert_proj` 레이어가 예약되어 있으나 현재 zero tensor를 입력받는다(Phase 1). 이 실험은 실제 한국어 BERT 피처를 연결하여 TTS 품질 향상을 검증한다.

---

## 현재 구조 이해

### BERT 피처가 모델에 주입되는 방식

```
텍스트 입력
    ↓ normalize_text()
    ↓ g2p()  →  phones, tones, word2ph
    ↓ extract_bert_feature(text, word2ph, ...)
        - BERT tokenize
        - forward (hidden_states[-3] 추출, shape: [seq_len, 1024])
        - word2ph 기준으로 phoneme 단위로 expand
        - 출력: [1024, total_phone_count]
    ↓
TextEncoder.forward(bert=[1024, L], ...)
    bert_emb = bert_proj(bert).transpose(1,2)   # Conv1d(1024→hidden_channels)
    x = emb(phones) + tone_emb + lang_emb + bert_emb + style_emb
```

### 관련 파일

| 파일 | 역할 | 한국어 현황 |
|------|------|-------------|
| `style_bert_vits2/nlp/bert_models.py` | 모델/토크나이저 로드 | KO 경로 미등록 |
| `style_bert_vits2/nlp/japanese/bert_feature.py` | JP BERT 추출 | 참조 구현 |
| `style_bert_vits2/nlp/__init__.py:54-59` | 언어별 디스패처 | KO → zero tensor 반환 |
| `style_bert_vits2/constants.py` | DEFAULT_BERT_MODEL_PATHS | KO 미포함 |
| `style_bert_vits2/models/models_ko.py` | 학습 모델 | `bert_proj` 예약됨, 0 입력 중 |
| `style_bert_vits2/models/infer.py` | 추론 파이프라인 | KO BERT 분기 없음 |

---

## BERT 모델 스펙 비교

### 현재 사용 중인 JP / ZH 모델

| 항목 | JP (`ku-nlp/deberta-v2-large-japanese-char-wwm`) | ZH (`hfl/chinese-roberta-wwm-ext-large`) |
|------|--------------------------------------------------|------------------------------------------|
| 아키텍처 | DeBERTa-v2 | RoBERTa (BERT 기반) |
| 파라미터 | ~300M | ~330M |
| hidden_size | 1024 | 1024 |
| 레이어 / 헤드 | 24L × 16h | 24L × 16h |
| max_position_embeddings | 512 | 512 |
| **토크나이저** | **SentencePiece, 가타카나 모라 단위** | **BertTokenizer (WordPiece), 한자 단위** |
| **토크나이저 세부** | BERT 입력 전 `text_to_sep_kata()`로 한자→가타카나 변환 (`bert_feature.py:45`). "私"→"ワタシ". 가타카나 1문자 = 1토큰 | 변환 없이 한자 원문 입력. `tokenize_chinese_chars=true`. 한자 1자 = 1토큰. 중국어 NLP 구조적 특성 |
| 어휘 크기 | 22,012 | 21,128 |
| 학습 코퍼스 | 171GB (Wikipedia + CC-100 + OSCAR) | 5.4B 단어 (Wikipedia + 백과·뉴스·Q&A) |
| word2ph 정렬 | **완벽 (1:1)** | **완벽 (1:1)** |
| HuggingFace | [ku-nlp/deberta-v2-large-japanese-char-wwm](https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm) | [hfl/chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) |
| GitHub | — | [ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) |

---

### 한국어 후보 모델

| 항목 | `snunlp/KR-BERT-char16424` | `klue/roberta-large` | `beomi/kcbert-large` |
|------|---------------------------|----------------------|----------------------|
| 아키텍처 | BERT (base) | RoBERTa (large) | BERT (large) |
| 파라미터 | ~99M | ~337M | ~337M |
| hidden_size | **768** | **1024** | **1024** |
| 레이어 / 헤드 | 12L × 12h | 24L × 16h | 24L × 16h |
| max_position_embeddings | 512 | 512 | **300** ⚠️ |
| **토크나이저** | **BidirectionalWordPiece, 음절(char) 단위** | **BertTokenizer (WordPiece), 형태소 기반** | BertTokenizer (WordPiece) |
| **토크나이저 세부** | 양방향 BPE 중 고빈도 선택. 어휘가 한글 음절로 구성 → 1음절 = 1토큰 성립 | vocab 구성 시 Mecab-ko 형태소 단위 BPE 적용. 추론 시 BertTokenizer. 형태소 경계 ≠ 음절 경계 | 뉴스 댓글 코퍼스 기반 WordPiece. 구어체 특화. 음절↔토큰 불일치 |
| 어휘 크기 | 16,424 | 32,000 | 30,000 |
| 학습 코퍼스 | 2.47GB (20M 문장, 범용) | 62GB (뉴스·위키·MODU·CC-100·구어체) | 12.5GB (**뉴스 댓글 특화**) |
| word2ph 정렬 | **완벽 (1:1) — JP/ZH 동일** | offset_mapping 구현 필요 | offset_mapping 필요 |
| bert_proj 호환 | Conv1d(**768**→hidden) — 수정 필요 | Conv1d(**1024**→hidden) — 무수정 | Conv1d(**1024**→hidden) — 무수정 |
| 탈락 사유 | — | — | max_position_embeddings=300 ❌ |
| HuggingFace | [snunlp/KR-BERT-char16424](https://huggingface.co/snunlp/KR-BERT-char16424) | [klue/roberta-large](https://huggingface.co/klue/roberta-large) | [beomi/kcbert-large](https://huggingface.co/beomi/kcbert-large) |
| GitHub | [snunlp/KR-BERT](https://github.com/snunlp/KR-BERT) | [KLUE-benchmark/KLUE](https://github.com/KLUE-benchmark/KLUE) | [Beomi/KcBERT](https://github.com/Beomi/KcBERT) |

---

### 조사 결과: 한국어 large + char-level 모델은 존재하지 않음

한국어 공개 모델 전수 조사 결과, **1024 hidden + 음절 단위 tokenizer**를 동시에 만족하는 모델은 없다. 음절 단위 모델은 전부 base 규격(768), large 규격(1024) 모델은 형태소·WordPiece 기반이다. 선택은 두 축 중 하나를 포기하는 구조로 귀결된다.

---

## 한국어 BERT 모델 선택

### 두 선택지의 트레이드오프

#### 선택지 A: `klue/roberta-large` — 품질 우선

- `bert_proj` 차원(1024) 유지, Phase 1 체크포인트 포맷과 호환
- KLUE 벤치마크 SOTA, 62GB 말뭉치(뉴스·위키·구어체 포함)
- **비용**: offset_mapping 기반 음절↔토큰 매핑 구현 필요 (`custom/notes_bert_alignment.md` 참고)
- JP/ZH의 단순 어서션 `len(word2ph) == len(text) + 2` 사용 불가, 별도 런타임 검증 로직 필요

#### 선택지 B: `snunlp/KR-BERT-char16424` — 구조 일관성 우선

- JP/ZH와 완전히 동일한 설계: 1음절 = 1토큰, 기존 어서션 그대로 재사용
- **비용**: `bert_proj`를 `Conv1d(768, hidden_channels, 1)`로 변경 → `models_ko.py` 한 줄 수정
- Phase 1 체크포인트의 `bert_proj` 가중치는 zero 입력으로 학습되어 어차피 무의미. 포맷 변경의 실질적 손실 없음
- 학습 데이터 2.47GB (klue 62GB 대비 소규모), base 규모. 표현 품질 상한이 낮을 수 있음
- TTS에서 BERT는 fine-grained NLU가 아닌 **운율·문맥 힌트** 역할. 12k 학습 문장 규모에서 base와 large의 차이가 결정적일지는 실험으로 확인 필요

### 결론 및 권장 전략

| | 선택지 A (`klue/roberta-large`) | 선택지 B (`KR-BERT-char16424`) |
|--|--|--|
| 구현 난이도 | 높음 (offset_mapping 매핑 로직) | 낮음 (JP/ZH 코드 거의 그대로) |
| 모델 품질 | large, 62GB | base, 2.47GB |
| bert_proj 수정 | 불필요 | `Conv1d(1024→768)` |
| 런타임 검증 | 새로 설계 필요 | 기존 어서션 재사용 |
| JP/ZH 구조 일관성 | 낮음 | 높음 |

> **권장**: 실험 초기에는 **선택지 B** (`snunlp/KR-BERT-char16424`)로 BERT 통합 파이프라인을 먼저 완성한다. 구현이 검증된 후 선택지 A로 교체해 품질 차이를 측정한다.
>
> `bert_proj` 차원 변경은 코드 한 줄이지만 offset_mapping 매핑 로직은 엣지 케이스가 많아 디버깅 비용이 크다. 파이프라인 검증과 품질 측정을 분리하는 것이 합리적이다.

---

## 구현 단계

### Step 1. 한국어 BERT 모델 다운로드

**Phase 3-A (선택지 B, 권장 시작점)**:
```bash
huggingface-cli download snunlp/KR-BERT-char16424 --local-dir bert/KR-BERT-char16424
```

**Phase 3-B (선택지 A, 추후 품질 비교용)**:
```bash
huggingface-cli download klue/roberta-large --local-dir bert/klue-roberta-large
```

`bert/` 폴더에 저장 (`constants.py`의 `BASE_DIR / "bert"` 규칙 준수).

### Step 2. `constants.py` 업데이트

`DEFAULT_BERT_MODEL_PATHS`에 KO 추가:
```python
# Phase 3-A
Languages.KO: BASE_DIR / "bert" / "KR-BERT-char16424",
# Phase 3-B (교체 시)
# Languages.KO: BASE_DIR / "bert" / "klue-roberta-large",
```

### Step 3. `bert_feature.py` 작성 (신규)

파일: `style_bert_vits2/nlp/korean/bert_feature.py`

참조: `style_bert_vits2/nlp/japanese/bert_feature.py`

핵심 로직:
```python
def extract_bert_feature(
    text: str,
    word2ph: list[int],
    device: str,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
) -> torch.Tensor:
    tokenizer = load_tokenizer(Languages.KO)
    model = load_model(Languages.KO)
    model = model.to(device)

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        res = model(**inputs, output_hidden_states=True)

    # hidden_states[-3]: 끝에서 3번째 hidden layer (shape: [1, seq_len, 1024])
    hidden = torch.cat(res["hidden_states"][-3:-2], dim=-1)[0]  # [seq_len, 1024]

    # word2ph 기준으로 phoneme 단위 expand
    # (tokenizer 토큰 수와 word2ph 길이 정렬 필요 — 검증 단계에서 확인)
    ...
    return bert_feature.T  # [1024, total_phone_count]
```

> **Phase 3-A (`KR-BERT-char16424`)**: 음절 단위 tokenizer이므로 JP/ZH와 동일하게 `res[i]` 직접 접근. `assert len(word2ph) == len(text) + 2` 그대로 사용 가능.
>
> **Phase 3-B (`klue/roberta-large`)로 교체 시**: Mecab 형태소 기반으로 직접 접근 불가. offset_mapping 기반 음절별 평균으로 교체 필요. 상세 내용은 아래 "토크나이저-음소 정렬 문제" 섹션 및 `custom/notes_bert_alignment.md` 참고.

### Step 4. `nlp/__init__.py` 디스패처 업데이트

```python
# 기존 (Phase 1)
elif language == Languages.KO:
    return torch.zeros(1024, sum(word2ph))

# 변경 후
elif language == Languages.KO:
    from style_bert_vits2.nlp.korean.bert_feature import extract_bert_feature
    return extract_bert_feature(text, word2ph, device, assist_text, assist_text_weight)
```

### Step 5. `infer.py` KO 분기 추가

```python
elif language_str == Languages.KO:
    bert = bert_ori
    ja_bert = torch.zeros(1024, len(phone))
    en_bert = torch.zeros(1024, len(phone))
```

### Step 6. 단위 테스트

```python
# custom/test_bert_feature.py
text = "저는 아무것도 하지 않고 앉아 있는 것을 싫어해요."
phones, tones, word2ph = g2p(normalize_text(text))
feat = extract_bert_feature(text, word2ph, "cpu")
assert feat.shape == (1024, len(phones)), f"Shape mismatch: {feat.shape}"
assert not feat.isnan().any()
assert not (feat == 0).all()
```

### Step 7. from-scratch 재학습

현재 `kss_exp_jamo` / `kss_exp_g2pk2` 실험(Phase 1, zero BERT)이 완료된 후:
- `kss_exp_jamo_bert` / `kss_exp_g2pk2_bert` 실험 디렉토리 생성
- 동일한 train.list 사용, bert 피처만 활성화
- Phase 1 결과와 비교: BERT 추가만으로 인한 품질 향상 측정

---

## 핵심 난관: 토크나이저 정렬 문제

### 배경

`extract_bert_feature`는 각 입력 문자의 BERT 벡터를 `word2ph[i]`번 반복해서 phoneme 시퀀스 길이에 맞춘다. 이 확장이 올바르려면 **BERT 토크나이저의 토큰 수 = `word2ph`의 길이**여야 한다.

### 일본어가 단순한 이유

`deberta-v2-large-japanese-char-wwm`은 **문자(char) 단위** tokenizer. 입력 문자 수 = 토큰 수가 거의 항상 성립.

### 한국어의 문제

`klue/roberta-large`는 **BPE/WordPiece** 기반. 음절 "않" 같은 경우 하나의 토큰이 될 수도, 여러 토큰이 될 수도 있다. `word2ph`는 음절(자모 분해 전) 기준이므로 토큰 수와 다를 수 있음.

### 해결 방법 후보

| 방법 | 장점 | 단점 |
|------|------|------|
| **음절 단위 토크나이저 사용** | word2ph와 1:1 대응 단순 | 한국어 음절 단위 사전학습 모델 희소 |
| **BPE 토큰 → 원래 문자 매핑** (offset_mapping) | 범용 | 복잡한 매핑 로직 필요 |
| **문자 단위 BERT fine-tune** | 정렬 보장 | 별도 학습 필요 |
| **토큰 평균 후 word2ph 확장** | 간단한 근사 | 정보 손실 가능성 |

> **권장 접근**: `tokenizer(text, return_offsets_mapping=True)`로 각 토큰의 원문 문자 범위를 구하고, 음절별로 해당 토큰들의 hidden state를 평균 내어 word2ph로 확장.

---

## 토크나이저-음소 정렬 문제

### 배경

`extract_bert_feature`는 각 음절의 BERT 벡터를 `word2ph[i]`번 반복해서 phoneme 시퀀스 길이에 맞춘다. 이 확장이 올바르려면 **BERT 토크나이저의 (특수 토큰 제외) 토큰 수 = `word2ph`의 길이(음절 수)**여야 한다.

### 일본어가 단순한 이유

`deberta-v2-large-japanese-char-wwm`은 **문자(char) 단위** tokenizer. 입력 문자 수 = 토큰 수가 거의 항상 성립.

### 한국어의 상황

`klue/roberta-large`는 BPE 기반이지만 **한국어 음절을 기본 단위**로 vocabulary를 구성하는 경향이 있다. "않아" → `[않, 아]` 처럼 음절 단위 분절이 대부분이지만 미등록 조합에서 서브워드가 발생할 수 있다.

실제 비율은 구현 전에 `custom/check_bert_alignment.py`로 측정:

```python
# train.list 전체에 대해 음절 수 vs BERT 토큰 수 비교
for line in train_lines:
    text, word2ph = parse(line)
    tokens = tokenizer(text, add_special_tokens=False)
    n_tokens  = len(tokens["input_ids"])
    n_syllables = len(word2ph)
    if n_tokens != n_syllables:
        log_mismatch(text, n_tokens, n_syllables)
```

불일치 비율이 낮으면 단순 assert + 로그로 처리하고, 높으면 offset_mapping 기반 매핑으로 전환한다.

### offset_mapping 기반 매핑 (범용 해결책)

```python
inputs = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
offsets = inputs["offset_mapping"]  # [(char_start, char_end), ...]
hidden  = bert_hidden_states        # [n_tokens, 1024]

# 음절별로 해당 토큰들의 hidden state를 평균
result = []
syllable_idx = 0
for syl_start, syl_end in syllable_spans(text):
    tok_indices = [i for i, (s, e) in enumerate(offsets)
                   if s >= syl_start and e <= syl_end]
    if tok_indices:
        result.append(hidden[tok_indices].mean(0))
    else:
        result.append(hidden[syllable_idx])  # fallback
    syllable_idx += 1
```

공백은 `word2ph`에 포함되지 않으므로 offset 계산 시 공백 위치를 건너뛰는 처리 필요.

### 디버깅 체크리스트

1. `n_tokens == len(word2ph)` 어서션 + 불일치 로그
2. `feat.shape == (1024, sum(word2ph))` 출력 shape 검증
3. `feat.isnan().any()` NaN 검사
4. `(feat == 0).all(dim=0)` — 특정 위치 전체 0 여부 (CLS/SEP 누락 시 발생)
5. 샘플 문장 단위로 토큰-음절 시각화 출력 (구현 초기 필수)

---

## BERT 학습 전략 비교

BERT를 도입할 때 어떤 파라미터를 언제 학습시킬지에 따라 학습 비용, 수렴 속도, 최종 품질이 달라진다.

### 전략 비교표

| 전략 | 방법 | 장점 | 단점 |
|------|------|------|------|
| **A. 완전 freeze** | BERT `requires_grad=False`, `bert_proj`만 학습 | GPU 메모리 절약 (~4GB), 학습 안정 | BERT가 TTS 도메인에 미적응, 품질 상한 존재 |
| **B. 처음부터 전체 unfreeze** | BERT 포함 전체 학습 | 도메인 적응 최대화 | +337M 파라미터, 과적합 위험, LR 튜닝 복잡 |
| **C. 점진적 unfreeze** | N step freeze → unfreeze (또는 레이어별 순차) | TTS 구조 먼저 안정화 후 BERT 적응 | unfreeze 시점·LR 추가 튜닝 필요 |
| **D. Phase 1 체크포인트 fine-tune** | 기존 체크포인트 로드 → BERT 연결 후 재학습 | 수렴된 G/D에서 시작, 빠른 수렴 기대 | zero→real BERT로 입력 분포 변화 → 초기 불안정 가능 |

### 권장 실험 순서

```
[완료] Phase 1 (G2PK2, zero BERT)  →  kss_exp_g2pk2
    ↓
[실험 D-1] Phase 1 체크포인트 로드 + BERT freeze fine-tune
    BERT 완전 고정, bert_proj + 나머지 TTS 파라미터만 학습
    ~20,000 step 추가
    → 목적: BERT 피처가 실제로 도움이 되는지 최소 비용으로 확인

    ↓ D-1이 개선을 보이면
[실험 D-2] D-1 체크포인트에서 BERT unfreeze
    BERT LR = TTS LR / 10 (catastrophic forgetting 방지)
    → 목적: BERT 도메인 적응 효과 측정

    ↓
비교: Phase1 vs D-1 vs D-2
```

**Phase 1 체크포인트 재사용이 유리한 이유**: `bert_proj` 가중치는 zero 입력 시절 학습 의미가 없었지만 모델의 나머지 부분(phoneme embedding, flow, decoder)은 이미 수렴. BERT만 연결하는 fine-tune이 from-scratch보다 빠를 가능성이 높다.

**주의**: `bert_proj` 가중치는 Phase 1 체크포인트에 포함되어 있으나 zero 입력에 대한 편향으로 초기화되어 있다. fine-tune 초기에 `bert_proj` LR을 높이거나 별도 초기화를 고려할 수 있다.

---

## 실험 계획

### 실험 구성

| 조건 | G2P | BERT | BERT 전략 | 실험 이름 |
|------|-----|------|-----------|-----------|
| A | Jamo | ❌ | — | `kss_exp_jamo` (진행 중) |
| B | G2PK2 | ❌ | — | `kss_exp_g2pk2` (진행 중) |
| C | G2PK2 | ✅ | freeze fine-tune | `kss_exp_g2pk2_bert_frozen` |
| D | G2PK2 | ✅ | unfreeze fine-tune | `kss_exp_g2pk2_bert_unfrozen` |

비교 축:
- A vs B: G2PK2 G2P의 효과
- B vs C: BERT 피처 추가의 효과 (freeze 기준)
- C vs D: BERT fine-tuning의 추가 효과

### 학습 설정

조건 A/B와 동일:
- 학습 데이터: 12,749 문장 (알파벳 필터링 후, val 20문장 제외)
- Batch size: 8, 목표 65,000 steps (~41 epoch)
- Val: 동일 20문장

조건 C/D:
- Phase 1 체크포인트에서 fine-tune 시작
- C: BERT freeze, 추가 ~20,000 step
- D: C 완료 후 BERT LR=1e-5 (TTS LR의 1/10)로 unfreeze

---

## 진행 상태

| 항목 | 상태 |
|------|------|
| Jamo/G2PK2 Phase 1 학습 | 🔄 진행 중 |
| BERT 모델 선택 (`klue/roberta-large`) | ✅ 결정 |
| `custom/check_bert_alignment.py` 작성 | ⬜ 미시작 |
| `bert_feature.py` 구현 | ⬜ 미시작 |
| 토크나이저 정렬 검증 | ⬜ 미시작 |
| `constants.py` / `__init__.py` 업데이트 | ⬜ 미시작 |
| `infer.py` KO 분기 추가 | ⬜ 미시작 |
| 조건 C: BERT freeze fine-tune | ⬜ 미시작 |
| 조건 D: BERT unfreeze fine-tune | ⬜ 미시작 |
| 조건 B vs C vs D 청취 비교 | ⬜ 미시작 |

---

## 참고

- 일본어 BERT 추출 구현: `style_bert_vits2/nlp/japanese/bert_feature.py`
- 모델 아키텍처에서 BERT 주입 위치: `style_bert_vits2/models/models_ko.py` (TextEncoder, `bert_proj`)
- Phase 1 설계 주석: `models_ko.py` — "bert_proj receives zero tensors, kept so checkpoint format matches Phase 3"
