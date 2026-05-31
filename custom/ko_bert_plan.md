# 한국어 BERT 통합 계획서

## 현황

| 항목 | 상태 |
|------|------|
| Phase 1 학습 (kss_exp_jamo, kss_exp_g2pk2, zero BERT) | 🔄 진행 중 |
| 정렬 알고리즘 설계 (`KO_BERT_SPEC.md`) | ✅ 완료 |
| 정렬 알고리즘 검증 (29/29 PASS) | ✅ 완료 |
| 독립 실험 저장소 (`ko_bert_alignment_experiment/`) | ✅ 완료 |
| Style-Bert-VITS2 통합 구현 (L2 PASS) | ✅ 완료 |
| BERT 연결 학습 (Phase 3) | ⬜ 미시작 |

---

## 알고리즘 요약 (`KO_BERT_SPEC.md` 기반)

klue/roberta-large는 형태소 기반 WordPiece이므로 토큰 수 N ≠ 음절 수 M+2.
3단계 연결로 BERT 토큰 인덱스 기준 `word2ph_tok`을 계산한다.

```
1단계  offset_mapping   →  토큰 ↔ 원문 음절 위치 매핑
2단계  g2pk2            →  원문 음절 ↔ 발음 음절 (1:1 위치 보존)
3단계  자모 수 합산     →  word2ph_tok[j] = 토큰 j가 커버하는 음절들의 발음 자모 수 합계
```

불변식:  `sum(word2ph_tok) == sum(jamo_counts) + 2`

---

## 코드 구현 계획

### 구현 대상 파일

```
style_bert_vits2/nlp/korean/
├── decompose.py           ← 신규. 자모 분해 + g2pk2 연동
├── alignment.py           ← 신규. word2ph_tok 계산 핵심 로직
└── bert_feature.py        ← 신규. JP/ZH bert_feature.py 상당

style_bert_vits2/
├── constants.py           ← DEFAULT_BERT_MODEL_PATHS에 KO 추가
├── nlp/__init__.py        ← KO 분기: zero tensor → extract_bert_feature 호출
└── models/infer.py        ← KO BERT 분기 추가
```

---

### Step 1. `decompose.py`

**역할**: 자모 분해 + g2pk2 발음 변환 유틸리티.
`ko_bert_alignment_experiment/decompose.py`를 이 경로로 이식.

```python
# style_bert_vits2/nlp/korean/decompose.py

def decompose_syllable(char: str) -> list[str]:
    """한국어 음절 1자를 KO_* phone 리스트로 분해. ㅇ 초성 생략."""

def text_to_jamo_counts(text: str, use_g2pk2: bool = True) -> list[int]:
    """
    원문 음절 위치 기준 자모 수 리스트 (len = M).
    use_g2pk2=True: g2pk2 발음 기준으로 자모 수 계산.
    음절 수 불일치 시 표기 기준 fallback.
    비한국어·공백은 skip.
    """
```

**의존성**: g2pk2, eunjeon (이미 pyproject.toml에 포함됨).

---

### Step 2. `alignment.py`

**역할**: offset_mapping으로 토큰별 word2ph 계산.
`ko_bert_alignment_experiment/alignment.py`를 이 경로로 이식.

```python
# style_bert_vits2/nlp/korean/alignment.py

def compute_word2ph_tok(
    text: str,
    tokenizer,
    jamo_counts: list[int],
) -> list[int]:
    """
    Returns [1(CLS), tok_1_jamo, ..., tok_k_jamo, 1(SEP)]
    len = N (BERT 토큰 수, CLS/SEP 포함)
    sum = sum(jamo_counts) + 2  (순수 한국어 기준)
    """

def apply_add_blank(word2ph: list[int]) -> list[int]:
    """word2ph[i] *= 2, word2ph[0] += 1"""
```

---

### Step 3. `bert_feature.py` (핵심)

JP `bert_feature.py`를 참조하되 word2ph 계산 방식을 교체한다.

```python
# style_bert_vits2/nlp/korean/bert_feature.py

def extract_bert_feature(
    text: str,
    word2ph: list[int],   # g2p() 출력 (음절 인덱스, add_blank 적용 후)
    device: str,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
) -> torch.Tensor:
    """
    Returns: Tensor shape (1024, L)
    L = sum(word2ph)  ← add_blank 적용 후 phoneme 수

    내부 처리:
      1. text_to_jamo_counts(text)         →  jamo_counts (len=M)
      2. compute_word2ph_tok(text, tok, jamo_counts)  →  word2ph_tok_pre (len=N)
      3. apply_add_blank(word2ph_tok_pre)  →  word2ph_tok  (sum=L)
      4. assert sum(word2ph_tok) == sum(word2ph) == L
      5. BERT forward  →  hidden_states[-3]  (N, 1024)
      6. res[j].repeat(word2ph_tok[j], 1)   →  (L, 1024) → .T → (1024, L)
    """
```

**JP와의 차이점**:

| 항목 | JP | KO |
|------|----|----|
| word2ph 계산 | g2p()가 반환한 것 그대로 | `compute_word2ph_tok`으로 내부 재계산 |
| assert | `len(word2ph) == len(text)+2` | `sum(word2ph_tok) == sum(word2ph)` |
| word2ph 인자 역할 | expand 인덱스로 직접 사용 | L 검증 목적 + add_blank 기준값 |

---

### Step 4. `constants.py` 업데이트

```python
DEFAULT_BERT_MODEL_PATHS = {
    Languages.JP: BASE_DIR / "bert" / "deberta-v2-large-japanese-char-wwm",
    Languages.EN: BASE_DIR / "bert" / "deberta-v3-large",
    Languages.ZH: BASE_DIR / "bert" / "chinese-roberta-wwm-ext-large",
    Languages.KO: BASE_DIR / "bert" / "klue-roberta-large",   # ← 추가
}
```

모델 다운로드:
```bash
uv run huggingface-cli download klue/roberta-large --local-dir bert/klue-roberta-large
```

---

### Step 5. `nlp/__init__.py` 교체

```python
# 기존 (Phase 1)
elif language == Languages.KO:
    import torch
    total_phones = sum(word2ph)
    return torch.zeros(1024, total_phones)

# 변경 후
elif language == Languages.KO:
    from style_bert_vits2.nlp.korean.bert_feature import extract_bert_feature
    return extract_bert_feature(text, word2ph, device, assist_text, assist_text_weight)
```

---

### Step 6. `infer.py` KO 분기 추가

JP Extra 모델처럼 KO도 ja_bert / en_bert를 zero로 채우는 분기 필요.

```python
elif language_str == Languages.KO:
    bert     = bert_ori
    ja_bert  = torch.zeros(1024, len(phone))
    en_bert  = torch.zeros(1024, len(phone))
    zh_bert  = torch.zeros(1024, len(phone))
```

---

## 검증 계획

### 레벨 1: 정렬 단위 테스트 (독립 저장소)

`ko_bert_alignment_experiment/`에서 수행. 이미 25/25 PASS.

| 테스트 | 기준 |
|--------|------|
| sum 불변식 | `sum(word2ph_tok) == sum(jamo_counts) + 2` |
| add_blank 후 sum | `sum == 2*(sum(jamo_counts)+2)+1` |
| 최솟값 보장 | `word2ph_tok[i] >= 1` for all i |
| 비한국어 fallback | 예외 없이 처리됨 |
| g2pk2 음절 수 보존 | `len(orig_syls) == len(pron_syls)` |

---

### 레벨 2: 파이프라인 단위 테스트 (Style-Bert-VITS2)

```python
# custom/test_ko_bert_feature.py

def test_bert_feature_shape():
    """extract_bert_feature 출력 shape 검증"""
    text = "안녕하세요"
    phones, tones, word2ph = g2p(normalize_text(text))
    word2ph_blank = apply_add_blank(word2ph)
    feat = extract_bert_feature(text, word2ph_blank, "cpu")
    assert feat.shape == (1024, sum(word2ph_blank))

def test_no_nan():
    feat = extract_bert_feature("저는 학생입니다.", word2ph_blank, "cpu")
    assert not feat.isnan().any()

def test_not_zero():
    feat = extract_bert_feature("저는 학생입니다.", word2ph_blank, "cpu")
    assert not (feat == 0).all()

def test_infer_pipeline():
    """infer.py 전체 파이프라인 — 합성 결과가 wav로 생성되는지"""
    # models/infer.py의 infer() 호출, 예외 없이 완료되면 통과
```

---

### 레벨 3: 형상 불변식 확인

infer.py에서 최종적으로 실행되는 assert 통과 여부:

```python
assert bert_ori.shape[-1] == len(phone)  # infer.py:140
```

KO BERT 통합 후 이 assert가 통과하면 전체 shape 정렬이 올바름.

---

### 레벨 4: 출력 음성 청취 테스트

레벨 1~3 통과 후 동일 텍스트·동일 스타일로:
- Phase 1 체크포인트 (zero BERT)
- Phase 3 체크포인트 (klue/roberta-large BERT)

를 합성하여 prosody·자연스러움 청취 비교.

---

## 실험 계획

### 실험 구성

| 조건 | G2P | BERT | 전략 | 실험명 |
|------|-----|------|------|--------|
| A | Jamo | ❌ | — | `kss_exp_jamo` (진행 중) |
| B | G2PK2 | ❌ | — | `kss_exp_g2pk2` (진행 중) |
| C | G2PK2 | ✅ | Phase 1 체크포인트 → BERT freeze fine-tune | `kss_exp_g2pk2_bert_frozen` |
| D | G2PK2 | ✅ | C 완료 후 BERT unfreeze | `kss_exp_g2pk2_bert_unfrozen` |

비교 축:
- A vs B: G2PK2 음소 변환의 효과
- B vs C: BERT 피처 추가 효과 (freeze 기준)
- C vs D: BERT fine-tuning 추가 효과

---

### 학습 설정

| 항목 | 값 |
|------|-----|
| 학습 데이터 | 12,749 문장 (알파벳 필터링 후) |
| Validation | 20문장 |
| Batch size | 8 |
| 목표 steps (A/B) | 65,000 (~41 epoch) |
| 추가 steps (C) | 20,000 (Phase 1 체크포인트에서 fine-tune) |
| 추가 steps (D) | 20,000 (C 체크포인트에서 BERT unfreeze) |
| BERT LR (D) | TTS LR × 0.1 (catastrophic forgetting 방지) |

---

### 체크포인트 재활용 근거

Phase 1의 `bert_proj` 가중치는 zero 입력 학습이라 무의미하나,
나머지 (phoneme embedding, flow, decoder)는 수렴됨.
→ BERT만 연결한 fine-tune이 from-scratch보다 빠른 수렴 기대.

Phase 1 체크포인트 로드 후 `bert_proj` 재초기화:
```python
# bert_proj는 zero 입력에 편향된 가중치이므로 재초기화
nn.init.kaiming_uniform_(model.enc_p.bert_proj.weight)
```

---

### 평가 지표

| 지표 | 방법 |
|------|------|
| 자연스러움 (MOS) | 10k / 30k / 65k step 체크포인트 청취 비교 |
| 운율 일관성 | 동일 문장 5회 합성 후 prosody 변동 청취 |
| 학습 수렴 속도 | TensorBoard total_loss 곡선 비교 |
| BERT 효과 | B vs C 동일 스텝 대비 loss 비교 |

---

## 진행 순서

```
[지금] Phase 1 학습 완료 대기
              ↓
[다음] bert/klue-roberta-large 모델 다운로드
              ↓
[다음] decompose.py + alignment.py 이식
       → bert_feature.py 구현
       → 레벨 1,2 검증 통과 확인
              ↓
[다음] constants.py / __init__.py / infer.py 업데이트
       → 레벨 3 (assert 통과) + 레벨 4 (합성 가능) 확인
              ↓
[다음] 조건 C: BERT freeze fine-tune 시작
              ↓
[다음] 조건 C vs B 청취 비교 → 개선 확인 시 조건 D로 진행
```

---

## 참고 파일

| 파일 | 내용 |
|------|------|
| `custom/KO_BERT_SPEC.md` | 정렬 알고리즘 명세 + 검증 결과 |
| `custom/notes_textencoder_pipeline.md` | JP/ZH/KO 파이프라인 전체 비교표 |
| `custom/notes_bert_alignment.md` | JP/ZH assert 분석 및 KO 해결책 |
| `custom/ko_bert_alignment_experiment/` | 독립 정렬 검증 저장소 |
| `docs/experiment_korean_bert.md` | 실험 전체 계획 (BERT 모델 선택 등) |
