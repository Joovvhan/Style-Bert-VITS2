# ko-bert-alignment-test

klue/roberta-large 토크나이저와 한국어 자모 G2P 간의 phoneme 길이 정렬 실험.
Style-Bert-VITS2 Korean BERT 통합(Phase 3) 사전 검증용 독립 저장소.

---

## 배경

Style-Bert-VITS2의 TextEncoder는 BERT hidden state를 phoneme 시퀀스 길이 L에 맞춰 expand한다.
일본어/중국어는 BERT 토큰 수 N = 글자 수 M + 2가 성립하므로 단순히 `res[i].repeat(word2ph[i])`로 처리된다.

한국어 `klue/roberta-large`는 형태소 기반 WordPiece tokenizer를 사용해 **N ≠ M + 2**인 경우가 빈번하다.
이 실험은 `offset_mapping`을 이용해 **token → 음절 → 자모 수** 역방향 매핑을 구축하고,
`sum(word2ph_tok) == L` (총 phoneme 수 보존)을 검증한다.

---

## 목표

| # | 목표 | 성공 기준 |
|---|------|-----------|
| 1 | offset_mapping 기반 word2ph_tok 계산 | `sum(word2ph_tok) == total jamo count` for 전체 test 문장 |
| 2 | add_blank 적용 후 정렬 유지 | `sum(word2ph_tok_blank) == 2 * total_jamo + 1` |
| 3 | 불일치 케이스 분류 | 토큰 하나가 여러 음절을 커버하는 패턴 목록화 |
| 4 | fallback 처리 검증 | 공백·구두점·미등록 문자에서 assert 미발생 |

---

## 범위 (Out of scope)

- BERT 모델 forward pass (hidden state 추출) — tokenizer 정렬만 검증
- Style-Bert-VITS2 전체 파이프라인
- 학습 또는 품질 평가

---

## 패키지 요구사항

```
# requirements.txt
transformers>=4.35.0
torch>=2.0.0          # tokenizer 전용이면 CPU only도 충분
```

선택:
```
pytest>=7.0           # 단위 테스트
tabulate              # 정렬 시각화 출력
```

모델 다운로드:
```bash
huggingface-cli download klue/roberta-large --local-dir ./klue-roberta-large
```

---

## 디렉토리 구조

```
ko-bert-alignment-test/
├── README.md
├── requirements.txt
├── decompose.py          # 한국어 자모 분해 (Style-Bert-VITS2 g2p.py에서 발췌)
├── alignment.py          # word2ph_tok 계산 핵심 로직
├── test_alignment.py     # 단위 테스트 및 케이스 검증
└── visualize.py          # 토큰↔음절 정렬 시각화
```

---

## 구현 함수 목록

### `decompose.py`

```python
def decompose_syllable(char: str) -> list[str]:
    """
    한국어 음절 1자를 자모 phone 리스트로 분해.
    ㅇ 초성은 무음이므로 생략.
    예: '가' → ['KO_g', 'KO_a']
        '안' → ['KO_a', 'KO_N']
        '닭' → ['KO_d', 'KO_a', 'KO_L', 'KO_G']  # ㄺ 겹받침
    """

def text_to_jamo_counts(text: str) -> list[int]:
    """
    텍스트의 각 음절별 자모 수 리스트 반환. 비한국어·비구두점 문자는 건너뜀.
    예: '안녕' → [2, 3]
        '읽었어요' → [3, 2, 1, 1]
    반환값 길이 = 처리된 음절 수 M
    """

def text_to_jamo_phones(text: str) -> list[str]:
    """
    텍스트를 자모 phone 시퀀스로 변환. '_' padding은 포함하지 않음.
    예: '안녕' → ['KO_a', 'KO_N', 'KO_n', 'KO_yeo', 'KO_NG']
    """
```

### `alignment.py`

```python
def compute_word2ph_tok(
    text: str,
    tokenizer,
    jamo_counts: list[int],
) -> list[int]:
    """
    klue/roberta-large offset_mapping을 이용해 토큰 인덱스 기준 word2ph 계산.

    Args:
        text:        BERT에 입력할 원문 텍스트 (공백 포함 가능)
        tokenizer:   klue/roberta-large tokenizer
        jamo_counts: text_to_jamo_counts(text) 결과 (M개)

    Returns:
        word2ph_tok: [1] + [토큰별 자모 수] + [1]  (len = N = 토큰 수 + 2)
        sum(word2ph_tok) = sum(jamo_counts) + 2  (CLS/SEP 각 1)

    처리 규칙:
        - 1토큰이 여러 음절 커버  → 해당 음절들의 자모 수 합산
        - 1음절이 여러 토큰으로   → 각 토큰에 자모 수를 distribute_phone식으로 배분
        - 공백 토큰               → 자모 0이므로 1 할당 (fallback)
        - 미매핑 토큰             → 1 할당 (fallback)
    """

def apply_add_blank(word2ph_tok: list[int]) -> list[int]:
    """
    word2ph_tok에 add_blank 변환 적용.
    word2ph_tok[i] *= 2 (전체), word2ph_tok[0] += 1
    예: [1,3,2,1] → [3,6,4,2]
    sum 검증: sum(result) == 2 * sum(word2ph_tok) + 1
    """

def verify(
    word2ph_tok: list[int],
    jamo_counts: list[int],
    after_blank: bool = False,
) -> bool:
    """
    sum 불변식 검증.
    after_blank=False: sum(word2ph_tok) == sum(jamo_counts) + 2
    after_blank=True:  sum(word2ph_tok) == 2 * (sum(jamo_counts) + 2) + 1
    """
```

### `visualize.py`

```python
def print_alignment(text: str, tokenizer, jamo_counts: list[int]) -> None:
    """
    토큰↔음절 매핑을 테이블로 출력. 디버깅 필수 도구.

    출력 예:
    토큰 idx | 토큰    | offset  | 커버 음절 | 자모 수
    ---------|---------|---------|-----------|--------
    0 (CLS)  | [CLS]   | —       | —         | 1
    1        | 읽      | (0,1)   | 읽         | 3
    2        | ##었    | (1,2)   | 었         | 2
    3        | ##어요  | (2,4)   | 어,요      | 1+1=2
    4 (SEP)  | [SEP]   | —       | —         | 1
    합계: sum=9, L_pre=9, L=19
    """

def print_mismatch_stats(test_sentences: list[str], tokenizer) -> None:
    """
    전체 테스트 문장 대비 N≠M+2인 비율과 패턴 요약 출력.
    """
```

### `test_alignment.py`

```python
# 검증할 케이스 목록
TEST_CASES = [
    # (text,            설명)
    ("가나다",          "단순 음절, 받침 없음"),
    ("안녕하세요",      "일반 인사말"),
    ("읽었어요",        "1토큰이 여러 음절 커버하는 케이스: ##어요"),
    ("닭볶음",          "겹받침 포함"),
    ("좋아요",          "ㅎ탈락 케이스 (Jamo G2P는 표기 기준)"),
    ("학교에 갑니다.",  "공백·구두점 포함"),
    ("AI로 TTS를 만든다", "영문자 혼합 — skip 처리 확인"),
    ("그 사람이 왜 그랬는지 모르겠어요.", "긴 문장"),
]

def test_sum_invariant():
    """sum(word2ph_tok) == total jamo + 2 for all TEST_CASES"""

def test_add_blank_sum():
    """sum(word2ph_tok_blank) == 2*(total jamo+2)+1 for all TEST_CASES"""

def test_no_zero_entry():
    """word2ph_tok[i] >= 1 for all i (모든 토큰에 최소 1 phoneme)"""

def test_fallback_on_nonkorean():
    """영문자·숫자 포함 문장에서 예외 없이 처리"""
```

---

## 핵심 알고리즘 설명

### offset_mapping 기반 word2ph_tok 계산

```python
# 개념 코드 (실제 구현의 참조)
inputs = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
offsets = inputs["offset_mapping"]       # [(start,end), ...] len=N
tokens  = tokenizer.convert_ids_to_tokens(inputs["input_ids"])

# 음절 위치 계산 (공백은 건너뜀)
syllable_spans = []   # [(char_start, char_end), ...]
char_pos = 0
for ch in text:
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:         # 한국어 음절
        syllable_spans.append((char_pos, char_pos + 1))
    elif ch in PUNCTUATIONS:
        syllable_spans.append((char_pos, char_pos + 1))
    # else: 공백·영문 → skip (jamo_counts에도 없음)
    char_pos += len(ch)

# 토큰별 커버 음절 파악
word2ph_tok = [1]                        # CLS
for tok_start, tok_end in offsets[1:-1]: # CLS/SEP 제외
    if tok_start == tok_end:             # 특수 토큰 방어
        word2ph_tok.append(1)
        continue
    covered = sum(
        jamo_counts[i]
        for i, (s, e) in enumerate(syllable_spans)
        if s >= tok_start and e <= tok_end
    )
    word2ph_tok.append(covered if covered > 0 else 1)   # fallback
word2ph_tok.append(1)                    # SEP

assert sum(word2ph_tok) == sum(jamo_counts) + 2
```

### 1음절 → 여러 토큰 분할 케이스 (distribute 필요)

"닭" 같은 음절이 `닭` `##ㄱ` 등으로 분리되면 반대 방향 문제 발생.
이 경우 한 음절의 자모 수를 여러 토큰에 배분해야 함.

```python
# 하나의 음절이 여러 토큰으로 분할되는 경우
# 예: 음절 '닭'(자모3개)이 토큰 ['닭','##ㄱ']으로 분리
# → distribute_phone(3, 2) = [2, 1] 또는 [1, 2]
```

klue/roberta-large에서 실제로 이 케이스가 발생하는지, 발생 빈도는 얼마인지 실험에서 측정.

---

## 예상 결과 및 체크리스트

- [ ] `TEST_CASES` 전체에서 sum 불변식 통과
- [ ] N≠M+2 문장 비율 측정 (train.list 적용 시 참고용)
- [ ] 1토큰 다음절 vs 1음절 다토큰 각 케이스 발생 빈도
- [ ] fallback(값=1) 발동 케이스 목록화
- [ ] add_blank 적용 후 sum 검증

결과를 바탕으로 `style_bert_vits2/nlp/korean/bert_feature.py` 최종 구현 전략을 확정한다.

---

## 참고

- Style-Bert-VITS2 JP 구현: `style_bert_vits2/nlp/japanese/bert_feature.py`
- 자모 분해 원본: `style_bert_vits2/nlp/korean/g2p.py`
- 정렬 문제 배경: `custom/notes_bert_alignment.md`
- 전체 파이프라인 표: `custom/notes_textencoder_pipeline.md`
