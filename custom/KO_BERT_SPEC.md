# 한국어 BERT 연동 명세 — Style-Bert-VITS2

## 목적

Style-Bert-VITS2의 `bert_feature.py`에 한국어 BERT(`klue/roberta-large`)를 연동한다.  
TTS TextEncoder는 BERT 출력 벡터를 phoneme(자모) 시퀀스 길이에 맞게 `expand`하는데,
이때 **"각 BERT 토큰이 몇 개의 자모에 대응하는가"**를 나타내는 `word2ph` 배열이 필요하다.

---

## 문제: N ≠ M + 2

일본어·중국어는 `토큰 수 N = 글자 수 M + 2(CLS/SEP)`가 성립해 단순하게 처리된다.  
`klue/roberta-large`는 형태소 기반 WordPiece를 사용하므로 N < M+2인 경우가 빈번하다.

```
"안녕하세요" (M=5음절)
  → [CLS] 안녕 ##하 ##세요 [SEP]   N=5, M+2=7  ← N < M+2
```

따라서 단순히 `word2ph[i] = 글자당 자모 수`로는 BERT 토큰 수와 맞출 수 없다.

---

## 해결: 3단계 연결 구조

토큰·원문·phoneme 세 가지는 서로 다른 기준을 가진다. 이를 연결하는 흐름은 다음과 같다.

### 1단계 — 토큰 ↔ 원문 음절: offset_mapping

tokenizer는 각 토큰이 원문의 어떤 char 범위를 커버하는지 `offset_mapping`으로 반환한다.  
이를 이용해 **"이 토큰이 어떤 음절 인덱스를 포함하는가"**를 역산한다.

```
"괜찮은"
 0  1  2   ← 원문 char index

 토큰 '괜찮'  offset (0,2) → 음절[0]=괜, 음절[1]=찮
 토큰 '##은'  offset (2,3) → 음절[2]=은
```

### 2단계 — 원문 음절 ↔ g2pk2 phoneme: 음절 수 보존

g2pk2는 실제 발음으로 변환하되 순수 한국어에서 **음절 수를 보존**한다.  
따라서 원문 i번째 음절 ↔ g2pk2 i번째 음절이 1:1 대응된다.  
자모 수는 g2pk2 발음 기준으로 계산한다.

```
원문:    괜    찮    은
          ↕     ↕     ↕    ← 위치는 원문 기준 (1:1)
g2pk2:  괜    차    는     ← 자모 수는 발음 기준
자모수:   3     2     2
```

| 원문 | g2pk2 발음 | 자모 수 변화 |
|------|-----------|------------|
| 읽었어요 | 일거써요 | 읽(3)→일(2), 었(2)→거(2), 어(1)→써(2), 요(1)→요(1) |
| 닭볶음 | 닥뽀끔 | 닭(3)→닥(3), 볶(3)→뽀(2), 음(2)→끔(3) |
| 좋아요 | 조아요 | 좋(2)→조(2), 아(1)→아(1), 요(1)→요(1) |

### 3단계 — 합산: word2ph_tok

토큰이 커버하는 음절들의 g2pk2 자모 수를 합산하면 해당 토큰의 `word2ph` 값이 된다.

```
토큰 '괜찮' → 음절[0]=괜(3) + 음절[1]=찮→차(2) = 5
토큰 '##은' → 음절[2]=은→는(2)                 = 2

word2ph_tok = [1(CLS), 5, 2, 1(SEP)]
```

### 전체 예시

```
"안녕하세요"
 0  1  2  3  4   ← 원문 char index

 토큰 '안녕'   offset (0,2) → 음절[0]=안, 음절[1]=녕  →  자모수 2+3=5
 토큰 '##하'   offset (2,3) → 음절[2]=하              →  자모수 2
 토큰 '##세요'  offset (3,5) → 음절[3]=세, 음절[4]=요  →  자모수 2+1=3

 word2ph_tok = [1,  5,  2,  3,  1]   (CLS / 안녕 / ##하 / ##세요 / SEP)
 sum = 12 = (2+3+2+2+1) + 2 = sum(jamo_counts) + 2  ✓
```

---

## sum 불변식

순수 한국어 텍스트에 대해 반드시 성립해야 하는 불변식:

```
sum(word2ph_tok) == sum(jamo_counts) + 2
```

`add_blank` 변환 후:

```
sum(word2ph_after_blank) == 2 * (sum(jamo_counts) + 2) + 1
```

영문·숫자 혼합 텍스트는 비한국어 토큰에 fallback=1을 적용하므로 불변식이 성립하지 않는다 (설계상 정상).

---

## 구현 파일 요약

### `decompose.py`

```python
PUNCTUATIONS = ["!", "?", "…", ",", ".", "'", "-"]
_KO_SYL_START = 0xAC00
_KO_SYL_END   = 0xD7A3

def decompose_syllable(char: str) -> list[str]:
    """한국어 음절 1자를 자모 phone 리스트로 분해. ㅇ 초성은 생략."""
    code = ord(char) - _KO_SYL_START
    final_idx   = code % 28
    medial_idx  = (code // 28) % 21
    initial_idx = code // 28 // 21
    phones = []
    if _INITIAL[initial_idx]:  phones.append(_INITIAL[initial_idx])
    phones.append(_MEDIAL[medial_idx])
    if _FINAL[final_idx]:      phones.append(_FINAL[final_idx])
    return phones

def text_to_jamo_counts(text: str, use_g2pk2: bool = True) -> list[int]:
    """
    원문 한국어 음절 위치 기준 자모 수 리스트.
    반환값 길이 M = 원문에서 처리된 한국어 음절 + 구두점 수.
    비한국어 문자(영문·숫자·공백)는 skip.
    """
    if use_g2pk2:
        pronounced = G2p()(text)
        orig_syls = [ch for ch in text     if korean_or_punct(ch)]
        pron_syls = [ch for ch in pronounced if korean_or_punct(ch)]
        if len(orig_syls) == len(pron_syls):   # 음절 수 보존 확인
            return [len(decompose_syllable(ch)) if is_korean(ch) else 1
                    for ch in pron_syls]
    # fallback: 원문 표기 기준
    return [len(decompose_syllable(ch)) if is_korean(ch) else 1
            for ch in text if korean_or_punct(ch)]
```

### `alignment.py`

```python
def _build_syllable_spans(text: str) -> list[tuple[int, int]]:
    """원문에서 한국어 음절·구두점의 char offset 범위 목록."""
    return [(i, i+1) for i, ch in enumerate(text) if korean_or_punct(ch)]

def compute_word2ph_tok(text, tokenizer, jamo_counts) -> list[int]:
    """
    Returns [1(CLS)] + [토큰별 자모 수] + [1(SEP)]
    sum == sum(jamo_counts) + 2  (순수 한국어 기준)
    """
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    syllable_spans = _build_syllable_spans(text)
    assert len(syllable_spans) == len(jamo_counts)

    body_offsets = offsets[1:-1]

    # 토큰 → 커버하는 음절 인덱스 목록
    tok_to_syl = []
    for tok_start, tok_end in body_offsets:
        if tok_start == tok_end:
            tok_to_syl.append([])
        else:
            covered = [i for i, (s, e) in enumerate(syllable_spans)
                       if s >= tok_start and e <= tok_end]
            tok_to_syl.append(covered)

    # 1음절이 여러 토큰으로 분할된 경우 자모 수를 균등 배분
    syl_to_toks = {}
    for tok_i, syls in enumerate(tok_to_syl):
        for syl_i in syls:
            syl_to_toks.setdefault(syl_i, []).append(tok_i)

    split_alloc = {}
    for syl_i, tok_list in syl_to_toks.items():
        if len(tok_list) > 1:
            allocated = _distribute(jamo_counts[syl_i], len(tok_list))
            for tok_i, val in zip(tok_list, allocated):
                split_alloc[(tok_i, syl_i)] = val

    word2ph = [1]  # CLS
    for tok_i, syls in enumerate(tok_to_syl):
        if not syls:
            word2ph.append(1)  # 비한국어·공백 토큰 fallback
            continue
        total = sum(
            split_alloc.get((tok_i, syl_i), jamo_counts[syl_i])
            for syl_i in syls
        )
        word2ph.append(total if total > 0 else 1)
    word2ph.append(1)  # SEP
    return word2ph

def apply_add_blank(word2ph: list[int]) -> list[int]:
    result = [v * 2 for v in word2ph]
    result[0] += 1
    return result
```

---

## bert_feature.py 구현 지침

### 함수 시그니처

```python
def get_bert_feature(
    text: str,
    word2ph: list[int],        # 위에서 계산한 word2ph_tok (CLS/SEP 포함)
    device: str = "cpu",
    model_path: str = "klue/roberta-large",
) -> torch.Tensor:
    """
    Returns: (hidden_size, phoneme_seq_len) tensor
    phoneme_seq_len = sum(word2ph)
    """
```

### 처리 흐름

```
text
 │
 ├─► text_to_jamo_counts(text)  →  jamo_counts  (len=M)
 │
 ├─► compute_word2ph_tok(text, tokenizer, jamo_counts)  →  word2ph_tok  (len=N)
 │       N = tokenizer(text)의 토큰 수 (CLS/SEP 포함)
 │
 ├─► tokenizer(text)  →  input_ids  →  BERT  →  hidden_states  (N, hidden_size)
 │
 └─► expand(hidden_states, word2ph_tok)  →  (sum(word2ph_tok), hidden_size)
         각 토큰 벡터를 word2ph_tok[i]번 반복
         → transpose  →  (hidden_size, sum(word2ph_tok))
```

### expand 예시

```python
def expand_by_word2ph(hidden: torch.Tensor, word2ph: list[int]) -> torch.Tensor:
    # hidden: (N, hidden_size)
    # word2ph: list of length N
    # returns: (sum(word2ph), hidden_size)
    return torch.cat([h.unsqueeze(0).expand(n, -1)
                      for h, n in zip(hidden, word2ph)], dim=0)
```

### 다른 언어 구현과의 차이점

| 항목 | 일본어·중국어 | 한국어 |
|------|--------------|--------|
| N = M+2 보장 | 항상 | 아님 (klue/roberta-large는 WordPiece) |
| word2ph 계산 | `[1] * N` 또는 글자별 phone 수 | `compute_word2ph_tok` 필요 |
| 발음 변환 | mecab / jieba | g2pk2 (음절 수 보존) |
| fallback | — | 비한국어 토큰 → 1 |

---

## 검증 결과

`pytest test_alignment.py -v` → **25 / 25 PASSED**

| 문장 | N (토큰) | M+2 | word2ph body | sum 검증 |
|------|----------|-----|--------------|---------|
| 안녕하세요 | 5 | 7 | [5, 2, 3] | 12=10+2 ✓ |
| 읽었어요 | 5 | 9 | [2, 2, 3] | 9=7+2 ✓ |
| 닭볶음 | 4 | 7 | [3, 5] | 10=8+2 ✓ |
| 좋아요 | 4 | 6 | [3, 1] | 6=4+2 ✓ |
| 학교에 갑니다. | 6 | 10 | [5, 1, 7, 1] | 16=14+2 ✓ |

---

## 환경 의존성

```toml
# pyproject.toml
[project]
requires-python = "==3.12.*"
dependencies = [
    "transformers>=4.51.3",
    "torch>=2.8.0",
    "g2pk2",
    "eunjeon",          # Windows에서 g2pk2가 요구하는 MeCab 래퍼
    "setuptools<72",    # eunjeon이 pkg_resources 사용 → setuptools 72+ 에서 제거됨
]
```

**주의**: `setuptools<72` 핀이 없으면 eunjeon import 실패 (`ModuleNotFoundError: pkg_resources`).

---

## 실험 저장소

`ko_bert_alignment_experiment/` — 위 알고리즘의 완전한 구현 및 테스트 포함.  
Style-Bert-VITS2 통합 시 `decompose.py`와 `alignment.py`를 직접 참조할 것.
