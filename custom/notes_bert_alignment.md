# BERT 토크나이저-word2ph 정렬 검증 노트

## 배경

`extract_bert_feature`는 BERT hidden state를 `word2ph` 인덱스로 직접 접근(`res[i]`)한다.
이 접근이 올바르려면 **BERT 토큰 수 == `word2ph` 길이**가 성립해야 한다.

---

## JP/ZH의 런타임 검증

### 코드 위치

- JP: `style_bert_vits2/nlp/japanese/bert_feature.py:70`
- ZH: `style_bert_vits2/nlp/chinese/bert_feature.py:63`

### 어서션

```python
assert len(word2ph) == len(text) + 2, text
```

`+2`는 BERT 특수 토큰 `[CLS]`·`[SEP]`.

```
입력:       "彼女は"      (len=3)
토큰:  [CLS] 彼 女 は [SEP]  (5개)
word2ph 길이:  3 + 2 = 5   ← 어서션 통과
res.shape:  [5, 1024]
res[0] = [CLS] 위치, res[1] = "彼", ...
```

만약 어떤 문자가 서브워드 2개로 쪼개지면 `res`는 6개지만 `word2ph`는 5개 → 어서션 실패.

### 어서션이 항상 통과하는 이유

| 언어 | 이유 |
|------|------|
| JP | BERT 입력 전 `text_to_sep_kata()`로 **한자→가타카나 변환** (`bert_feature.py:45`). "私"(1한자)→"ワタシ"(3가타카나). assert의 `len(text)`는 변환 후 가타카나 길이 기준. 가타카나 char 단위 SentencePiece vocabulary이므로 1가타카나 = 1토큰 성립 |
| ZH | 한자 원문 그대로 입력. `tokenize_chinese_chars=true` 설정으로 한자 1자 = 1토큰 보장. 중국어 텍스트는 한자 나열이므로 언어 구조상 항상 성립 |

---

## 한국어에서의 문제

### `klue/roberta-large` tokenizer

Mecab-ko 형태소 분석 후 WordPiece(BPE) 적용. 토큰 단위 = **형태소 수준**.

```
입력:   "읽었어요"      (len=4, 음절 4개)
Mecab:  읽 / 었 / 어요   (형태소 3개)
토큰:   [CLS] 읽 ##었 ##어요 [SEP]   (5개)
word2ph 길이: 4 + 2 = 6   ← 5 ≠ 6, 어서션 실패
```

JP/ZH의 `assert len(word2ph) == len(text) + 2`를 그대로 사용할 수 없다.

### 어서션 대체 방안

JP/ZH식 등호 조건 대신, 토큰 수와 음절 수 불일치를 측정하는 검증 스크립트를 별도 실행:

```
custom/check_bert_alignment.py   ← 구현 예정
```

전체 train.list에 대해 `n_tokens != len(word2ph) - 2` 인 문장 비율과 패턴을 집계.
결과에 따라 `extract_bert_feature`의 정렬 방식(offset_mapping 기반)을 결정.

---

## offset_mapping 기반 정렬 (한국어 구현 방향)

```python
inputs = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
offsets = inputs["offset_mapping"]   # [(char_start, char_end), ...]
hidden  = bert_hidden[1:-1]          # CLS, SEP 제거

# 음절별로 해당 토큰 hidden state를 평균
syllable_features = []
char_pos = 0
for syl in text:
    syl_start = char_pos
    syl_end   = char_pos + len(syl)
    tok_idx   = [i for i, (s, e) in enumerate(offsets)
                 if s >= syl_start and e <= syl_end]
    if tok_idx:
        syllable_features.append(hidden[tok_idx].mean(0))
    else:
        # fallback: 가장 가까운 토큰
        syllable_features.append(hidden[min(len(hidden)-1, char_pos)])
    char_pos += len(syl)

# 결과 shape: [len(text), 1024]
# 이후 word2ph로 phoneme 단위 expand (JP/ZH와 동일)
```

이 방식에서 런타임 검증은:

```python
assert len(syllable_features) == len(text), "음절 수 불일치"
assert len(word2ph) == len(text) + 2,       "word2ph 길이 불일치"
# +2: CLS/SEP 위치에 zero vector 또는 mean vector를 앞뒤로 붙이는 관례 유지
```

---

## 요약

| | JP/ZH | KO (`klue/roberta-large`) |
|--|-------|--------------------------|
| 어서션 | `len(word2ph) == len(text) + 2` | 동일 어서션 사용 불가 |
| 정렬 방식 | 토큰 인덱스 = 문자 인덱스 (1:1) | offset_mapping으로 음절↔토큰 매핑 |
| 구현 복잡도 | 없음 (직접 `res[i]` 접근) | 음절별 토큰 탐색 + 평균 필요 |
| 검증 스크립트 | 불필요 | `custom/check_bert_alignment.py` 선행 필요 |
