# TextEncoder 입력 컴포넌트 상세

## 기호 정의

| 기호 | 의미 |
|------|------|
| M | BERT 입력 텍스트의 글자(음절) 수 (공백 제외, CLS/SEP 미포함) |
| N | BERT 토큰 수 = M + 2 (CLS/SEP 포함) = `len(word2ph)` |
| L_pre | add_blank 적용 전 phoneme 수 = `sum(word2ph)` (before add_blank) |
| L | add_blank 후 phoneme 수 = 2×L_pre + 1 = `sum(word2ph)` (after add_blank) |
| H | `hidden_channels` (모델 config, 기본값 192) |

`word2ph`의 길이는 add_blank 전후 모두 N으로 일정하다. 변하는 것은 각 원소의 값(배수)과 합계(L_pre → L).

---

## 전체 파이프라인 표

| 단계 | 컴포넌트 | JP | ZH | KO Jamo (Phase 1) | KO Jamo Update (Phase 3 목표) | 출력 길이 / Shape | 코드 위치 |
|------|----------|----|----|-------------------|-------------------------------|-------------------|-----------|
| **① 입력** | 원문 텍스트 | 일본어 혼합문자 (한자·히라가나) | 중국어 한자 | 한국어 한글 음절 | 동일 | 가변 | — |
| **② 정규화** | `normalize_text(text)` | 전각→반각, 이체자 통일 | 번체→간체, 숫자 변환 | 구현됨 (기호·공백 처리) | 동일 | → `norm_text` (M 글자) | `*/normalizer.py` |
| **③ G2P** | `g2p(norm_text)` | pyopenjtalk → sep_kata → 서브모라 phoneme | pypinyin → 병음 → 구성음 | 음절별 자모 분해 | **동일 (변경 없음)** | `phones`[L_pre], `tones`[L_pre], `word2ph`[N] | `*/g2p.py` |
| | phoneme 단위 | 서브모라: ガ→(g,a), ン→(N) | 병음: 学→(x,u,e), 生→(sh,eng) | 자모: 가→(KO_g,KO_a), 안→(KO_a,KO_N) | **동일** | — | `mora_list.py` / `g2p.py` |
| | `word2ph` 생성 | `distribute_phone` 라운드로빈 + 앞뒤 [1] | 한자별 음소 수 + 앞뒤 [1] | 음절별 자모 수 + 앞뒤 [1] | **동일** | len=N, sum=L_pre | `g2p.py:144` |
| | `word2ph` 예시 | "学生は"(M=3): [1,4,3,2,1] sum=11 | "学生"(M=2): [1,3,2,1] sum=7 | "안녕"(M=2): [1,2,3,1] sum=7 | **동일** | N=M+2 | — |
| **④ Phoneme 시퀀스** | `phones` | `[_,g,a,k,u,s,e,i,w,a,_]` | `[_,x,u,e,sh,eng,_]` | `[_,KO_a,KO_N,KO_n,KO_yeo,KO_NG,_]` | **동일** | [L_pre] | — |
| | `tones` | 0 또는 1 | 0~4 (성조) | **전부 0** (성조 없음) | **동일** | [L_pre] | — |
| **⑤ add_blank** | `intersperse(phones, 0)` | blank 삽입 | 동일 | 동일 | **동일** | [L]=2×L_pre+1 | `infer.py:124` |
| | `word2ph` 조정 | `[i]*=2`, `[0]+=1` | 동일 | 동일 | **동일** | len=N, sum→L | `infer.py:127-128` |
| | 예시 | "学生は": [3,8,6,4,2] sum=23 | "学生": [3,6,4,2] sum=15 | "안녕": [3,4,6,2] sum=15 | **동일** | — | — |
| **⑥ BERT 입력 텍스트** | BERT에 전달되는 `text` | `"".join(text_to_sep_kata()[0])` = 원문 한자·히라가나 | `norm_text` 그대로 | **구현되지 않음** | `norm_text` 그대로 (음절 직접 입력) | M 글자 | `bert_feature.py:45` |
| | 변환 이유 | sep_kata는 phoneme 추출 전용. BERT 입력과 무관 | 한자=BERT char 단위, 변환 불필요 | — | 한글 음절=BERT char 단위 (Strategy B 기준), 변환 불필요 | — | — |
| **⑦ BERT 토크나이징** | `tokenizer(text)` | SentencePiece (char-wwm vocab) | BertTokenizer (`tokenize_chinese_chars=True`) | **구현되지 않음** | klue/roberta-large BertTokenizer (형태소 기반 WordPiece) | N 토큰 (N ≠ M+2 가능) | — |
| | 1글자=1토큰 여부 | char-wwm vocab에 각 글자 단독 등록 → 보장 | `tokenize_chinese_chars=True` → 보장 | — | **보장되지 않음** → offset_mapping으로 토큰↔음절 매핑 필요 | N = M+2 미보장 | `bert_feature.py:70/:63` |
| | 토큰 예시 | `[CLS,学,生,は,SEP]` (N=5) | `[CLS,学,生,SEP]` (N=4) | **구현되지 않음** | `[CLS,읽,##었,##어요,SEP]` (N=5, M=4 음절) — N≠M+2 | — | — |
| **⑧ BERT forward** | `model(**inputs)` | DeBERTa-v2-large (24L×16h, **1024**) | RoBERTa-large (24L×16h, **1024**) | **구현되지 않음** | klue/roberta-large (24L×16h, **1024**) | `res`: [N, **1024**] | — |
| | 추출 레이어 | `hidden_states[-3]` | 동일 | — | 동일 | [N, 1024] | `bert_feature.py:61/:54` |
| **⑨ word2ph_tok 계산** | bert_feature 내부 | word2ph 그대로 사용 (syllable=token) | 동일 | **구현되지 않음** → `zeros(1024, L)` | offset_mapping으로 토큰별 음절 파악 → word2ph_tok[j] = 해당 음절 자모 수 합계 → add_blank 조정 → `res[j].repeat(word2ph_tok[j], 1)` | [1024, L] | `nlp/__init__.py:58` |
| | 런타임 검증 | `assert len(word2ph) == len(text)+2` | 동일 | 없음 | `assert sum(word2ph_tok) == L` (총 phoneme 수 불변) | — | `bert_feature.py:70/:63` |
| **⑩ bert_proj** | `Conv1d(dim, H, 1)` | `Conv1d(1024, H, 1)` | `Conv1d(1024, H, 1)` | `Conv1d(1024, H, 1)` 예약, zero 입력 | `Conv1d(1024, H, 1)` — **무수정** | [1024,L]→[H,L] | `models_ko.py` |
| **⑪ TextEncoder 합산** | `emb(phones)` | 음소 임베딩 조회 | 동일 | 동일 | 동일 | [L,H] | — |
| | `tone_emb(tones)` | 0/1 + offset | 0~4 + offset | **전부 0** + offset | **동일** | [L,H] | — |
| | `language_emb(lang_ids)` | JP 상수 | ZH 상수 | KO 상수 | 동일 | [L,H] (사실상 bias) | — |
| | `bert_emb` | `bert_proj(bert).T` | 동일 | **zero** | **실제값** (`bert_proj(bert).T`) | [L,H] | — |
| | `style_emb` | `Linear(256→H)` broadcast | 동일 | 동일 | 동일 | [L,H] | — |
| | **TextEncoder 입력** | `(emb+tone+lang+bert+style)×√H` | 동일 | bert=0이므로 사실상 4개 합산 | **5개 전부 합산 (JP/ZH와 동일)** | **[L, H]** | `models_jp_extra.py:420-428` |

---

## 모델 선택 근거: klue/roberta-large 단일 전략

JP/ZH가 사용하는 모델을 보면, **음절/음소 전용 tokenizer를 쓴 것이 아니라 해당 언어에서 일반적인 BERT의 tokenizer가 자연스럽게 char 단위로 정렬되는 것**을 활용한 것이다.

| | JP | ZH | KO (목표) |
|--|----|----|-----------|
| 모델 | deberta-v2-large-japanese-**char-wwm** | chinese-roberta-wwm-ext-large | **klue/roberta-large** |
| 토크나이저 | 일반 Japanese BERT, 언어 구조상 char 단위 | 일반 Chinese BERT + `tokenize_chinese_chars=True` | 일반 Korean BERT, 형태소 기반 WordPiece |
| 파라미터 | ~300M | ~330M | ~337M |
| 학습 데이터 | 171GB | 5.4B 단어 | 62GB |
| hidden_size | 1024 | 1024 | **1024** |
| bert_proj | Conv1d(1024, H, 1) | Conv1d(1024, H, 1) | **Conv1d(1024, H, 1) — 무수정** |

`snunlp/KR-BERT-char16424`는 음절 정렬을 인위적으로 보장하기 위해 특수 설계된 base 모델로, JP/ZH의 방식과 목적이 다르다. klue/roberta-large가 목적·규모·학습 데이터에서 가장 근사한 대응 모델이다.

### word2ph 인덱스 체계 변경

JP/ZH에서는 `g2p()→word2ph`와 `BERT tokenizer→토큰`이 동일한 인덱스(syllable=token)를 공유하지만, KO + klue/roberta-large에서는 형태소 단위 분할로 인해 두 인덱스가 달라진다.

```
JP/ZH (일치):
  g2p()  word2ph[i] = i번째 글자의 phoneme 수   len = M+2 = N (BERT 토큰 수)
  BERT   토큰[i] = i번째 글자                   N = M+2

KO (불일치):
  g2p()  word2ph_syl[i] = i번째 음절의 자모 수  len = M+2
  BERT   토큰[j] = j번째 형태소 (j 범위 ≠ M+2)  N ≠ M+2 가능

해결:
  g2p() → word2ph_syl  (add_blank에 계속 사용, 변경 없음)
  bert_feature.py 내부에서:
    - tokenizer(text, return_offsets_mapping=True)
    - 각 토큰의 offset으로 커버하는 음절 범위 파악
    - word2ph_tok[j] = 해당 음절들의 자모 수 합계
    - add_blank 조정: word2ph_tok[j] *= 2, word2ph_tok[0] += 1
    - assert sum(word2ph_tok) == L  ← 총 phoneme 수 불변
```

JP/ZH의 `assert len(word2ph) == len(text)+2` 대신, KO는 `assert sum(word2ph_tok) == L`로 검증한다.

---

## 구체 예시 비교

### JP: "学生は" (M=3)

```
G2P    word2ph: [1,4,3,2,1]   N=5, L_pre=11
blank  word2ph: [3,8,6,4,2]   L=23
BERT   [CLS,学,生,は,SEP] → res:[5,1024] → expand → [1024,23]
proj   [1024,23]→[H,23]   →   x:[23,H]=[L,H]
```

### KO Jamo (Phase 1): "안녕" (M=2)

```
G2P    word2ph: [1,2,3,1]   N=4, L_pre=7
blank  word2ph: [3,4,6,2]   L=15
BERT   → zeros(1024,15)
proj   [1024,15]→[H,15]   →   x:[15,H]=[L,H]  (bert_emb=0)
```

### KO Jamo Update [B]: "안녕" (M=2)

```
G2P    word2ph: [1,2,3,1]   N=4, L_pre=7   (변경 없음)
blank  word2ph: [3,4,6,2]   L=15           (변경 없음)
BERT   [CLS,안,녕,SEP] → res:[4,768] → expand → [768,15]
proj   [768,15]→[H,15]   →   x:[15,H]=[L,H]  (bert_emb 실제값)
```

---

## 핵심 불변식 요약

| 불변식 | JP/ZH | KO Phase 1 | KO Update (klue/roberta-large) |
|--------|-------|------------|--------------------------------|
| `len(word2ph) == len(text)+2` | 성립 (syllable=token) | 검증 없음 | **불성립** — 형태소 N ≠ M+2. 대신 `sum(word2ph_tok)==L`로 검증 |
| `sum(word2ph) == len(phones)` | 성립 | 성립 | 성립 (word2ph_tok 합계=L) |
| `bert.shape[-1] == len(phones)` | 성립 | 성립 (zero) | **성립** (실제값, L) |
| bert_proj 입력 dim | 1024 | 1024 (zero) | **1024 — 무수정** |

---

## Phase 4 누락 사항 전체 점검

`symbols_ko.py` 확인 결과, 심볼·tone·language_id 관련 항목은 이미 구현 완료:

| 항목 | 상태 | 비고 |
|------|------|------|
| `KO_*` 심볼 66개 (초성18+중성21+종성27) | ✅ | `symbols_ko.py` |
| `LANGUAGE_ID_MAP_KO["KO"] = 3` | ✅ | — |
| `LANGUAGE_TONE_START_MAP_KO["KO"] = 12` | ✅ | ZH6+JP2+EN4 이후 |
| `NUM_KO_TONES = 1` | ✅ 의도적 | 한국어 성조 없음 → tone_emb 1칸 = 상수 bias. 구조 문제 없음 |
| `cleaned_text_to_sequence_ko()` | ✅ | `symbols_ko.py:123` |
| BERT 파이프라인 (⑥~⑨) | ❌ | Phase 3 핵심 목표 |
| word2ph 토큰 인덱스 재설계 | ❌ | `bert_feature.py` 내부 구현 필요 |

## Phase 3 구현 대상 파일

| 파일 | 변경 내용 |
|------|-----------|
| `style_bert_vits2/nlp/korean/bert_feature.py` | 신규 작성. offset_mapping으로 word2ph_tok 내부 계산 |
| `style_bert_vits2/constants.py` | `DEFAULT_BERT_MODEL_PATHS[KO] = "bert/klue-roberta-large"` 추가 |
| `style_bert_vits2/nlp/__init__.py:55-59` | zero tensor → `extract_bert_feature` 호출로 교체 |
| `style_bert_vits2/models/models_ko.py` | **변경 없음** (`Conv1d(1024, H, 1)` 그대로) |
| `style_bert_vits2/models/infer.py` | KO 분기 추가 |

상세 내용: `custom/notes_bert_alignment.md`, `docs/experiment_korean_bert.md`
