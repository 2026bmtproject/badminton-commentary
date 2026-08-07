# Planner 與 Generator 實作報告

## 1. 目標與完成範圍

本里程碑將確定性的 Rally Facts 轉為可驗證的繁體中文賽評，完成：

1. 以 importance 決定是否評論、語氣與最大句數。
2. 建立每個 rally 可引用的 fact ID allowlist。
3. 將 fact、plan 與球員名稱組成結構化 LLM 輸入。
4. 使用 Fake Provider 完成不依賴網路的端到端測試。
5. 透過相同 `LLMProvider` 介面支援 Gemini Provider。
6. 驗證 LLM JSON、segment、fact ID 與句數限制。

本里程碑沒有加入 TTS、Web UI、資料庫，也沒有重新執行任何電腦視覺模型。

## 2. 程式結構

核心檔案：

```text
src/badminton_commentary/
├── schemas.py
├── analysis/
│   └── rally_analyzer.py
├── generation/
│   ├── __init__.py
│   ├── planner.py
│   ├── commentator.py
│   ├── validator.py
│   └── batch.py
└── prompts/
    └── commentator.txt

scripts/
├── build_ttyvsasy_facts.py
└── generate_ttyvsasy_commentary.py

tests/
├── test_planner.py
├── test_commentator.py
└── test_generation_pipeline.py
```

`planner.py` 是純 Python 規則，不呼叫 LLM。`commentator.py` 是單一 rally 的
LLM 邊界。`validator.py` 執行結構後的語意規則；`batch.py` 負責依序規劃及產生所有符合條件的 rally。開發 scripts
只負責讀寫 TTYvsASY fixture，不承載核心決策邏輯。

## 3. 資料契約

### 3.1 CommentaryPlan

```python
class CommentaryPlan(StrictModel):
    segment_index: int
    should_comment: bool
    style: Literal["neutral", "analytical", "excited", "concise"]
    focus: list[str]
    max_sentences: int
    allowed_fact_ids: list[str]
```

### 3.2 GeneratedCommentary

```python
class GeneratedCommentary(StrictModel):
    segment_index: int
    text: str
    source_fact_ids: list[str]
```

文字長度限制為 1 至 240 字元；`source_fact_ids` 至少一筆。所有 model 都繼承
`StrictModel`，未知欄位會被拒絕。

### 3.3 批次輸出

```json
{
  "lines": [
    {
      "segment_index": 0,
      "text": "戴資穎與安洗瑩目前戰成15比15。",
      "source_fact_ids": ["rally:0:score"]
    }
  ]
}
```

## 4. Planner 演算法

Planner 的輸入是 `ScoredRallyFact`，輸出是唯一且可重現的 `CommentaryPlan`。

### 4.1 Rally Analyzer

Planner 前新增一層確定性的 Rally Analyzer。它不呼叫 LLM，也不推論 rally 勝者。

| Confidence | 處理方式 |
|---:|---|
| `>= 0.70` | `reliable`，可以肯定描述 |
| `0.50–0.69` | `cautious`，只能使用可能、似乎等措辭 |
| `< 0.50` | 從 Generator fact catalog 排除 |

Analyzer 記錄 `opening_observed_stroke` 與 `final_observed_stroke`。後者只代表最後一筆
可用的模型觀測，不代表最後一拍或致勝球。

第一版 pattern 包含 `net_exchange`、`attack_sequence`、`clear_exchange` 與
`varied_strokes`。每個 pattern 都保存 `supporting_fact_ids`，可以回查原始 stroke。

### 4.2 Importance 門檻

| Importance | 是否評論 | Style | 最大句數 |
|---:|---|---|---:|
| `>= 0.70` | 是 | `excited` | 2 |
| `>= 0.50` | 是 | `analytical` | 2 |
| `>= 0.25` | 是 | `concise` | 1 |
| `< 0.25` | 否 | `neutral` | 1 |

核心規則：

```python
if importance >= 0.7:
    should_comment, style, max_sentences = True, "excited", 2
elif importance >= 0.5:
    should_comment, style, max_sentences = True, "analytical", 2
elif importance >= 0.25:
    should_comment, style, max_sentences = True, "concise", 1
else:
    should_comment, style, max_sentences = False, "neutral", 1
```

`focus` 直接複製 Importance Scorer 的 `reasons`，例如 `close_score`、
`late_game_score`、`long_rally`，因此每個語氣決策都能追溯。

### 4.3 Fact allowlist

Planner 只為實際存在的資料產生 ID：

- 完整比分：`rally:{segment}:score`
- 至少一筆 event：`rally:{segment}:length`
- Analyzer 選出的 stroke：`rally:{segment}:stroke:{event_index}`
- 可追溯 pattern：`rally:{segment}:pattern:{pattern_name}`
- 存在 highlight：`rally:{segment}:highlight`

Planner 依 style 最多選擇 1、2 或 3 個 notable strokes，以及最多 1 或 2 個 pattern，
不再把整段所有 stroke 都交給 LLM。缺少資料時，相應 ID 不會出現在 allowlist。

## 5. Generator 演算法

### 5.1 Fact catalog

Generator 不把任意的整場資料交給 LLM，而是把 allowlist 對應的資料整理成：

```json
{
  "rally:2:score": {"a": 20, "b": 20},
  "rally:2:length": 14,
  "rally:2:stroke:7": {
    "frame": 420,
    "time_sec": 14.0,
    "player": "a",
    "stroke_type": "殺球",
    "confidence": 0.91
  }
}
```

送往 provider 的 user prompt 包含：

- `prompt_version`: `commentator-v1`
- `players`: `a/b` 到顯示名稱的對應
- `plan`: 已驗證的 CommentaryPlan
- `fact_catalog`: allowlist 允許的事實

system prompt 固定存放於 `src/badminton_commentary/prompts/commentator.txt`，不需要在
PowerShell 指令中手動輸入 prompt。

### 5.2 回傳驗證

Provider 回傳後依序檢查：

1. 必須能解析為 JSON；也容忍模型偶爾加上的單層 Markdown code fence。
2. 必須通過 `GeneratedCommentary` Pydantic schema。
3. `segment_index` 必須與 plan 相同。
4. `source_fact_ids` 不得超出 `allowed_fact_ids`。
5. 以 `。！？!?` 分句後不得超過 `max_sentences`。
6. 沒有 outcome fact 時，拒絕「致勝」、「拿下這一分」、「得分」等結論。
7. 引用 cautious stroke 時，文字必須包含不確定措辭。

任一檢查失敗都會拋出 `CommentaryGenerationError`，不會把未驗證文字寫成正式結果。

## 6. Provider 設計

Generator 只依賴：

```python
class LLMProvider(Protocol):
    def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...
```

因此 Fake 與 Gemini 共用完全相同的 Planner、prompt、fact catalog 和驗證器。

### 6.1 Fake Provider

Fake 模式為每個 rally 建立穩定 JSON，只引用 score fact。用途是測試資料流與契約，
不是評估自然語言品質。

```powershell
uv run python .\scripts\build_ttyvsasy_facts.py
uv run python .\scripts\generate_ttyvsasy_commentary.py `
  --provider fake `
  --config .\config.yaml.example
```

輸出位於每組 clip 的 `commentary_fake.json`。

### 6.2 Gemini Provider

先設定 `config.yaml`，格式參照 `config.yaml.example`，並將 API key 放在環境變數：

```powershell
$env:GEMINI_API_KEY = "你的 API key"

uv run python .\scripts\generate_ttyvsasy_commentary.py `
  --provider gemini `
  --config .\config.yaml
```

預設模型為 `gemini-flash-latest`。Gemini 仍需通過與 Fake 相同的 JSON 與 fact 引用
驗證。本次自動測試沒有呼叫真實 API，因此不需要網路或 API key，也不消耗額度。

## 7. TTYvsASY 驗證結果

三組各產生 5 個 plan。依目前 Importance 規則，15 個 rally 都達到 `0.25`，因此都會
產生 Fake commentary。

每組同時輸出 `rally_analyses.json`，記錄 confidence band、排除數量、notable
strokes、patterns、supporting fact IDs 與 warnings。

| Clip | Rally 數 | Importance 範圍 | 最高 style |
|---|---:|---:|---|
| `seg0039-0043` | 5 | 0.25–0.75 | excited |
| `seg0052-0056` | 5 | 0.25–0.40 | concise |
| `seg0140-0144` | 5 | 0.50–0.75 | excited |

上游 `player=null` 的 7 筆 stroke 已在 commentary input 階段排除，但仍保留於原始
stage。Generator 不會猜測這些 stroke 的球員。

## 8. 測試覆蓋

新增測試涵蓋：

- 高、低 importance 的 Planner 決策。
- 只允許存在的 fact ID。
- Fake Provider 的有效結構化輸出。
- Markdown fenced JSON。
- 非 JSON 與缺欄位回應。
- 不允許的 fact ID。
- 超出最大句數。
- 批次 Fake Provider 生成。
- scored output 缺少 importance 時拒絕。

## 9. 已知限制與下一步

目前已有第一層語意防護，但尚未完成完整 Validator。下一步至少需要：

1. 玩家代號／姓名檢查。
2. 文字中比分與 score fact 的一致性檢查。
3. 跨 rally 重複內容檢查。
4. 更完整的繁體中文句數與長度策略。
5. Gemini token usage logging。

在 Validator 完成前，Gemini 輸出適合作為開發預覽，不應直接視為正式
`commentary.json`。
