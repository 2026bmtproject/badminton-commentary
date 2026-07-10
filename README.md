# Badminton Commentary

羽球比賽 AI 賽評生成模組。

本專案會接收主系統已完成的比賽分析結果，例如比分、擊球事件、球種分類與觀眾歡呼程度，將它們整理成可靠的比賽事實，再使用 LLM 產生自然、簡潔且不捏造資訊的羽球賽評。

本專案是以下主系統的獨立子專案：

```text
2026bmtproject/badminton-analysis-system
```

Repository：

```text
2026bmtproject/badminton-commentary
```

---

## 1. 專案目標

輸入一場羽球比賽各分析階段產生的 JSON，輸出與影片片段對齊的賽評文字。

目標流程：

```text
主系統分析結果
    ↓
Fact Builder
    ↓
Importance Scorer
    ↓
Commentary Planner
    ↓
Commentary Generator
    ↓
Validator
    ↓
commentary.json
```

本專案目前只處理：

* 比賽事實整理
* 回合重要性判斷
* 賽評內容規劃
* LLM 賽評文字生成
* 輸出驗證
* `commentary.json` 產生

目前不處理：

* TTS 語音合成
* 聲音模仿
* 影片剪輯
* 字幕燒錄
* 音訊與影片混音
* 上游電腦視覺或音訊模型訓練

---

## 2. 與主系統的關係

主系統負責：

* `match_segmentation`：回合切割
* `score_recognition`：比分辨識
* `court_detection`：球場辨識
* `pose`：人體骨架
* `shuttle_tracking`：羽球軌跡
* `event_detection`：擊球事件
* `stroke_classification`：球種辨識
* `audio_highlight`：觀眾歡呼或精彩程度
* `commentary`：呼叫本專案產生賽評

本專案不應複製主系統的模型，也不應直接修改主系統內其他分析階段。

主系統中的 `modules/commentary/` 應保持為薄薄的 Adapter，負責：

1. 找到各階段輸出的 JSON。
2. 將路徑或資料傳給本套件。
3. 執行 commentary pipeline。
4. 將結果寫入：

```text
matches/{match}/stages/commentary/commentary.json
```

本專案應保持能夠單獨執行與測試，不應依賴完整影片分析流程才能啟動。

---

## 3. 核心設計原則

### 3.1 LLM 不負責建立事實

LLM 不應自行推測：

* 比分
* 球員身分
* 擊球者
* 球種
* 回合長度
* 比賽局數
* 是否為局點或賽末點
* 觀眾反應強度

這些資訊必須由確定性的 Python 程式從輸入 JSON 建立。

### 3.2 所有賽評必須可追溯

每段內部賽評資料應記錄它使用了哪些事實，例如：

```json
{
  "source_fact_ids": [
    "rally:12:score",
    "rally:12:stroke:8",
    "rally:12:highlight"
  ]
}
```

正式輸出若需符合主系統的簡化契約，可以由 Adapter 移除這些除錯欄位。

### 3.3 低信心資料不可被肯定描述

例如球種模型 confidence 很低時，不應生成：

```text
A 選手以一記精準的反手切球控制網前。
```

可以改成較保守的描述：

```text
A 選手在網前做出細膩處理。
```

或者完全不提該球種。

### 3.4 先完成可測試的確定性流程

第一版必須能使用 fixture 和 fake LLM provider 執行，不應要求真實 API Key 才能跑測試。

### 3.5 Provider 不得與核心邏輯耦合

OpenAI、Gemini 或其他 LLM 應實作共同介面。

核心 pipeline 不應直接 import 特定廠商 SDK。

---

## 4. 建議目錄結構

```text
badminton-commentary/
├── README.md
├── pyproject.toml
├── uv.lock
├── config.example.yaml
│
├── src/
│   └── badminton_commentary/
│       ├── __init__.py
│       ├── cli.py
│       ├── pipeline.py
│       ├── schemas.py
│       ├── config.py
│       │
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── fact_builder.py
│       │   ├── importance.py
│       │   └── rally_analyzer.py
│       │
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── planner.py
│       │   ├── commentator.py
│       │   └── validator.py
│       │
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── fake.py
│       │   ├── openai.py
│       │   └── gemini.py
│       │
│       └── prompts/
│           ├── planner.txt
│           └── commentator.txt
│
├── fixtures/
│   └── sample_match/
│       ├── segments.json
│       ├── scores.json
│       ├── events.json
│       ├── strokes.json
│       └── highlights.json
│
└── tests/
    ├── test_schemas.py
    ├── test_fact_builder.py
    ├── test_importance.py
    ├── test_planner.py
    ├── test_validator.py
    └── test_pipeline.py
```

不要為了符合目錄草稿而建立大量空檔案。只有在開始實作相關功能時才建立模組。

---

## 5. 上游輸入契約

輸入格式以主系統的 `modules/contracts.py` 為準。

本 README 只記錄目前預期格式。若主系統契約更新，應同步修改本專案 schema 與 fixture。

### 5.1 `segments.json`

每個 segment 對應一個候選 rally 影片片段。

```json
{
  "fps": 30.0,
  "segments": [
    {
      "start_frame": 1200,
      "end_frame": 1470,
      "start_sec": 40.0,
      "end_sec": 49.0,
      "duration_sec": 9.0
    }
  ]
}
```

陣列索引即為 `segment_index`。

### 5.2 `scores.json`

```json
{
  "rallies": [
    {
      "segment_index": 0,
      "score_a": 18,
      "score_b": 18,
      "server": "a",
      "game_index": 1
    }
  ]
}
```

目前比分應視為該回合開始前或主系統定義的比分狀態。

在契約尚未完全確認前，不得自行假設它一定是回合前或回合後比分。需要將此語意寫進測試與 fixture。

### 5.3 `events.json`

```json
{
  "events": [
    {
      "frame": 1260,
      "player": "a",
      "segment_index": 0
    },
    {
      "frame": 1295,
      "player": "b",
      "segment_index": 0
    }
  ]
}
```

`player` 只允許：

```text
a
b
```

球員代號固定綁定選手，不因換邊而改變。

### 5.4 `strokes.json`

```json
{
  "strokes": [
    {
      "event_index": 0,
      "stroke_type": "serve",
      "confidence": 0.96
    },
    {
      "event_index": 1,
      "stroke_type": "clear",
      "confidence": 0.87
    }
  ]
}
```

`event_index` 指向 `events.json` 中的陣列索引。

`StrokeLabel` 本身沒有：

* `segment_index`
* `frame`
* `player`

因此 Fact Builder 必須透過 `event_index` 與 `events.json` join。

不得假設 `strokes` 和 `events` 的陣列位置一定完全相同，應以 `event_index` 為準。

### 5.5 `highlights.json`

```json
{
  "highlights": [
    {
      "segment_index": 0,
      "score": 0.82
    }
  ]
}
```

`score` 代表該片段的觀眾反應、歡呼程度或精彩程度。

實際語意仍需與 `audio_highlight` 模組確認。賽評模組不可把它直接宣稱為「現場觀眾歡呼」，除非上游明確保證此欄位就是歡呼機率。

---

## 6. 內部 Rally Facts

Fact Builder 應將多個 JSON 合併成統一資料結構。

建議內部格式：

```json
{
  "segment_index": 12,
  "game_index": 1,
  "start_sec": 183.4,
  "end_sec": 192.7,
  "duration_sec": 9.3,
  "score": {
    "a": 19,
    "b": 19
  },
  "server": "a",
  "events": [
    {
      "event_index": 34,
      "frame": 5510,
      "time_sec": 183.67,
      "player": "a",
      "stroke_type": "serve",
      "stroke_confidence": 0.96
    },
    {
      "event_index": 35,
      "frame": 5562,
      "time_sec": 185.4,
      "player": "b",
      "stroke_type": "clear",
      "stroke_confidence": 0.87
    }
  ],
  "rally_length": 14,
  "highlight_score": 0.82,
  "tags": [
    "close_score",
    "long_rally",
    "high_excitement"
  ],
  "importance": {
    "score": 0.87,
    "reasons": [
      "比分接近",
      "回合拍數較長",
      "精彩程度較高"
    ]
  }
}
```

實際 Python schema 應使用 Pydantic model，並且：

* 驗證 `segment_index`。
* 驗證 `player` 值。
* 驗證 confidence 和 score 範圍。
* 驗證 `event_index` 是否存在。
* 對缺失資料有明確處理方式。
* 不因單一球種資料缺失而讓整個 match 失敗。
* 不默默吞掉無效索引。

---

## 7. Importance Scorer

Importance Scorer 第一版應使用確定性規則，不使用 LLM。

可能特徵：

* 比分是否接近
* 是否接近一局尾聲
* 是否可能為局點或賽末點
* rally 擊球次數
* rally 持續時間
* highlight score
* 是否出現連續進攻
* 是否以殺球或網前球結束
* 模型 confidence 是否足夠

輸出範例：

```json
{
  "score": 0.87,
  "reasons": [
    "19-19 的關鍵比分",
    "回合長度高於平均",
    "highlight score 高"
  ]
}
```

重要性分數只能用於：

* 決定是否產生賽評
* 決定文字長度
* 決定語氣強度
* 決定是否保留在精華中

它不代表模型預測正確率。

初版規則應保持簡單且可測試，避免加入大量沒有資料支持的魔法常數。

---

## 8. Commentary Planner

Planner 的工作是決定「要講什麼」，不是直接生成最終句子。

建議輸出：

```json
{
  "segment_index": 12,
  "should_comment": true,
  "style": "excited",
  "focus": [
    "close_score",
    "long_rally",
    "final_attack"
  ],
  "max_sentences": 2,
  "allowed_fact_ids": [
    "rally:12:score",
    "rally:12:length",
    "rally:12:stroke:47"
  ]
}
```

允許的 style：

```text
neutral
analytical
excited
concise
```

Planner 不得：

* 創造輸入中不存在的事件
* 修改比分
* 指定無法由 facts 支持的敘事
* 推測球員心理狀態
* 推測戰術意圖為確定事實
* 使用真實球員姓名，除非上游提供姓名對照

---

## 9. Commentary Generator

Generator 根據 Rally Facts 和 Commentary Plan 產生最終文字。

輸出應簡潔、口語自然，適合之後進行 TTS。

範例：

```text
雙方在十九平展開多拍拉鋸，A 選手持續施壓，最後拿下這個關鍵回合。
```

避免：

```text
在這場史詩級、令人窒息、無與倫比的世界頂尖對決中，
A 選手憑藉超乎常人的意志與無可匹敵的技巧完成驚天逆轉。
```

除非資料真的支持，否則不得使用：

* 驚天逆轉
* 完全壓制
* 毫無還手之力
* 體力耗盡
* 心態崩潰
* 戰術改變
* 故意欺騙
* 預判對手
* 世界級
* 生涯最佳

### 賽評語言

第一版預設：

```text
Traditional Chinese
```

使用臺灣常見羽球用語。

不要混用簡體中文。

---

## 10. Validator

Validator 必須是確定性程式，不應只靠另一個 LLM 說「看起來沒問題」。

至少檢查：

* `segment_index` 是否存在。
* `start_sec` 是否與 segment 對應。
* 是否出現不存在的球員代號。
* 是否報出 facts 中不存在的比分。
* 是否提到未被允許的球種。
* 是否超過最大句數或字數。
* 是否為空字串。
* 是否與前幾個回合高度重複。
* 低 confidence 球種是否被過度肯定。
* 是否包含未允許的球員姓名。
* 是否包含明顯無法驗證的心理或戰術推測。

Validator 發現問題時可採用：

1. 拒絕該段輸出。
2. 要求 Generator 重試。
3. 降級成模板式賽評。

必須設定最大重試次數，禁止無限重試。

---

## 11. 正式輸出契約

目前主系統期待：

```json
{
  "lines": [
    {
      "segment_index": 12,
      "start_sec": 183.4,
      "text": "雙方在十九平展開多拍拉鋸，A 選手最後拿下關鍵一分。"
    }
  ]
}
```

MVP 的 `commentary.json` 必須至少包含：

```text
segment_index
start_sec
text
```

內部可以保留較完整資料：

```json
{
  "segment_index": 12,
  "start_sec": 183.4,
  "end_sec": 188.2,
  "text": "雙方在十九平展開多拍拉鋸，A 選手最後拿下關鍵一分。",
  "importance": 0.87,
  "style": "excited",
  "source_fact_ids": [
    "rally:12:score",
    "rally:12:length"
  ]
}
```

但在主系統契約正式更新前，整合 Adapter 應輸出相容的最小格式。

不要擅自修改主系統資料契約。

---

## 12. Provider 介面

建議建立抽象介面：

```python
from typing import Protocol


class LLMProvider(Protocol):
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        ...
```

實作至少包含：

```text
FakeProvider
OpenAIProvider
GeminiProvider
```

測試必須使用 `FakeProvider`，不得在 pytest 中呼叫真實 API。

Provider 應負責：

* API 呼叫
* timeout
* retry
* response text extraction
* 廠商錯誤轉換

Provider 不應負責：

* rally facts 建立
* importance 計算
* prompt 的業務規則
* commentary validation
* JSON 輸出路徑

---

## 13. CLI 目標

預期使用方式：

```powershell
uv run badminton-commentary `
  --input .\fixtures\sample_match `
  --output .\output\commentary.json `
  --provider fake
```

真實 LLM：

```powershell
uv run badminton-commentary `
  --input .\fixtures\sample_match `
  --output .\output\commentary.json `
  --provider openai
```

CLI 至少應支援：

```text
--input
--output
--provider
--config
--overwrite
```

錯誤時應：

* 顯示是哪個輸入檔缺失。
* 顯示是哪個 schema 驗證失敗。
* 回傳非零 exit code。
* 不產生半完成的正式輸出。
* 不把 API Key 印到 console。

---

## 14. 開發環境

### Python

為了與主系統相容，目前使用：

```text
Python >=3.10,<3.11
```

### 安裝

```powershell
uv sync
```

### 加入套件

```powershell
uv add pydantic pyyaml
uv add --dev pytest ruff
```

### 執行測試

```powershell
uv run pytest
```

### Lint

```powershell
uv run ruff check .
```

### Format

```powershell
uv run ruff format .
```

---

## 15. Windows 與中文路徑注意事項

目前專案可能位於：

```text
F:\CODE\專題\badminton-commentary
```

Python 3.10 在繁體中文 Windows 上，可能使用 CP950 讀取 virtual environment 中的 `.pth` 檔。

如果 `.pth` 使用 UTF-8 儲存且內容含有中文路徑，可能出現：

```text
UnicodeDecodeError: 'cp950' codec can't decode byte ...
```

目前可將 `.pth` 重新寫為 CP950：

```powershell
$path = '.\.venv\Lib\site-packages\badminton_commentary.pth'
$content = "F:\CODE\專題\badminton-commentary\src`r`n"

[System.IO.File]::WriteAllText(
    (Resolve-Path $path),
    $content,
    [System.Text.Encoding]::GetEncoding(950)
)
```

但刪除或重建 `.venv` 後可能需要重新處理。

長期較穩定的解法是把 repo 放到純英文路徑，例如：

```text
F:\CODE\projects\badminton-commentary
```

Codex 不應為了解決此問題而修改套件 import 架構或加入 `sys.path` hack。

禁止在正式程式碼中加入：

```python
sys.path.append(...)
```

---

## 16. 第一個里程碑

第一個里程碑不使用真實 LLM API。

完成條件：

1. 建立 Pydantic 輸入 schema。
2. 建立一組完整 fixture。
3. 讀取五個輸入 JSON。
4. 依 `event_index` join events 和 strokes。
5. 建立統一 Rally Facts。
6. 計算基本 importance score。
7. 使用 `FakeProvider` 產生固定 commentary。
8. 通過 Validator。
9. 輸出合法 `commentary.json`。
10. 所有測試通過。

預期指令：

```powershell
uv run badminton-commentary `
  --input .\fixtures\sample_match `
  --output .\output\commentary.json `
  --provider fake
```

預期結果：

```json
{
  "lines": [
    {
      "segment_index": 0,
      "start_sec": 40.0,
      "text": "雙方在比分接近時展開多拍拉鋸，A 選手最後拿下這個回合。"
    }
  ]
}
```

---

