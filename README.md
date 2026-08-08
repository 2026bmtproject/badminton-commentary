# Badminton Commentary

將 Badminton Analysis System 已建立的單一 `RallyFact` 轉成可追溯的繁體中文即時賽評。

正式 runtime 回傳 structured JSON；不負責影片分析、Pose、TTS、FFmpeg 或前端播放。

## Production pipeline

```text
Upstream match analysis
        ↓
     RallyFact
        ↓
Stroke Event Analyzer ── Event Planner
Rally Analyzer        ── Rally Planner
        ↓
Rally Commentary Service
        ↓
Gemini / interchangeable Provider
        ↓
Pydantic + provenance Validators
        ↓
RallyCommentaryBundle JSON
        ↓
frontend / Badminton Analysis System adapter
```

正式使用情境是一位使用者選取一個已分析完成的 rally。該 rally 所有 selected stroke
events 與 summary 會合併成一次 Provider request；不會每個 stroke 各呼叫一次 Gemini。

## Public API

主要 API 是：

```python
from badminton_commentary import RallyCommentaryService
from badminton_commentary.providers import GeminiProvider
from badminton_commentary.schemas import RallyFact

provider = GeminiProvider.from_config(config.provider.gemini)
service = RallyCommentaryService(
    provider=provider,
    player_names={"a": "戴資穎", "b": "安洗瑩"},
)

commentary = service.generate(rally_fact=rally_fact)
```

也可以使用 functional API：

```python
from badminton_commentary import generate_rally_commentary

commentary = generate_rally_commentary(
    rally_fact=rally_fact,
    provider=provider,
    player_names={"a": "戴資穎", "b": "安洗瑩"},
)
```

上層系統不需要知道 fixture 路徑、TTYvsASY、ASS 或 FFmpeg。

### Input

正式 service 接受一個已通過 Pydantic 驗證的 `RallyFact`：

```json
{
  "segment_index": 37,
  "game_index": 1,
  "start_sec": 0.0,
  "end_sec": 8.0,
  "duration_sec": 8.0,
  "score": {"a": 20, "b": 20},
  "server": "a",
  "events": [
    {
      "event_index": 5,
      "frame": 64,
      "time_sec": 2.14,
      "player": "a",
      "stroke_type": "小球",
      "stroke_confidence": 0.91
    }
  ],
  "rally_length": 1,
  "highlight_score": null
}
```

`RallyFact` 通常由上游 adapter 或 `build_rally_facts()` 透過 `event_index` join 建立。

### Output

```json
{
  "segment_index": 37,
  "events": [
    {
      "segment_index": 37,
      "stroke_index": 5,
      "frame": 64,
      "time_sec": 2.14,
      "text": "戴資穎突然把球放短。",
      "source_fact_ids": ["rally:37:stroke:5"]
    }
  ],
  "summary": {
    "segment_index": 37,
    "text": "雙方目前戰成 20 比 20。",
    "source_fact_ids": ["rally:37:score"]
  }
}
```

`stroke_index`、`frame`、`time_sec` 由 Python 寫入。模型回傳的 stroke order 必須與
Planner 完全一致，所有文字 claim 都必須由 `source_fact_ids` 支持。

## Importance

使用者選取 rally 並呼叫 service，決定是否啟動 commentary pipeline。Importance 不再
作為整個 service 的入口 gate，而是控制輸出密度與詳細程度：

- 低 importance：只保留 `speaking_score >= 0.9` 的高 salience events，summary 僅使用比分或拍數等基本事實。
- 中 importance：使用標準 speaking policy，summary 簡短。
- 高 importance：保留標準 event density，summary 可使用更多 pattern、較完整且能有
  適度情緒。

Service 預設使用 deterministic `score_importance()`；上游已有 ImportanceResult 時可透過
`importance=` 傳入。

## Grounding guarantees

- Fact Builder、Analyzers、Planners 與 Validators 都是 deterministic Python。
- LLM 不負責建立比分、球種、擊球者、順序、時間或 tactical relation。
- Local sequence 必須由相鄰 `event_index` 支持。
- 普通發球與低資訊量 strokes 不會為了填滿賽評而被選用。
- 禁止自行宣稱致勝球、最後一拍、得分原因、戰術意圖、因果或球員移動。
- 低 confidence stroke 不會以肯定語氣輸出。
- 所有 Provider JSON 都必須通過 Pydantic 與 provenance validation。

## Generic CLI

Package CLI 處理單一 `RallyFact` JSON：

```powershell
uv run badminton-commentary `
  --input .\rally_fact.json `
  --output .\outputs\rally_37.json `
  --provider gemini `
  --config .\config.yaml `
  --player-a "戴資穎" `
  --player-b "安洗瑩"
```

CLI 不包含 TTYvsASY、批次處理多個 rallies 或 FFmpeg 邏輯。使用 FakeProvider 時必須用
`--fake-response` 提供預先建立的 batch response JSON。

## Configuration

複製範例設定：

```powershell
Copy-Item .\config.yaml.example .\config.yaml
```

API key 必須放在環境變數，不可寫入 repository。`config.yaml` 只記錄環境變數名稱、
model、timeout 與 retry 設定。

## Repository structure

```text
badminton-commentary/
├── src/badminton_commentary/
│   ├── analysis/           # deterministic facts and patterns
│   ├── generation/         # planners, batch generator, validators
│   ├── providers/          # replaceable LLM providers
│   ├── services/           # production public orchestration boundary
│   ├── cli.py              # single-rally generic CLI
│   ├── schemas.py          # Pydantic data contracts
│   ├── config.py
│   └── subtitles.py        # optional artifact utility, not service dependency
├── experiments/
│   └── ttyvsasy/           # evaluation harness and ignored local workspace
├── fixtures/
│   └── sample_match/       # small stable test fixture
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── outputs/                # generated artifacts, ignored by Git
├── config.yaml.example
└── pyproject.toml
```

## Experiments

TTYvsASY 的三組、共 15 rallies workflow 是 evaluation harness，不是 production API：

```text
experiments/ttyvsasy
        ↓ fixture preparation
production package API
        ↓
outputs/ttyvsasy JSON / ASS / rendered demo video
```

完整指令與資料配置請見
[experiments/ttyvsasy/README.md](experiments/ttyvsasy/README.md)。

ASS 與 FFmpeg 只用於 demo、evaluation、展示影片；`RallyCommentaryService` 不 import 或
執行這些功能。

## Legacy compatibility

以下功能暫時保留，production service 不依賴它們：

- `generation.event_commentator.generate_stroke_commentary()`：舊每 stroke Provider API。
- `generation.event_batch.generate_event_driven_commentary()`：舊逐 event orchestration。
- `generation.commentator.generate_commentary()` 與 `generation.batch`：舊 summary-only API。

新的整合應使用 `RallyCommentaryService`。

## Development

```powershell
uv sync
uv run pytest
uv run ruff check .
```

Python 版本：`>=3.10,<3.11`。

在含繁體中文的 Windows 路徑中，Python 3.10 可能以 CP950 讀取 editable install 的
`.pth`。若 `uv sync` 後出現 `UnicodeDecodeError`，可依下列方式重寫：

```powershell
$path = '.\.venv\Lib\site-packages\badminton_commentary.pth'
$content = (Resolve-Path '.\src').Path + "`r`n"
[System.IO.File]::WriteAllText(
    (Resolve-Path $path),
    $content,
    [System.Text.Encoding]::GetEncoding(950)
)
```
