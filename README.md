# Badminton Commentary

將 Badminton Analysis System 的 stage outputs 轉成可追溯的繁體中文即時賽評。

正式 runtime 回傳 structured JSON；讀取既有 stage artifacts，但不執行電腦視覺模型、
TTS、FFmpeg 或前端播放。

## Production pipeline

```text
Badminton Analysis System
│
├── match_segmentation ─┐
├── event_detection ────┤
├── score_recognition ──┤
├── stroke_classification
│                       │
├── court_detection ────┤
├── shuttle_tracking ───┤
└── pose ────────────────┘
                        ↓
                Upstream Stage Adapter
                        ↓
          RallyFact + CompactRallyFacts
                        ↓
          ┌──────── current ────────┐
Stroke Event Analyzer ── Event Planner
Rally Analyzer        ── Rally Planner
          └─────────────────────────┘
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

Current Milestone 3:
CompactRallyFacts → Gemini Tactical Analyzer → TacticalFact[]

Next milestone:
TacticalFact[] → Planner → Gemini Commentator → RallyCommentaryBundle
```

`RallyFact` 是 badminton-commentary 內部的 canonical domain representation，不是要求
Badminton Analysis System 預先建立的外部資料格式。`CompactRallyFacts` 則是 selected
segment 的 verified multimodal representation；目前可送入獨立的 Gemini Tactical
Analyzer，但 TacticalFact 尚未接入 production Planner／Commentator。

正式使用情境是一位使用者選取一個已分析完成的 rally。使用者的選擇本身就是啟動條件；
該 rally 中所有具有球種與 confidence 的 strokes（包含普通發球與低 confidence 結果）
都會依時間順序放進同一次 Provider request。具有明確 player mapping 的 stroke 必須各自
產生一個 event；無法確認 player 的 stroke 只作為完整序列上下文，模型不得猜測擊球者。
Rally summary 也合併在同一次請求，不會每個 stroke 各呼叫一次 Gemini。

## Public API

主系統的高階 API 直接接受已解析的 stage models：

```python
from badminton_commentary import RallyCommentaryService
from badminton_commentary.adapters import (
    CourtPositionToPlayer,
    StagePaths,
    read_upstream_stages,
)

stages = read_upstream_stages(
    StagePaths.from_stage_root(match_path / "stages"),
    segment_index=37,
)
service = RallyCommentaryService(
    provider=commentary_provider,
    tactical_provider=tactical_provider,
    player_names=player_names,
)

bundle = service.generate_from_stages(
    stages=stages,
    segment_index=37,
    court_position_to_player=CourtPositionToPlayer(top="b", bottom="a"),
)
```

Filesystem reader 是外層便利功能；核心 adapter 接受 typed `UpstreamStageData`，不自行尋找
固定資料夾。指定 `segment_index` 時，reader 會串流 pose/shuttle 大型 JSON array，只保留
該 segment，並讀取小型 court calibration；未指定時維持舊行為，只讀四個核心 stages。

主系統原始 `stroke_classification.player` 是場上位置 `top/bottom`，而比分使用固定球員
代號 `a/b`。目前 stages 沒有提供兩者的身分關係，因此 caller 必須明確提供所選 segment
的 `CourtPositionToPlayer`；adapter 不會依換邊或擊球順序猜測。

需要檢查中間結果時，可以停在 `RallyFact`：

```python
rally_fact = service.prepare_rally_fact(
    stages=stages,
    segment_index=37,
    court_position_to_player=position_mapping,
)

compact_facts = service.prepare_compact_facts(
    stages=stages,
    segment_index=37,
    court_position_to_player=position_mapping,
)

tactical_facts = service.analyze_tactics(compact_facts=compact_facts)
```

Compact schema、演算法與品質 gate 詳見
[docs/compact-facts.md](docs/compact-facts.md)；TacticalFact schema、prompt boundary 與
provenance gates 詳見 [docs/tactical-analyzer.md](docs/tactical-analyzer.md)。

既有較低階 API 保留：

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

上層系統不需要建立 `commentary_input/*.json`，也不需要知道 fixture、TTYvsASY、ASS 或
FFmpeg。

### Input

較低階 service 接受一個已通過 Pydantic 驗證的 `RallyFact`：

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

Stage Adapter 依絕對 frame 選出 requested segment，驗證
`stroke_classification.event_index → event_detection.events[]`、frame 與 segment_index，
再交給 Fact Builder 建立 `RallyFact`。缺少 optional highlight stage 時使用 `None`；不會
從其他 stage 推論 highlight。

實際 schema、join 演算法與 TTYvsASY fixture-specific 邊界詳見
[docs/upstream-stage-adapter.md](docs/upstream-stage-adapter.md)。

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

## Selection、Importance 與逐拍輸出

使用者選取 rally 並呼叫 service，決定是否啟動 commentary pipeline。Production 的逐拍
事件不再使用 Importance 或 salience 篩選：普通發球、一般回球與低 confidence stroke 都
會送給 Provider，且具有 player mapping 的每拍都必須輸出。

Production `RallyCommentaryService` 不再呼叫 `score_importance()`；summary 使用固定的
user-selected rally planner，依可用的 score、拍數、pattern 與 highlight facts 規劃內容。
既有 `ImportanceResult` 與 `plan_commentary()` 保留給舊 summary-only API、fixtures 與相容性，
不會刪除任何 production 逐拍 event。`speaking_score` 也只保留給舊的稀疏輸出流程與受控
驚嘆號判斷，不是 production event inclusion gate。

## Grounding guarantees

- Fact Builder、stroke/rally Analyzers、Planners 與 Validators 是 deterministic Python；
  Gemini Tactical Analyzer 由 LLM 提出 candidate，再由 deterministic gates 驗證。
- LLM 不負責建立比分、球種、擊球者、順序、時間或其他觀測 facts；Tactical Analyzer
  只提出可追溯的戰術候選，程式會驗證 evidence、event range 與 pattern-specific evidence。
- Local sequence 必須由相鄰 `event_index` 支持。
- 所有具有球種與 confidence 的 strokes 都進入 ordered context，包含普通發球。
- 具有 player mapping 的每個 stroke 都有對應 event；缺少 mapping 時不得猜測擊球者。
- 禁止自行宣稱致勝球、最後一拍、得分原因、戰術意圖、因果或球員移動。
- 低 confidence stroke 保留，但必須使用「可能」、「似乎」或「辨識結果」等保守措辭。
- Provider 只有措辭違反語言安全規則時，batch 會以同一組 grounded facts 建立 deterministic
  fallback；provenance、順序、比分與 current stroke 錯誤仍會拒絕整批。
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
model、timeout 與 retry 設定。Commentator 與 Tactical Analyzer 可使用不同模型；範例預設
分別為 `gemini-flash-latest` 與 `gemini-3.1-pro-preview`。若 Pro Preview 暫時回傳容量或
服務錯誤，Tactical Analyzer 會自動降級至 stable `gemini-3.6-flash`，並在輸出記錄實際模型。

## Repository structure

```text
badminton-commentary/
├── src/badminton_commentary/
│   ├── analysis/           # deterministic patterns + validated tactical analyzer
│   ├── adapters/           # main-system stage schemas/readers/normalization
│   ├── facts/              # compact multimodal and tactical schemas
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
│   ├── sample_match/       # normalized Fact Builder fixture
│   └── upstream_stages/    # actual main-system stage structure fixture
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

目前架構、演算法、Git 演進與 implementation audit 見
[docs/reports/algorithm_development_report.md](docs/reports/algorithm_development_report.md)；
逐項驗證矩陣見
[docs/reports/implementation_verification_report.md](docs/reports/implementation_verification_report.md)；
2D pose 正反手 confidence-gate 的人工審核與失敗分析見
[docs/reports/forehand_backhand_confidence_gate_review.md](docs/reports/forehand_backhand_confidence_gate_review.md)；
orientation A/B/C/D 比較見
[docs/reports/forehand_backhand_orientation_ablation.md](docs/reports/forehand_backhand_orientation_ablation.md)。

```powershell
uv sync
uv run pytest
uv run ruff check .
```

Python 版本固定為 `3.12.x`。Repository 內的 `.python-version` 會讓 uv 在執行
`uv sync` 時自動選擇或安裝相容的 Python 3.12。

```powershell
uv run python --version
```

預期輸出為 `Python 3.12.x`。
