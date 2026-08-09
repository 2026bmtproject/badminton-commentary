# TTYvsASY evaluation workflow

這個目錄保留三組連續五個 rallies 的研究、品質評估與展示影片流程。它是 production
package 的 consumer，不是正式 runtime pipeline。

## Purpose

- 驗證 15 個 rallies 的 Fact Builder、Analyzers、Planners 與 provenance。
- 比較 Fake/Gemini commentary 品質。
- 量測每 rally Gemini latency 與總時間。
- 產生 ASS 與燒錄字幕的 demo MP4。

## Local input

本地資料放在被 Git 忽略的 `experiments/ttyvsasy/workspace/`：

```text
workspace/
├── stages/                  # full-match upstream stage outputs
└── selected_clips/
    ├── seg0039-0043/
    ├── seg0052-0056/
    └── seg0140-0144/
```

每組 selected clip 包含 `source_mapping.json`、各 stage artifacts 與原始合併影片。
Pose/MMPose 不屬於本 commentary experiment，也不再是 split script 的輸入。

`build_facts.py` 直接以 production `StagePaths`、Upstream Stage Adapter 和 Fact Builder
讀取 `selected_clips/{group}/stages/`。既有 `commentary_input/` 僅是研究期間留下的
normalized inspection artifacts，不是 production 或目前 experiment build 的必要輸入。

## Outputs

所有生成物放在被 Git 忽略的：

```text
outputs/ttyvsasy/{group}/
├── rally_facts.json
├── rally_analyses.json
├── commentary_plans.json
├── commentary_fake_event_driven.json
├── commentary_gemini_event_driven.json
├── subtitles/
└── video/
```

## Commands

從完整 stages 重新切分三組資料：

```powershell
uv run python .\experiments\ttyvsasy\scripts\split_stages.py
```

建立 deterministic facts、analysis 與 plans：

```powershell
uv run python .\experiments\ttyvsasy\scripts\build_facts.py
```

直接從完整主系統 stages 測試單一 rally（不經 `commentary_input` 或預建 facts）：

```powershell
uv run python .\experiments\ttyvsasy\scripts\generate_selected_rally.py `
  --segment-index 39 `
  --top-player a `
  --bottom-player b `
  --provider fake `
  --overwrite
```

使用 Gemini：

```powershell
uv run python .\experiments\ttyvsasy\scripts\generate_selected_rally.py `
  --segment-index 39 `
  --top-player a `
  --bottom-player b `
  --provider gemini `
  --config .\config.yaml `
  --overwrite
```

`top/bottom` 是該 segment 的場上位置；請依 identity frame／已知換邊狀態明確指定。
目前已確認的三組 evaluation mapping：

- segment 39–43：`top=a`、`bottom=b`
- segment 52–56：`top=b`、`bottom=a`
- segment 140–144：`top=b`、`bottom=a`

腳本會在 commentary JSON 旁保留 `rally_fact.json` 與 `compact_facts.json`；可分別用
`--fact-output`、`--compact-output` 改變位置。Compact facts 目前只供 inspection 與下一階段
Tactical Analyzer 使用，尚未送進現有 Gemini Commentator。

FakeProvider 離線 smoke test：

```powershell
uv run python .\experiments\ttyvsasy\scripts\generate_commentary.py `
  --provider fake `
  --mode event-driven
```

Gemini evaluation：

```powershell
uv run python .\experiments\ttyvsasy\scripts\generate_commentary.py `
  --provider gemini `
  --mode event-driven `
  --config .\config.yaml
```

同一 rally 的所有可描述 strokes（包含發球、一般回球與低 confidence 結果）及 summary
透過 `RallyCommentaryService` 一次送出，因此完整三組正常是 15 次 Gemini calls。stroke
數量會增加單次 request 的內容與輸出長度，但不會增加每個 rally 的 API call 次數。

只建立 ASS：

```powershell
uv run python .\experiments\ttyvsasy\scripts\burn_subtitles.py `
  --provider gemini `
  --subtitles-only `
  --overwrite
```

燒錄 demo MP4：

```powershell
uv run python .\experiments\ttyvsasy\scripts\burn_subtitles.py `
  --provider gemini `
  --overwrite
```

FFmpeg 只在最後一個 demo 步驟使用，不是 production service dependency。
