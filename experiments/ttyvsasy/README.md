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

每組 selected clip 包含 `source_mapping.json`、`commentary_input/` 與原始合併影片。
Pose/MMPose 不屬於本 commentary experiment，也不再是 split script 的輸入。

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

同一 rally 的 selected events 與 summary 透過 `RallyCommentaryService` 一次送出，因此
完整三組正常是 15 次 Gemini calls。

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
