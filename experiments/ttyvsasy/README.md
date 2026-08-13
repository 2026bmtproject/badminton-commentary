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

## Pose / court projection frame overlay

場地深度分類允許球員站在自己底線後方最多 1.5m；這些投影保留原始場地座標，
並以 `projected_point_behind_own_baseline` 標記，但戰術區域歸為 `rear`。
此容許範圍只套用於深度方向，橫向超出 0..6.1m 的投影仍會被拒絕。

若要直接在原始比賽畫面檢查 event 的 pose 與 homography 投影，可執行：

```powershell
uv run python .\experiments\ttyvsasy\scripts\overlay_pose_projection.py `
  --segment-index 144 `
  --event-index 1280 `
  --event-index 1292 `
  --top-player b `
  --bottom-player a
```

此工具必須使用與最新 stages 相同的完整來源影片；預設為
`experiments/ttyvsasy/workspace/video/TTYvsASY.mp4`。不要使用舊的 SEG144
短片，因為它與目前 stages 的絕對 frame 時間軸不一致。

輸出位於
`outputs/ttyvsasy/direct_rallyfact/seg0144/geometry/frame_overlays/`：

- 青色：court calibration quadrilateral
- 黃色直線：由 court plane 投影回畫面的球網位置
- 綠色：event hitter pose 與 bbox
- 黃色圓點：左右腳踝
- 紅色十字：腳踝中點，也是送入 inverse homography 的 image point
- 右上小圖：投影後的 court-plane 座標；超出球場時仍保留在框外以便診斷

每個 event 會產生一張可獨立開啟的 SVG；`index.html` 可並排檢視全部結果。

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

打包單一 segment，測試讓 Gemini 不經 Adapter / Fact Builder、直接從 stage slices 產生
`RallyFact`：

```powershell
uv run python .\experiments\ttyvsasy\scripts\package_direct_rallyfact.py `
  --segment-index 144 `
  --top-player b `
  --bottom-player a `
  --config .\config.yaml `
  --overwrite
```

`--config` 只是和其他實驗 CLI 保持相同呼叫介面；此腳本不呼叫 provider，也不讀取 API
key。

目前 transport input 為 `direct-rallyfact-event-centric-v4`。Python 仍讀取每一拍
-8..+10 的完整擊球方 pose window，但 LLM-facing `rally_stage_input.json` 只包含：

- `pose_features`：step width、knee angle、body height、torso lean、左右 wrist
  reach 與 body displacement 的 deterministic 2D geometry。
- `pose_keyframes`：固定 delta `[-8, -4, 0, 4, 8, 10]`，來源缺幀時直接略過。
- `rally_stage_input_debug.json`：保留完整 raw pose window，僅供 provenance、疊圖與
  geometry debug，不送入 LLM。

Geometry confidence 只在計算時 clamp 到 0..1，不修改 upstream raw pose。Python 不產生
lunge、jump、low reach、hitting arm 或正反手等語義；這些姿態／移動語義仍交由 LLM
判讀。輸入 package 升為 v4，LLM 輸出 schema 仍維持
`experimental-enriched-rally-fact-v3`。

要真正執行一次 Experimental Enriched RallyFact v3 Gemini 分析，使用專用 runner：

```powershell
uv run python .\experiments\ttyvsasy\scripts\run_direct_rallyfact_v3.py `
  --segment-index 144 `
  --top-player b `
  --bottom-player a `
  --config .\config.yaml `
  --overwrite
```

此命令會先重建 event-centric package，再進行一次 logical provider call。Gemini
回傳會先保留為 `gemini_response_v3_raw.txt`，通過 Pydantic schema、segment/event、
deterministic court observation 與 tactical evidence provenance 驗證後，才寫入
`gemini_enriched_rally_fact_v3.json`。執行時間與實際模型記錄在
`gemini_v3_run_metadata.json`。

目前使用 `Experimental Enriched RallyFact v3 — Compact Prompt`。它保留 event、
stroke、court 與 shuttle 的核心 observation，並明確支援 `front_court_exchange`、
`rear_court_exchange` 與 `repeated_posture_pattern`；tactical candidate 仍必須通過
event/frame provenance 驗證。

v3 pose observation 只保留 posture 與 secondary cues，不再輸出
`hitting_arm_candidate`。v4 input 先以肩、腕、髖、膝、踝計算 pose geometry，再由
LLM 產生 posture interpretation。

將已驗證的 v3 輸出、動態 pose skeleton、event observations 與 tactical candidates
疊到正確 SEG144 影片：

```powershell
uv run python .\experiments\ttyvsasy\scripts\visualize_enriched_rallyfact_v3.py `
  --fact .\outputs\ttyvsasy\direct_rallyfact\seg0144\gemini_enriched_rally_fact_v3.json `
  --package .\outputs\ttyvsasy\direct_rallyfact\seg0144\rally_stage_input.json `
  --debug-package .\outputs\ttyvsasy\direct_rallyfact\seg0144\rally_stage_input_debug.json `
  --video .\outputs\ttyvsasy\from_stages\seg0144\TTYvsASY_seg0144_corrected.mp4 `
  --model-label "Gemini 3.1 Pro" `
  --overwrite
```

腳本會先執行 v3 schema 與 package provenance 驗證，也會要求影片恰好是 517 frames、
1920×1080、30 FPS，避免再次使用錯誤的舊 SEG144 短片。

若 package 已經建立，只想在暫時性 503/504 後更換模型重試，可加上
`--reuse-package --model <model-name>`，避免再次載入完整 stages。

輸出位於 `outputs/ttyvsasy/direct_rallyfact/seg0144/` 與同名 ZIP。主要檔案是：

```text
rally_stage_input.json               # event-centric deterministic stage slice
rally_stage_input_debug.json         # full -8..+10 raw pose windows, debug only
prompt.txt                            # Experimental Enriched RallyFact v3 prompt
prompt_with_rally_stage_input.txt     # 可直接貼入 Gemini 的合併版本
manifest.json                         # counts、hashes 與縮減比例
```

v4 不會把完整 19-frame raw pose window 傳給 LLM。每個 event 只附帶 deterministic
`pose_features` 與最多六個固定-delta keyframes；每個 keyframe 有 10 個具名 keypoints
（肩、腕、髖、膝、踝，不含鼻子與手肘）。Shuttle 仍保留指定 method 的 ±6 frames
有效可見點；無效 records 只計入 `excluded_points`。

Court 的 inverse homography、擊球位置、player-relative front/mid/rear 與同球員前後變化改由
slicer 進行 deterministic geometry preprocessing；raw homography 不送入 LLM。深度定義為
`0=net/front`、`1=baseline/rear`，前後變化使用 `0.08` normalized depth（約 0.54m）門檻避免
雜訊。`confirmed=false` 仍可使用，只有 `detection_failed=true` 或無法 inverse/project 時才
不產生 court position。

Court source 依序使用 event frame 的雙腳踝中點、單腳踝、bbox bottom center；event frame
沒有該球員 pose 時才選 pose window 內距離最近的真實 record。腳踝最低 confidence 是
`0.5`；單腳 fallback 對 confidence 乘 `0.75`，bbox fallback 固定為 `0.35` 並記錄
limitation。所有 confidence 規則均由 Python deterministic 計算。

Slicer 不產生 Compact Facts、hitting-arm label、posture label、正反手、shuttle direction 或
戰術判斷；其中 `court_point_m`、`depth_zone`、
`position_change_from_previous_same_player_hit` 是數值幾何 quantization，不是 tactical
reasoning。LLM 預期輸出 `experimental-enriched-rally-fact-v3`，且 v3 明確禁止推論正反手。

套件刻意不包含現有 Fact Builder 的答案，避免污染 direct stages-to-RallyFact 實驗。

視覺檢查 deterministic court projection：

```powershell
uv run python .\experiments\ttyvsasy\scripts\visualize_rally_geometry.py `
  --segment-index 144 `
  --top-player b `
  --bottom-player a `
  --config .\config.yaml
```

輸出在 `outputs/ttyvsasy/direct_rallyfact/seg0144/geometry/`：

- `court_geometry.html`：球場圖與逐 event 表格，最適合人工檢查。
- `court_geometry.svg`：可縮放的球場投影圖。
- `court_geometry_report.json`：image point、court point、合法範圍與 rejection reason。

圖中藍色與橘色分別是 top/bottom 的有效點，紫色是球場內但落在錯誤球員半場的點，紅色
叉號是 6.1×13.4m 球場外投影。報告會分別記錄 `within_court_bounds` 與
`within_player_half`，方便判斷是 homography 邊界問題或 player-side 問題。

已保存的 direct RallyFact prompt/output runs 位於：

```text
experiments/ttyvsasy/evaluations/direct_rallyfact/
└── seg0144/
    └── v2/
        ├── prompt.txt
        ├── gemini_output.json
        ├── metadata.json
        └── README.md
```

這些是研究 artifacts，不會自動進入 production Planner 或 Commentator。

`top/bottom` 是該 segment 的場上位置；請依 identity frame／已知換邊狀態明確指定。
目前已確認的三組 evaluation mapping：

- segment 39–43：`top=a`、`bottom=b`
- segment 52–56：`top=b`、`bottom=a`
- segment 140–144：`top=b`、`bottom=a`

腳本會在 commentary JSON 旁保留 `rally_fact.json`、`compact_facts.json` 與
`tactical_facts.json`；可分別用 `--fact-output`、`--compact-output`、`--tactical-output`
改變位置。每個所選 rally 會呼叫 Tactical Analyzer 一次，再呼叫現有 Commentator 一次；
Planner 尚未使用 tactical facts。

Tactical Analyzer 預設使用 `config.yaml` 的 `tactical_analyzer.model`，範例是
`gemini-3.1-pro-preview`；若 Preview 暫時不可用，會依 `fallback_models` 改用
`gemini-3.6-flash`。Commentator 則使用 `provider.gemini.model`。終端與
`tactical_facts.json` 都會記錄實際 tactical model。

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
