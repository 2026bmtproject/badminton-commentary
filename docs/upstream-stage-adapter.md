# Upstream Stage Adapter

這份文件記錄 commentary 對 Badminton Analysis System stage artifacts 的正式 integration
contract。內容來自實際 TTYvsASY artifacts 與主系統 `modules/contracts.py`，不是依資料夾名稱
猜測。

## 實際 consumed schema

| Stage | Commentary 使用欄位 | 實際語意 |
| --- | --- | --- |
| `match_segmentation/segments.json` | `fps`, `segments[].start_frame/end_frame/start_sec/end_sec/duration_sec` | segment index 是 `segments` 陣列位置；frame 是原片絕對 frame |
| `event_detection/events.json` | `events[].frame` | 每拍只有絕對 frame，沒有 player 或 segment index |
| `score_recognition/scores.json` | `rallies[].segment_index/score_a/score_b/server/game_index` | `a/b` 是固定球員／記分板列，不是場上 top/bottom |
| `stroke_classification/strokes.json` | `event_index/frame/segment_index/player/stroke_type/confidence` | `event_index` 指向 `events` 陣列；player 是場上 `top/bottom` 或 unknown |
| `audio_highlight/highlights.json` | `segment_index/score` | optional；目前 TTYvsASY 沒有這個正式 stage |

主系統的 segment 秒數目前以三位小數輸出。因 `start_sec`、`end_sec`、`duration_sec` 分別
量化到毫秒，`duration_sec` 與 `end_sec - start_sec` 允許最多 `0.001` 秒差異；超過此範圍
仍視為契約錯誤。Adapter 不會修改上游檔案。

`court_detection`、`shuttle_tracking`、`pose` 目前不是 commentary fact source。`StagePaths`
保留 optional path hook，但 reader 不解析它們，Fact Builder、Analyzer 與 prompt 都不依賴它們。

## Player identity boundary

`stroke_classification` 只能回答擊球者位於 `top` 或 `bottom`；`RallyFact` 與比分使用固定
`a/b` identity。現有四個 consumed stages 沒有提供兩者的 mapping，而且球員換邊後 mapping
會改變。

因此 adapter 要求 caller 對所選 segment 提供：

```python
CourtPositionToPlayer(top="b", bottom="a")
```

缺少 mapping 時不會依發球、擊球交替或比分猜測。若 BST 明確輸出 `player: null`，stroke
會保留在 `RallyFact` 與 Gemini ordered context 中，但不會建立具名逐拍 event；Generator
不得猜測擊球者。

## Adapter algorithm

對 requested `segment_index`：

1. 驗證 segment 存在並取得絕對 frame range。
2. 只選取 frame 落在該 range 的 raw hit events。
3. 以 raw `event_index` 找到 stroke，驗證 event/stroke frame 與 segment index 一致。
4. 使用 caller 提供的 mapping 將 `top/bottom` 正規化成 `a/b`。
5. 只取同一 segment 的 score 與 optional highlight。
6. 建立 Fact Builder 所需的 normalized Pydantic inputs。
7. 呼叫 production `build_rally_facts()` 完成 score/event/stroke join。
8. 回傳單一 `RallyFact`，保留原始全場 event index 供 provenance 使用。

若 `scores.json` 的 `sub_scores` 表示一個 segment 內恢復出多個 rallies，目前會明確拒絕；
現有 `RallyFact` 尚無 sub-rally identifier，不能安全地自行選擇其中一個。

## TTYvsASY fixture-specific boundary

仍留在 `experiments/ttyvsasy/scripts/split_stages.py` 的工作：

- 固定選擇三組連續五個 segments。
- 依 `source_mapping.json` 將全場 frame 轉成合併 clip frame。
- 重排 local segment/event indexes。
- 依人工確認的 identity frame 寫入每組 player mapping。
- 複製 demo 所需的 shuttle/court artifacts。

已移入 production adapter 的工作：

- 解析主系統四個 consumed stage schemas。
- 依 segment frame range 選 event。
- 驗證並 join event/stroke。
- `top/bottom` 到 `a/b` 的 typed mapping。
- score 與 optional highlight selection。
- normalized inputs 到 `RallyFact` 的 Fact Builder 呼叫。

`commentary_input/*.json` 只保留作歷史 inspection/debug artifacts；正式主系統與目前
TTYvsASY `build_facts.py` 都不依賴它們。
