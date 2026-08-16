# SYQ evaluation workspace

## Status：已淘汰

2026-08-16 決定捨棄這場比賽。來源影片只有 640×360，畫面不足以支援可靠的逐幀正反手人工審核，因此：

- 不進行六個 rally 的人工標註。
- 不計算或引用 selective accuracy、coverage、fixed accuracy。
- 不把這場資料當成 C 的跨比賽 hold-out。
- 已產生的 candidate 與 review viewer 只保留為可重建的失敗實驗產物。
- 原始影片與 stages 屬於使用者資料，不因淘汰決策自動刪除。

以下內容保留作為 input quality-control 與淘汰歷程紀錄。

這個目錄預留給第二場比賽 `SYQ`，用來驗證目前凍結的正反手候選策略：

```text
orientation_policy             = court_prior
racket_joint_min_score         = 0.50
minimum accepted racket frames = 3
```

`SYQ` 是獨立 hold-out match，不得在人工審核前依照它的結果調整上述參數。

## 放置輸入

大型資料全部放在被 Git 忽略的 `workspace/`：

```text
experiments/syq/
├── README.md
└── workspace/                       # ignored by Git
    ├── video/
    │   └── SYQ.mp4                  # 建議名稱，可使用其他檔名
    └── stages/
        ├── match_segmentation/
        │   └── segments.json
        ├── event_detection/
        │   └── events.json
        ├── score_recognition/
        │   └── scores.json
        ├── stroke_classification/
        │   └── strokes.json
        ├── pose/
        │   └── pose.json
        ├── court_detection/
        │   └── court.json           # optional for this experiment
        └── shuttle_tracking/
            └── shuttle.json         # optional for this experiment
```

請直接放主 Badminton Analysis System 的原始 stage artifacts，不要先轉成 `commentary_input/*.json`，也不要重新命名 JSON 欄位。各 stage 的 `status.json` 可以一起保留。

正反手實驗需要前四個正式 commentary stages 加上 `pose/pose.json`；目前不使用 court 與 shuttle 產生正反手結果。

## 已完成的輸入檢查

2026-08-16 audit 結果：

| Item | Result |
| --- | ---: |
| Video | 640×360、30 FPS、155400 frames |
| Segments / score rallies | 118 / 118 |
| Events / strokes | 1045 / 1045 |
| Stroke-event join errors | 0 |
| Valid / missing pose records | 101252 / 618 |
| Shuttle points / schema errors | 101870 / 0 |
| Court calibration | present，但 upstream `confirmed=false` |

影片 FPS、segment 最大 frame 與全部 stroke/event/segment join 均一致。Court 不參與本輪正反手分類。

可重跑：

```powershell
uv run python .\experiments\syq\scripts\audit_inputs.py
```

## 已取消的跨場次 hold-out

在執行 C classifier 前，已依時間區段、rally 長度、stroke 數與輸入 pose availability 選定：

```text
SEG12, SEG25, SEG51, SEG72, SEG92, SEG111
```

球員代號固定為 `a=SHI Y.Q.`、`b=FAIHAN`，並依比賽換邊逐 segment 記錄 top/bottom mapping。完整證據 frame 與 mapping 位於 `holdout_selection.json`。

C review set 曾產生在：

```text
outputs/syq/forehand_backhand/holdout_c_review/
```

這批 viewer 不應再用於正式人工審核。原入口僅供檢查淘汰原因：

```powershell
Start-Process .\outputs\syq\forehand_backhand\holdout_c_review\index.html
```

六段共 76 拍，C 在人工審核前產生 24 個 unknown candidate；由於輸入解析度不合格，這些數字不進入模型品質評估。

重建 review set：

```powershell
uv run python .\experiments\syq\scripts\generate_c_holdout_review.py
```

## 建置遵循的流程

本次沒有在資料到位後直接產生 accuracy，而是依序完成：

1. 驗證七個 stage schema 與 event/frame/segment join。
2. 確認影片 FPS、frame count 與 `match_segmentation` 一致。
3. 由畫面確認 `top`、`bottom` 對應的球員代號；不得從檔名猜 mapping。
4. 在不查看正反手結果的前提下，依 rally 長度挑選 hold-out segments。
5. 使用凍結的 C + 3-frame gate 產生逐幀人工 reference viewer。
6. 分開報告 coverage、fixed-denominator accuracy 與 selective accuracy。

## 單一 rally smoke test 範例

單一 segment 仍可重用目前的實驗引擎：

```powershell
uv run python .\experiments\ttyvsasy\scripts\experiment_forehand_backhand.py `
  --stage-root .\experiments\syq\workspace\stages `
  --video .\experiments\syq\workspace\video\syq.mp4 `
  --segment-index <SEGMENT_INDEX> `
  --top-player <a-or-b> `
  --bottom-player <a-or-b> `
  --orientation-policy court_prior `
  --output .\outputs\syq\forehand_backhand\segXXXX_C_court_prior
```

`<SEGMENT_INDEX>` 與 player mapping 應依 `holdout_selection.json` 或實際畫面填寫。輸出一律放在被 Git 忽略的 `outputs/syq/`。
