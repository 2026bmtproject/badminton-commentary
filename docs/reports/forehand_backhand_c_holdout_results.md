# C 版本正反手 Hold-out 人工審核結果

日期：2026-08-15

## 資料驗證

六份人工 reference 均通過以下驗證：

- schema 為 `experimental-forehand-backhand-human-reference-v3`
- `segment_index` 與 C 版結果一致
- event 集合完整且沒有重複
- local frame、player、stroke、candidate side、margin 與被審核結果一致
- 沒有 unreviewed event

SEG130、132、136、139、146 是未參與 C 選型的新 hold-out。SEG144 曾參與 orientation ablation，只能作為指定複審，不能混入主要泛化數字。

## 主要結果：新 hold-out

| 指標 | 結果 |
| --- | ---: |
| Rally | 5 |
| Total events | 67 |
| Human-labeled side | 61 |
| Human uncertain | 6 |
| Human-decidable rate | 91.04% |
| C binary predictions | 48 |
| C unknown on labeled events | 13 |
| Coverage | 78.69% |
| Correct binary predictions | 47 |
| Binary errors | 1 |
| Fixed-denominator accuracy | 47/61 = 77.05% |
| Selective accuracy | 47/48 = 97.92% |

固定分母 accuracy 使用全部 61 個人工可判定 event，C 的 unknown 算錯。Selective accuracy 只計算 C 有輸出正手或反手的 48 拍。

### Confusion matrix

| Reference \\ Prediction | Forehand | Backhand | Unknown |
| --- | ---: | ---: | ---: |
| Forehand | 29 | 1 | 9 |
| Backhand | 0 | 18 | 4 |

### 分 rally 結果

| Segment | Human labeled | Coverage | Fixed accuracy | Selective accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 130 | 20 | 90.00% | 85.00% | 94.44% |
| 132 | 7 | 85.71% | 85.71% | 100.00% |
| 136 | 14 | 78.57% | 78.57% | 100.00% |
| 139 | 13 | 61.54% | 61.54% | 100.00% |
| 146 | 7 | 71.43% | 71.43% | 100.00% |

## 唯一的新 hold-out 二元誤判

```text
segment       = 130
event_index   = 1116
player        = a
stroke_type   = 高遠球
prediction    = backhand
reference     = forehand
margin        = 0.8431
accepted pose = 3 frames
```

這是高 margin、剛好通過 3-frame gate 的錯誤，不能以低 margin 解釋。C 雖大幅改善原 orientation branch 的系統性錯誤，但 court prior only 仍不是完美的正反手規則。

## Coverage 是目前主要瓶頸

13 個人工可判定 event 被 C 輸出為 unknown，其 accepted racket frames 分布為：

| Accepted frames | Events |
| ---: | ---: |
| 0 | 3 |
| 1 | 4 |
| 2 | 6 |

目前 gate 要求至少 3 frames，所以這 13 拍全部被拒絕。不能直接把門檻降到 2，因為需要先用已取得的 explicit reference 做 threshold ablation，確認新增 coverage 是否引入二元誤判。

觀察到明顯的不對稱：

| Player | Human labeled | Coverage | Fixed accuracy | Selective accuracy |
| --- | ---: | ---: | ---: | ---: |
| a | 29 | 65.52% | 62.07% | 94.74% |
| b | 32 | 90.62% | 90.62% | 100.00% |

這表示 gate／pose availability 對 player a 的拒絕率較高，但目前資料不能證明原因是球場遠近、遮擋、追蹤或 pose model；必須回看 rejected-frame diagnostics 才能歸因。

依 stroke type，高遠球的 coverage 最低：10 個人工可判定高遠球只有 5 個得到 binary prediction，且其中 1 個錯誤。因此高遠球 fixed accuracy 為 40%，不宜只引用全體 97.92% selective accuracy。

## SEG144 指定複審

| 指標 | 結果 |
| --- | ---: |
| Human-labeled side | 16 |
| Human uncertain | 1 |
| Coverage | 13/16 = 81.25% |
| Fixed accuracy | 12/16 = 75.00% |
| Selective accuracy | 12/13 = 92.31% |

二元錯誤為 event 1293：C 預測 forehand，新 v3 reference 標為 backhand。三個人工可判定但 candidate unknown 的事件為 1285、1289、1292。

SEG144 舊版 review 與新版 v3 在 16 個共同可判定 event 中只有 event 1293 不一致：舊 review 推導為 forehand，新 review 明確標為 backhand。因此 SEG144 指標需保留 annotation disagreement 註記；event 1293 適合再做一次盲式 adjudication。

## 決策

1. C 的 orientation 策略在新 hold-out 上得到 97.92% selective accuracy，支持淘汰原本的 pose orientation flip branch。
2. 目前整體 fixed accuracy 仍只有 77.05%，主要損失來自 confidence gate coverage，不應宣稱系統已有 97.92% 的逐拍正反手完整辨識率。
3. 正反手仍保持 experimental，不進入 verified `RallyFact`。
4. 下一個實驗應固定 C orientation，只比較 `min_racket_frames=1/2/3` 與 joint-confidence threshold；使用這批 v3 reference 作開發分析後，仍需另一場比賽做最終驗證。
5. SEG144 event 1293 先完成人工 adjudication，再決定該 rally 的最終 reference。

## 可追溯輸出

- `outputs/ttyvsasy/forehand_backhand/holdout_c_review/c_holdout_review_summary.json`
- `outputs/ttyvsasy/forehand_backhand/holdout_c_review/c_holdout_review_summary.md`
- 各 `segXXXX_C_court_prior/human_reference.json`

重跑分析：

```powershell
uv run python .\experiments\ttyvsasy\scripts\analyze_forehand_backhand_c_holdout_review.py `
  --review-files `
  "F:\Downloads\seg0130_forehand_backhand_human_review.json" `
  "F:\Downloads\seg0132_forehand_backhand_human_review.json" `
  "F:\Downloads\seg0136_forehand_backhand_human_review.json" `
  "F:\Downloads\seg0139_forehand_backhand_human_review.json" `
  "F:\Downloads\seg0144_forehand_backhand_human_review (1).json" `
  "F:\Downloads\seg0146_forehand_backhand_human_review.json"
```

