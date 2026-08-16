# C 版本正反手 Hold-out 人工審核 Protocol

日期：2026-08-15

## 固定候選版本

本輪在人工標註前凍結以下設定：

```text
orientation_policy             = court_prior
base pose threshold            = 0.30
racket joint minimum score     = 0.50
minimum accepted racket frames = 3
hit window                     = ±2 frames
```

這就是 A/B/C/D ablation 的 C 版本。人工審核期間不得再依照結果調整參數。

## Rally 選擇

| Segment | Hits | Duration | C candidate unknown | 用途 |
| ---: | ---: | ---: | ---: | --- |
| 130 | 22 | 21.97 sec | 3 | 新 hold-out |
| 132 | 7 | 10.07 sec | 1 | 新 hold-out |
| 136 | 14 | 22.00 sec | 3 | 新 hold-out |
| 139 | 16 | 13.83 sec | 6 | 新 hold-out |
| 144 | 17 | 17.23 sec | 4 | 指定的 C 版複審 |
| 146 | 8 | 15.30 sec | 3 | 新 hold-out |

總計 84 拍，其中 20 拍為 C candidate unknown。SEG130、132、136、139、146 沒有參與 A/B/C/D 選型；選擇依據只有 rally 長度與 hit 數，沒有查看 C 預測或人工 verdict。SEG144 的舊標註曾參與 ablation，因此必須單獨報告，不能混入新的 hold-out 泛化指標。

## 人工標註格式 v3

舊版 reviewer 記錄候選是否正確，遇到錯誤 unknown 時無法得知真正側別。本輪改成直接記錄人工 reference：

| Key | 人工標註 |
| --- | --- |
| `F` | forehand／正手 |
| `B` | backhand／反手 |
| `U` | 畫面不足，無法判定 |
| `E` | 匯出該 rally 的 JSON |

輸出 schema：

```text
experimental-forehand-backhand-human-reference-v3
```

每拍會保留 candidate side、人工 `reference_side`、`review_status` 與由兩者計算出的 verdict。即使 candidate 是 unknown，只要人工能判斷，仍可留下明確 forehand/backhand reference。

## 審核入口

```powershell
Start-Process .\outputs\ttyvsasy\forehand_backhand\holdout_c_review\index.html
```

標註時至少查看 hit frame 前後數幀。若骨架疊圖錯誤但原始影片仍足以判斷，可以依原始動作標正反手；若動作、球拍或擊球時序不足，標記 `U`，不要猜測。

每個 rally 完成後按 `E`，會下載：

```text
segXXXX_forehand_backhand_human_review.json
```

## 統計規則

完成後需分開計算：

1. 新 hold-out：SEG130、132、136、139、146。
2. 指定複審：SEG144。
3. 固定分母 accuracy：人工可判定的 reference 為分母，candidate unknown 算錯。
4. selective accuracy：只計算 C 有輸出 forehand/backhand 的拍。
5. coverage：C 有輸出 forehand/backhand 的比例。
6. undecidable rate：人工標記 `U` 的比例。

在新 hold-out 結果完成前，C 仍是 experimental candidate，不得加入 verified `RallyFact`。

六個 rally 的人工審核已完成；統計與錯誤分析見 [forehand_backhand_c_holdout_results.md](forehand_backhand_c_holdout_results.md)。
