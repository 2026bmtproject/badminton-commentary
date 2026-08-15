# 正反手 Confidence Gate 人工審核紀錄

日期：2026-08-13

## 實驗目的

上一輪 SEG144 的錯誤案例顯示，低品質骨架仍可能被規則式正反手分類器輸出為確定結果。本輪加入 racket shoulder、elbow、wrist 的逐幀 confidence gate；hit window 內少於 3 個合格 frame 時輸出 `unknown`，並以 SEG140、SEG141、SEG143、SEG144 進行人工審核。

這是單場比賽內的開發實驗，不是跨場次的 production accuracy 驗證。

## 實驗設定

```text
base pose threshold             = 0.30
racket shoulder/elbow/wrist min = 0.50
required accepted frames        = 3
hit window                      = ±2 frames
```

分類器的 `heuristic_margin` 只是幾何規則分數，不是校準後的機率。人工審核中的：

- `correct` / `incorrect`：判斷非 null 的正反手輸出。
- unknown 被標為 `correct` 或 `uncertain`：視為合理棄權。
- unknown 被標為 `incorrect`：視為不必要棄權；因 review v2 沒有記錄正確側別，不能放入正反手 confusion matrix。

## 輸入相容性

| Segment | 結果 |
| ---: | --- |
| 140 | confidence-gated v2 review 與結果相符 |
| 141 | confidence-gated v2 review 與結果相符 |
| 143 | confidence-gated v2 review 與結果相符 |
| 144 | 上傳檔為舊版 v1 baseline；event 1285 的 side 與 margin 不符新版結果，因此不納入新版彙整 |

SEG144 必須從 `seg0144_confidence_gate/frame_review.html` 重新匯出，才能加入同一組統計。舊檔仍可作為修改前 baseline，但不能冒充 confidence-gated 結果。

## 人工審核結果

有效資料為 SEG140、SEG141、SEG143，共 40 拍。

| 指標 | 結果 |
| --- | ---: |
| Binary predictions | 31 |
| Unknown predictions | 9 |
| Classifier coverage | 77.50% |
| Binary correct / incorrect | 23 / 8 |
| Selective accuracy | 74.19% |
| Appropriate / unnecessary abstentions | 2 / 7 |
| Abstention appropriateness | 22.22% |

分 rally 結果：

| Segment | Hits | Coverage | Selective accuracy | Appropriate / unnecessary unknown |
| ---: | ---: | ---: | ---: | ---: |
| 140 | 21 | 71.43% | 86.67% | 1 / 5 |
| 141 | 5 | 80.00% | 100.00% | 0 / 1 |
| 143 | 14 | 85.71% | 50.00% | 1 / 1 |

結果在不同 rally 間差異很大，因此不能用 SEG141 的 100% 推論整體可靠。SEG143 的 12 個二元輸出中有 6 個錯誤。

## 關鍵診斷：orientation flip 是主要失敗點

| Orientation decision | Binary predictions | Correct | Incorrect | Selective accuracy |
| --- | ---: | ---: | ---: | ---: |
| followed court prior | 23 | 23 | 0 | 100.00% |
| flipped from court prior | 8 | 0 | 8 | 0.00% |

8 個二元誤判全部具有 `body_flipped_from_court_prior=true`，而未翻轉的 23 個二元輸出全部正確。錯誤的 margin 為 0.7004 至 0.9287，多數 frame 也通過 racket-arm confidence gate，所以這不是「低 confidence 骨架漏過 gate」可以解釋的現象。

目前最合理的工程判讀是：朝向翻轉規則把 image-space lateral axis 反轉後，將實際 backhand 系統性輸出成 forehand。這個結論是由本次資料得到的強關聯，仍需以更多人工標註 rally 驗證，不能直接當成已證明的因果。

二元 confusion matrix 也呈現同一偏差：

| Reference \\ Prediction | Forehand | Backhand |
| --- | ---: | ---: |
| Forehand | 16 | 0 |
| Backhand | 8 | 7 |

## Confidence gate 的實際效果

Confidence gate 有效達成的是「骨架腕、肘、肩不足時可以輸出 unknown」，但目前 threshold 過於保守：9 次棄權中只有 2 次被人工視為合理，7 次被標為不必要棄權。它降低錯骨架產生確定輸出的風險，代價是 coverage 降低，而且沒有處理 orientation flip 的系統性錯誤。

因此本輪結論不是調低或調高單一 confidence threshold，而是將兩類問題分開：

1. pose reliability gate 決定資料是否足以分類。
2. orientation / side rule 決定合格骨架的正反手語義。

只調 confidence gate 無法修正第二類錯誤。

## 對 commentary pipeline 的決策

目前正反手結果仍是 experimental candidate，不應加入 verified `RallyFact`、Gemini Tactical Analyzer 或 Commentator prompt。若需要保留實驗輸出，必須允許 `unknown` 並附上規則版本、診斷資訊與限制，不得包裝成可靠比賽事實。

`body_flipped_from_court_prior` 的 A/B/C/D ablation 已完成，結果見 [forehand_backhand_orientation_ablation.md](forehand_backhand_orientation_ablation.md)。court-prior-only 在這組 development labels 上最佳，但仍必須以新的人工審核 rally 做 hold-out 驗證，不能用相同資料宣稱泛化能力。

## 可追溯產物

- `outputs/ttyvsasy/forehand_backhand/confidence_gate_review_summary.json`
- `outputs/ttyvsasy/forehand_backhand/confidence_gate_review_summary.md`
- `outputs/ttyvsasy/forehand_backhand/seg0140_confidence_gate/review_metrics.json`
- `outputs/ttyvsasy/forehand_backhand/seg0141_confidence_gate/review_metrics.json`
- `outputs/ttyvsasy/forehand_backhand/seg0143_confidence_gate/review_metrics.json`

重跑彙整：

```powershell
uv run python .\experiments\ttyvsasy\scripts\analyze_forehand_backhand_review_set.py `
  --review-dir F:\Downloads `
  --segments 140 141 143 144
```

`outputs/` 是可重建且被 git ignore 的實驗產物；本文件與分析程式才是 repository 內的可追溯紀錄。
