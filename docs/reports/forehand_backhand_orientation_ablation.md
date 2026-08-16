# 正反手 Orientation A/B/C/D Ablation

日期：2026-08-15

## 目的

前一輪人工審核發現，8 個二元誤判全部集中於 `body_flipped_from_court_prior=true`。本輪直接比較四個版本：

| Variant | 設定 |
| --- | --- |
| A | 原版：face weight = 0.35，使用 pose orientation vote |
| B | face weight = 0，仍使用 pose orientation vote |
| C | 禁止推翻 court prior，orientation 永遠採用 top/bottom prior |
| D | 原始 vote 與 court prior 不一致時，反轉 `orientation_sign` |

## 人工 reference 集合

評估使用 SEG140、141、143、144，共 47 個能從人工 review 推導出明確 forehand/backhand reference 的 event：

- SEG140、141、143：來自 confidence-gated A 結果的人工 review，共 31 拍。
- SEG144：來自修改前 baseline review，共 16 拍；以相同 `event_index` 對應目前各 variant。
- 原候選為 unknown 且 review 沒有記錄 corrected side 的 rows 無法推導 reference，因此排除。

為避免某個版本大量輸出 unknown 而得到虛高準確率，主要 accuracy 固定使用同一組 47 拍作為分母，unknown 算錯；另外才報告只計算非 null 輸出的 selective accuracy。

## 結果

| Variant | Correct | Fixed-set accuracy | Coverage | Selective accuracy | 與 A 不同的預測 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 36 / 47 | 76.60% | 44 / 47 = 93.62% | 81.82% | 0 |
| B | 36 / 47 | 76.60% | 44 / 47 = 93.62% | 81.82% | 0 |
| C | 44 / 47 | 93.62% | 44 / 47 = 93.62% | 100.00% | 8 |
| D | 44 / 47 | 93.62% | 44 / 47 = 93.62% | 100.00% | 8 |

固定集合的三個未命中是 confidence gate 輸出的 unknown，而不是 C/D orientation 分類錯誤。

### 分 rally accuracy

| Variant | SEG140 | SEG141 | SEG143 | SEG144 |
| --- | ---: | ---: | ---: | ---: |
| A | 13/15 = 86.67% | 4/4 = 100% | 6/12 = 50.00% | 13/16 = 81.25% |
| B | 13/15 = 86.67% | 4/4 = 100% | 6/12 = 50.00% | 13/16 = 81.25% |
| C | 15/15 = 100% | 4/4 = 100% | 12/12 = 100% | 13/16 = 81.25% |
| D | 15/15 = 100% | 4/4 = 100% | 12/12 = 100% | 13/16 = 81.25% |

### Confusion matrix

A/B：

| Reference \\ Prediction | Forehand | Backhand | Unknown |
| --- | ---: | ---: | ---: |
| Forehand | 25 | 0 | 2 |
| Backhand | 8 | 11 | 1 |

C/D：

| Reference \\ Prediction | Forehand | Backhand | Unknown |
| --- | ---: | ---: | ---: |
| Forehand | 25 | 0 | 2 |
| Backhand | 0 | 19 | 1 |

## 解讀

### B：face cue 不是本批錯誤的決定因素

將 face weight 從 0.35 改成 0，47 個 reference event 的最終預測完全沒有改變，accuracy 與 A 相同。這表示 face confidence proxy 雖然語義薄弱，但移除它不足以修正本批 orientation 問題；肩膀、髖部與 court prior 的合成 vote 仍得到相同方向。

### C 與 D 在目前定義下數學等價

orientation sign 只有 `-1` 與 `+1`。若 voted sign 與 court prior 不一致，將 voted sign 反轉後必然等於 court-prior sign；若兩者一致則不反轉。因此：

```text
if vote != prior:
    orientation = -vote
else:
    orientation = vote

等價於：

orientation = prior
```

實際重跑也確認 C、D 的 47 個預測完全相同。實作時應選語義較直接的 C，沒有必要保留兩套等價 policy。

### C/D 修正了 development set 的 orientation 錯誤

C/D 相對 A 改變 8 拍，正好將 A 的 8 個 backhand→forehand 誤判全部修正。這支持「目前 pose orientation vote 的 disagreement branch 有系統性反向」的診斷。

但這批人工標籤正是用來發現該問題的資料，因此 100% selective accuracy 是 development-set 結果，不是 hold-out accuracy。直接以 C 取代 A 後再用相同 47 拍宣稱泛化，會形成 evaluation leakage。

## 決策

1. 下一個候選版本採用 C（court prior only），因為它與 D 等價且語義更清楚。
2. 暫不將正反手結果升級為 verified `RallyFact`。
3. 下一輪必須選擇尚未人工審核的新 rally，先凍結 C 的參數，再進行 hold-out review。
4. 若新 rally 中確實存在球員轉身、round-the-head 或大幅 body rotation，court prior only 可能失敗；這正是 hold-out 必須覆蓋的案例。

後續 hold-out 已凍結 SEG130、132、136、139、146，另加入使用者指定的 SEG144 C 版複審；完整標註方式與統計邊界見 [forehand_backhand_c_holdout_review_protocol.md](forehand_backhand_c_holdout_review_protocol.md)。

## 重跑方式

```powershell
uv run python .\experiments\ttyvsasy\scripts\compare_forehand_backhand_orientation_ablation.py `
  --review-dir F:\Downloads `
  --segments 140 141 143 144
```

可重建輸出：

- `outputs/ttyvsasy/forehand_backhand/orientation_ablation/orientation_ablation_summary.json`
- `outputs/ttyvsasy/forehand_backhand/orientation_ablation/orientation_ablation_summary.md`
- `outputs/ttyvsasy/forehand_backhand/orientation_ablation/A_original/`
- `outputs/ttyvsasy/forehand_backhand/orientation_ablation/B_face_weight_0/`
- `outputs/ttyvsasy/forehand_backhand/orientation_ablation/C_court_prior_only/`
- `outputs/ttyvsasy/forehand_backhand/orientation_ablation/D_invert_true_branch/`
