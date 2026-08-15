# 正反手 2D Pose Heuristic：演算法分析與 SEG144 實驗

評估日期：2026-08-13  
來源腳本：`F:/Downloads/forehand_backhand.py`  
本 repo 實驗入口：`experiments/ttyvsasy/scripts/experiment_forehand_backhand.py`

## 結論

這個演算法比「手腕 x 座標在左或右」合理，因為它先建立 torso-based body frame、估計身體朝向，再把肩／肘／腕投影到持拍側方向，並用擊球前後多幀加權降低單幀抖動。它適合產生一個可人工檢查的 **forehand/backhand candidate**。

但它仍不能把正反手變成 verified fact。單視角 2D skeleton 沒有 racket face、grip、實際 contact point、手腕旋前／旋後、3D torso rotation 與可靠 handedness；`heuristic_margin` 也不是校準過的機率。Production Planner／Commentator 目前不應直接使用結果。

## 1. 原始演算法

### 1.1 Body frame 與尺度

原始腳本由雙肩中點與雙髖中點建立向上方向：

```text
up = unit(shoulder_center - hip_center)
image_lateral = perpendicular(up), forced toward image-right
```

尺度不是只使用 shoulder width，而是：

```text
scale = max(
  torso_length,
  shoulder_width × 1.2,
  bbox_height × 0.30,
  12 pixels
)
```

此 fallback 可降低側身時 shoulder width 塌縮及彎腰時 torso length 變短造成的數值爆炸；代價是 bbox jitter 會直接影響 normalized lateral distances。

### 1.2 身體朝向投票

肩線、髖線、face confidence 與 top/bottom prior 共同判斷「anatomical right 在影像哪一側」：

```text
flip_vote =
    1.00 × shoulder_projection
  + 0.60 × hip_projection
  + 0.35 × face_proxy
  + 0.25 × court-side prior
```

整個 `±2` frame window 先共用一個 orientation sign，再進行逐幀 stroke-side 計算。這點是合理的時序約束，可避免一擊內 orientation sign 因 pose jitter 反覆切換。

風險如下：

- Face keypoint confidence 不是 facing direction 的可靠 measurement；背面也可能被模型補出高 confidence，低 confidence 也可能只是 motion blur。
- `top → toward camera`、`bottom → away` 只是弱先驗；球員轉身、側身與頭頂區回球會違反。
- Shoulder/hip anatomical labels 若被 pose estimator 左右交換，主 evidence 會整體翻面。
- `flip_confidence` 是權重總和經 `flip_full=0.8` 截斷，不是由 labeled data 校準的 confidence。

### 1.3 持拍臂訊號

右手選手使用 right shoulder/elbow/wrist；左手選手鏡射。三個 normalized cue 為：

```text
d_wrist = dot(wrist - racket_shoulder, racket_side) / scale
d_elbow = dot(elbow - racket_shoulder, racket_side) / scale
d_fore  = dot(wrist - elbow, racket_side) / scale
```

各 cue 經 `tanh(value / 0.35)` 壓縮。平球權重為 `(elbow=0, wrist=0.45, forearm=0.55)`；過頂球權重為 `(0.35, 0.30, 0.35)`。正值判正手、負值判反手。

這個設計的優點是輸出可解釋，可以回看 wrist/forearm 是哪個 cue 拉動結果。核心限制則是「手腕相對持拍肩在哪一側」和「拍面是正手或反手」並不等價；準備、觸球、隨揮三個 phase 會出現不同幾何。

### 1.4 Overhead 與 round-the-head

當 wrist 高於 shoulder 超過 `0.15 × scale` 時切換 overhead 權重。若 elbow 仍在持拍側 `>0.10`，但 wrist 已越過另一側 `<-0.05`，則把結果提升到至少 `+0.35`，視為 round-the-head forehand。

這能處理一類明顯反例，但三個門檻均為手寫 heuristic。缺少 racket/contact evidence 時，可能把 overhead backhand、錯幀或 pose left/right swap 強制改成 forehand。

### 1.5 多幀投票與 unknown gate

預設 window 是擊球前後 `±2` frames，時間權重為：

```text
w_time(k) = exp(-k² / (2 × 1.5²))
```

逐幀 score 再乘持拍臂最低 joint confidence。窗口內沒有可用骨架時，可向外找至 `±6` frames；只取最近的一個 fallback frame。Orientation 不明確時，aggregate score 會被：

```text
0.35 + 0.65 × flip_confidence
```

壓低。最終絕對值小於 `0.08` 回傳 unknown。Unknown gate 是必要的安全設計；但 `0.08` 尚未用 labeled dataset 選定。

## 2. 適配到目前 repo

原始檔不能直接執行，因為它依賴另一個 repository 的 `modules.artifacts`、`modules.contracts` 與 NumPy。本次實驗的適配版本：

- 透過現有 `StagePaths`／`read_upstream_stages()` 讀取 stages。
- 只分析 selected segment，保持原始全場 `event_index`。
- 以 stroke 的 `event_index` join hit frame，並驗證 stroke/event frame 一致。
- 使用 pose stage 的 COCO-17 keypoints，不增加 runtime NumPy dependency。
- top/bottom 經 `CourtPositionToPlayer` 轉成 player identity。
- 左右手用 `--left-handed-player a|b` 綁定 identity，不用會換場的 top/bottom 規則。
- 輸出欄位命名為 `heuristic_margin`，明載不是 probability。
- 實驗輸出不加入 `RallyFact`、`CompactRallyFacts` 或 Gemini prompt。

## 3. SEG144 初步結果

設定：segment 144、top=`b`、bottom=`a`，兩位 player 暫按右手持拍；原始腳本其他預設參數不變。

| 結果 | 拍數 |
| --- | ---: |
| forehand | 10 |
| backhand | 6 |
| unknown | 1 |

Player a 為 4 forehand / 4 backhand；player b 為 6 forehand / 2 backhand / 1 unknown。Median heuristic margin 為 0.4901。這些是 classifier distribution，不是 accuracy。

逐拍候選：

| Event | Player | Stroke | Candidate | Margin |
| ---: | --- | --- | --- | ---: |
| 1280 | b/top | 殺球 | forehand | 0.3821 |
| 1281 | a/bottom | 小球 | forehand | 0.7733 |
| 1282 | b/top | 平快球 | backhand | 0.8889 |
| 1283 | a/bottom | 切球 | forehand | 0.4386 |
| 1284 | b/top | 小球 | forehand | 0.3499 |
| 1285 | a/bottom | 高遠球 | backhand | 0.4901 |
| 1286 | b/top | 切球 | forehand | 0.7933 |
| 1287 | a/bottom | 小球 | backhand | 0.4847 |
| 1288 | b/top | 小球 | forehand | 0.8482 |
| 1289 | a/bottom | 小球 | backhand | 0.3089 |
| 1290 | b/top | 高遠球 | forehand | 0.8462 |
| 1291 | a/bottom | 高遠球 | forehand | 0.5812 |
| 1292 | b/top | 切球 | forehand | 0.3385 |
| 1293 | a/bottom | 小球 | forehand | 0.3477 |
| 1294 | b/top | 平快球 | unknown | 0.0253 |
| 1295 | a/bottom | 小球 | backhand | 0.5946 |
| 1296 | b/top | 撲球 | backhand | 0.7362 |

初步視覺 smoke check：event 1280 的 overhead forehand candidate 與畫面表面上一致；event 1282 的跨身側向擊球使 backhand candidate 具有合理性；event 1294 的分數接近零且姿勢在邊界，回傳 unknown 比硬猜安全。完整人工 review 結果如下節。

## 4. SEG144 人工 review 結果

人工標註涵蓋全部 17 拍，結果為 15 correct、1 incorrect、1 uncertain，沒有 unreviewed event。

| Metric | Result |
| --- | ---: |
| Classifier coverage（非 unknown） | 16/17 = 94.12% |
| Human-decidable coverage | 16/17 = 94.12% |
| Selective accuracy | 15/16 = 93.75% |
| Overall correct fraction | 15/17 = 88.24% |
| Forehand precision / recall / F1 | 100% / 90.91% / 95.24% |
| Backhand precision / recall / F1 | 83.33% / 100% / 90.91% |
| Macro F1 | 93.07% |

Selective accuracy 排除 uncertain／unreviewed；它是這個 abstaining classifier 較合理的單 rally 指標。Overall correct fraction 把 uncertain 分母保留，但不能把 uncertain 當成錯誤或正確。

Confusion matrix：

| Reference \\ Prediction | Forehand | Backhand |
| --- | ---: | ---: |
| Forehand | 10 | 1 |
| Backhand | 0 | 5 |

Review schema 對錯誤只保存 `incorrect`，沒有 explicit corrected side；因為此處是 binary 且 prediction 非 null，矩陣把 opposite side 推為 reference。未來若增加其他 class 或要標 unknown 的真實側別，viewer 應直接收 `reference_side`，不能繼續依賴此推論。

### 4.1 分組觀察

- Player a：7/8 correct，87.5%。唯一錯誤在 player a。
- Player b：8 個人工可判定候選全對，另有 1 個 unknown/uncertain；不能用 100% 推論跨 rally 表現。
- 小球 7/7、切球 3/3、高遠球 2/3；其他球種只有 1–2 拍，樣本太少。
- Margin `<0.08` 只有 event 1294，classifier abstain 且人工也 uncertain。
- Margin `0.08–0.40` 為 5/5、`0.40–0.60` 為 4/5、`>=0.60` 為 6/6。此分布不單調，17 拍不足以宣稱 margin 已校準。

### 4.2 唯一錯誤：event 1285

Event 1285 是 player a 的高遠球；candidate 為 backhand、margin 0.4901，人工 verdict 為 incorrect，因此 binary reference 推為 forehand。Hit-frame diagnostics：

```text
wrist_lateral   = -0.101
elbow_lateral   = +0.119
forearm_lateral = -0.220
wrist_height    = -0.380
overhead        = false
```

因 `wrist_height` 為負，演算法沒有進 overhead branch；flat weights 又把 elbow weight 設為 0，所以 wrist 與 forearm 共同把結果拉向 backhand。Window sensitivity `±1, ±2, ±3, ±4, ±6, ±8` 全部仍判 backhand，margin 約 0.457–0.493，而且各可用 frame score 均為負。這表示單純擴大 averaging window 不會修正錯誤。

較可能的失敗機制是：2D projection／body rotation 使 forehand follow-through 的 wrist 落到模型定義的跨身側，或 event timestamp 沒對到能代表拍面的 contact phase。這正是「wrist 在持拍肩哪側」不能完全等價於 forehand/backhand 的反例。

不建議因這一拍直接調整 `min_margin` 或硬把高遠球改為 forehand。較合理的後續實驗是：

1. 以 shuttle proximity、wrist speed/reach peak 或人工 contact frame 選 phase，而非固定平均 hit `±2`。
2. 對 pose anatomical left/right swap 與大幅 torso rotation 建立 rejection gate。
3. 收集更多高遠球／overhead labels，確認這是系統性錯誤而非單例。
4. 若能取得 racket/contact evidence，再評估 2D geometry 是否仍有必要。

### 4.3 Unknown event 1294

Event 1294 的 classifier margin 0.0253，小於 0.08 gate，人工也標為 uncertain。這不能證明 threshold 最佳，但至少在本案例中 abstention 避免了一次不可驗證的硬判斷。

機器可讀結果保存在：

- `outputs/ttyvsasy/forehand_backhand/seg0144/human_review.json`
- `outputs/ttyvsasy/forehand_backhand/seg0144/review_metrics.json`
- `outputs/ttyvsasy/forehand_backhand/seg0144/review_analysis.md`

## 5. 人工檢查流程

執行：

```powershell
uv run python .\experiments\ttyvsasy\scripts\experiment_forehand_backhand.py
Start-Process .\outputs\ttyvsasy\forehand_backhand\seg0144\frame_review.html
```

Viewer 快捷鍵：

| Key | 動作 |
| --- | --- |
| `←` / `→` | 前／後一幀 |
| `Shift` + `←` / `→` | 前／後十幀 |
| `P` / `N` | 上／下一個 hit event |
| `Space` | 播放／暫停 |
| `C` | 人工標記 candidate 正確 |
| `X` | 人工標記 candidate 錯誤 |
| `U` | 人工標記無法從畫面判斷 |
| `E` | 匯出 `seg0144_forehand_backhand_human_review.json` |

人工標記保存在 browser `localStorage`，直到按 `E` 匯出。檢查時不要只看單一 hit frame；至少看 `-2..+2` frames 的黃色持拍臂與球拍／球的相對位置。若原始 hit timestamp 偏移，應先記為 uncertain，而不是把錯誤全部歸因於正反手幾何。

## 6. 採用標準

在沒有人工 labels 前，本演算法維持 experimental。建議收集至少多場、兩位球員、top/bottom、網前／平抽／過頂等分層 labels，再計算：

- coverage：非 unknown 比例
- accuracy：只對可由畫面判定且非 unknown 的 hits
- confusion matrix：forehand ↔ backhand
- error rate by court position、stroke type、overhead、pose confidence
- margin calibration：margin 區間與實際正確率是否單調

只有當不同 match 的 hold-out 結果穩定，才考慮建立 `ForehandBackhandCandidate` schema；即使如此，Commentator 也應依 confidence/limitations 使用候選措辭，而不是把它當無條件 verified fact。

## 7. Confidence-gated v2 實驗

SEG144 的唯一錯誤 event 1285 顯示：原本雖以 joint confidence 加權，仍會讓五個品質偏低但方向一致的錯骨架累積成 margin 0.4901。V2 將 confidence 從 soft weight 提升為 explicit abstention gate：

```text
base pose threshold             = 0.30
racket shoulder/elbow/wrist min = 0.50
required accepted frames        = 3
window                          = hit ±2 frames

accepted frames < 3
        ↓
unknown: insufficient high-confidence racket-arm frames
```

只有同一 frame 的持拍側 shoulder、elbow、wrist 最低 confidence 至少 0.50，該 frame 才能參與正反手分數。少於三個 accepted frames 時直接 unknown；被拒絕 frames 仍保存在 diagnostics 與 overlay，供人工查看，不會被用來投票。

這個 gate 不會把錯骨架修好，而是用 coverage 換 precision。它也不能防止「高 confidence 但解剖位置錯誤」，所以仍需 temporal bone consistency 與跨 rally review。

### 7.1 SEG144 before / after

| 版本 | Forehand | Backhand | Unknown | Coverage | 依舊 review 的非 unknown 表現 |
| --- | ---: | ---: | ---: | ---: | --- |
| 原始 confidence weighting | 10 | 6 | 1 | 94.12% | 15/16 correct |
| V2 explicit gate | 9 | 4 | 4 | 76.47% | 既有 13 個保留候選均為 correct；需重新 review |

V2 把錯誤 event 1285 降為 unknown，也把原本正確的 1289、1292 降為 unknown；1294 維持 unknown。這正是 abstention 的預期代價，不能只報 accuracy 而隱藏 coverage 下降。

### 7.2 新增 review set

為避免只針對 SEG144 調參，另選相同 `top=b / bottom=a` 換邊區間的三個 rallies：

| Segment | 選擇理由 | Hits | Forehand | Backhand | Unknown | Coverage | Null pose records skipped |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 140 | 21 拍長 rally；含發球、網前、高遠、殺球 | 21 | 11 | 4 | 6 | 71.43% | 4 |
| 141 | 5 拍短 rally；測試少樣本與缺 pose 降級 | 5 | 2 | 2 | 1 | 80.00% | 14 |
| 143 | 14 拍；含平快、切球、殺球與網前交換 | 14 | 11 | 1 | 2 | 85.71% | 0 |
| 144 | 原 baseline rally | 17 | 9 | 4 | 4 | 76.47% | 0 |

`pose.json` 在 SEG140／141 含 `keypoints=null, bbox=null` records。Production typed reader 依契約仍會拒絕它們；只有本實驗 reader 跳過並記錄數量。實驗不插值、不建立假骨架，相關 event 若沒有至少三個高品質 frames 就回 unknown。

Review fixtures 位於：

```text
outputs/ttyvsasy/forehand_backhand/
├── seg0140_confidence_gate/
├── seg0141_confidence_gate/
├── seg0143_confidence_gate/
└── seg0144_confidence_gate/
```

SEG140、141、143 的新一輪人工審核已完成；SEG144 上傳檔仍是修改前的 v1 baseline，不能納入 confidence-gated 統計。完整結果、統計語意與 orientation flip 診斷記錄於 [forehand_backhand_confidence_gate_review.md](forehand_backhand_confidence_gate_review.md)。
