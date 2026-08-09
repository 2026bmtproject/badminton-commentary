# Compact / Verified Facts

`CompactRallyFacts` 是 Stage Adapter 與未來 Gemini Tactical Analyzer 之間的 typed contract。
這一層只做 deterministic Python 計算，不呼叫 LLM，也不修改上游 artifacts。

## Input isolation

呼叫：

```python
stages = read_upstream_stages(paths, segment_index=144)
compact = build_compact_rally_facts(
    stages=stages,
    segment_index=144,
    court_position_to_player=mapping,
)
```

Pose 與 shuttle reader 逐筆掃描大型 JSON array，只把 `segment_index=144` 的 records 建成
Pydantic models。所有 Compact events 保留原始全場 `event_index`，因此可追溯回 stage records。

## Pose features

Pose 使用 COCO 17-keypoint indexes。每個 hit event 只取同場上位置、最接近 event frame 的
pose，預設最多相差 2 frames。輸出：

- usable keypoint count 與 mean confidence；raw score 大於 1 時 clamp 並留下 limitation。
- body center image coordinate。
- 左右 shoulder-to-wrist extension，以 torso length 正規化。
- stance width / shoulder width。
- shoulder angle。
- `left/right/unknown` hitting-arm geometry candidate。

這些特徵不等於 forehand/backhand；Compact facts 明確附帶
`forehand_backhand_not_inferred`。

## Court features

Court homography 實際方向是 court coordinates 到 image coordinates，builder 反矩陣後才將
腳踝中點投影到 6.1 m × 13.4 m court。缺腳踝時可降級使用 bbox bottom center，quality 為
`cautious`。

預設 policy 將 court calibration 視為已由使用者／上游流程驗證，因此即使 artifact 的
`confirmed=false` 也會嘗試投影，並在 fact limitation 記錄
`court_calibration_prevalidated_by_policy`。若需要嚴格遵守上游旗標，可設定：

```python
CompactFactConfig(require_court_confirmation=True)
```

Calibration 仍必須 `detection_failed=false`、唯一且矩陣可逆，才會輸出：

- normalized x/y。
- left/center/right。
- 對該球員而言的 rear/mid/front。
- 與同一球員上次 hit position 的 court-plane displacement。

嚴格模式下，TTYvsASY 的 `confirmed=false` 會得到 `court_position=null` 與
`court_calibration_unconfirmed` warning。

## Shuttle features

同一 frame 可能同時有 `inpaint` 與 `viterbi`。Builder 只使用
`stroke_classification.shuttle_method` 指定的方法，不能混合兩條軌跡。每個 event 使用前後
各 6 frames，輸出：

- usable sample ratio 與 quality。
- incoming/outgoing unit vector。
- image-space direction，例如 `down`、`up_left`。

不計算速度。由於 shuttle 具有高度，不能直接用地面 court homography 假裝成三維落點，
因此目前明確標記 `image_coordinates_only`。

## Quality and degradation

- 缺 pose：保留 stroke，`pose=null`。
- pose confidence 不足：保留 compact pose，quality 為 cautious/unavailable。
- court 未確認或投影失敗：不產生 court fact。
- shuttle 不可見：保留 window metadata，quality 為 unavailable。
- 任何 vision stage 缺失都不會刪除原始 stroke。

Compact JSON 不包含 raw keypoints、完整 homography 或逐 frame shuttle point arrays。目前它
尚未進入 Gemini prompt；下一個 milestone 才會建立 Tactical Analyzer 與 `TacticalFact[]`。
