# Event-driven 羽球賽評 MVP

## 目標

賽評輸出拆成兩層：

```text
Per-stroke commentary
+ Rally-level tactical summary
```

Per-stroke 層只看當前 stroke、前 2–4 個可用 stroke、相鄰 local sequence facts 與
比分背景；rally summary 繼續使用 Rally Analyzer 的長尺度 patterns。

## 資料流

```text
events.json + strokes.json
        ↓ event_index join
Fact Builder / RallyFactEvent
        ↓
Stroke Event Analyzer ────────────── Rally Analyzer
        ↓                                  ↓
StrokeEventAnalysis                  RallyAnalysis
        ↓                                  ↓
Event Planner                         Rally Planner
        └──────────────┬───────────────────┘
                       ↓
             Rally Batch Generator
             （每個 rally 一次 LLM）
                       ↓
                events + summary
```

Gemini 不負責建立 tactical relation；它只將 rule-based facts 轉為自然繁中。

## Local sequence facts

第一版支援：

- `rear_exchange_continuation`
- `rear_court_stroke_to_front_court_stroke`
- `net_exchange_continuation`
- `flat_exchange_continuation`
- `net_to_lift_transition`
- `lift_to_attack_transition`
- `drop_lift_attack_sequence`

所有 sequence 只允許原始 event index 連續的 2–3 拍。跳過低 confidence event 後，
不得把不相鄰的 stroke 重新接成 sequence。

## Speaking policy

每個 usable stroke 都有 `speaking_score`，取 current stroke salience 與 local fact
salience 的最大值。初始門檻為 `0.65`：

- 普通發球：通常跳過。
- 重複高遠球：通常跳過或由後續 sequence 合併。
- 小球、切球、平快球：較容易發話。
- 殺球、撲球、高 salience transition：優先發話。

若下一個相鄰 local fact 包含當前 stroke 且 salience 更高，當前 stroke 先不發話，
由下一個 commentary unit 合併描述。非高 salience 事件另有最小間隔及重複 focus cooldown。

## 時序與 provenance

Gemini 每個 rally 一次回傳：

```json
{
  "segment_index": 2,
  "events": [
    {
      "stroke_index": 9,
      "text": "網前球挑高後，下一拍接上殺球！",
      "source_fact_ids": [
        "rally:2:local:7-9:drop_lift_attack_sequence",
        "rally:2:stroke:7",
        "rally:2:stroke:8",
        "rally:2:stroke:9"
      ]
    }
  ],
  "summary": {
    "segment_index": 2,
    "text": "挑球之後緊接著轉為進攻球。",
    "source_fact_ids": ["rally:2:pattern:lift_to_attack_transition"]
  }
}
```

`segment_index`、`stroke_index`、`frame`、`time_sec` 全由 Python 從 event analysis
校對或寫入。模型回傳的 event `stroke_index` 必須與計畫完全相同且順序一致；
`frame` 和 `time_sec` 不交給模型生成。local fact 若被引用，Validator 要求同時引用其全部
`supporting_fact_ids`，並強制包含 current stroke fact。文字若包含比分，還必須引用
score fact，且數值要與該 rally 的 `RallyScore` 完全一致。

## 輸出格式

```json
{
  "rallies": [
    {
      "segment_index": 2,
      "events": [
        {
          "segment_index": 2,
          "stroke_index": 9,
          "frame": 120,
          "time_sec": 4.0,
          "text": "安洗瑩以平快球回擊。",
          "source_fact_ids": ["rally:2:stroke:9"]
        }
      ],
      "summary": {
        "segment_index": 2,
        "text": "球路從後場球轉為網前處理。",
        "source_fact_ids": [
          "rally:2:pattern:rear_court_stroke_to_front_court_stroke"
        ]
      }
    }
  ]
}
```

## 執行

Fake 離線驗證：

```powershell
uv run python .\experiments\ttyvsasy\scripts\generate_commentary.py `
  --provider fake `
  --mode event-driven `
  --config .\config.yaml.example
```

Gemini：

```powershell
uv run python .\experiments\ttyvsasy\scripts\generate_commentary.py `
  --provider gemini `
  --mode event-driven `
  --config .\config.yaml
```

舊 rally-only 模式仍可使用 `--mode summary`。

## 安全限制

兩層 Generator 共用語言安全規則，禁止最後一拍、致勝球、得分原因、被迫、抓到
機會、戰術意圖、主被動、因果、球員移動及 schema/debug wording。只有 speaking
score `>= 0.9` 的即時 unit 能使用一個驚嘆號。

## MVP 限制

- 目前是離線 timestamp-aligned event commentary，尚未實作串流播放或延遲排程。
- 尚未使用 pose，不能描述真實移動、跳躍或防守姿態。
- 「防回去了」需要額外 deterministic defensive-retrieval fact，目前不允許生成。
- 目前每個 rally 呼叫一次 Gemini；三組 TTYvsASY fixture 共 15 個 rally，因此由
  原本 50 次降為 15 次。後續仍可加入 checkpoint 與 prompt cache。
