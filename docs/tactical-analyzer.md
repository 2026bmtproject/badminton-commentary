# Tactical Analyzer

Milestone 3 將 `CompactRallyFacts` 交給一個可替換的 LLM provider，透過單次 request
產生最多五筆 tactical candidates。Analyzer 不直接產生賽評，也尚未改變 Planner。

```text
CompactRallyFacts
        ↓ one provider call
Gemini Tactical Analyzer
        ↓ Pydantic + deterministic provenance/semantic gates
TacticalAnalysisResult
        └── TacticalFact[]
```

## Model roles

`config.yaml.example` 將兩種用途分開設定：

```yaml
provider:
  gemini:
    model: gemini-flash-latest

tactical_analyzer:
  model: gemini-3.1-pro-preview
  fallback_models:
    - gemini-3.6-flash
  max_facts: 5
```

Commentator 繼續使用低延遲 Flash；Tactical Analyzer 優先使用推理品質較高的 Pro Preview。
若 Preview 在 SDK retry 後仍回傳 404、500、502、503 或 504，會以同一份 prompt 自動改用
stable `gemini-3.6-flash`。429 spending cap / rate limit、authentication 與 validation error
不會 fallback，以免掩蓋真正的帳務或輸入問題。

兩種用途共用 API key、timeout 與 retry 設定，但會建立不同 `GeminiProvider` instance。
實際成功模型會寫入 `TacticalAnalysisResult.provider_model`；若發生降級，也會加入
`provider_model_fallback:<primary>-><fallback>` warning。模型名稱只是可調設定，不屬於
上游 domain contract。

## TacticalFact contract

每筆 fact 包含：

- deterministic `fact_id`，由程式產生，不接受模型自行命名。
- `pattern_type`。
- 自然中文 `description`。
- `confidence` 與 `salience`。
- `start_event_index` / `end_event_index`。
- 涉及的 `players`。
- 2 至 12 個 `evidence_fact_ids`。
- 明確的 `limitations`。

目前 pattern taxonomy：

- `sustained_attack`
- `rear_to_front_stroke_transition`
- `attack_transition`
- `defense_to_counterattack_candidate`
- `front_back_court_displacement`
- `attacking_initiative_candidate`
- `notable_stroke_sequence`
- `rally_tactical_theme`

`candidate` 後綴代表這是有證據的戰術解讀，不是上游 CV 已直接驗證的觀測值。

## Validation

Provider JSON 先經過 Pydantic，再執行 deterministic gates：

1. `segment_index` 必須與輸入相同。
2. start/end 必須是所選 rally 的真實 event index。
3. 所有 evidence ID 必須存在於 Compact Facts。
4. evidence 必須落在宣告的 event range，且至少跨兩個 events。
5. players 必須出現在 evidence events。
6. pattern-specific evidence 必須成立，例如 sustained attack 需要同一球員至少兩個進攻球；
   rear-to-front 需要後場與網前球種；front-back displacement 需要不同 depth zone 的 court facts。
7. 拒絕目前無法支持的正反手、球速、致勝球、最後一拍、得分原因與因果描述。
8. facts 為空是合法結果；少於兩個 events 時不呼叫 provider。

單筆 candidate 未通過第 2 至 7 項時會被捨棄，原因寫成
`rejected_tactical_fact:<source-index>:<pattern-type>:<reason-code>` warning；同一 response
內其他通過驗證的 facts 仍會保留。整份 JSON 無法通過 Pydantic、segment 不一致或超過
`max_facts` 才會中止整次分析。

輸入 prompt 不包含 raw keypoints、court homography 或完整逐 frame shuttle arrays。

## Current boundary

Milestone 3 只產生並保存 `tactical_facts.json`。既有 Planner / Commentator 尚未消費這些
facts，避免同時改變分析與生成品質。下一個里程碑才會把高 salience TacticalFact 放入
Planner 的 allowed facts，再由 Commentator 轉為賽評文字。
