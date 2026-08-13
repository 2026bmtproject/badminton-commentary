# 技術報告來源清單

盤點日期：2026-08-13

本清單記錄 `algorithm_development_report.md` 與
`implementation_verification_report.md` 的可追溯來源。盤點優先級依序為目前程式碼、
schema／tests／config、目前生成 artifact、現行文件、歷史文件與 Git history。

## 掃描範圍與結果

使用 `rg --files` 掃描 repository 中的 `.md`、`.txt`、`.json`、`.yaml`、`.yml`、
`.toml` 與 `.py`，排除 ignored/generated `outputs/`、大型 fixture payload 及
`experiments/ttyvsasy/workspace/`。共得到 95 個候選來源：

| 類型 | 數量 | 說明 |
| --- | ---: | --- |
| Python | 75 | production、experiment scripts 與 tests |
| Markdown | 9 | README 與既有設計／實作文件 |
| Text | 7 | prompts |
| JSON | 2 | tracked v2 Gemini experiment output 與 metadata |
| YAML | 1 | 本機 config；未擷取 API key 或秘密值 |
| TOML | 1 | Python 與 dependency contract |

`conversation/`、`conversations/`、`chat/`、`chat_logs/`、`logs/`、`history/`、
`archive/` 等目錄未在可存取 workspace 中找到。因此報告不把目前互動或模型記憶
當作歷史證據。

> Conversation records were not found in the accessible workspace.

## Current implementation：production

| Source | Type | Relevance | Historical / Current | Notes |
| --- | --- | --- | --- | --- |
| `pyproject.toml` | TOML | Python 3.12 與直接 dependencies | Current | 只有 google-genai、Pydantic、PyYAML |
| `README.md` | Markdown | 對外 API、runtime boundary、grounding | Current with stale statements | 個別里程碑敘述需由 code 驗證 |
| `src/badminton_commentary/schemas.py` | Python | RallyFact、plan、generated output、bundle schema | Current | production canonical contracts |
| `src/badminton_commentary/adapters/upstream.py` | Python | upstream stage models、reader、segment selection、join | Current | 四個 core stages；vision optional |
| `src/badminton_commentary/adapters/vision.py` | Python | pose/court/shuttle typed reader | Current | pose/shuttle 依 segment 串流 |
| `src/badminton_commentary/analysis/fact_builder.py` | Python | normalized JSON Fact Builder | Current / legacy-compatible | 純 Python，不呼叫 LLM |
| `src/badminton_commentary/facts/schemas.py` | Python | CompactRallyFacts schema | Current | 仍含 hitting-arm geometry candidate |
| `src/badminton_commentary/facts/builder.py` | Python | production Compact Facts | Current | nearest-pose、court homography、shuttle vector |
| `src/badminton_commentary/facts/tactical.py` | Python | TacticalFact[] schema | Current, uncommitted | 最多五個 validated candidates |
| `src/badminton_commentary/analysis/tactical_analyzer.py` | Python | Gemini tactical analysis + provenance gates | Current, uncommitted | 一次 provider call；不直接產生 commentary |
| `src/badminton_commentary/analysis/rally_analyzer.py` | Python | rule-based stroke patterns | Current | stroke-derived，不是 pose movement |
| `src/badminton_commentary/analysis/stroke_event_analyzer.py` | Python | 每拍 local facts 與 confidence band | Current | production all-strokes path |
| `src/badminton_commentary/generation/planner.py` | Python | summary planning | Current | user-selected 與 importance legacy 分支 |
| `src/badminton_commentary/generation/event_planner.py` | Python | 每拍 deterministic plan | Current | 可 force 每個 mapped stroke 輸出 |
| `src/badminton_commentary/generation/rally_batch_commentator.py` | Python | 單 rally、單次 provider batch | Current | output validation + language fallback |
| `src/badminton_commentary/generation/validator.py` | Python | summary provenance/language validation | Current | 禁止 unsupported movement/causality |
| `src/badminton_commentary/generation/event_validator.py` | Python | stroke event validation | Current | low-confidence uncertainty wording |
| `src/badminton_commentary/providers/gemini.py` | Python | Gemini timeout/retry/model fallback | Current, uncommitted changes | API key 只由 environment 讀取 |
| `src/badminton_commentary/providers/fake.py` | Python | deterministic offline provider | Current | tests 不依賴網路 |
| `src/badminton_commentary/services/rally_commentary.py` | Python | production orchestration boundary | Current, uncommitted changes | tactical branch 尚未接入 commentator |
| `src/badminton_commentary/config.py` | Python | provider/tactical config schema | Current, uncommitted changes | tactical max facts 1..5 |

## Current implementation：direct v4 experiment

| Source | Type | Relevance | Historical / Current | Notes |
| --- | --- | --- | --- | --- |
| `src/badminton_commentary/analysis/pose_geometry.py` | Python | deterministic 2D pose geometry | Experimental current | 不在 production service execution path |
| `experiments/ttyvsasy/scripts/package_direct_rallyfact.py` | Python | event-centric v4 package | Experimental current | full debug / compact LLM input 分流 |
| `experiments/ttyvsasy/scripts/run_direct_rallyfact_v3.py` | Python | Experimental Enriched RallyFact v3 runner | Experimental current | input v4、output semantic schema v3 |
| `experiments/ttyvsasy/prompts/direct_rallyfact_event_centric_v3.txt` | Prompt | direct multimodal fusion prompt | Experimental current | LLM 解讀 posture/shuttle/tactics |
| `experiments/ttyvsasy/scripts/visualize_enriched_rallyfact_v3.py` | Python | 姿態／戰術影片疊圖 | Evaluation utility | 依賴外部 FFmpeg executable |
| `experiments/ttyvsasy/README.md` | Markdown | TTYvsASY workflow | Current experiment guide | 明確非 production API |
| `outputs/ttyvsasy/direct_rallyfact/seg0144_v4_prompt_test/rally_stage_input.json` | Generated JSON | 實際 compact v4 artifact | Current local evidence | 17 events；不進 Git |
| `outputs/ttyvsasy/direct_rallyfact/seg0144_v4_prompt_test/rally_stage_input_debug.json` | Generated JSON | full raw pose debug artifact | Current local evidence | 323 pose records；不進 LLM |
| `outputs/ttyvsasy/direct_rallyfact/seg0144_v4_prompt_test/gemini_enriched_rally_fact_v3.json` | Generated JSON | 使用者提供的 Gemini output | Current local evaluation | schema/provenance 通過；不代表語義正確 |
| `outputs/ttyvsasy/direct_rallyfact/seg0144_v4_prompt_test/gemini_v3_run_metadata.json` | Generated JSON | 實際 provider failure metadata | Current local evidence | `gemini-flash-latest` 單次 call，79.736 秒後 504 |
| `outputs/ttyvsasy/direct_rallyfact/seg0144_v4_prompt_test/evaluation_v4_prompt.md` | Generated report | prompt quality review | Current local evaluation | pose conditional、shuttle not ready |

## Tests and executable verification

| Source | Type | Relevance | Historical / Current | Notes |
| --- | --- | --- | --- | --- |
| `tests/unit/test_schemas.py` | Test | core Pydantic constraints | Current | index/player/confidence validation |
| `tests/unit/test_upstream_adapter.py` | Test | segment isolation與 joins | Current | 防止相鄰 rally 混入 |
| `tests/unit/test_vision_adapter.py` | Test | vision reader/schema | Current | confidence >1 可保留為 raw value |
| `tests/unit/test_compact_facts.py` | Test | production compact features | Current | court prevalidation policy |
| `tests/unit/test_pose_geometry.py` | Test | v4 pose numeric geometry | Experimental current | 90°/180°、1.5 ratio、missing data |
| `tests/unit/test_direct_rallyfact_package.py` | Test | v4 transport/package | Experimental current | 17 events、compact/debug separation |
| `tests/unit/test_direct_rallyfact_v3_runner.py` | Test | output schema/provenance | Experimental current | evidence frame allowlist |
| `tests/unit/test_tactical_analyzer.py` | Test | tactical schema/gates | Current, uncommitted | candidate rejection reasons |
| `tests/integration/test_rally_commentary_service.py` | Test | one-call production service | Current | no experiment/subtitle runtime dependency |
| `tests/integration/test_upstream_stage_service.py` | Test | stages → service | Current | compact and tactical provider paths |
| `tests/regression/test_tactical_regressions.py` | Test | wording/provenance regressions | Current | stroke semantics and unsupported claims |
| `tools/report_implementation_audit.py` | Audit | runtime/schema/constants inspection | Added for this report | offline；optional artifact check |
| `docs/reports/implementation_audit_result.json` | Audit output | 16 executable checks | Added for this report | SEG144 package included |

## Existing design and development records

| Source | Type | Relevance | Historical / Current | Notes |
| --- | --- | --- | --- | --- |
| `docs/planner-generator-report.md` | Markdown | early Planner/Generator design | Historical + legacy current | importance-centric sections非 production all-strokes gate |
| `docs/event-driven-commentary-report.md` | Markdown | event-driven migration | Historical + current | records per-stroke evolution |
| `docs/upstream-stage-adapter.md` | Markdown | stage integration contract | Current | validated against adapter code |
| `docs/compact-facts.md` | Markdown | Compact Facts algorithm | Current production branch | differs from direct v4 geometry |
| `docs/tactical-analyzer.md` | Markdown | Tactical Analyzer boundary | Current, uncommitted | tactical facts not yet consumed by Commentator |
| `experiments/ttyvsasy/evaluations/direct_rallyfact/seg0144/v2/README.md` | Markdown | v2 direct experiment | Historical | records 2D pose limits |
| `experiments/ttyvsasy/evaluations/direct_rallyfact/seg0144/v2/gemini_output.json` | JSON | earlier model output | Historical | qualitative experiment only |
| `experiments/ttyvsasy/prompts/direct_rallyfact_event_centric_v2.txt` | Prompt | earlier direct prompt | Historical | superseded by compact v3 prompt/v4 transport |

## Git history used

| Commit | Date | Relevance | Classification |
| --- | --- | --- | --- |
| `153e8f8` | 2026-08-07 | TTYvsASY Planner/Generator pipeline | Historical foundation |
| `1c80c3e` | 2026-08-07 | rule-based higher-level stroke patterns | Historical evolution / current code lineage |
| `39aef89` | 2026-08-07 | event-driven per-stroke commentary | Historical evolution / current code lineage |
| `b2d2ad8` | 2026-08-08 | single-rally production service、batch call、repo cleanup | Current architecture boundary |
| `8aaef6a` | 2026-08-08 | Python 3.12 pin | Current environment contract |
| `4d453d6` | 2026-08-09 | upstream stage adapter | Current integration boundary |
| `69d3a7f` | 2026-08-09 | vision adapter + CompactRallyFacts | Current multimodal production branch |
| `37851fd` | 2026-08-09 | court calibration policy | Current production court behavior |

目前 Tactical Analyzer、direct v4 package 與 pose geometry 位於 working tree，尚未出現在
上述 Git commits。報告將它們標為「current working tree」或「experimental current」，不會
誤稱為已提交的歷史里程碑。
