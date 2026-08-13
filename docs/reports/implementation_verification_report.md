# Implementation Verification Report

查核日期：2026-08-13  
查核對象：`algorithm_development_report.md` 中的架構、演算法、參數、輸入輸出與歷程敘述

## 1. 方法

驗證優先順序為：目前 executable code 與 Pydantic schema、unit/integration/regression tests、runtime audit、目前 artifacts、現行文件、Git history。README 或歷史報告若與程式碼衝突，以程式碼與測試為準。

狀態定義：

| 狀態 | 定義 |
| --- | --- |
| `VERIFIED` | 可由目前程式、schema、test 或 artifact 直接證明 |
| `PARTIALLY_VERIFIED` | 結構或單例成立，但品質／一般化結論證據不足 |
| `HISTORICAL_ONLY` | 只描述過去流程，不是目前推薦 production path |
| `NOT_IMPLEMENTED` | 文件中的目標尚無可執行串接 |
| `CONTRADICTED` | 現有文件 claim 與目前程式碼衝突 |
| `INSUFFICIENT_EVIDENCE` | workspace 沒有足夠來源支持歷史或品質結論 |

新增的 `tools/report_implementation_audit.py` 只 import schema、執行一個 synthetic pose geometry sample、讀取常數與 optional JSON artifact；不讀 API key、不呼叫 LLM、不執行 CV。結果保存於 `docs/reports/implementation_audit_result.json`。

## 2. Claim-by-claim matrix

| ID | Claim | 狀態 | 主要證據 | 報告處理 |
| --- | --- | --- | --- | --- |
| V001 | Production 以四個核心 stage 建立 commentary domain input | VERIFIED | `adapters/upstream.py: UpstreamStageData` | 寫為 CURRENT |
| V002 | court、pose、shuttle 是 optional，且不自動進 canonical RallyFact | VERIFIED | `StagePaths`、`read_upstream_stages()`、`build_rally_fact_from_stages()` | 明確區分 RallyFact/Compact facts |
| V003 | top/bottom 必須顯式映射到 a/b | VERIFIED | `CourtPositionToPlayer` validator/resolve | 不宣稱自動 identity recognition |
| V004 | 事件依 segment frame boundary 選取，stroke 用全場 event index join | VERIFIED | `build_rally_fact_from_stages()`、`test_upstream_adapter.py` | 記錄 adjacent-rally isolation |
| V005 | Fact Builder 為純 Python，不呼叫 LLM、不修改 input | VERIFIED | `analysis/fact_builder.py`、unit tests | 寫為 deterministic join |
| V006 | 使用者選定 rally 的 Commentator 對全拍與 summary 只呼叫 provider 一次 | VERIFIED | `rally_batch_commentator.py` 只有一個 `provider.generate()`；service integration test | 不與 tactical call 合併計數 |
| V007 | Production event inclusion 不以 Importance/speaking score 過濾 | VERIFIED | `include_all_strokes=True`、`force_commentary=True` | Importance 僅列 legacy/summary context |
| V008 | LLM JSON 經 Pydantic、index/order、provenance 與語言安全驗證 | VERIFIED | batch commentator、event/summary validators | 寫入 grounding guarantees |
| V009 | 語言安全違規可 fallback，但 provenance/order 錯誤不會被吞掉 | VERIFIED | `CommentaryLanguageSafetyError` 專門 catch；其餘轉為 generation error | 分開描述兩種失敗 |
| V010 | Production Compact pose 預設找 `±2` frame 最近 pose，使用單 frame | VERIFIED | `CompactFactConfig.pose_max_frame_delta=2`、`facts/builder.py` | 不誤寫為 v4 window |
| V011 | Production `CompactPoseFact` 仍有 `hitting_arm_candidate` | VERIFIED | schema field；audit A007 | 修正「全 repo 已移除」的錯誤印象 |
| V012 | Production court 尺寸 6.1 × 13.4 m，使用 inverse homography | VERIFIED | `facts/builder.py` constants/functions/tests | 寫為 CURRENT |
| V013 | Production court 不使用 direct v4 的 1.5 m baseline extension | VERIFIED | production builder 無該常數；direct package 有該常數 | 兩條演算法分表 |
| V014 | Production shuttle 預設 `±6` frames，只建立 image-space path | VERIFIED | `CompactFactConfig`、compact fact schema/builder | 不宣稱 speed 或可靠 reversal |
| V015 | Tactical Analyzer 使用一次專用 provider call，最多五個 facts | VERIFIED | `analyze_tactical_facts()`、config、schema、audit A013 | 寫為獨立 CURRENT branch |
| V016 | Tactical candidates 有 pattern-specific evidence gates | VERIFIED | `_validate_pattern_evidence()`、tactical tests | 記錄 rule-based gate 的限制 |
| V017 | TacticalFact 已被 production Planner/Commentator 消費 | NOT_IMPLEMENTED | `RallyCommentaryService.generate()` 未呼叫 `analyze_tactics()`；batch payload 無 TacticalFact | 報告以虛線及缺口標示 |
| V018 | README 所述「CompactRallyFacts 尚未送入 Gemini prompt」適用所有路徑 | CONTRADICTED | Tactical Analyzer `_compact_payload()` 後呼叫 provider | 修正為「已進 Tactical Analyzer，未進 Commentator」 |
| V019 | README 所述「Analyzers 都是 deterministic Python」適用目前全部 analyzer | CONTRADICTED | `analysis/tactical_analyzer.py` 呼叫 provider | 改寫為 LLM candidates + deterministic gates |
| V020 | Direct transport 是 v4，但預期模型輸出 schema 仍為 v3 | VERIFIED | package constants、audit A011/A012 | 同時記錄兩個版本名稱 |
| V021 | Direct v4 pose window 為 `-8..+10`，固定 keyframes 為 `-8,-4,0,4,8,10` | VERIFIED | package/pose constants、audit A009/A011 | 只標 EXPERIMENTAL |
| V022 | Direct v4 geometry 有六組 numeric features，沒有 deterministic posture/hitting-arm label | VERIFIED | `PoseGeometryFeatures` schema、audit A008 | 與 production CompactPoseFact 分開 |
| V023 | Direct v4 計算 confidence threshold 為 0.35，並 clamp 計算 confidence | VERIFIED | `pose_geometry.py`、synthetic audit A010、unit tests | 同時註明 raw payload 不回寫 |
| V024 | Direct v4 compact LLM JSON 不含 full pose_window | VERIFIED | `_to_llm_input()`、artifact、audit A016 | debug/compact 分離 |
| V025 | SEG144 artifact 保留 17/17 events 且 schema/provenance 通過 | VERIFIED | `evaluation_v4_prompt.md`、audit artifact check | 限定為單一 local artifact |
| V026 | SEG144 compact/debug 大小約為 96,688 / 279,506 bytes（2.89×） | VERIFIED | local artifact file sizes/package metadata | 不泛化為整體 token savings |
| V027 | V4 Gemini pose semantics 有部分 numeric evidence，但 Commentator readiness 仍是 conditional | PARTIALLY_VERIFIED | 部分 cues 支持；evaluation 同時記錄多個 overclaim | 結論寫為 schema-ready、非 fully commentary-ready |
| V028 | Shuttle observation 可追溯，但 reversal semantic reliability 未被證明 | PARTIALLY_VERIFIED | 有 schema/provenance，但 16/17 reversal 分布可疑且無 deterministic trajectory features | 明確禁止目前送入 Commentator |
| V029 | 目前 2D skeleton 足以穩定判斷正手／反手 | INSUFFICIENT_EVIDENCE | v2 紀錄與 v4 限制相反；沒有 ground truth benchmark | 記錄為不可支持的 claim |
| V030 | 早期 Importance-based sparse event flow 是目前 production selection | HISTORICAL_ONLY | 歷史 docs/commits；目前 batch 強制保留全 stroke | 報告只置於演進段落 |
| V031 | Production runtime 需要 OpenCV/Torch/MMPose/TTS/FFmpeg | NOT_IMPLEMENTED | `pyproject.toml` runtime deps 與 service imports 無此依賴 | 記錄為 experiment/future boundary |
| V032 | Workspace 中存在可供歷程查核的 conversation records | INSUFFICIENT_EVIDENCE | 掃描沒有 conversation/chat/log/history/archive 類目錄 | 報告明載未找到，不推測其內容 |

狀態統計：

| 狀態 | 數量 |
| --- | ---: |
| VERIFIED | 23 |
| PARTIALLY_VERIFIED | 2 |
| HISTORICAL_ONLY | 1 |
| NOT_IMPLEMENTED | 2 |
| CONTRADICTED | 2 |
| INSUFFICIENT_EVIDENCE | 2 |

註：矩陣共 32 個 claims；狀態統計也應合計 32。若後續程式變更，應重新執行 audit 並更新此矩陣，不應只改報告文字。

## 3. 可執行 audit 結果

執行命令：

```powershell
uv run python .\tools\report_implementation_audit.py `
  --package .\outputs\ttyvsasy\direct_rallyfact\seg0144_v4_prompt_test\rally_stage_input.json `
  --output .\docs\reports\implementation_audit_result.json
```

結果：16 `VERIFIED`、0 `FAILED`、0 `NOT_CHECKED`。檢查涵蓋 Python pin、核心 schemas、service methods、Compact/Tactical contracts、production hitting-arm 現況、direct v4 constants、synthetic torso-normalized step width、confidence clamp、Gemini config 及 SEG144 compact transport。

針對 audit 程式的測試：

```text
uv run pytest tests/unit/test_report_implementation_audit.py -q
2 passed

uv run ruff check tools/report_implementation_audit.py tests/unit/test_report_implementation_audit.py
All checks passed!
```

完整 repository tests 與 Ruff 結果記錄於 [report_generation_run.md](report_generation_run.md)。

## 4. Discrepancies Found

| Report claim / 初始文件 claim | Actual implementation | Severity | Correction |
| --- | --- | --- | --- |
| Production pose 使用 `-8/+10` window 與六組 v4 geometry | Production 只在 `±2` frame 找最近單 pose；完整 window 是 direct v4 | algorithm mismatch | 主報告拆成 Production Compact Facts 與 Direct v4 兩節 |
| Hitting arm 已從目前 pose facts 全面移除 | Production `CompactPoseFact` 仍有 candidate；direct v4 geometry 才移除 | schema mismatch | 兩條 schema 分別記錄，禁止互相套用 |
| CompactRallyFacts 尚未送入任何 Gemini prompt | Tactical Analyzer 已序列化 Compact Facts 並呼叫 provider | documentation stale | 改成「已進 Tactical Analyzer，未進 Commentator」 |
| 所有 Analyzers 都是 deterministic Python | Tactical Analyzer 呼叫 LLM；只有 evidence gates 是 deterministic | documentation stale | 把 rule-based analyzers 與 LLM-assisted analyzer 分開 |
| `TacticalFact[] → Planner → Commentator` 是 current closed pipeline | Service 的 tactical 與 generation methods 是平行入口，沒有資料串接 | not-implemented architecture | 架構圖使用虛線並列為下一個 milestone |
| SEG144 enriched output 已足以直接提供 Commentator | Pose 有 overclaim、shuttle 16/17 reversal 可疑、stroke description 有 mojibake | evidence/quality mismatch | 改為 schema-ready、not fully commentary-ready |

## 5. 查核發現與 Phase 1 修正

最重要的 discrepancy 不是數值誤差，而是 execution path 混用：

1. **Production Compact Facts 與 direct v4 不是同一套 pose algorithm。** 前者最近 `±2` frame、單 pose、保留 hitting-arm candidate；後者 `-8/+10` window、六組 geometry、送 LLM 的 compact keyframes 且移除 hitting-arm semantic。
2. **Tactical Analyzer 已存在，但沒有串進 Commentator。** 因此可以測 `CompactFacts → TacticalFact[]`，不能宣稱最終 architecture 已閉環。
3. **Analyzer 不再全是 deterministic。** Stroke/Rally analyzers 是 rule-based；Tactical Analyzer 呼叫 Gemini，但用 deterministic gates 驗證結果。
4. **SEG144 實驗只證明 schema/provenance，不證明 correctness。** Pose 有支持與反例；shuttle 的 reversal 分布尤其不可直接採用。

上述修正已回寫 `algorithm_development_report.md` 的架構圖、演算法表、實驗結論與 Implementation Verification 摘要。

## 6. 尚未能驗證的部分

- 沒有 conversation records，無法確認未記入 Git/docs 的歷史決策理由。
- 沒有 pose/shuttle/tactical ground-truth dataset，無法計算 precision、recall、calibration 或跨比賽泛化。
- 本次 audit 不呼叫 Gemini，無法保證外部模型 availability、latency、quota 或同 prompt 的輸出穩定性。
- Direct v4 與 Tactical Analyzer 位於未提交 working tree；在 commit 前不能把它們當作可由 Git commit hash 重現的 released milestone。
- 主 Badminton Analysis System 的 producers 不在此 repository；只能驗證現有 fixture 與 adapter 所宣告的 consumed schema，不能證明所有上游版本永遠相容。
