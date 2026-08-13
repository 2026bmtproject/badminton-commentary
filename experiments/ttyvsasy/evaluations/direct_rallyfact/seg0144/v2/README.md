# SEG144 direct all-stages experiment — v2

This directory preserves one prompt/output pair for the experimental direct conversion:

```text
seven upstream stage slices
        ↓ Gemini multimodal analysis
experimental-enriched-rally-fact-v2
```

## Artifacts

- `prompt.txt`: exact supplied prompt, including the v2 output contract.
- `gemini_output.json`: exact provider output.
- `metadata.json`: hashes, basic JSON validation and run-level notes.

The full stage payload is intentionally not duplicated in this tracked directory because pose and shuttle
slices are large. It can be regenerated from the ignored local workspace with
`experiments/ttyvsasy/scripts/package_direct_rallyfact.py`.

## Basic validation

```text
schema_version: experimental-enriched-rally-fact-v2
segment_index: 144
events: 17
rally_length: 17
tactical_candidates: 3
JSON: valid
```

## Important run observation

Although the prompt describes all seven stages, this output sets every court and shuttle observation to
`null`, reports those stages as missing, and reports pose data as truncated before events 1294–1296. This
record therefore captures the provider output faithfully, but it should not be treated as proof that a
complete all-stages request was successfully consumed. Confirm the actual uploaded/request payload before
using this run for a v1/v2 quality comparison.

## Pose evidence conclusion

As of 2026-08-11, this experiment treats single-view 2D skeleton data as insufficient evidence for a
reliable forehand/backhand classification. Shoulder, elbow and wrist coordinates do not establish racket
orientation, grip, contact side, body rotation in 3D or the player's handedness at the moment of contact.

Therefore, a Gemini-generated `forehand_backhand_candidate` from the current pose stage is an unsupported
research hypothesis, not a verified rally fact. It must not be passed to Planner or Commentator as a
grounded claim. The field should remain `unknown`/`null` or be excluded until stronger evidence or labeled
validation is available.

No production schema or pipeline behavior is changed by this stored experiment.
