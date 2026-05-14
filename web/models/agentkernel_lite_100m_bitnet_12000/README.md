# Agent Kernel Lite 100M BitNet Browser Bundle

Source artifact: `artifacts/pocketpal_controller_100m_v68_weighted_title_repair_from_v67`.

This bundle builds on the stable v54/v62 decoder behavior, keeps the v65/v66 generation-safe controller path, and applies v68 weighted intent-head repair using the stage18 title/confusion curriculum. The training script now weights intent-head cross entropy by each row's `loss_weight`, so repair examples actually influence the controller.

Validation:
- Required generation gates: 10/10 pass (`tmp/pocketpal_gate_v68_weighted_title_repair_required.json`)
- Intent head, stage15 balanced eval: 98.70% (`tmp/pocketpal_intent_head_v68_on_stage15_balanced_eval.json`)
- Intent head, stage16 hard-negative eval: 90.41% (`tmp/pocketpal_intent_head_v68_hard_negative_eval.json`)
- Intent head, stage18 title/confusion-repair eval: 94.26% (`tmp/pocketpal_intent_head_v68_stage18_eval.json`)

The intent head is an assistive controller signal. It is much stronger than v66, but routing should still use confidence thresholds and conservative fallbacks until random-mode generation improves.
