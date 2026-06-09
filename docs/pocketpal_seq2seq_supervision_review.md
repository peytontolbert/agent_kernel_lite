# PocketPal Tiny Seq2Seq Training Review

Generated: 2026-05-18T13:35:21.968154+00:00

## Executive Findings

- The tiny seq2seq controller is not primarily failing at coarse routing. Agent gates are green in the latest run, retrieval top-1 is near 0.795, and the recorded intent head evaluation reaches about 0.8995 accuracy.
- The weak layer is faithful direct generation. The best recent direct-agent prompt pass rate found by this audit is 0.417 from `v192b`, while the latest direct eval is 0.308 from `v201a`.
- The latest failure replay run appears to regress generation quality: it preserves gate pass status but direct prompt pass rate drops, with low-recall and malformed JSON/content drift still present.
- The eval history is fragmented across `/tmp` JSON files rather than attached to bundle manifests. This makes promotion decisions easy to lose and allows a model with green small gates to hide broad generation regressions.
- Several dataset manifests still use JSONL. That is acceptable for small smoke evals, but full training/eval corpora should be promoted to Parquet to reduce disk and memory pressure.

## Latest Signals

- Latest gates: version `v192b`, passed `True`, required `10/10`.
- Latest direct prompts: version `v201a`, pass rate `0.308`, mean recall `0.330`, zero-pass tasks `5`.
- Latest retrieval top-1: version `v201a`, accuracy `0.795`, mean margin `0.092`, negative margins `1`.
- Latest intent head: version `v200a`, accuracy `0.899`.

## Recent Artifacts

| version | objective | steps | eval loss | train examples | eval examples | frozen encoder | frozen decoder |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| v197a | pocketpal_v196_weak_skill_drill | 80 | 0.104 | 83277 | 2605 | False | False |
| v197b_step40 | pocketpal_v196_weak_skill_drill |  | n/a | 83277 | 2605 | False | False |
| v198a | pocketpal_v198_structured_exactness_drill | 80 | 2.591 | 22214 | 1788 | False | False |
| v199a | pocketpal_v199_task_discrimination_drill | 120 | 1.325 | 38777 | 1096 | False | False |
| v200a | pocketpal_v172d_broad_plus_greeting_mix | 240 | 0.199 | 352659 | 25009 | False | False |
| v200b | pocketpal_v172d_broad_plus_greeting_mix | 360 | 0.200 | 352659 | 25009 | False | False |
| v200b_step240 | pocketpal_v172d_broad_plus_greeting_mix |  | n/a | 352659 | 25009 | False | False |
| v201a | pocketpal_v201_failure_replay_from_v200a_eval | 100 | 1.914 | 32277 | 813 | False | False |

## Diagnosis

The model is behaving like a useful router plus a weak constrained generator. It can often select a surface, pass tiny canned gates, and embed retrieval candidates, but it still corrupts content under direct prompt pressure. The repeated failures show task drift (`rewrite` becoming `summary` or `translation`), malformed JSON, copied fragments from unrelated examples, and low recall on extraction/json/translation/risk tasks.

The current training ladder also appears to optimize one failure class at a time while regressing others. `v201a` failure replay improved neither the broad direct prompt score nor the content recall compared with `v197a` or `v200a`; it only kept the narrow gate suite green.

## Required Training Gate

Every candidate bundle should be rejected unless it records, in a single promotion JSON and Parquet row:

1. Agent gates pass all required cases.
2. Direct-agent prompt pass rate improves or stays within a small regression budget against the promoted baseline.
3. No protected task has zero pass rate.
4. Malformed outputs are zero or explicitly below a configured threshold.
5. Retrieval top-1 does not regress on harness-skill examples.
6. Intent head confusion does not regress on protected route pairs: rewrite/action_items, summary/extraction, json/rewrite, extraction/casual.
7. Browser/WASM export parity is checked before promotion.

Run model-backed evals from the `ai` conda environment so the evaluator uses CUDA PyTorch:

```bash
conda run -n ai python scripts/evaluate_pocketpal_controller.py --bundle-dir artifacts/<candidate_bundle>
```

## Model Direction

Do not ask the decoder to carry the whole controller contract. Keep the tiny model as a hybrid action model:

- encoder heads decide intent, action validity, retrieval need, confidence, OOD, and verification need;
- retrieval head supplies the relevant skill/operator context;
- decoder emits only a constrained decision object or short content when the route is simple;
- deterministic templates handle source echo, saved data references, missing slot questions, and extension requests;
- high-entropy content tasks use retrieved exemplars or a larger teacher, while this model chooses the action and constraints.

## Files Emitted

- Artifacts parquet: `/data/agent_kernel_lite/artifacts/pocketpal_seq2seq_review/20260518_133521/pocketpal_artifacts.parquet`
- Eval parquet: `/data/agent_kernel_lite/artifacts/pocketpal_seq2seq_review/20260518_133521/pocketpal_eval_trends.parquet`
- Summary JSON: `/data/agent_kernel_lite/artifacts/pocketpal_seq2seq_review/20260518_133521/pocketpal_seq2seq_review_summary.json`
