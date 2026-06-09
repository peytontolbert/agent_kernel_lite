#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_pocketpal_agent_gate_repair_dataset import GATE_INTENTS, TARGETS
from scripts.build_pocketpal_stage11_broad_agent_mode_dataset import INTENT_LABELS
from scripts.evaluate_pocketpal_agent_gates import GATES


def _payload(action: str, content: str, metadata: dict[str, Any]) -> str:
    return json.dumps(
        {"action": action, "content": content, "proposal_metadata": metadata},
        ensure_ascii=False,
        sort_keys=True,
    )


def _split(source_id: str, eval_fraction: float) -> str:
    bucket = int(hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < max(0.0, min(0.5, float(eval_fraction))) else "train"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _gate_prompts() -> dict[str, str]:
    return {str(gate.get("id") or ""): str(gate["prompt"]) for gate in GATES if not bool(gate.get("experimental"))}


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    report = json.loads(Path(args.dynamics_json).read_text(encoding="utf-8"))
    prompts = _gate_prompts()
    rows: list[dict[str, Any]] = []
    for gate_report in report.get("reports", []):
        gate_id = str(gate_report.get("id") or "")
        if gate_id not in TARGETS or gate_id not in prompts:
            continue
        target = TARGETS[gate_id]
        action = str(target["action"])
        content = str(target["content"])
        metadata = dict(target.get("metadata") or {})
        target_nll = float((gate_report.get("target_trace") or {}).get("mean_nll", 0.0) or 0.0)
        attractor_scores = gate_report.get("attractor_scores") or []
        if not attractor_scores:
            continue
        best_bad = attractor_scores[0]
        bad_content = str(best_bad.get("content") or "")
        bad_nll = float(best_bad.get("mean_nll", 0.0) or 0.0)
        severity = max(0.0, target_nll - bad_nll + float(args.margin_floor))
        repeats = min(int(args.max_repeats), max(1, int(round(float(args.repeat_scale) * severity))))
        if severity <= 0 and not bool(args.keep_nonnegative):
            continue
        target_payload = _payload(action, content, metadata)
        negative_payload = _payload(action, bad_content, metadata)
        for repeat in range(repeats):
            source_id = f"pocketpal_renorm_basin_{gate_id}_{repeat:04d}"
            intent = GATE_INTENTS.get(gate_id, "")
            rows.append(
                {
                    "source_id": source_id,
                    "source_type": "pocketpal_renorm_basin_repair",
                    "task_type": str(metadata.get("task_type") or "renorm_basin_repair"),
                    "action": action,
                    "intent_label": intent,
                    "intent_label_id": INTENT_LABELS.get(intent, -1),
                    "encoder_text": prompts[gate_id],
                    "decoder_text": target_payload,
                    "negative_decoder_text": negative_payload,
                    "negative_loss_weight": float(args.negative_weight) * max(0.25, severity),
                    "weight": float(args.weight) * max(0.25, severity),
                    "basin_target_nll": target_nll,
                    "basin_bad_nll": bad_nll,
                    "basin_margin": bad_nll - target_nll,
                    "basin_severity": severity,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamics-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--weight", type=float, default=10.0)
    parser.add_argument("--negative-weight", type=float, default=8.0)
    parser.add_argument("--repeat-scale", type=float, default=8.0)
    parser.add_argument("--max-repeats", type=int, default=80)
    parser.add_argument("--margin-floor", type=float, default=0.15)
    parser.add_argument("--keep-nonnegative", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    rows = build_rows(args)
    train_rows = [row for row in rows if _split(str(row["source_id"]), float(args.eval_fraction)) == "train"]
    eval_rows = [row for row in rows if _split(str(row["source_id"]), float(args.eval_fraction)) == "eval"]
    if not eval_rows and train_rows:
        eval_rows.append(train_rows.pop())

    train_path = output_dir / "pocketpal_renorm_basin_repair_train.jsonl"
    eval_path = output_dir / "pocketpal_renorm_basin_repair_eval.jsonl"
    manifest_path = output_dir / "pocketpal_renorm_basin_repair_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    action_counts: dict[str, int] = {}
    for row in rows:
        action_counts[str(row["action"])] = action_counts.get(str(row["action"]), 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": "pocketpal_renorm_basin_repair",
        "dataset_format": "jsonl",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "source_counts": {"pocketpal_renorm_basin_repair": len(rows)},
        "target_action_counts": dict(sorted(action_counts.items())),
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "dynamics_json": str(Path(args.dynamics_json).resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
