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


ATTRACTORS = [
    "Hello, I hope you are well.",
    "Could you please send those documents as soon as possible?",
    "- Maria: own launch slides",
    "Follow-Up on Friday Contract Review",
    "Design reviewed the search flow and agreed links should be clickable.",
    "Your launch code is ORBIT-t May TestFlight build.",
]


def _payload(action: str, content: str, metadata: dict[str, Any]) -> str:
    return json.dumps(
        {
            "action": action,
            "content": content,
            "proposal_metadata": metadata,
        },
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


def build_rows(repeats: int, *, weight: float, negative_weight: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat in range(max(1, int(repeats))):
        for gate in GATES:
            gate_id = str(gate.get("id") or "")
            if bool(gate.get("experimental")) or gate_id not in TARGETS:
                continue
            target = TARGETS[gate_id]
            action = str(target["action"])
            content = str(target["content"])
            metadata = dict(target.get("metadata") or {})
            target_payload = _payload(action, content, metadata)
            for attractor_index, attractor in enumerate(ATTRACTORS):
                if attractor == content:
                    continue
                source_id = f"pocketpal_gate_contrastive_{repeat:04d}_{gate_id}_{attractor_index:02d}"
                intent = GATE_INTENTS.get(gate_id, "")
                rows.append(
                    {
                        "source_id": source_id,
                        "source_type": "pocketpal_gate_contrastive",
                        "task_type": str(metadata.get("task_type") or "agent_gate_contrastive"),
                        "action": action,
                        "intent_label": intent,
                        "intent_label_id": INTENT_LABELS.get(intent, -1),
                        "encoder_text": str(gate["prompt"]),
                        "decoder_text": target_payload,
                        "negative_decoder_text": _payload(action, attractor, metadata),
                        "negative_loss_weight": float(negative_weight),
                        "weight": float(weight),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_gate_contrastive_v1")
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--weight", type=float, default=16.0)
    parser.add_argument("--negative-weight", type=float, default=12.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    rows = build_rows(int(args.repeats), weight=float(args.weight), negative_weight=float(args.negative_weight))
    train_rows = [row for row in rows if _split(str(row["source_id"]), float(args.eval_fraction)) == "train"]
    eval_rows = [row for row in rows if _split(str(row["source_id"]), float(args.eval_fraction)) == "eval"]
    if not eval_rows and train_rows:
        eval_rows.append(train_rows.pop())

    train_path = output_dir / "pocketpal_gate_contrastive_train.jsonl"
    eval_path = output_dir / "pocketpal_gate_contrastive_eval.jsonl"
    manifest_path = output_dir / "pocketpal_gate_contrastive_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    action_counts: dict[str, int] = {}
    for row in rows:
        action_counts[str(row["action"])] = action_counts.get(str(row["action"]), 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": "pocketpal_gate_contrastive",
        "dataset_format": "jsonl",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "source_counts": {"pocketpal_gate_contrastive": len(rows)},
        "target_action_counts": dict(sorted(action_counts.items())),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
