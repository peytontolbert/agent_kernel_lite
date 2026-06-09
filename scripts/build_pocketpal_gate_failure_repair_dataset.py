#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


IDEAL_CONTENT: dict[str, str] = {
    "runtime_plain_greeting": "I'm doing well. How can I help?",
    "active_agent_rewrite_greeting": "Hello, I hope you are well.",
    "active_agent_bullet_summary_same_input": "- Greeting: Hi, how are you?",
    "professional_email_rewrite": "Hi John, could you please send the report by Friday? The client is asking, and we are behind. Thank you.",
    "casual_retention": "It's going well. How's it going for you?",
    "source_echo_no_slots": "Source text: vendor invoice INV-2048 is blocked until finance approves $1,200",
    "ask_missing_text": "What text should I rewrite?",
    "saved_data_use": "Your launch code is ORBIT-42 for the May TestFlight build.",
    "missing_user_data": "I do not have saved data for your hotel confirmation code. Which saved data source should I use?",
    "web_search_request": "Requesting approval to search the web.",
    "web_result_synthesis": "PocketPal beta build 42 is planned for June 3 with web search enabled for up to five sources. The checklist says links must be clickable and agent actions require local approval.",
    "source_slots_experimental": "Source text: vendor invoice INV-2048 is blocked until finance approves $1,200",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_gates() -> list[dict[str, Any]]:
    path = _repo_root() / "scripts" / "evaluate_pocketpal_agent_gates.py"
    spec = importlib.util.spec_from_file_location("evaluate_pocketpal_agent_gates", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load gates: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.GATES)


def _payload(action: str, content: str, gate: dict[str, Any]) -> str:
    metadata: dict[str, Any] = {"task_type": "gate_failure_repair"}
    if action == "extension_request":
        metadata.update(
            {
                "extension_id": "web_search",
                "capability": "web.search",
                "query": str(gate.get("text_slots", {}).get("SOURCE_TEXT") or ""),
                "max_sources": 5,
                "requires_user_approval": True,
            }
        )
    return json.dumps({"action": action, "content": content, "proposal_metadata": metadata}, ensure_ascii=False)


def _load_bad_outputs(paths: list[Path]) -> dict[str, list[str]]:
    by_gate: dict[str, list[str]] = {}
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for result in data.get("results", []):
            gate_id = str(result.get("id") or "")
            output = str(result.get("output") or "").strip()
            if gate_id and output:
                by_gate.setdefault(gate_id, []).append(output)
    return by_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--failure-json", action="append", default=[])
    parser.add_argument("--repeat-train", type=int, default=80)
    parser.add_argument("--repeat-eval", type=int, default=8)
    parser.add_argument("--include-experimental", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    gates = _load_gates()
    bad_outputs = _load_bad_outputs([Path(item).expanduser().resolve() for item in args.failure_json])
    counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()

    def rows_for(split: str, repeat: int):
        for gate in gates:
            if bool(gate.get("experimental")) and not bool(args.include_experimental):
                continue
            gate_id = str(gate["id"])
            if gate_id not in IDEAL_CONTENT:
                continue
            action = str(gate.get("action") or "respond")
            target = _payload(action, IDEAL_CONTENT[gate_id], gate)
            negatives = bad_outputs.get(gate_id) or []
            if not negatives:
                negatives = [""]
            for n in range(max(1, int(repeat))):
                negative = negatives[n % len(negatives)]
                row = {
                    "example_id": f"{gate_id}_{split}_{n:04d}",
                    "split": split,
                    "source_type": "pocketpal_gate_failure_repair",
                    "source_id": gate_id,
                    "task_type": "gate_failure_repair",
                    "encoder_text": str(gate["prompt"]),
                    "decoder_text": target,
                    "negative_decoder_text": negative,
                    "negative_loss_weight": 1.0 if negative else 0.0,
                    "action": action,
                    "weight": 3.0,
                }
                counts[split] += 1
                action_counts[action] += 1
                yield row

    with train_path.open("w", encoding="utf-8") as handle:
        for row in rows_for("train", int(args.repeat_train)):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with eval_path.open("w", encoding="utf-8") as handle:
        for row in rows_for("eval", int(args.repeat_eval)):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "objective": "pocketpal_gate_failure_repair",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "train_examples": int(counts["train"]),
        "eval_examples": int(counts["eval"]),
        "total_examples": int(counts["train"] + counts["eval"]),
        "target_action_counts": dict(sorted(action_counts.items())),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
