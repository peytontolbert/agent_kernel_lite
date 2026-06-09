#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = {
    "casual": 5,
    "source_echo": 6,
    "saved_data": 7,
    "ask_user": 8,
    "summary": 9,
    "web_search": 4,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_gates():
    path = _repo_root() / "scripts" / "evaluate_pocketpal_agent_gates.py"
    spec = importlib.util.spec_from_file_location("evaluate_pocketpal_agent_gates", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load gates module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(action: str, content: str, task_type: str) -> str:
    return json.dumps({"action": action, "content": content, "proposal_metadata": {"task_type": task_type}}, ensure_ascii=False, sort_keys=True)


def _add(rows: list[dict[str, Any]], split: str, idx: int, prompt: str, action: str, content: str, task_type: str, intent: str, weight: float, negative: str = "") -> int:
    rows.append(
        {
            "action": action,
            "decoder_text": _payload(action, content, task_type),
            "encoder_text": prompt,
            "example_id": f"v185_{task_type}_{split}_{idx:06d}",
            "expected_content": content,
            "intent_label": intent,
            "intent_label_id": LABELS.get(intent, -1),
            "negative_decoder_text": negative,
            "negative_loss_weight": 1.0 if negative else 0.0,
            "source_id": f"v185_{task_type}",
            "source_type": "pocketpal_v185_source_casual_microrepair",
            "split": split,
            "task_type": task_type,
            "weight": float(weight),
        }
    )
    return idx + 1


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _hash_float(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def _rows(split: str, repeats: int) -> list[dict[str, Any]]:
    gates = _load_gates()
    rows: list[dict[str, Any]] = []
    idx = 0
    source_cases = [
        "vendor invoice INV-2048 is blocked until finance approves $1,200",
        "Nora will bring badge GATE-17 to the south entrance on Thursday at 2 PM",
        "Casey needs permit P-443 before the 8:30 AM delivery window",
        "Upload build 42 after Priya confirms links are clickable",
        "Avery owes receipt R-778 for $86.45 before Monday",
        "Room C12 is reserved for Blake at 11:15 AM on June 3",
        "Ticket AKL-902 remains open until Jordan uploads logs.zip",
        "The launch memo is due Friday and owner is Harper",
    ]
    casual_cases = [
        ("How's it going?", "It's going well. What would you like help with?"),
        ("hi how are you?", "I'm doing well. What would you like help with?"),
        ("what's up?", "Not much. What can I help with?"),
        ("are you there?", "Yes, I'm here. What would you like help with?"),
    ]
    for _ in range(repeats):
        for text in source_cases:
            prompt = gates._agent_prompt(
                name="Source Echo Agent",
                instruction="Return the exact user-provided source text with a short label. Preserve all names, dates, values, and wording.",
                user_text=text,
                text_slots={"SOURCE_TEXT": text},
            )
            idx = _add(
                rows,
                split,
                idx,
                prompt,
                "respond",
                f"Source text: {text}",
                "active_agent_source_echo",
                "source_echo",
                72.0,
                _payload("respond", "- Owner: [[NAME]]\n- Object: [[ITEM]]\n- Date: [[DATE]]", "negative_extract_instead_of_copy"),
            )
        for text, content in casual_cases:
            idx = _add(
                rows,
                split,
                idx,
                gates._agent_prompt(
                    name="Casual Assistant",
                    instruction="Answer casual messages naturally and briefly. Do not use stale context unless the user asks about it.",
                    user_text=text,
                    stale_context="Selected paper [P1]: unrelated optimization notes from a previous research turn.",
                    text_slots={"SOURCE_TEXT": text},
                ),
                "respond",
                content,
                "active_agent_casual",
                "casual",
                54.0,
                _payload("respond", "Source text: [[SOURCE_TEXT]]", "negative_source_echo_for_casual"),
            )
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v184-manifest", default="tmp/pocketpal_v184_boundary_repair_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-repeats", type=int, default=140)
    parser.add_argument("--eval-repeats", type=int, default=12)
    parser.add_argument("--protect", type=int, default=4000)
    args = parser.parse_args()

    train_rows = _rows("train", int(args.train_repeats))
    eval_rows = _rows("eval", int(args.eval_repeats))
    manifest = json.loads(Path(args.v184_manifest).read_text(encoding="utf-8"))
    protect_rows = list(_iter_jsonl(Path(manifest["train_dataset_path"])))
    protect_rows.sort(key=lambda row: _hash_float(str(row.get("example_id") or row.get("encoder_text"))))
    for index, row in enumerate(protect_rows[: int(args.protect)]):
        out = dict(row)
        out["example_id"] = f"{row.get('example_id')}_v185_protect_{index:06d}"
        out["source_type"] = f"{row.get('source_type', 'unknown')}_v185_protection"
        out["weight"] = min(max(float(row.get("weight") or 1.0), 2.0), 10.0)
        train_rows.append(out)

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write(train_path, train_rows)
    _write(eval_path, eval_rows)
    all_rows = train_rows + eval_rows
    out_manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path),
        "eval_examples": len(eval_rows),
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_v185_source_casual_microrepair",
        "task_type_counts": dict(sorted(Counter(str(row.get("task_type") or "") for row in all_rows).items())),
        "total_examples": len(all_rows),
        "train_dataset_path": str(train_path),
        "train_examples": len(train_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(out_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
