#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


INTENT_LABELS = {
    "rewrite": 2,
    "web_search": 4,
    "casual": 5,
    "source_echo": 6,
    "saved_data": 7,
    "ask_user": 8,
    "summary": 9,
    "checklist": 11,
    "risks": 12,
    "json": 13,
    "action_items": 14,
    "extraction": 15,
    "subject": 16,
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


def _payload(action: str, content: str, task_type: str, metadata: dict[str, Any] | None = None) -> str:
    proposal_metadata = {"task_type": task_type}
    if metadata:
        proposal_metadata.update(metadata)
    return json.dumps({"action": action, "content": content, "proposal_metadata": proposal_metadata}, ensure_ascii=False, sort_keys=True)


def _add(
    rows: list[dict[str, Any]],
    split: str,
    idx: int,
    prompt: str,
    action: str,
    content: str,
    task_type: str,
    intent: str,
    weight: float,
    negative: str = "",
    metadata: dict[str, Any] | None = None,
) -> int:
    rows.append(
        {
            "action": action,
            "decoder_text": _payload(action, content, task_type, metadata),
            "encoder_text": prompt,
            "example_id": f"v184_{task_type}_{split}_{idx:06d}",
            "expected_content": content,
            "intent_label": intent,
            "intent_label_id": INTENT_LABELS.get(intent, -1),
            "negative_decoder_text": negative,
            "negative_loss_weight": 1.0 if negative else 0.0,
            "source_id": f"v184_{task_type}",
            "source_type": "pocketpal_v184_boundary_repair_mix",
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


def _normalize_row(row: dict[str, Any], suffix: str, weight_floor: float, weight_cap: float) -> dict[str, Any]:
    out = dict(row)
    out["example_id"] = f"{row.get('example_id') or row.get('source_id')}_{suffix}"
    out["source_type"] = f"{row.get('source_type', 'unknown')}_v184_protection"
    out["weight"] = min(max(float(row.get("weight") or 1.0), weight_floor), weight_cap)
    return out


def _repair_rows(split: str, repeats: int) -> list[dict[str, Any]]:
    gates = _load_gates()
    rows: list[dict[str, Any]] = []
    idx = 0
    web_meta = {
        "capability": "web.search",
        "extension_id": "web_search",
        "max_sources": 5,
        "query": "search the web for current TestFlight upload limits",
        "requires_user_approval": True,
    }
    source_cases = [
        "vendor invoice INV-2048 is blocked until finance approves $1,200",
        "Nora will bring badge GATE-17 to the south entrance on Thursday at 2 PM",
        "Casey needs permit P-443 before the 8:30 AM delivery window",
        "Upload build 42 after Priya confirms links are clickable",
    ]
    missing_cases = [
        "what is my hotel confirmation code",
        "what gate code do I use for the conference",
        "what is the wifi password I saved",
        "where did I store the renewal receipt",
    ]
    saved_cases = [
        ("what badge code do I use", "[D1] saved note: Nora's badge code is GATE-17 for the south entrance."),
        ("what is my launch code", "[D1] saved note (note, chunk 1): Launch code is ORBIT-42 for the May TestFlight build."),
        ("when is the upload", "[D1] saved note: TestFlight upload is planned for Thursday at 2 PM."),
    ]
    classify_cases = [
        ("Find current Swift WKWebView examples online.", "web_search"),
        ("Can you search the web for today's App Store Connect outage status?", "web_search"),
        ("Can you rewrite this note professionally?", "writing"),
        ("Please approve invoice INV-2048 for $1,200.", "finance"),
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
                42.0,
                _payload("respond", "- Owner: [[NAME]]\n- Object: [[ITEM]]", "negative_extraction_attractor"),
            )
        for text in missing_cases:
            prompt = gates._agent_prompt(
                name="Saved Data Assistant",
                instruction="Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. If no relevant saved data is available, ask one concise question.",
                user_text=text,
                stale_context="Retrieved evidence [1]: unrelated web-search result about insurance markets.",
                text_slots={"SOURCE_TEXT": text},
            )
            idx = _add(
                rows,
                split,
                idx,
                prompt,
                "ask_user",
                "I do not have relevant saved data for that. What saved data should I use?",
                "active_agent_missing_user_data",
                "ask_user",
                46.0,
                _payload("respond", "I found this in your saved data: [[DATA_CONTEXT]]", "negative_saved_data_without_data_context"),
            )
        for user_text, data in saved_cases:
            prompt = gates._agent_prompt(
                name="Saved Data Assistant",
                instruction="Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. If no relevant saved data is available, ask one concise question.",
                user_text=user_text,
                stale_context="Selected paper [P1]: unrelated transformer attention paper.",
                text_slots={"SOURCE_TEXT": user_text, "DATA_CONTEXT": data},
            )
            idx = _add(
                rows,
                split,
                idx,
                prompt,
                "respond",
                "I found this in your saved data: [[DATA_CONTEXT]]",
                "active_agent_saved_data",
                "saved_data",
                40.0,
                _payload("ask_user", "I do not have relevant saved data for that. What saved data should I use?", "negative_missing_when_data_exists"),
            )
        for text, label in classify_cases:
            prompt = gates._agent_prompt(
                name="Classifier Agent",
                instruction="Classify the user's text into exactly one label: travel, finance, schedule, writing, web_search. Return only the label.",
                user_text=text,
                text_slots={"SOURCE_TEXT": text},
            )
            idx = _add(
                rows,
                split,
                idx,
                prompt,
                "respond",
                label,
                "active_agent_classify",
                "web_search" if label == "web_search" else "casual",
                30.0,
                _payload("respond", "writing" if label == "web_search" else "web_search", "negative_classifier_boundary"),
            )
        anchors = [
            (gates._runtime_plain_chat_prompt("Hi how are you?"), "respond", "I'm doing well. What would you like help with?", "runtime_plain_chat", "casual", 18.0, None),
            (
                gates._agent_prompt(
                    name="Professional Email Rewriter",
                    instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                    user_text="Hi how are you?",
                    text_slots={"SOURCE_TEXT": "Hi how are you?"},
                ),
                "respond",
                "Hello, I hope you are well.",
                "active_agent_rewrite_greeting",
                "rewrite",
                22.0,
                None,
            ),
            (
                gates._agent_prompt(
                    name="Professional Email Rewriter",
                    instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                    user_text="rewrite this",
                ),
                "ask_user",
                "What text should I rewrite?",
                "active_agent_missing_text",
                "ask_user",
                20.0,
                None,
            ),
            (gates._web_agent_prompt(user_text="search the web for current TestFlight upload limits"), "extension_request", "Requesting approval to search the web.", "runtime_web_search_request", "web_search", 24.0, web_meta),
        ]
        for prompt, action, content, task_type, intent, weight, metadata in anchors:
            idx = _add(rows, split, idx, prompt, action, content, task_type, intent, weight, metadata=metadata)
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v183-manifest", default="tmp/pocketpal_v183_broad_agent_repair_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--retrieval-manifest", default="tmp/pocketpal_v182_openclaw_hermes_retrieval_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-repeats", type=int, default=220)
    parser.add_argument("--eval-repeats", type=int, default=20)
    parser.add_argument("--broad-protect", type=int, default=12000)
    parser.add_argument("--retrieval-protect", type=int, default=2500)
    args = parser.parse_args()

    v183_manifest = json.loads(Path(args.v183_manifest).read_text(encoding="utf-8"))
    retrieval_manifest = json.loads(Path(args.retrieval_manifest).read_text(encoding="utf-8"))
    train_rows = _repair_rows("train", int(args.train_repeats))
    eval_rows = _repair_rows("eval", int(args.eval_repeats))

    broad_rows = list(_iter_jsonl(Path(v183_manifest["train_dataset_path"])))
    broad_rows.sort(key=lambda row: _hash_float(str(row.get("example_id") or row.get("source_id") or row.get("encoder_text"))))
    for index, row in enumerate(broad_rows[: int(args.broad_protect)]):
        train_rows.append(_normalize_row(row, f"v184_broad_{index:06d}", weight_floor=2.5, weight_cap=8.0))

    retrieval_added = 0
    for row in _iter_jsonl(Path(retrieval_manifest["train_dataset_path"])):
        if retrieval_added >= int(args.retrieval_protect):
            break
        if str(row.get("retrieval_doc_text") or "").strip():
            protected = dict(row)
            protected["example_id"] = f"{row.get('example_id')}_v184_retrieval"
            protected["source_type"] = "v182_retrieval_protection_v184"
            protected["weight"] = 0.0
            train_rows.append(protected)
            retrieval_added += 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write(train_path, train_rows)
    _write(eval_path, eval_rows)
    all_rows = train_rows + eval_rows
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path),
        "eval_examples": len(eval_rows),
        "intent_labels": INTENT_LABELS,
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_v184_boundary_repair_mix",
        "task_type_counts": dict(sorted(Counter(str(row.get("task_type") or "") for row in all_rows).items())),
        "total_examples": len(all_rows),
        "train_dataset_path": str(train_path),
        "train_examples": len(train_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
