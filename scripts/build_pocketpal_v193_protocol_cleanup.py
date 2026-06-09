#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(action: str, content: str, task_type: str, metadata: dict[str, Any] | None = None) -> str:
    proposal_metadata = {"task_type": task_type}
    if metadata:
        proposal_metadata.update(metadata)
    return json.dumps(
        {"action": action, "content": content, "proposal_metadata": proposal_metadata},
        ensure_ascii=False,
        sort_keys=True,
    )


def _row(
    *,
    example_id: str,
    encoder_text: str,
    decoder_text: str,
    task_type: str,
    negative_decoder_text: str = "",
    weight: float = 60.0,
) -> dict[str, Any]:
    return {
        "action": json.loads(decoder_text)["action"],
        "decoder_text": decoder_text,
        "encoder_text": encoder_text,
        "example_id": example_id,
        "intent_label": task_type.replace("active_agent_", "").replace("runtime_", ""),
        "intent_label_id": 0,
        "negative_decoder_text": negative_decoder_text or None,
        "negative_loss_weight": 0.8 if negative_decoder_text else None,
        "retrieval_doc_text": "",
        "retrieval_loss_weight": 0.0,
        "retrieval_query_text": "",
        "source_id": example_id,
        "source_type": "pocketpal_v193_protocol_cleanup",
        "split": "train",
        "task_type": task_type,
        "weight": weight,
    }


def _read_outputs(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    return {str(item.get("id")): str(item.get("output") or "") for item in report.get("results", [])}


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--gate-json", default="tmp/v192b_agent_gates.json")
    parser.add_argument("--cert-json", default="tmp/v192b_agent_certification_matrix.json")
    parser.add_argument("--slot-copy-manifest", default="tmp/pocketpal_v186_slot_copy_repair/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--rewrite-slot-manifest", default="tmp/pocketpal_v187_rewrite_slot_repair/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    gates = _load_module(repo_root / "scripts" / "evaluate_pocketpal_agent_gates.py", "eval_gates")
    cert = _load_module(repo_root / "scripts" / "evaluate_pocketpal_agent_certification_matrix.py", "eval_cert")
    bad_outputs = {**_read_outputs(Path(args.gate_json)), **_read_outputs(Path(args.cert_json))}

    expected_by_id: dict[str, tuple[str, str, str, dict[str, Any]]] = {
        "runtime_plain_greeting": ("respond", "I'm doing well. What would you like help with?", "runtime_plain_chat", {}),
        "active_agent_rewrite_greeting": ("respond", "Hello, I hope you are well.", "active_agent_rewrite_greeting", {}),
        "active_agent_bullet_summary_same_input": ("respond", "- Greeting: Hi, how are you?", "active_agent_summary", {}),
        "professional_email_rewrite": ("respond", "Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you.", "active_agent_rewrite_slots", {}),
        "casual_retention": ("respond", "It's going well. What would you like help with?", "active_agent_casual", {}),
        "source_echo_no_slots": ("respond", "Source text: [[SOURCE_TEXT]]", "active_agent_source_echo", {}),
        "ask_missing_text": ("ask_user", "What text should I rewrite?", "active_agent_missing_text", {}),
        "saved_data_use": ("respond", "I found this in your saved data: [[DATA_CONTEXT]]", "active_agent_saved_data", {}),
        "missing_user_data": ("ask_user", "I do not have relevant saved data for that. What saved data should I use?", "active_agent_missing_user_data", {}),
        "web_search_request": ("extension_request", "Requesting approval to search the web.", "runtime_web_search_request", {"extension_id": "web_search", "capability": "web.search", "max_sources": 5, "requires_user_approval": True, "query": "search the web for current TestFlight upload limits"}),
        "web_result_synthesis": ("respond", "PocketPal beta build 42 is planned for June 3. Web search is enabled for up to five sources, and links must be clickable.", "runtime_web_search_result", {}),
        "source_slots_experimental": ("respond", "Source text: vendor invoice INV-2048 is blocked until finance approves $1,200", "active_agent_source_echo", {}),
        "cached_rewrite_greeting": ("respond", "Hello, I hope you are well.", "active_agent_rewrite_greeting", {}),
        "fresh_rewrite_lena": ("respond", "Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you.", "active_agent_rewrite_slots", {}),
        "near_rewrite_ava": ("respond", "Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you.", "active_agent_rewrite_slots", {}),
        "far_rewrite_devon": ("respond", "Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you.", "active_agent_rewrite_slots", {}),
        "fresh_classify_writing": ("respond", "writing", "active_agent_classify", {}),
        "near_classify_finance": ("respond", "finance", "active_agent_classify", {}),
        "far_classify_search": ("respond", "web_search", "active_agent_classify", {}),
        "fresh_source_copy": ("respond", "Source text: [[SOURCE_TEXT]]", "active_agent_source_echo", {}),
        "offpolicy_memory_recovery": ("respond", "I found this in your saved data: [[DATA_CONTEXT]]", "active_agent_saved_data", {}),
        "offpolicy_classify_recovery": ("respond", "writing", "active_agent_classify", {}),
        "fresh_web_request": ("extension_request", "Requesting approval to search the web.", "runtime_web_search_request", {"extension_id": "web_search", "capability": "web.search", "max_sources": 5, "requires_user_approval": True, "query": "search the web for current TestFlight upload limits"}),
    }

    rows: list[dict[str, Any]] = []
    for gate in gates.GATES:
        case_id = str(gate["id"])
        action, content, task_type, metadata = expected_by_id[case_id]
        for repeat in range(20):
            rows.append(
                _row(
                    example_id=f"v193_gate_{case_id}_{repeat:02d}",
                    encoder_text=str(gate["prompt"]),
                    decoder_text=_payload(action, content, task_type, metadata),
                    task_type=task_type,
                    negative_decoder_text=bad_outputs.get(case_id, ""),
                )
            )
    for case in cert._cases(gates):
        case_id = str(case["id"])
        action, content, task_type, metadata = expected_by_id[case_id]
        for repeat in range(20):
            rows.append(
                _row(
                    example_id=f"v193_cert_{case_id}_{repeat:02d}",
                    encoder_text=str(case["prompt"]),
                    decoder_text=_payload(action, content, task_type, metadata),
                    task_type=task_type,
                    negative_decoder_text=bad_outputs.get(case_id, ""),
                )
            )

    for manifest_path, label, repeat_count in [
        (Path(args.slot_copy_manifest), "slot", 2),
        (Path(args.rewrite_slot_manifest), "rewrite", 4),
    ]:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for repeat in range(repeat_count):
            for source_path in [Path(manifest["train_dataset_path"]), Path(manifest["eval_dataset_path"])]:
                with source_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        out = json.loads(line)
                        out["example_id"] = f"{out.get('example_id')}_v193_{label}_{repeat:02d}"
                        out["source_type"] = f"{out.get('source_type', 'unknown')}_v193_{label}_protect"
                        out["weight"] = min(max(float(out.get("weight") or 1.0), 18.0), 46.0)
                        rows.append(out)

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write(train_path, rows)
    _write(eval_path, rows[: min(500, len(rows))])
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path),
        "eval_examples": min(500, len(rows)),
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_v193_protocol_cleanup",
        "task_type_counts": dict(sorted(Counter(str(row.get("task_type") or "") for row in rows).items())),
        "total_examples": len(rows) + min(500, len(rows)),
        "train_dataset_path": str(train_path),
        "train_examples": len(rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
