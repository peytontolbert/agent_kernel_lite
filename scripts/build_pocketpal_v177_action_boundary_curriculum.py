#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


INTENT_LABELS = {
    "plan": 0,
    "action_items": 1,
    "rewrite": 2,
    "translation": 3,
    "web_search": 4,
    "casual": 5,
    "source_echo": 6,
    "saved_data": 7,
    "ask_user": 8,
    "summary": 9,
    "title": 10,
    "checklist": 11,
    "risks": 12,
    "json": 13,
    "ranking": 14,
    "extraction": 15,
    "subject": 16,
    "brainstorm": 17,
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


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def _hash_split(key: str, eval_fraction: float) -> str:
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if value < eval_fraction else "train"


def _payload(action: str, content: str, task_type: str, metadata: dict[str, Any] | None = None) -> str:
    proposal_metadata = {"task_type": task_type}
    if metadata:
        proposal_metadata.update(metadata)
    return json.dumps(
        {"action": action, "content": content, "proposal_metadata": proposal_metadata},
        ensure_ascii=False,
        sort_keys=True,
    )


def _decision_content(text: str) -> tuple[str, str]:
    try:
        payload = json.loads(str(text or ""))
    except json.JSONDecodeError:
        return "respond", str(text or "").strip()
    if not isinstance(payload, dict):
        return "respond", str(text or "").strip()
    return str(payload.get("action") or "respond"), str(payload.get("content") or "").strip()


def _add_row(
    rows: list[dict[str, Any]],
    *,
    encoder_text: str,
    action: str,
    content: str,
    task_type: str,
    intent: str,
    source_id: str,
    split: str,
    weight: float,
    negative_action: str = "",
    negative_content: str = "",
    negative_weight: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> None:
    negative_decoder_text = ""
    if negative_action and negative_content:
        negative_decoder_text = _payload(negative_action, negative_content, task_type, metadata)
    rows.append(
        {
            "action": action,
            "decoder_text": _payload(action, content, task_type, metadata),
            "encoder_text": encoder_text,
            "example_id": source_id,
            "expected_content": content,
            "intent_label": intent,
            "intent_label_id": INTENT_LABELS.get(intent, -1),
            "negative_decoder_text": negative_decoder_text,
            "negative_loss_weight": float(negative_weight),
            "source_id": source_id,
            "source_type": "pocketpal_v177_action_boundary_curriculum",
            "split": split,
            "task_type": task_type,
            "weight": float(weight),
        }
    )


def _add_boundary_anchors(rows: list[dict[str, Any]], repeats: int, eval_fraction: float) -> None:
    gates = _load_gates()
    ask_specs = [
        ("rewrite this", "Please provide the text you want rewritten."),
        ("make this professional", "Please provide the text you want made professional."),
        ("summarize this", "Please provide the text you want summarized."),
        ("translate this", "Please provide the text you want translated and the target language."),
        ("extract the facts", "Please provide the text you want me to extract facts from."),
    ]
    for repeat in range(repeats):
        for index, (user_text, content) in enumerate(ask_specs):
            prompt = gates._agent_prompt(
                name="Professional Email Rewriter",
                instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                user_text=user_text,
            )
            source_id = f"v177_ask_missing_text_{repeat:04d}_{index:02d}"
            _add_row(
                rows,
                encoder_text=prompt,
                action="ask_user",
                content=content,
                task_type="active_agent_missing_text",
                intent="ask_user",
                source_id=source_id,
                split=_hash_split(source_id, eval_fraction),
                weight=42.0,
                negative_action="respond",
                negative_content="Hello, I hope you are well.",
                negative_weight=1.0,
            )

    rewrite_specs = [
        ("Devon", "roadmap update", "next Tuesday", "Planning needs it", "devon get the roadmap update ready next tuesday because planning needs it"),
        ("Lena", "budget draft", "June 3", "Finance is waiting", "yo lena send the budget draft by june 3 because finance is waiting"),
        ("Ava", "invoice", "tomorrow morning", "Finance needs it", "ava please send the invoice tomorrow morning because finance needs it"),
        ("Mira", "launch notes", "Friday", "Ops needs them", "mira send the launch notes friday because ops needs them"),
    ]
    for repeat in range(max(1, repeats // 2)):
        for index, (name, item, deadline, reason, user_text) in enumerate(rewrite_specs):
            slots = {"NAME": name, "ITEM": item, "DEADLINE": deadline, "REASON": reason}
            prompt = gates._agent_prompt(
                name="Professional Email Rewriter",
                instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                user_text=user_text,
                text_slots=slots,
            )
            content = f"Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you."
            source_id = f"v177_rewrite_slot_anchor_{repeat:04d}_{index:02d}"
            _add_row(
                rows,
                encoder_text=prompt,
                action="respond",
                content=content,
                task_type="active_agent_rewrite",
                intent="rewrite",
                source_id=source_id,
                split=_hash_split(source_id, eval_fraction),
                weight=34.0,
                negative_action="respond",
                negative_content="Maria, Devin, and Sam each have launch tasks with near-term deadlines.",
                negative_weight=1.0,
            )

    classify_specs = [
        ("Please approve invoice INV-2048 for $1,200.", "finance"),
        ("Can you rewrite this note professionally?", "writing"),
        ("Find current Swift WKWebView examples online.", "web_search"),
        ("Book flights to Austin next week.", "travel"),
        ("Move my planning meeting to Tuesday.", "schedule"),
    ]
    for repeat in range(repeats):
        for index, (text, label) in enumerate(classify_specs):
            prompt = gates._agent_prompt(
                name="Classifier Agent",
                instruction="Classify the user's text into exactly one label: travel, finance, schedule, writing, web_search. Return only the label.",
                user_text=text,
                text_slots={"SOURCE_TEXT": text},
            )
            wrong = "web_search" if label != "web_search" else "finance"
            source_id = f"v177_classify_anchor_{repeat:04d}_{index:02d}"
            _add_row(
                rows,
                encoder_text=prompt,
                action="respond",
                content=label,
                task_type="active_agent_classify",
                intent="web_search" if label == "web_search" else "casual",
                source_id=source_id,
                split=_hash_split(source_id, eval_fraction),
                weight=38.0,
                negative_action="respond",
                negative_content=wrong,
                negative_weight=1.0,
            )

    for repeat in range(max(1, repeats // 3)):
        user_text = "search the web for current Swift WKWebView navigation policy examples"
        source_id = f"v177_web_extension_anchor_{repeat:04d}"
        _add_row(
            rows,
            encoder_text=gates._web_agent_prompt(user_text=user_text),
            action="extension_request",
            content="Requesting approval to search the web.",
            task_type="runtime_web_search_request",
            intent="web_search",
            source_id=source_id,
            split=_hash_split(source_id, eval_fraction),
            weight=28.0,
            negative_action="respond",
            negative_content="web_search",
            negative_weight=1.0,
            metadata={
                "capability": "web.search",
                "extension_id": "web_search",
                "max_sources": 5,
                "query": user_text,
                "requires_user_approval": True,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-repeat", type=int, default=1)
    parser.add_argument("--anchor-repeat", type=int, default=420)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    args = parser.parse_args()

    source_manifest = json.loads(Path(args.input_manifest).read_text(encoding="utf-8"))
    source_paths = [Path(source_manifest["train_dataset_path"]), Path(source_manifest["eval_dataset_path"])]
    rows: list[dict[str, Any]] = []
    for path in source_paths:
        for row in _iter_jsonl(path):
            task_type = str(row.get("task_type") or "")
            action, content = _decision_content(str(row.get("decoder_text") or ""))
            if not content:
                continue
            for repeat in range(int(args.base_repeat)):
                converted = dict(row)
                converted["action"] = action
                converted["decoder_text"] = _payload(action, content, task_type)
                converted["expected_content"] = content
                converted["example_id"] = f"{row.get('example_id') or row.get('source_id')}_v177base_{repeat:02d}"
                converted["source_type"] = "pocketpal_v177_action_boundary_base"
                converted["split"] = _hash_split(str(converted["example_id"]), float(args.eval_fraction))
                converted["weight"] = min(max(float(row.get("weight") or 1.0), 1.0), 18.0)
                rows.append(converted)

    _add_boundary_anchors(rows, int(args.anchor_repeat), float(args.eval_fraction))

    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "eval"]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    for path, split_rows in [(train_path, train_rows), (eval_path, eval_rows)]:
        with path.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path),
        "eval_examples": len(eval_rows),
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_v177_action_boundary_curriculum",
        "source_manifest_path": str(Path(args.input_manifest).resolve()),
        "target_action_counts": dict(sorted(Counter(str(row.get("action") or "") for row in rows).items())),
        "task_type_counts": dict(sorted(Counter(str(row.get("task_type") or "") for row in rows).items())),
        "total_examples": len(rows),
        "train_dataset_path": str(train_path),
        "train_examples": len(train_rows),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
