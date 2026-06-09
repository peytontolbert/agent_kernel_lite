#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd


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


def _compact(value: object, *, limit: int = 1400) -> str:
    if value is None:
        raw = ""
    elif isinstance(value, (list, tuple, set)):
        raw = ", ".join(str(item) for item in value if str(item).strip())
    elif hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            raw = ", ".join(str(item) for item in converted if str(item).strip())
        else:
            raw = str(converted)
    else:
        raw = str(value)
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:limit].rstrip()


def _one_line(value: object, *, limit: int = 260) -> str:
    return " ".join(_compact(value, limit=limit * 2).split())[:limit].rstrip()


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _stable_sample(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= int(limit):
        return rows
    rng = random.Random(int(seed))
    return rng.sample(rows, int(limit))


def _skill_doc(row: pd.Series) -> str:
    pieces = [
        f"repo={_one_line(row.get('source_repo'), limit=120)}",
        f"path={_one_line(row.get('source_path'), limit=180)}",
        f"kind={_one_line(row.get('skill_kind'), limit=80)} primitive={_one_line(row.get('primitive_type'), limit=80)}",
        f"summary={_compact(row.get('annotation_summary'), limit=420)}",
        f"use_when={_compact(row.get('annotation_use_when'), limit=420)}",
        f"patch_relevance={_compact(row.get('annotation_patch_relevance'), limit=360)}",
        f"risks={_compact(row.get('annotation_risks'), limit=260)}",
        f"verify={_compact(row.get('annotation_verification_hints'), limit=360)}",
    ]
    excerpt = _compact(row.get("source_excerpt"), limit=700)
    if excerpt:
        pieces.append(f"excerpt={excerpt}")
    return "\n".join(piece for piece in pieces if piece.split("=", 1)[-1].strip())


def _skill_query(row: pd.Series) -> str:
    summary = _one_line(row.get("annotation_summary"), limit=160)
    use_when = _one_line(row.get("annotation_use_when"), limit=180)
    primitive = _one_line(row.get("primitive_type"), limit=80)
    path = _one_line(row.get("source_path"), limit=120)
    return f"Need a PocketPal operator for {summary}. Use when: {use_when}. Primitive: {primitive}. Path: {path}".strip()


def _operator_decoder(row: pd.Series) -> str:
    operator = _one_line(row.get("primitive_type"), limit=80) or "retrieved_skill"
    verifier = _one_line(row.get("annotation_verification_hints"), limit=220) or "verify the selected skill matches the user request"
    risk = _one_line(row.get("annotation_risks"), limit=180) or "no known risk"
    payload = {
        "action": "gather_context",
        "content": "<AK_RETRIEVE> <AK_RET_SKILLS> <AK_VERIFY>",
        "proposal_metadata": {
            "task_type": "harness_skill_operator_policy",
            "operator": operator,
            "verifier": verifier,
            "risk": risk,
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _skill_encoder(row: pd.Series) -> str:
    return "\n".join(
        [
            "<AK_CHAT> <AK_RETRIEVE> PocketPal local controller skill selection.",
            "<AK_AGENT_ACTIVE>",
            "Agent name: PocketPal Controller",
            "Agent instruction: Select the relevant local skill/operator, choose a verifier, and require approval for tools when needed.",
            "Retrieval policy: auto",
            "Tool policy: ask_before_extensions",
            "Action policy: retrieve_then_constrain",
            "</AK_AGENT_ACTIVE>",
            "<AK_CONTEXT> Installed local skill catalog is available through retrieval.",
            f"<AK_USER> {_skill_query(row)}",
        ]
    )


def _skill_rows(skill_path: Path, *, limit: int, eval_limit: int, negatives: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    df = pd.read_parquet(skill_path, engine="pyarrow")
    df = df.fillna("")
    if len(df) > int(limit) + int(eval_limit):
        df = df.sample(n=int(limit) + int(eval_limit), random_state=int(seed)).reset_index(drop=True)
    docs = [_skill_doc(row) for _, row in df.iterrows()]
    rng = random.Random(int(seed) + 17)
    rows: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(df.iterrows()):
        doc = docs[index]
        if not doc.strip():
            continue
        negative_indices: set[int] = set()
        while len(negative_indices) < min(int(negatives), max(0, len(docs) - 1)):
            candidate = rng.randrange(0, len(docs))
            if candidate != index:
                negative_indices.add(candidate)
        negative_docs = [docs[i] for i in sorted(negative_indices)]
        confidence = float(row.get("annotation_confidence") or 0.75)
        if confidence <= 0.0:
            confidence = 0.70
        source_repo = str(row.get("source_repo") or "")
        needs_verification = 0.95 if _one_line(row.get("annotation_verification_hints"), limit=80) else 0.75
        permissions = _one_line(row.get("required_permissions"), limit=240).lower()
        requires_approval = any(token in permissions for token in ["shell", "write", "network"])
        rows.append(
            {
                "action": "gather_context",
                "answer_confidence_target": min(0.95, max(0.55, confidence)),
                "decoder_text": _operator_decoder(row),
                "encoder_text": _skill_encoder(row),
                "example_id": f"v202_skill_operator_{index:06d}",
                "expected_content": "<AK_RETRIEVE>",
                "intent_label": "",
                "intent_label_id": -1,
                "negative_decoder_text": "",
                "negative_loss_weight": 0.0,
                "needs_verification_target": needs_verification,
                "ood_evidence_target": 0.10,
                "ood_query_target": 0.08,
                "paper_action_validity_target": 0.80 if requires_approval else 0.92,
                "query_confidence_target": min(0.95, max(0.55, confidence)),
                "retrieval_coverage_target": 0.92,
                "retrieval_doc_text": doc,
                "retrieval_loss_weight": 1.0,
                "retrieval_negative_doc_texts": json.dumps(negative_docs, ensure_ascii=False),
                "retrieval_query_text": _skill_query(row),
                "source_id": str(row.get("id") or f"skill_{index:06d}"),
                "source_type": f"v202_openclaw_hermes_operator_{source_repo.replace('/', '_')}",
                "split": "eval" if index < int(eval_limit) else "train",
                "task_type": "harness_skill_operator_policy",
                "weight": 0.0,
            }
        )
    train = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "eval"]
    return train, eval_rows


def _broad_rows(manifest_path: Path, *, train_limit: int, eval_limit: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = [dict(row) for row in _iter_jsonl(Path(manifest["train_dataset_path"]))]
    eval_rows = [dict(row) for row in _iter_jsonl(Path(manifest["eval_dataset_path"]))]
    train = _stable_sample(train, int(train_limit), int(seed))
    eval_rows = _stable_sample(eval_rows, int(eval_limit), int(seed) + 1)
    for row in train + eval_rows:
        row["source_type"] = f"v202_broad_controller_protect_{row.get('source_type', '')}"
        row["weight"] = float(row.get("weight", 1.0) or 1.0)
        row.setdefault("retrieval_loss_weight", 0.0)
    return train, eval_rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad-manifest", default="tmp/pocketpal_v172d_broad_plus_greeting_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--skill-parquet", default="/data/repo_skills_miner/artifacts/hf_openclaw_hermes_skills/data/train.parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--broad-train", type=int, default=80000)
    parser.add_argument("--broad-eval", type=int, default=6000)
    parser.add_argument("--skill-train", type=int, default=50000)
    parser.add_argument("--skill-eval", type=int, default=4000)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()

    broad_train, broad_eval, broad_manifest = _broad_rows(
        Path(args.broad_manifest),
        train_limit=int(args.broad_train),
        eval_limit=int(args.broad_eval),
        seed=int(args.seed),
    )
    skill_train, skill_eval = _skill_rows(
        Path(args.skill_parquet),
        limit=int(args.skill_train),
        eval_limit=int(args.skill_eval),
        negatives=int(args.negatives),
        seed=int(args.seed),
    )
    train_rows = broad_train + skill_train
    eval_rows = broad_eval + skill_eval
    random.Random(int(args.seed) + 3).shuffle(train_rows)
    random.Random(int(args.seed) + 4).shuffle(eval_rows)

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path),
        "eval_examples": len(eval_rows),
        "intent_labels": dict(broad_manifest.get("intent_labels") or INTENT_LABELS),
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_v202_hybrid_controller_operator_policy",
        "retrieval_sources": [str(Path(args.skill_parquet).resolve())],
        "source_counts": {
            "broad_controller_protection_train": len(broad_train),
            "broad_controller_protection_eval": len(broad_eval),
            "openclaw_hermes_operator_train": len(skill_train),
            "openclaw_hermes_operator_eval": len(skill_eval),
        },
        "schema": {
            "decoder_text": "constrained action JSON; v202 training can set decoder loss to zero",
            "policy_heads": "confidence/retrieval/ood/verification/action-validity targets",
            "retrieval": "skill query, positive skill doc, hard negative skill docs",
        },
        "total_examples": len(train_rows) + len(eval_rows),
        "train_dataset_path": str(train_path),
        "train_examples": len(train_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
