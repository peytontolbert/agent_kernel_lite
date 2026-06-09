#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

from pocketpal_structured_decode import json_to_structured_tokens


STRUCTURED_JSON_PREFIX = "<AK_STRUCTURED> <AK_ACTION_RESPOND> <AK_TASK_TYPE> active_agent_json "


def _stable_id(*parts: object) -> str:
    return hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _payload_with_content(row: dict[str, Any], content: str | None = None) -> str:
    raw = str(row.get("decoder_text") or row.get("json_decoder_text") or "").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"action": "respond", "proposal_metadata": {"task_type": "active_agent_json"}}
    if not isinstance(payload, dict):
        payload = {"action": "respond", "proposal_metadata": {"task_type": "active_agent_json"}}
    payload["action"] = str(payload.get("action") or "respond")
    metadata = payload.get("proposal_metadata") if isinstance(payload.get("proposal_metadata"), dict) else {}
    metadata["task_type"] = "active_agent_json"
    payload["proposal_metadata"] = metadata
    if content is not None:
        payload["content"] = str(content)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _structured_suffix(row: dict[str, Any], *, content: str | None = None) -> str:
    target = json_to_structured_tokens(_payload_with_content(row, content=content))
    if target.startswith(STRUCTURED_JSON_PREFIX):
        return target[len(STRUCTURED_JSON_PREFIX) :].strip()
    return target


def _json_row(row: dict[str, Any], *, index: int, weight: float, source_tag: str, content: str | None = None) -> dict[str, Any]:
    out = deepcopy(row)
    suffix = _structured_suffix(out, content=content)
    out["decoder_train_prefix"] = STRUCTURED_JSON_PREFIX.strip()
    out["decoder_text"] = suffix
    out["json_decoder_text"] = _payload_with_content(row, content=content)
    out["expected_content"] = str(content if content is not None else out.get("expected_content") or "")
    out["negative_decoder_text"] = ""
    out["negative_loss_weight"] = 0.0
    out["decoder_loss_weight"] = 1.0
    out["weight"] = float(weight)
    out["split"] = "train"
    out["source_type"] = f"structured_json_{source_tag}"
    out["example_id"] = _stable_id("structured_json", source_tag, index, out.get("source_id"), suffix)
    out["source_id"] = f"{out.get('source_id', 'row')}_structured_json_{source_tag}_{index:05d}"
    return out


def _anchor_row(row: dict[str, Any], *, index: int, weight_cap: float) -> dict[str, Any]:
    out = deepcopy(row)
    out.pop("decoder_train_prefix", None)
    out["split"] = "train"
    out["weight"] = min(max(float(out.get("weight") or 1.0), 1.0), float(weight_cap))
    out["source_type"] = f"{out.get('source_type', 'row')}_structured_json_anchor"
    out["example_id"] = _stable_id("structured_json_anchor", index, out.get("source_id"), out.get("encoder_text"), out.get("decoder_text"))
    out["source_id"] = f"{out.get('source_id', 'row')}_structured_json_anchor_{index:05d}"
    return out


def _synthetic_encoder(source_text: str, *, instruction: str = "Classify the user request as compact JSON.") -> str:
    return (
        "<AK_CHAT> <AK_RESPOND> PocketPal user-configured agent example.\n"
        "<AK_AGENT_ACTIVE>\n"
        "Agent name: JSON Classifier\n"
        f"Agent instruction: {instruction}\n"
        "Retrieval policy: auto\n"
        "Tool policy: ask_before_extensions\n"
        "Action policy: respond_or_ask\n"
        "The active agent instruction is the primary task contract for this turn.\n"
        "</AK_AGENT_ACTIVE>\n"
        "<AK_TASK_HINT> intent=json task=active_agent_json source_text_required=true\n"
        "<AK_CONTEXT> Saved user data: none\n"
        "<AK_PROFILE> User text slots:\n"
        f"<AK_SLOT> <AK_SLOT_NAME>=SOURCE_TEXT <AK_SLOT_VALUE>={source_text}\n"
        "Available placeholders for this turn: [[SOURCE_TEXT]].\n"
        "Use only the available placeholders listed above. Do not invent unavailable placeholders such as [[NAME]], [[ITEM]], [[DEADLINE]], or [[REASON]] unless they are listed for this turn.\n"
        "<AK_CONTEXT> Stale selected paper context: Selected paper [P1]: unrelated research paper context.\n"
        "Use stale paper context only when the current user request asks about that paper or research evidence.\n"
        f"<AK_USER> {source_text}\n"
        "Return compact JSON with the correct action and content for the active agent."
    )


def _synthetic_json_rows(*, repeats: int, weight: float) -> list[dict[str, Any]]:
    examples = [
        ("Rank these tasks by urgency.", '{"intent":"ranking","criterion":"urgency"}'),
        ("Sort these tasks by priority.", '{"intent":"ranking","criterion":"urgency"}'),
        ("Prioritize the launch tasks.", '{"intent":"ranking","criterion":"urgency"}'),
        ("Find current news about TestFlight processing.", '{"intent":"web_search","freshness":"current"}'),
        ("Look up the latest TestFlight processing status.", '{"intent":"web_search","freshness":"current"}'),
        ("Search online for recent app review delays.", '{"intent":"web_search","freshness":"current"}'),
        ("Please make this more professional.", '{"intent":"rewrite","tone":"professional"}'),
        ("Rewrite this in a professional tone.", '{"intent":"rewrite","tone":"professional"}'),
        ("Polish this message for work.", '{"intent":"rewrite","tone":"professional"}'),
        ("Translate this into Spanish.", '{"intent":"translation","target_language":"spanish"}'),
        ("Translate this to French.", '{"intent":"translation","target_language":"french"}'),
        ("Extract the owner and deadline.", '{"intent":"extraction","fields":["owner","deadline"]}'),
        ("Pull out the owner and due date.", '{"intent":"extraction","fields":["owner","deadline"]}'),
    ]
    rows: list[dict[str, Any]] = []
    for repeat in range(max(0, int(repeats))):
        for index, (source_text, content) in enumerate(examples):
            payload = {
                "action": "respond",
                "content": content,
                "proposal_metadata": {"task_type": "active_agent_json"},
            }
            raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            suffix = json_to_structured_tokens(raw)
            if suffix.startswith(STRUCTURED_JSON_PREFIX):
                suffix = suffix[len(STRUCTURED_JSON_PREFIX) :].strip()
            rows.append(
                {
                    "action": "respond",
                    "decoder_train_prefix": STRUCTURED_JSON_PREFIX.strip(),
                    "decoder_text": suffix,
                    "json_decoder_text": raw,
                    "encoder_text": _synthetic_encoder(source_text),
                    "expected_content": content,
                    "intent_label": "json",
                    "intent_label_id": 13,
                    "negative_decoder_text": "",
                    "negative_loss_weight": 0.0,
                    "decoder_loss_weight": 1.0,
                    "retrieval_loss_weight": 0.0,
                    "retrieval_query_text": "",
                    "retrieval_doc_text": "",
                    "source_id": f"synthetic_structured_json_{index:02d}_{repeat:04d}",
                    "source_type": "structured_json_synthetic",
                    "split": "train",
                    "task_type": "active_agent_json",
                    "weight": float(weight),
                    "example_id": _stable_id("structured_json_synthetic", index, repeat, source_text, content),
                }
            )
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(int(args.seed))
    source_manifest_path = Path(args.source_manifest).expanduser().resolve()
    source_manifest = _load_manifest(source_manifest_path)
    train_rows = list(_iter_jsonl(Path(source_manifest["train_dataset_path"])))
    eval_rows = list(_iter_jsonl(Path(source_manifest["eval_dataset_path"])))
    by_source = {str(row.get("source_id") or ""): row for row in train_rows + eval_rows}

    rows: list[dict[str, Any]] = []
    synthetic_rows = _synthetic_json_rows(repeats=int(args.synthetic_repeats), weight=float(args.synthetic_weight))
    rows.extend(synthetic_rows)
    json_candidates = [row for row in train_rows if str(row.get("task_type") or "") == "active_agent_json"]
    rng.shuffle(json_candidates)
    for index, row in enumerate(json_candidates[: max(0, int(args.json_rows))]):
        rows.append(_json_row(row, index=index, weight=float(args.json_weight), source_tag="anchor"))

    failure_count = 0
    for failure_path in [Path(path).expanduser().resolve() for path in list(args.failure_json or [])]:
        if not failure_path.exists():
            continue
        failures = json.loads(failure_path.read_text(encoding="utf-8")).get("failures", [])
        for failure in failures:
            if str(failure.get("task_type") or "") != "active_agent_json":
                continue
            row = by_source.get(str(failure.get("source_id") or ""))
            if row is None:
                continue
            expected = str(failure.get("expected") or "").strip()
            for repeat in range(max(1, int(args.failure_repeat))):
                rows.append(
                    _json_row(
                        row,
                        index=failure_count * max(1, int(args.failure_repeat)) + repeat,
                        weight=float(args.failure_weight),
                        source_tag="failure",
                        content=expected or None,
                    )
                )
            failure_count += 1

    anchors = list(train_rows)
    rng.shuffle(anchors)
    rows.extend(
        _anchor_row(row, index=index, weight_cap=float(args.anchor_weight_cap))
        for index, row in enumerate(anchors[: max(0, int(args.anchor_rows))])
    )
    rng.shuffle(rows)

    out_eval: list[dict[str, Any]] = []
    for index, row in enumerate(eval_rows[: min(len(eval_rows), int(args.eval_rows))]):
        copied = deepcopy(row)
        copied.pop("decoder_train_prefix", None)
        copied["split"] = "eval"
        copied["example_id"] = _stable_id("structured_json_eval", index, copied.get("source_id"), copied.get("encoder_text"))
        out_eval.append(copied)

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    train_count = _write_jsonl(train_path, rows)
    eval_count = _write_jsonl(eval_path, out_eval)
    manifest = {
        **source_manifest,
        "artifact_kind": "agentkernel_lite_structured_json_curriculum",
        "objective": "pocketpal_structured_json_decoder_alignment",
        "source_manifest_path": str(source_manifest_path),
        "source_failure_json": [str(Path(path).expanduser().resolve()) for path in list(args.failure_json or [])],
        "structured_json_prefix": STRUCTURED_JSON_PREFIX.strip(),
        "json_rows": int(min(len(json_candidates), max(0, int(args.json_rows)))),
        "json_weight": float(args.json_weight),
        "synthetic_json_rows": int(len(synthetic_rows)),
        "synthetic_repeats": int(args.synthetic_repeats),
        "synthetic_weight": float(args.synthetic_weight),
        "failure_json_rows": int(failure_count),
        "failure_repeat": int(args.failure_repeat),
        "failure_weight": float(args.failure_weight),
        "anchor_rows": int(args.anchor_rows),
        "anchor_weight_cap": float(args.anchor_weight_cap),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "train_examples": int(train_count),
        "eval_examples": int(eval_count),
        "total_examples": int(train_count + eval_count),
        "manifest_path": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--failure-json", action="append", default=[])
    parser.add_argument("--json-rows", type=int, default=2400)
    parser.add_argument("--json-weight", type=float, default=6.0)
    parser.add_argument("--synthetic-repeats", type=int, default=0)
    parser.add_argument("--synthetic-weight", type=float, default=10.0)
    parser.add_argument("--failure-repeat", type=int, default=8)
    parser.add_argument("--failure-weight", type=float, default=32.0)
    parser.add_argument("--anchor-rows", type=int, default=26000)
    parser.add_argument("--anchor-weight-cap", type=float, default=8.0)
    parser.add_argument("--eval-rows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=414)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
