#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from pocketpal_structured_decode import CORE_TOKENS, structured_tokens_to_json, json_to_structured_tokens


CASES = [
    {
        "name": "respond_text",
        "json": {"action": "respond", "content": "Hello.", "proposal_metadata": {"task_type": "active_agent_rewrite"}},
        "must": ['"action":"respond"', '"content":"Hello."', '"task_type":"active_agent_rewrite"'],
    },
    {
        "name": "ask_user",
        "json": {"action": "ask_user", "content": "What text should I rewrite?", "proposal_metadata": {"task_type": "ask_missing_text"}},
        "must": ['"action":"ask_user"', "What text should I rewrite?"],
    },
    {
        "name": "extension_request",
        "json": {"action": "extension_request", "content": "Requesting approval to search the web.", "proposal_metadata": {"task_type": "web_search_request"}},
        "must": ['"action":"extension_request"', "Requesting approval"],
    },
    {
        "name": "save_memory",
        "json": {"action": "save_memory", "content": "Remember launch code ORBIT-42.", "proposal_metadata": {"task_type": "save_memory"}},
        "must": ['"action":"save_memory"', "ORBIT-42"],
    },
    {
        "name": "copy_source",
        "json": {"action": "respond", "content": "vendor invoice INV-2048", "proposal_metadata": {"task_type": "source_slot_copy"}},
        "source_text": "vendor invoice INV-2048",
        "use_copy_source": True,
        "must": ["vendor invoice INV-2048"],
    },
    {
        "name": "intent_payload",
        "tokens": "<AK_STRUCTURED> <AK_ACTION_RESPOND> <AK_TASK_TYPE> active_agent_json <AK_INTENT> web_search <AK_FRESHNESS> current <AK_END>",
        "must": ['"intent":"web_search"', '"freshness":"current"'],
    },
    {
        "name": "field_payload",
        "tokens": "<AK_STRUCTURED> <AK_ACTION_RESPOND> <AK_TASK_TYPE> active_agent_extraction <AK_FIELD> <AK_FIELD_NAME> owner <AK_FIELD_VALUE> Priya <AK_FIELD> <AK_FIELD_NAME> date <AK_FIELD_VALUE> Friday <AK_END>",
        "must": ['"owner":"Priya"', '"date":"Friday"'],
    },
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_sampler(repo_root: Path):
    path = repo_root / "scripts" / "sample_agentkernel_lite_encdec.py"
    spec = importlib.util.spec_from_file_location("sample_agentkernel_lite_encdec", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load sampler script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_round_trips() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in CASES:
        source_text = str(case.get("source_text") or "")
        if "tokens" in case:
            tokens = str(case["tokens"])
        else:
            tokens = json_to_structured_tokens(
                json.dumps(case["json"], ensure_ascii=False, separators=(",", ":")),
                source_text=source_text,
                use_copy_source=bool(case.get("use_copy_source")),
            )
        rendered = structured_tokens_to_json(tokens, source_text=source_text)
        try:
            parsed = json.loads(rendered)
            content = str(parsed.get("content") or "")
        except Exception:
            content = ""
        failures = [needle for needle in case["must"] if needle not in rendered and needle not in content]
        results.append({"name": case["name"], "passed": not failures, "tokens": tokens, "json": rendered, "failures": failures})
    return results


def _check_bundle_tokens(bundle_dir: Path, repo_root: Path) -> list[dict[str, Any]]:
    sampler = _load_sampler(repo_root)
    manifest = sampler._load_manifest(bundle_dir)
    tokenizer = sampler._load_tokenizer(manifest)
    rows: list[dict[str, Any]] = []
    for token in CORE_TOKENS:
        ids = [
            int(token_id)
            for token_id in tokenizer.encode(token, max_length=16)
            if int(token_id)
            not in {
                int(tokenizer.bos_token_id),
                int(tokenizer.eos_token_id),
                int(tokenizer.pad_token_id),
            }
        ]
        rows.append({"token": token, "ids": ids, "atomic": len(ids) == 1})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--bundle-dir", default="")
    parser.add_argument("--require-atomic", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    round_trips = _check_round_trips()
    token_rows: list[dict[str, Any]] = []
    if str(args.bundle_dir).strip():
        token_rows = _check_bundle_tokens(Path(args.bundle_dir).expanduser().resolve(), repo_root)
    summary = {
        "round_trips": round_trips,
        "round_trip_passed": sum(1 for row in round_trips if row["passed"]),
        "round_trip_total": len(round_trips),
        "bundle_token_rows": token_rows,
        "bundle_atomic_passed": sum(1 for row in token_rows if row.get("atomic")),
        "bundle_atomic_total": len(token_rows),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    failed = [row for row in round_trips if not row["passed"]]
    if failed:
        raise SystemExit(1)
    if bool(args.require_atomic) and any(not row.get("atomic") for row in token_rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
