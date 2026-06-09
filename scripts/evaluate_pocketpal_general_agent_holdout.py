#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contains(text: str, needle: str) -> bool:
    return str(needle).lower() in str(text).lower()


def _strict_json_dict(text: str) -> bool:
    try:
        return isinstance(json.loads(str(text or "").strip()), dict)
    except json.JSONDecodeError:
        return False


def _cases(ev) -> list[dict[str, Any]]:
    return [
        {
            "id": "rewrite_new_fact_pattern",
            "prompt": ev._agent_prompt(
                name="Professional Email Rewriter",
                instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                user_text="yo lena send the budget draft by june 3 because finance is waiting",
                text_slots={"NAME": "Lena", "ITEM": "budget draft", "DEADLINE": "June 3", "REASON": "finance is waiting"},
            ),
            "text_slots": {"NAME": "Lena", "ITEM": "budget draft", "DEADLINE": "June 3", "REASON": "finance is waiting"},
            "action": "respond",
            "must": ["Lena", "budget draft", "June 3", "finance"],
            "must_not": ["John", "report", "Friday"],
        },
        {
            "id": "source_copy_new_value",
            "prompt": ev._agent_prompt(
                name="Source Echo Agent",
                instruction="Return the exact user-provided source text with a short label. Preserve all names, dates, values, and wording.",
                user_text="Nora's access code is GATE-17 for Thursday at 2 PM",
                text_slots={"SOURCE_TEXT": "Nora's access code is GATE-17 for Thursday at 2 PM"},
            ),
            "text_slots": {"SOURCE_TEXT": "Nora's access code is GATE-17 for Thursday at 2 PM"},
            "action": "respond",
            "must": ["Nora", "GATE-17", "Thursday", "2 PM"],
            "must_not": ["INV-20", "John", "report"],
        },
        {
            "id": "classify_writing",
            "prompt": ev._agent_prompt(
                name="Classifier Agent",
                instruction="Classify the user's text into exactly one label: travel, finance, schedule, writing, web_search. Return only the label.",
                user_text="Can you rewrite this note professionally?",
                text_slots={"SOURCE_TEXT": "Can you rewrite this note professionally?"},
            ),
            "text_slots": {"SOURCE_TEXT": "Can you rewrite this note professionally?"},
            "action": "respond",
            "must": ["writing"],
            "must_not": ["finance", "travel"],
        },
        {
            "id": "memory_uses_saved_data",
            "prompt": ev._agent_prompt(
                name="Saved Data Assistant",
                instruction="Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. If no relevant saved data is available, ask one concise question.",
                user_text="what badge code do I use",
                stale_context="Selected paper [P1]: unrelated optimization notes.",
                text_slots={"SOURCE_TEXT": "what badge code do I use", "DATA_CONTEXT": "[D1] saved note: Nora's badge code is GATE-17 for the south entrance."},
            ),
            "text_slots": {"SOURCE_TEXT": "what badge code do I use", "DATA_CONTEXT": "[D1] saved note: Nora's badge code is GATE-17 for the south entrance."},
            "action": "respond",
            "must": ["GATE-17", "south entrance"],
            "must_not": ["paper", "optimization"],
        },
        {
            "id": "search_request_new_topic",
            "prompt": ev._web_agent_prompt(user_text="search the web for current Swift WKWebView navigation policy examples"),
            "action": "extension_request",
            "must": ["search"],
            "must_not": ["cannot browse"],
            "metadata_must": {"extension_id": "web_search", "capability": "web.search", "max_sources": 5, "requires_user_approval": True},
        },
    ]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    sampler = _load_module(repo_root / "scripts" / "sample_agentkernel_lite_encdec.py", "sample_agentkernel_lite_encdec")
    gates = _load_module(repo_root / "scripts" / "evaluate_pocketpal_agent_gates.py", "evaluate_pocketpal_agent_gates")
    sampler._install_paths(repo_root)

    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    manifest = sampler._load_manifest(bundle_dir)
    config = load_config(str(manifest["model_dir"]))
    tokenizer = sampler._load_tokenizer(manifest)
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    sampler._materialize_lazy_modules(model)
    load_pretrained(model, str(manifest["model_dir"]), strict=True)
    device = torch.device(str(args.device))
    model.to(device).eval()

    results: list[dict[str, Any]] = []
    passed = 0
    for case in _cases(gates):
        decoder_prefix = gates._decoder_prefix_for_gate(case, enabled=bool(args.use_action_prefix)) or ""
        output = sampler._generate(
            model,
            tokenizer,
            str(case["prompt"]),
            decoder_prefix=decoder_prefix,
            device=device,
            max_encoder_tokens=int(args.max_encoder_tokens),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
        )
        parsed = gates._parse_decision(output)
        content = str((parsed or {}).get("content", "") or "")
        content = gates._expand_text_slots(content, case.get("text_slots"))
        failures: list[str] = []
        if not _strict_json_dict(output):
            failures.append("malformed_json")
        if parsed is None:
            failures.append("invalid_json")
        elif str(parsed.get("action", "") or "") != str(case["action"]):
            failures.append(f"action:{parsed.get('action')!r}")
        for needle in case.get("must", []):
            if not _contains(content, str(needle)):
                failures.append(f"missing:{needle}")
        for needle in case.get("must_not", []):
            if _contains(content, str(needle)):
                failures.append(f"forbidden:{needle}")
        metadata = (parsed or {}).get("proposal_metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        for key, expected in case.get("metadata_must", {}).items():
            actual = gates._metadata_value(metadata, str(key))
            if bool(args.allow_runtime_extension_fallback) and key in {"extension_id", "capability", "max_sources", "requires_user_approval"}:
                continue
            if actual != expected:
                failures.append(f"metadata:{key}:{actual!r}")
        ok = not failures
        passed += 1 if ok else 0
        results.append({"id": case["id"], "passed": ok, "failures": failures, "output": output, "content": content})

    return {
        "bundle_dir": str(bundle_dir),
        "passed": int(passed),
        "total": len(results),
        "pass_rate": float(passed) / float(len(results) or 1),
        "ok": passed == len(results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-encoder-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--use-action-prefix", type=int, choices=(0, 1), default=1)
    parser.add_argument("--allow-runtime-extension-fallback", type=int, choices=(0, 1), default=1)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    summary = evaluate(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if str(args.output_json).strip():
        Path(args.output_json).expanduser().resolve().write_text(text + "\n", encoding="utf-8")
    print(text)
    if not bool(summary["ok"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
