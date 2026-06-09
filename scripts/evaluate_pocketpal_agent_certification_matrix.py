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


def _case(case_id: str, regime: str, prompt: str, action: str, must: list[str], must_not: list[str] | None = None, text_slots: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "id": case_id,
        "regime": regime,
        "prompt": prompt,
        "action": action,
        "must": must,
        "must_not": must_not or [],
        "text_slots": text_slots or {},
    }


def _cases(ev) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out.append(
        _case(
            "cached_rewrite_greeting",
            "cached",
            ev._agent_prompt(
                name="Professional Email Rewriter",
                instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                user_text="Hi how are you?",
                text_slots={"SOURCE_TEXT": "Hi how are you?"},
            ),
            "respond",
            ["Hello", "well"],
            ["John", "report"],
            {"SOURCE_TEXT": "Hi how are you?"},
        )
    )
    rewrite_specs = [
        ("fresh_rewrite_lena", "fresh", "Lena", "budget draft", "June 3", "Finance is waiting", "yo lena send the budget draft by june 3 because finance is waiting"),
        ("near_rewrite_ava", "near", "Ava", "invoice", "tomorrow morning", "Finance needs it", "ava please send the invoice tomorrow morning because finance needs it"),
        ("far_rewrite_devon", "far", "Devon", "roadmap update", "next Tuesday", "Planning needs it", "devon get the roadmap update ready next tuesday because planning needs it"),
    ]
    for case_id, regime, name, item, deadline, reason, user_text in rewrite_specs:
        slots = {"NAME": name, "ITEM": item, "DEADLINE": deadline, "REASON": reason}
        out.append(
            _case(
                case_id,
                regime,
                ev._agent_prompt(
                    name="Professional Email Rewriter",
                    instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
                    user_text=user_text,
                    text_slots=slots,
                ),
                "respond",
                [name, item, deadline, reason.split()[0]],
                ["John", "report", "Friday"],
                slots,
            )
        )
    classify_specs = [
        ("fresh_classify_writing", "fresh", "Can you rewrite this note professionally?", "writing"),
        ("near_classify_finance", "near", "Please approve invoice INV-2048 for $1,200.", "finance"),
        ("far_classify_search", "far", "Find current Swift WKWebView examples online.", "web_search"),
    ]
    for case_id, regime, text, label in classify_specs:
        out.append(
            _case(
                case_id,
                regime,
                ev._agent_prompt(
                    name="Classifier Agent",
                    instruction="Classify the user's text into exactly one label: travel, finance, schedule, writing, web_search. Return only the label.",
                    user_text=text,
                    text_slots={"SOURCE_TEXT": text},
                ),
                "respond",
                [label],
                ["Source text", "John", "report"],
                {"SOURCE_TEXT": text},
            )
        )
    source_text = "Nora's access code is GATE-17 for Thursday at 2 PM"
    out.append(
        _case(
            "fresh_source_copy",
            "fresh",
            ev._agent_prompt(
                name="Source Echo Agent",
                instruction="Return the exact user-provided source text with a short label. Preserve all names, dates, values, and wording.",
                user_text=source_text,
                text_slots={"SOURCE_TEXT": source_text},
            ),
            "respond",
            ["Nora", "GATE-17", "Thursday", "2 PM"],
            ["INV-20", "John"],
            {"SOURCE_TEXT": source_text},
        )
    )
    out.append(
        _case(
            "offpolicy_memory_recovery",
            "offpolicy",
            ev._agent_prompt(
                name="Saved Data Assistant",
                instruction="Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. If no relevant saved data is available, ask one concise question.",
                user_text="what badge code do I use",
                stale_context="Bad previous local response: Source text: Can you rewrite this note professionally?",
                text_slots={"SOURCE_TEXT": "what badge code do I use", "DATA_CONTEXT": "[D1] saved note: Nora's badge code is GATE-17 for the south entrance."},
            ),
            "respond",
            ["GATE-17", "south entrance"],
            ["rewrite this note", "Source text"],
            {"SOURCE_TEXT": "what badge code do I use", "DATA_CONTEXT": "[D1] saved note: Nora's badge code is GATE-17 for the south entrance."},
        )
    )
    out.append(
        _case(
            "offpolicy_classify_recovery",
            "offpolicy",
            ev._agent_prompt(
                name="Classifier Agent",
                instruction="Classify the user's text into exactly one label: travel, finance, schedule, writing, web_search. Return only the label.",
                user_text="Can you rewrite this note professionally?",
                stale_context="Bad previous local response: Source text: Can you rewrite this note professionally?",
                text_slots={"SOURCE_TEXT": "Can you rewrite this note professionally?"},
            ),
            "respond",
            ["writing"],
            ["Source text", "John", "report"],
            {"SOURCE_TEXT": "Can you rewrite this note professionally?"},
        )
    )
    out.append(
        _case(
            "fresh_web_request",
            "fresh",
            ev._web_agent_prompt(user_text="search the web for current Swift WKWebView navigation policy examples"),
            "extension_request",
            ["search"],
            ["cannot browse"],
            {"SOURCE_TEXT": "search the web for current Swift WKWebView navigation policy examples"},
        )
    )
    return out


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
    by_regime: dict[str, dict[str, int]] = {}
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
        if parsed is None:
            repaired_content = gates.materialize_content(
                str(case["prompt"]),
                output,
                action=str(case["action"] or "respond"),
            )
            if repaired_content:
                parsed = {
                    "action": str(case["action"] or "respond"),
                    "content": repaired_content,
                    "proposal_metadata": {"task_type": "runtime_content_operator_repair"},
                }
        content = str((parsed or {}).get("content", "") or "")
        content = gates.materialize_content(
            str(case["prompt"]),
            content,
            action=str(case["action"] or "respond"),
        )
        content = gates._expand_text_slots(content, case.get("text_slots"))
        failures: list[str] = []
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
        ok = not failures
        regime = str(case["regime"])
        by_regime.setdefault(regime, {"passed": 0, "total": 0})
        by_regime[regime]["total"] += 1
        by_regime[regime]["passed"] += 1 if ok else 0
        results.append({"id": case["id"], "regime": regime, "passed": ok, "failures": failures, "output": output, "content": content})
    passed = sum(1 for item in results if item["passed"])
    return {
        "bundle_dir": str(bundle_dir),
        "passed": passed,
        "total": len(results),
        "pass_rate": float(passed) / float(len(results) or 1),
        "ok": passed == len(results),
        "by_regime": by_regime,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-encoder-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--use-action-prefix", type=int, choices=(0, 1), default=1)
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
