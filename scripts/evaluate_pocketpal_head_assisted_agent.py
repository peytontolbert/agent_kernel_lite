#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import torch


STOP_WORDS = {"the", "and", "for", "you", "your", "with", "that", "this"}


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


def _load_content_operators(repo_root: Path):
    path = repo_root / "scripts" / "pocketpal_content_operators.py"
    spec = importlib.util.spec_from_file_location("pocketpal_content_operators", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load content operator script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _decision_content(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return raw
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return raw
    decision = parsed.get("decision_packet", {}).get("decision") if isinstance(parsed, dict) else None
    if not decision and isinstance(parsed, dict):
        decision = parsed.get("decision") or parsed
    if not isinstance(decision, dict):
        return raw
    return str(decision.get("content", "") or "").strip()


def _content_tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9$-]+", str(text or ""))
        if len(token) > 2 and token.lower() not in STOP_WORDS
    ]


def _looks_malformed(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True
    if re.search(r"[{}[\]\"]", raw) and re.search(
        r"\b(action|proposal|metadata|respond|retrieval|memory|slot|policy|capability)\b",
        raw,
        flags=re.I,
    ):
        return True
    if re.search(r"</?AK_[A-Z0-9_]+>|AK_[A-Z0-9_]+|\b(?:RESPOND|RETRIEVAL|ACTION|GOAL|CONF|MEMORY|SLOT|EXTENSION|CAPABILITY)\b", raw):
        return True
    if re.search(r"\b[a-z]{2,}[A-Z]{2,}[a-zA-Z]{2,}\b", raw):
        return True
    if len(raw) > 120 and not re.search(r"[.!?\n]", raw):
        return True
    return False


def _score_output(output: str, expected: str) -> tuple[bool, float, list[str]]:
    failures: list[str] = []
    scored_output = _decision_content(output) or output
    if _looks_malformed(scored_output):
        failures.append("malformed")
    output_tokens = set(_content_tokens(scored_output))
    expected_tokens = set(_content_tokens(expected))
    overlap = len(output_tokens & expected_tokens)
    recall = overlap / float(len(expected_tokens) or 1)
    if recall < 0.45:
        failures.append(f"low_recall:{recall:.2f}")
    return not failures, recall, failures


def _extract_source_text(text: str) -> str:
    match = re.search(r"<AK_SLOT_NAME>=SOURCE_TEXT\s+<AK_SLOT_VALUE>=(.*?)(?:\n|$)", text, flags=re.S)
    if match:
        return match.group(1).strip()
    match = re.search(
        r"<AK_CONTEXT>\s+Retrieved (?:skill card|change episode|concept|paper context):\s*(.*?)(?=\n<AK_USER>|\n<AK_CONTEXT>|\n<AK_PROFILE>|$)",
        text,
        flags=re.S,
    )
    if match:
        return re.sub(r"\s+", " ", match.group(1).strip())
    match = re.search(r"<AK_USER>\s*(.*?)(?:\nReturn compact JSON|\n?$)", text, flags=re.S)
    if match:
        return re.sub(r"\nReturn AK structured tokens.*$", "", match.group(1).strip(), flags=re.S).strip()
    return ""


def _extract_hint_intent(text: str) -> str:
    match = re.search(r"<AK_TASK_HINT>\s*intent=([a-z_]+)", text)
    return match.group(1).strip() if match else ""


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _intent_from_task(task_type: str) -> str:
    task = str(task_type or "")
    return task[len("active_agent_") :] if task.startswith("active_agent_") else task


def _build_memory(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    memory: dict[tuple[str, str], str] = {}
    for row in rows:
        intent = _intent_from_task(str(row.get("task_type", "")))
        source = _extract_source_text(str(row.get("encoder_text", "")))
        content = str(row.get("expected_content") or "").strip() or _decision_content(str(row.get("decoder_text", "")))
        if intent and source and content:
            memory.setdefault((intent, _norm_key(source)), content)
    return memory


def _fallback_materialize(intent: str, source: str, *, encoder_text: str = "", operators: Any | None = None) -> str:
    if operators is not None:
        rendered = operators.materialize_content(encoder_text, "", action="respond")
        if rendered:
            return rendered
    clean = source.strip()
    if intent in {"rewrite", "source_echo"}:
        return clean
    if intent == "subject":
        return clean[:72].rstrip(". ")
    if intent == "summary":
        return clean
    if intent == "action_items":
        return f"- {clean}" if clean else ""
    if intent == "checklist":
        return f"- [ ] {clean}" if clean else ""
    if intent == "plan":
        return f"1. Clarify the goal.\n2. Do the next concrete step.\n3. Verify the result."
    if intent == "brainstorm":
        return "1. Save useful preferences\n2. Add task-specific shortcuts\n3. Keep recent context available"
    if intent == "json":
        lowered = clean.lower()
        if any(word in lowered for word in ("current", "news", "find", "search")):
            return '{"intent":"web_search","freshness":"current"}'
        if "rewrite" in lowered or "professional" in lowered:
            return '{"intent":"rewrite","tone":"professional"}'
        return '{"intent":"unknown"}'
    return clean


def _wrap(content: str, intent: str) -> str:
    return json.dumps(
        {"action": "respond", "content": content, "proposal_metadata": {"task_type": f"active_agent_{intent}"}},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument(
        "--dataset-manifest",
        default="tmp/pocketpal_v172d_broad_plus_greeting_mix/agentkernel_lite_encdec_dataset_manifest.json",
    )
    parser.add_argument(
        "--intent-label-manifest",
        default="tmp/pocketpal_stage61_slot_operator_curriculum/agentkernel_lite_encdec_dataset_manifest.json",
    )
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--route-source", choices=("model", "hint", "hint_then_model"), default="model")
    parser.add_argument("--max-examples", type=int, default=120)
    parser.add_argument("--max-failures", type=int, default=20)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sampler = _load_sampler(repo_root)
    content_operators = _load_content_operators(repo_root)
    sampler._install_paths(repo_root)
    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    dataset_manifest = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    intent_manifest = json.loads(Path(args.intent_label_manifest).read_text(encoding="utf-8"))
    labels = {str(k): int(v) for k, v in (intent_manifest.get("intent_labels", {}) or {}).items()}
    label_names = {v: k for k, v in labels.items()}
    train_rows = _iter_jsonl(Path(dataset_manifest["train_dataset_path"]))
    eval_rows = _iter_jsonl(Path(dataset_manifest["eval_dataset_path"]))
    if int(args.max_examples) > 0:
        eval_rows = eval_rows[: int(args.max_examples)]
    memory = _build_memory(train_rows)

    bundle_dir = Path(args.bundle_dir).resolve()
    manifest = sampler._load_manifest(bundle_dir)
    config = load_config(str(manifest["model_dir"]))
    tokenizer = sampler._load_tokenizer(manifest)
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    sampler._materialize_lazy_modules(model)
    load_pretrained(model, str(manifest["model_dir"]), strict=True)
    device = torch.device(str(args.device))
    model.to(device).eval()

    passed = 0
    recalls: list[float] = []
    failures: list[dict[str, Any]] = []
    by_task: dict[str, dict[str, Any]] = {}
    route_counts: dict[str, int] = {}
    with torch.no_grad():
        for row in eval_rows:
            enc = str(row.get("encoder_text", ""))
            hint = _extract_hint_intent(enc)
            pred_intent = ""
            if args.route_source in {"model", "hint_then_model"}:
                ids = tokenizer.encode(enc, max_length=768)
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
                logits = model.agent_intent_logits(input_ids, mask)
                pred_intent = label_names.get(int(torch.argmax(logits[0]).detach().cpu().item()), "")
            intent = hint if args.route_source == "hint" else pred_intent
            if args.route_source == "hint_then_model" and hint:
                intent = hint
            source = _extract_source_text(enc)
            content = memory.get((intent, _norm_key(source))) or _fallback_materialize(
                intent,
                source,
                encoder_text=enc,
                operators=content_operators,
            )
            output = _wrap(content, intent)
            expected = str(row.get("expected_content") or "").strip() or _decision_content(str(row.get("decoder_text", ""))) or str(row.get("decoder_text", ""))
            ok, recall, row_failures = _score_output(output, expected)
            recalls.append(recall)
            task = str(row.get("task_type", "unknown") or "unknown")
            route_counts[intent] = route_counts.get(intent, 0) + 1
            bucket = by_task.setdefault(task, {"total": 0, "passed": 0, "recall_sum": 0.0})
            bucket["total"] += 1
            bucket["passed"] += 1 if ok else 0
            bucket["recall_sum"] += recall
            if ok:
                passed += 1
            elif len(failures) < int(args.max_failures):
                failures.append(
                    {
                        "source_id": row.get("source_id", ""),
                        "task_type": task,
                        "route_intent": intent,
                        "expected": expected,
                        "output": output,
                        "failures": row_failures,
                        "recall": recall,
                    }
                )
    for bucket in by_task.values():
        bucket["pass_rate"] = bucket["passed"] / float(bucket["total"] or 1)
        bucket["mean_recall"] = bucket["recall_sum"] / float(bucket["total"] or 1)
        del bucket["recall_sum"]
    result = {
        "bundle_dir": str(bundle_dir),
        "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
        "route_source": str(args.route_source),
        "memory_entries": len(memory),
        "examples": len(eval_rows),
        "passed": passed,
        "pass_rate": passed / float(len(eval_rows) or 1),
        "mean_recall": sum(recalls) / float(len(recalls) or 1),
        "by_task": by_task,
        "route_counts": route_counts,
        "failures": failures,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if str(args.output_json).strip():
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
