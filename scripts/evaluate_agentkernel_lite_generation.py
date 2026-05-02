#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from sample_agentkernel_lite_encdec import (  # noqa: E402
    _generate,
    _install_paths,
    _load_manifest,
    _load_tokenizer,
    _materialize_lazy_modules,
)


DEFAULT_PROBES: list[dict[str, Any]] = [
    {
        "id": "query_rewrite_multi_agent_planning",
        "prompt": (
            "<AK_CHAT> <AK_GATHER_CONTEXT> <AK_QUERY_REWRITE> "
            "AgentKernel Lite retrieval training example. Rewrite the user request into a compact "
            "research-library search query. Do not answer yet. <AK_USER> Find the best papers about "
            "multi-agent planning with grounded evidence. Return a structured decision with "
            "action=gather_context."
        ),
        "expected_action": "gather_context",
        "required": ["multi", "agent"],
    },
    {
        "id": "rerank_neural_retrieval_scientific_assistant",
        "prompt": (
            "<AK_CHAT> <AK_GATHER_CONTEXT> <AK_RERANK> AgentKernel Lite retrieval training example. "
            "Select the candidate that best matches the user request. Do not invent a paper outside "
            "the candidate list. <AK_USER> neural retrieval scientific assistant <AK_CONTEXT> "
            "Candidates: <AK_CANDIDATE> P1: Neural Retrieval for Scientific Assistants cs.AI | 2026 "
            "This paper studies neural retrieval for scientific assistant systems, candidate ranking, "
            "and grounded answer synthesis from long research documents. <AK_CANDIDATE> P2: "
            "Distributed Storage Recovery cs.DC | 2026 This paper studies recovery algorithms for "
            "replicated storage systems after partial node failures. Return a structured decision "
            "with action=gather_context and the selected candidate id."
        ),
        "expected_action": "gather_context",
        "required": ["selected_candidate_id=P1"],
        "forbidden": ["selected_candidate_id=P2"],
    },
    {
        "id": "answer_from_evidence",
        "prompt": (
            "<AK_CHAT> <AK_RESPOND> <AK_ANSWER> AgentKernel Lite research answer training example. "
            "<AK_USER> What is the main contribution of Neural Retrieval for Scientific Assistants? "
            "<AK_EVIDENCE> [1]: Neural Retrieval for Scientific Assistants This paper studies neural "
            "retrieval for scientific assistant systems, including candidate ranking and grounded "
            "answer synthesis from long research documents. Answer directly, cite [1] for supported "
            "claims, and do not list unrelated papers."
        ),
        "expected_action": "respond",
        "required": ["[1]", "neural", "retrieval"],
    },
    {
        "id": "selected_paper_followup_uses_active_context",
        "prompt": (
            "<AK_CHAT> <AK_RESPOND> <AK_CONTEXT> <AK_ANSWER>\n"
            "AgentKernel Lite selected-context answer training example.\n"
            "Answer the user's follow-up from the paper that is already loaded in chat. Do not say "
            "retrieval failed and do not introduce unrelated papers.\n"
            "<AK_CONTEXT> Active context:\nContext id P1\nSelected paper P1: "
            "Neural Retrieval for Scientific Assistants\n2501.00001 | cs.AI | 2026\n"
            "<AK_EVIDENCE> [P1]: This paper studies neural retrieval for scientific assistant "
            "systems, including candidate ranking and grounded answer synthesis from long research "
            "documents.\n<AK_USER> Tell me more about this paper please.\n"
            "Answer directly, cite [P1] for supported claims, and keep the response "
            "conversational."
        ),
        "expected_action": "respond",
        "required": ["[P1]", "neural", "retrieval"],
        "forbidden": ["retrieval failed", "do not have enough"],
    },
    {
        "id": "ordinary_chat_answers_directly",
        "prompt": (
            "<AK_CHAT> <AK_RESPOND> AgentKernel Lite chat example. <AK_USER> Tell me about "
            "multi-agent intelligence in plain language. Return a structured decision with "
            "action=respond."
        ),
        "expected_action": "respond",
        "required": ["agent"],
        "forbidden": ["did not add paper context", "retrieval failed"],
    },
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_probes(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_PROBES
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("probes file must be a JSON list")
    return [dict(item) for item in payload if isinstance(item, dict)]


def _parse_decision(text: str) -> dict[str, str]:
    raw = str(text or "").strip()
    if not raw:
        return {"action": "", "content": "", "format": "empty"}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return {
            "action": str(payload.get("action", "") or "").strip(),
            "content": str(payload.get("content", payload.get("answer", "")) or "").strip(),
            "format": "json",
        }
    action_match = re.search(r"(?im)^\s*Action:\s*([A-Za-z0-9_.-]+)\s*$", raw)
    content_match = re.search(r"(?ims)^\s*Content:\s*(.*)$", raw)
    return {
        "action": action_match.group(1).strip() if action_match else "",
        "content": content_match.group(1).strip() if content_match else raw,
        "format": "line" if action_match or content_match else "plain",
    }


def _decode_health(text: str) -> dict[str, Any]:
    tokens = re.findall(r"[A-Za-z0-9_\\.-]+|\S", str(text or ""))
    if not tokens:
        return {
            "token_count": 0,
            "unique_ratio": 0.0,
            "max_token_run": 0,
            "max_bigram_run": 0,
            "artifact_count": 0,
            "malformed_citation_count": 0,
            "sentence_like": False,
        }
    max_run = 1
    current_run = 1
    for left, right in zip(tokens, tokens[1:]):
        current_run = current_run + 1 if left == right else 1
        max_run = max(max_run, current_run)
    bigrams = list(zip(tokens, tokens[1:]))
    max_bigram_run = 0
    if bigrams:
        max_bigram_run = 1
        current_bigram_run = 1
        for left, right in zip(bigrams, bigrams[1:]):
            current_bigram_run = current_bigram_run + 1 if left == right else 1
            max_bigram_run = max(max_bigram_run, current_bigram_run)
    artifact_patterns = [
        r"(?i)(?:AST){4,}",
        r"(?i)(?:Wemathbthough\s+repro){2,}",
        r"\b(?:TOP|RARN|RSTA|HARR|DERR|BENR|ANRSTA|IASTAST)\b",
        r"(?i)\b[a-z]{2,}(?:anted|ede|vivalence|alant|ialed)[a-z]{2,}\b",
        r"(?i)\b(?:studytrieval|retrievetrieval|gretrieve|recoverage)\b",
        r"(?i)\b(?:formatalized|dynamicator|selected-powered|in-powered)\b",
        r"[^\x00-\x7f]{2,}",
    ]
    artifact_count = sum(len(re.findall(pattern, str(text or ""))) for pattern in artifact_patterns)
    malformed_citations = [
        match.group(0)
        for match in re.finditer(r"\[([^\]]+)\]", str(text or ""))
        if not re.fullmatch(r"(?:\d{1,2}|P\d{1,2})", match.group(1).strip())
    ]
    alpha_words = re.findall(r"[A-Za-z][A-Za-z'-]{1,}", str(text or ""))
    common_words = {
        "the", "a", "an", "and", "or", "is", "are", "it", "this", "that",
        "because", "with", "from", "about", "paper", "evidence", "answer",
        "focuses", "uses", "shows", "supports", "based", "retrieved",
    }
    common_ratio = (
        sum(1 for word in alpha_words if word.lower() in common_words) / max(1, len(alpha_words))
    )
    return {
        "token_count": len(tokens),
        "unique_ratio": len(set(tokens)) / max(1, len(tokens)),
        "max_token_run": max_run,
        "max_bigram_run": max_bigram_run,
        "artifact_count": artifact_count,
        "malformed_citation_count": len(malformed_citations),
        "sentence_like": bool(re.search(r"[A-Za-z][^.?!]{20,}[.?!]", str(text or ""))),
        "common_word_ratio": common_ratio,
    }


def _score_probe(probe: dict[str, Any], output: str) -> dict[str, Any]:
    parsed = _parse_decision(output)
    haystack = output.lower()
    expected_action = str(probe.get("expected_action", "") or "").strip().lower()
    required = [str(item) for item in probe.get("required", []) if str(item)]
    required_any = [str(item) for item in probe.get("required_any", []) if str(item)]
    forbidden = [str(item) for item in probe.get("forbidden", []) if str(item)]
    missing = [item for item in required if item.lower() not in haystack]
    missing_any = bool(required_any) and not any(item.lower() in haystack for item in required_any)
    present_forbidden = [item for item in forbidden if item.lower() in haystack]
    action_ok = not expected_action or parsed["action"].strip().lower() == expected_action
    format_ok = parsed["format"] in {"line", "json"}
    health = _decode_health(output)
    base_health_ok = (
        int(health["token_count"]) > 0
        and float(health["unique_ratio"]) >= 0.18
        and int(health["max_token_run"]) <= 8
        and int(health["max_bigram_run"]) <= 4
        and int(health["artifact_count"]) == 0
        and int(health["malformed_citation_count"]) == 0
    )
    if expected_action == "gather_context":
        # Retrieval decisions are often terse search queries or candidate IDs.
        # Do not require answer-like sentence structure for those control actions.
        health_ok = base_health_ok
    else:
        health_ok = (
            base_health_ok
            and bool(health["sentence_like"])
            and float(health["common_word_ratio"]) >= 0.08
        )
    passed = bool(action_ok and format_ok and not missing and not missing_any and not present_forbidden and health_ok)
    return {
        "id": probe.get("id", ""),
        "passed": passed,
        "action_ok": action_ok,
        "format_ok": format_ok,
        "expected_action": expected_action,
        "parsed_action": parsed["action"],
        "parsed_format": parsed["format"],
        "missing_required": missing,
        "missing_required_any": required_any if missing_any else [],
        "present_forbidden": present_forbidden,
        "decode_health": health,
        "output": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--probes-file", default="")
    parser.add_argument("--probe-id", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-encoder-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--apply-bitnet", type=int, choices=(0, 1), default=0)
    parser.add_argument("--bitnet-include", default="")
    parser.add_argument("--bitnet-exclude", default="")
    parser.add_argument("--decoder-prefix", default="")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    _install_paths(repo_root)

    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    torch.manual_seed(int(args.seed))
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    manifest = _load_manifest(bundle_dir)
    model_dir = Path(str(manifest["model_dir"]))
    config = load_config(str(model_dir))
    tokenizer = _load_tokenizer(manifest)
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    _materialize_lazy_modules(model)
    load_pretrained(model, str(model_dir), strict=True)
    if bool(args.apply_bitnet):
        from compress.apply import apply_compression

        apply_compression(
            model,
            quant={
                "scheme": "bitnet",
                "include": [item.strip() for item in str(args.bitnet_include).split(",") if item.strip()] or None,
                "exclude": [item.strip() for item in str(args.bitnet_exclude).split(",") if item.strip()] or None,
                "weight_opt": "none",
                "activation_quant": "none",
                "spin": False,
                "spin_random": True,
                "spin_seed": 0,
            },
        )
    device = torch.device(str(args.device))
    model.to(device).eval()

    probes_path = Path(args.probes_file).expanduser().resolve() if str(args.probes_file).strip() else None
    results = []
    probes = _load_probes(probes_path)
    if str(args.probe_id).strip():
        wanted = {item.strip() for item in str(args.probe_id).split(",") if item.strip()}
        probes = [probe for probe in probes if str(probe.get("id", "")) in wanted]
    with torch.no_grad():
        for probe in probes:
            output = _generate(
                model,
                tokenizer,
                str(probe.get("prompt", "")),
                decoder_prefix=str(args.decoder_prefix),
                device=device,
                max_encoder_tokens=int(args.max_encoder_tokens),
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                repetition_penalty=float(args.repetition_penalty),
            )
            results.append(_score_probe(probe, output))

    summary = {
        "bundle_dir": str(bundle_dir),
        "device": str(device),
        "apply_bitnet": bool(args.apply_bitnet),
        "probe_count": len(results),
        "passed": sum(1 for item in results if item["passed"]),
        "failed": sum(1 for item in results if not item["passed"]),
        "pass_rate": (sum(1 for item in results if item["passed"]) / len(results)) if results else 0.0,
        "results": results,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if str(args.output_json).strip():
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
