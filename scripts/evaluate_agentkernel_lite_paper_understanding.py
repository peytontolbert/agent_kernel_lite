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

from evaluate_agentkernel_lite_generation import _decode_health, _parse_decision  # noqa: E402
from sample_agentkernel_lite_encdec import (  # noqa: E402
    _generate,
    _install_paths,
    _load_manifest,
    _load_tokenizer,
    _materialize_lazy_modules,
)


PAPER_CASES: list[dict[str, Any]] = [
    {
        "id": "selected_comm_only_when_necessary_detail",
        "title": "Communicate only when necessary: Cooperative tasking for multi-agent systems",
        "paper_id": "1106.3134",
        "citation": "[P1]",
        "abstract": (
            "The paper deals with design of interactions among agents to make an undecomposable task "
            "automaton decomposable and achievable in a top-down framework. It identifies root causes "
            "of undecomposability as over-communications that should be deleted or lack of communications "
            "that require sharing events."
        ),
        "question": "explain this paper in detail",
        "must_include": ["[P1]", "communication", "decomposable", "agents"],
        "concept_groups": [
            ["communication", "communications", "sharing"],
            ["decomposable", "decomposability", "undecomposable"],
            ["task", "automaton"],
            ["agents", "multi-agent"],
        ],
        "forbidden": ["retrieval failed", "unrelated", "wanking", "mister", "racket", "envend"],
    },
    {
        "id": "selected_theory_of_mind_llm_agents",
        "title": "Theory of Mind for Multi-Agent Collaboration via Large Language Models",
        "paper_id": "2310.10701",
        "citation": "[P1]",
        "abstract": (
            "This study evaluates LLM-based agents in a multi-agent cooperative text game with Theory "
            "of Mind inference tasks, comparing their performance with multi-agent reinforcement learning "
            "and planning-based baselines. It observes evidence of emergent collaborative behaviors and "
            "higher-order Theory of Mind capabilities among LLM agents."
        ),
        "question": "what are the main takeaways from this paper?",
        "must_include": ["[P1]", "Theory of Mind", "LLM", "multi-agent"],
        "concept_groups": [
            ["theory of mind", "tom"],
            ["llm", "large language"],
            ["collaboration", "collaborative", "cooperative"],
            ["baseline", "baselines", "reinforcement", "planning"],
        ],
        "forbidden": ["retrieval failed", "unrelated", "traffic", "robotic"],
    },
    {
        "id": "selected_mobileflow_gui_agent",
        "title": "MobileFlow: A Multimodal LLM For Mobile GUI Agent",
        "paper_id": "2407.04346",
        "citation": "[P1]",
        "abstract": (
            "The paper studies multimodal large language models for mobile GUI agents. It focuses on GUI "
            "comprehension and user action analysis while reducing dependence on page layout information "
            "from system APIs, which can pose privacy risks."
        ),
        "question": "summarize the method and why it matters",
        "must_include": ["[P1]", "mobile", "GUI", "privacy"],
        "concept_groups": [
            ["mobile", "gui"],
            ["multimodal", "vision"],
            ["action", "actions"],
            ["privacy", "api", "layout"],
        ],
        "forbidden": ["retrieval failed", "unrelated", "traffic"],
    },
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_prompt(case: dict[str, Any]) -> str:
    title = str(case["title"])
    paper_id = str(case["paper_id"])
    abstract = str(case["abstract"])
    question = str(case["question"])
    return "\n".join(
        [
            "<AK_CHAT> <AK_RESPOND> <AK_ACTIVE_CONTEXT> <AK_SELECTED_PAPER> <AK_USE_CONTEXT> <AK_NO_RETRIEVAL>",
            "<AK_LOOP> <AK_STATE> mode=chat selected_context=1 retrieval=active",
            "<AK_HISTORY>",
            "User: Add [1] to chat.",
            "Assistant: Loaded the selected paper [P1].",
            "<AK_READING_NOTES>",
            "Selected-context rule: answer from the active selected paper P1, cite [P1], and do not call this fresh retrieval.",
            f"[P1] {title}",
            f"Active paper reading: {abstract}",
            "<AK_CONTEXT>",
            "<AK_EVIDENCE> <AK_EVIDENCE_ID> P1",
            f"<AK_TITLE> {title}",
            f"<AK_PAPER_ID> {paper_id}",
            "<AK_CATEGORY> cs.AI",
            "<AK_YEAR> 2024",
            f"<AK_ABSTRACT> {abstract}",
            f"<AK_USER> {question}",
            "Answer from the active paper only, cite [P1], and do not retrieve new papers.",
        ]
    )


def _contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _score_case(case: dict[str, Any], output: str) -> dict[str, Any]:
    parsed = _parse_decision(output)
    content = parsed["content"]
    lowered = output.lower()
    must_include = [str(item) for item in case.get("must_include", [])]
    missing_required = [item for item in must_include if item.lower() not in lowered]
    forbidden = [str(item) for item in case.get("forbidden", [])]
    present_forbidden = [item for item in forbidden if item.lower() in lowered]
    concept_groups = list(case.get("concept_groups", []))
    concept_hits = sum(1 for group in concept_groups if _contains_any(output, [str(item) for item in group]))
    concept_coverage = concept_hits / max(1, len(concept_groups))
    title = str(case["title"])
    title_words = [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", title)]
    title_hit_ratio = (
        sum(1 for word in title_words if word in lowered) / max(1, len(title_words))
    )
    health = _decode_health(output)
    action_ok = parsed["action"].lower() == "respond"
    citation_ok = str(case.get("citation", "[P1]")).lower() in lowered
    no_new_retrieval = (
        "gather_context" not in lowered
        and "<ak_retrieve" not in lowered
        and "retrieve_new" not in lowered
        and "retrieved papers" not in lowered
    )
    content_ok = (
        bool(content)
        and bool(health["sentence_like"])
        and int(health["artifact_count"]) == 0
        and int(health["malformed_citation_count"]) == 0
        and float(health["unique_ratio"]) >= 0.28
        and int(health["max_token_run"]) <= 5
        and int(health["max_bigram_run"]) <= 3
    )
    passed = (
        action_ok
        and citation_ok
        and no_new_retrieval
        and content_ok
        and not missing_required
        and not present_forbidden
        and concept_coverage >= 0.75
        and title_hit_ratio >= 0.25
    )
    return {
        "id": str(case["id"]),
        "passed": bool(passed),
        "action_ok": bool(action_ok),
        "citation_ok": bool(citation_ok),
        "no_new_retrieval": bool(no_new_retrieval),
        "content_ok": bool(content_ok),
        "concept_coverage": float(concept_coverage),
        "title_hit_ratio": float(title_hit_ratio),
        "missing_required": missing_required,
        "present_forbidden": present_forbidden,
        "decode_health": health,
        "output": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--output-json", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-encoder-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    _install_paths(repo_root)
    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    manifest = _load_manifest(bundle_dir)
    model_dir = Path(str(manifest["model_dir"]))
    config = load_config(str(model_dir))
    tokenizer = _load_tokenizer(manifest)
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    _materialize_lazy_modules(model)
    load_pretrained(model, str(model_dir), strict=True)
    device = torch.device(str(args.device))
    model.to(device).eval()

    results = []
    with torch.no_grad():
        for case in PAPER_CASES:
            output = _generate(
                model,
                tokenizer,
                _build_prompt(case),
                decoder_prefix="",
                device=device,
                max_encoder_tokens=int(args.max_encoder_tokens),
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                repetition_penalty=float(args.repetition_penalty),
            )
            results.append(_score_case(case, output))
    passed = sum(1 for item in results if item["passed"])
    summary = {
        "bundle_dir": str(bundle_dir),
        "probe_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "mean_concept_coverage": sum(float(item["concept_coverage"]) for item in results) / len(results),
        "mean_title_hit_ratio": sum(float(item["title_hit_ratio"]) for item in results) / len(results),
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
