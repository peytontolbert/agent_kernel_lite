#!/usr/bin/env python3
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
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


PAPERS: dict[str, dict[str, Any]] = {
    "1": {
        "title": "Communicate only when necessary: Cooperative tasking for multi-agent systems",
        "category": "cs.MA",
        "abstract": (
            "The paper studies cooperative control of multi-agent systems. It focuses on local control "
            "and interaction rules, decomposability of task automata, over-communications that should be "
            "deleted, and lack of communications that require sharing events."
        ),
        "rationale_terms": ["communication", "decomposability", "task automata", "cooperative control"],
    },
    "2": {
        "title": "A new approach of designing Multi-Agent Systems",
        "category": "cs.MA",
        "abstract": (
            "The paper presents a practical approach to design multi-agent systems with component-oriented "
            "architecture and model-driven engineering."
        ),
        "rationale_terms": ["design", "component-oriented", "model-driven"],
    },
    "3": {
        "title": "Theory of Mind for Multi-Agent Collaboration via Large Language Models",
        "category": "cs.CL",
        "abstract": (
            "This study evaluates LLM-based agents in a multi-agent cooperative text game with Theory of Mind "
            "inference tasks, comparing performance with multi-agent reinforcement learning and planning-based "
            "baselines. It observes emergent collaborative behaviors and high-order Theory of Mind capabilities "
            "among LLM agents."
        ),
        "rationale_terms": ["LLM", "Theory of Mind", "collaboration", "cooperative text game"],
    },
}


def _context() -> str:
    notes = [
        "<AK_READING_NOTES>",
        "Recommendation rule: compare evidence, prefer direct topic matches, cite evidence ids, and copy exact titles from evidence.",
    ]
    for paper_id, paper in PAPERS.items():
        notes.append(f"[{paper_id}] {paper['title']}")
        notes.append(f"Relevant reading: {paper['abstract']}")
    blocks = ["\n".join(notes), "<AK_CONTEXT>"]
    for paper_id, paper in PAPERS.items():
        blocks.append(
            "\n".join(
                [
                    f"<AK_EVIDENCE> <AK_EVIDENCE_ID> {paper_id}",
                    f"<AK_TITLE> {paper['title']}",
                    f"<AK_PAPER_ID> {paper_id}",
                    f"<AK_CATEGORY> {paper['category']}",
                    "<AK_YEAR> 2024",
                    f"<AK_ABSTRACT> {paper['abstract']}",
                ]
            )
        )
    return "\n".join(blocks)


CASES: list[dict[str, Any]] = [
    {
        "id": "ambiguous_best_multi_agent",
        "query": "what's the best multi agent paper?",
        "mode": "ambiguous",
        "acceptable": ["1", "3"],
        "must_distinguish": [
            {"paper_id": "1", "terms": ["general", "classic", "systems", "communication", "decomposability"]},
            {"paper_id": "3", "terms": ["LLM", "Theory of Mind", "collaboration"]},
        ],
    },
    {
        "id": "classic_multi_agent_systems",
        "query": "what's the best classic multi-agent systems paper?",
        "mode": "single",
        "acceptable": ["1"],
    },
    {
        "id": "best_communication_multi_agent_systems",
        "query": "best paper about communication in multi-agent systems",
        "mode": "single",
        "acceptable": ["1"],
    },
    {
        "id": "best_multi_agent_llm",
        "query": "what's the best multi-agent llm paper?",
        "mode": "single",
        "acceptable": ["3"],
    },
    {
        "id": "recommend_multi_agent_llm_papers",
        "query": "recommend multi-agent LLM papers",
        "mode": "single",
        "acceptable": ["3"],
    },
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _prompt(case: dict[str, Any]) -> str:
    return "\n".join(
        [
            "<AK_CHAT> <AK_RESPOND> <AK_CONTEXT> <AK_READING_NOTES> <AK_ANSWER>",
            "<AK_LOOP> <AK_STATE> mode=chat selected_context=0 retrieval=ranked",
            f"<AK_USER> {case['query']}",
            _context(),
            "Answer directly. Recommend only papers present in evidence. Copy exact titles. Cite the chosen paper.",
        ]
    )


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def _contains_term(text: str, term: str) -> bool:
    return _norm(term) in _norm(text)


def _citation_ids(text: str) -> list[str]:
    ids = []
    for match in re.finditer(r"\[(\d{1,2})\]", text or ""):
        value = str(match.group(1))
        if value in PAPERS:
            ids.append(value)
    return ids


def _title_similarity(output: str, title: str) -> float:
    out = _norm(output)
    expected = _norm(title)
    if expected in out:
        return 1.0
    best = 0.0
    window = max(len(expected) + 20, 80)
    for start in range(0, max(1, len(out) - window + 1), 20):
        candidate = out[start : start + window]
        best = max(best, SequenceMatcher(None, expected, candidate).ratio())
    words = [word for word in re.findall(r"[a-z][a-z0-9-]{3,}", expected) if word not in {"with", "from", "only", "paper"}]
    bag_ratio = sum(1 for word in words if word in out) / max(1, len(words))
    return max(best, bag_ratio)


def _paper_score(output: str, paper_id: str) -> dict[str, Any]:
    paper = PAPERS[paper_id]
    citation_ok = f"[{paper_id}]" in output
    title_similarity = _title_similarity(output, str(paper["title"]))
    title_ok = title_similarity >= 0.86
    rationale_terms = list(paper["rationale_terms"])
    rationale_hits = [term for term in rationale_terms if _contains_term(output, str(term))]
    rationale_score = len(rationale_hits) / max(1, len(rationale_terms))
    return {
        "paper_id": paper_id,
        "citation_ok": bool(citation_ok),
        "title_ok": bool(title_ok),
        "title_similarity": float(title_similarity),
        "rationale_score": float(rationale_score),
        "rationale_hits": rationale_hits,
        "title": paper["title"],
    }


def _ambiguous_score(case: dict[str, Any], output: str) -> dict[str, Any]:
    choice_scores = [_paper_score(output, paper_id) for paper_id in case["acceptable"]]
    cited = set(_citation_ids(output))
    has_any_valid_choice = any(item["citation_ok"] and item["title_ok"] for item in choice_scores)
    distinguish_checks = []
    for rule in case.get("must_distinguish", []):
        paper_id = str(rule["paper_id"])
        paper_present = f"[{paper_id}]" in output or _title_similarity(output, PAPERS[paper_id]["title"]) >= 0.86
        term_hits = [term for term in rule.get("terms", []) if _contains_term(output, str(term))]
        distinguish_checks.append(
            {
                "paper_id": paper_id,
                "paper_present": bool(paper_present),
                "term_hits": term_hits,
                "passed": bool(paper_present and len(term_hits) >= 2),
            }
        )
    # Ambiguous "best" is allowed to pick one paper, but the best behavior is to
    # state the split: [1] for classic/general MAS and [3] for LLM-agent work.
    distinction_ok = sum(1 for item in distinguish_checks if item["passed"]) >= 2
    wrong_citations = sorted(cited - set(PAPERS))
    return {
        "mode": "ambiguous",
        "choice_scores": choice_scores,
        "has_any_valid_choice": bool(has_any_valid_choice),
        "distinction_ok": bool(distinction_ok),
        "distinguish_checks": distinguish_checks,
        "wrong_citations": wrong_citations,
        "passed": bool(has_any_valid_choice and distinction_ok and not wrong_citations),
    }


def _single_score(case: dict[str, Any], output: str) -> dict[str, Any]:
    acceptable = [str(item) for item in case["acceptable"]]
    choice_scores = [_paper_score(output, paper_id) for paper_id in acceptable]
    best = max(choice_scores, key=lambda item: (item["citation_ok"], item["title_similarity"], item["rationale_score"]))
    cited = set(_citation_ids(output))
    unacceptable_cited = sorted(cited - set(acceptable))
    # It is fine to mention another paper as background only if the chosen paper
    # is clearly cited and titled. This avoids rewarding mixed-up answers where
    # [1] is cited with [3]'s title or vice versa.
    chosen_ok = bool(best["citation_ok"] and best["title_ok"] and best["rationale_score"] >= 0.5)
    return {
        "mode": "single",
        "choice_scores": choice_scores,
        "best_choice": best,
        "unacceptable_cited": unacceptable_cited,
        "passed": bool(chosen_ok and not unacceptable_cited),
    }


def _score(case: dict[str, Any], output: str) -> dict[str, Any]:
    parsed = _parse_decision(output)
    health = _decode_health(output)
    content_ok = (
        parsed["action"].lower() == "respond"
        and parsed["format"] in {"line", "json"}
        and bool(health["sentence_like"])
        and int(health["artifact_count"]) == 0
        and int(health["malformed_citation_count"]) == 0
        and float(health["unique_ratio"]) >= 0.35
        and int(health["max_token_run"]) <= 5
        and int(health["max_bigram_run"]) <= 3
    )
    recommendation = _ambiguous_score(case, output) if case.get("mode") == "ambiguous" else _single_score(case, output)
    passed = bool(content_ok and recommendation["passed"])
    return {
        "id": str(case["id"]),
        "passed": passed,
        "content_ok": bool(content_ok),
        "recommendation_ok": bool(recommendation["passed"]),
        "recommendation": recommendation,
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
        for case in CASES:
            output = _generate(
                model,
                tokenizer,
                _prompt(case),
                decoder_prefix="",
                device=device,
                max_encoder_tokens=int(args.max_encoder_tokens),
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                repetition_penalty=float(args.repetition_penalty),
            )
            results.append(_score(case, output))
    passed = sum(1 for item in results if item["passed"])
    title_scores = [
        score["title_similarity"]
        for result in results
        for score in result["recommendation"].get("choice_scores", [])
    ]
    summary = {
        "bundle_dir": str(bundle_dir),
        "probe_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "mean_title_similarity": sum(float(item) for item in title_scores) / len(title_scores) if title_scores else 0.0,
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
