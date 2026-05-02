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
from evaluate_agentkernel_lite_recommendation_quality import (  # noqa: E402
    CASES,
    PAPERS,
    _context,
    _score,
)
from sample_agentkernel_lite_encdec import (  # noqa: E402
    _generate,
    _install_paths,
    _load_manifest,
    _load_tokenizer,
    _materialize_lazy_modules,
)


def _load_model(bundle_dir: Path, repo_root: Path, device: torch.device):
    _install_paths(repo_root)
    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    manifest = _load_manifest(bundle_dir)
    model_dir = Path(str(manifest["model_dir"]))
    config = load_config(str(model_dir))
    tokenizer = _load_tokenizer(manifest)
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    _materialize_lazy_modules(model)
    load_pretrained(model, str(model_dir), strict=True)
    model.to(device).eval()
    return model, tokenizer


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _selector_prompt(case: dict[str, Any]) -> str:
    candidate_rows = []
    for paper_id, paper in PAPERS.items():
        candidate_rows.append(
            " | ".join(
                [
                    f"<AK_CANDIDATE_ID> {paper_id}",
                    f"<AK_TITLE> {paper['title']}",
                    f"<AK_CATEGORY> {paper['category']}",
                    f"<AK_TERMS> {'; '.join(str(term) for term in paper.get('rationale_terms', []))}",
                    f"<AK_ABSTRACT> {paper['abstract']}",
                ]
            )
        )
    return "\n".join(
        [
            "<AK_CHAT> <AK_GATHER_CONTEXT> <AK_RERANK> <AK_CANDIDATES>",
            "<AK_LOOP> <AK_STATE> mode=chat selected_context=0 retrieval=ranked",
            f"<AK_USER> {case['query']}",
            "\n".join(candidate_rows),
            (
                "Select the evidence id that best matches the user's intent. "
                "Output only a gather_context decision with selected_candidate_id."
            ),
        ]
    )


def _answer_prompt(case: dict[str, Any], selected_ids: list[str]) -> str:
    if case.get("mode") == "ambiguous" and {"1", "3"}.issubset(set(selected_ids)):
        ids = ["1", "3"]
    else:
        ids = selected_ids[:1]
    selected_context = []
    for paper_id in ids:
        paper = PAPERS[str(paper_id)]
        selected_context.append(
            "\n".join(
                [
                    f"<AK_EVIDENCE> <AK_EVIDENCE_ID> {paper_id}",
                    f"<AK_TITLE> {paper['title']}",
                    f"<AK_CATEGORY> {paper['category']}",
                    f"<AK_ABSTRACT> {paper['abstract']}",
                ]
            )
        )
    return "\n".join(
        [
            "<AK_CHAT> <AK_RESPOND> <AK_CONTEXT> <AK_READING_NOTES> <AK_ANSWER>",
            "<AK_LOOP> <AK_STATE> mode=chat selected_context=1 retrieval=selected",
            f"<AK_USER> {case['query']}",
            "<AK_CONTEXT>",
            "\n\n".join(selected_context),
            (
                "Answer directly from the selected evidence. Copy exact titles from evidence, "
                "cite selected ids, and do not mention papers that are not in selected evidence."
            ),
        ]
    )


def _selected_ids(selector_output: str) -> list[str]:
    ids: list[str] = []
    for match in re.finditer(r"selected_candidate_id\s*=\s*(?:P)?(\d{1,2})", selector_output, flags=re.I):
        paper_id = str(match.group(1))
        if paper_id in PAPERS and paper_id not in ids:
            ids.append(paper_id)
    if ids:
        return ids
    parsed = _parse_decision(selector_output)
    for match in re.finditer(r"\[(\d{1,2})\]|\bP?(\d{1,2})\b", parsed["content"]):
        paper_id = str(match.group(1) or match.group(2))
        if paper_id in PAPERS and paper_id not in ids:
            ids.append(paper_id)
    return ids


def _selector_score(case: dict[str, Any], selector_output: str) -> dict[str, Any]:
    parsed = _parse_decision(selector_output)
    health = _decode_health(selector_output)
    selected_ids = _selected_ids(selector_output)
    acceptable = {str(item) for item in case["acceptable"]}
    if case.get("mode") == "ambiguous":
        # Ambiguous broad "best" can select the general MAS paper, the LLM-agent
        # paper, or both. The answer pass still checks that the distinction is made.
        ok_selection = bool(set(selected_ids) & acceptable)
    else:
        ok_selection = bool(selected_ids and selected_ids[0] in acceptable)
    return {
        "passed": bool(
            parsed["action"].lower() == "gather_context"
            and parsed["format"] in {"line", "json"}
            and ok_selection
            and int(health["artifact_count"]) == 0
            and int(health["malformed_citation_count"]) == 0
        ),
        "selected_ids": selected_ids,
        "acceptable": sorted(acceptable),
        "parsed_action": parsed["action"],
        "parsed_format": parsed["format"],
        "decode_health": health,
        "output": selector_output,
    }


def _fallback_ids(case: dict[str, Any]) -> list[str]:
    acceptable = [str(item) for item in case["acceptable"]]
    if case.get("mode") == "ambiguous":
        return ["1", "3"]
    return acceptable[:1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--selector-bundle-dir", default="")
    parser.add_argument("--answer-bundle-dir", default="")
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--output-json", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-encoder-tokens", type=int, default=768)
    parser.add_argument("--selector-max-new-tokens", type=int, default=96)
    parser.add_argument("--answer-max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--allow-oracle-selection",
        type=int,
        choices=(0, 1),
        default=0,
        help="Use expected paper ids for the answer pass if selector fails. Useful for isolating answer quality.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    device = torch.device(str(args.device))
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    selector_bundle_dir = Path(args.selector_bundle_dir).expanduser().resolve() if str(args.selector_bundle_dir).strip() else bundle_dir
    answer_bundle_dir = Path(args.answer_bundle_dir).expanduser().resolve() if str(args.answer_bundle_dir).strip() else bundle_dir
    selector_model, selector_tokenizer = _load_model(selector_bundle_dir, repo_root, device)
    if answer_bundle_dir == selector_bundle_dir:
        answer_model, answer_tokenizer = selector_model, selector_tokenizer
    else:
        answer_model, answer_tokenizer = _load_model(answer_bundle_dir, repo_root, device)

    results = []
    with torch.no_grad():
        for case in CASES:
            selector_output = _generate(
                selector_model,
                selector_tokenizer,
                _selector_prompt(case),
                decoder_prefix="",
                device=device,
                max_encoder_tokens=int(args.max_encoder_tokens),
                max_new_tokens=int(args.selector_max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                repetition_penalty=float(args.repetition_penalty),
            )
            selector = _selector_score(case, selector_output)
            selected_ids = list(selector["selected_ids"])
            if not selected_ids or (not selector["passed"] and int(args.allow_oracle_selection)):
                selected_ids = _fallback_ids(case)
            answer_output = _generate(
                answer_model,
                answer_tokenizer,
                _answer_prompt(case, selected_ids),
                decoder_prefix="",
                device=device,
                max_encoder_tokens=int(args.max_encoder_tokens),
                max_new_tokens=int(args.answer_max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                repetition_penalty=float(args.repetition_penalty),
            )
            answer = _score(case, answer_output)
            results.append(
                {
                    "id": str(case["id"]),
                    "passed": bool(selector["passed"] and answer["passed"]),
                    "selector_passed": bool(selector["passed"]),
                    "answer_passed": bool(answer["passed"]),
                    "selected_ids_used": selected_ids,
                    "selector": selector,
                    "answer": answer,
                }
            )

    selector_passed = sum(1 for item in results if item["selector_passed"])
    answer_passed = sum(1 for item in results if item["answer_passed"])
    end_to_end_passed = sum(1 for item in results if item["passed"])
    summary = {
        "bundle_dir": str(bundle_dir),
        "selector_bundle_dir": str(selector_bundle_dir),
        "answer_bundle_dir": str(answer_bundle_dir),
        "probe_count": len(results),
        "selector_passed": selector_passed,
        "answer_passed": answer_passed,
        "end_to_end_passed": end_to_end_passed,
        "selector_pass_rate": selector_passed / len(results) if results else 0.0,
        "answer_pass_rate": answer_passed / len(results) if results else 0.0,
        "end_to_end_pass_rate": end_to_end_passed / len(results) if results else 0.0,
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
