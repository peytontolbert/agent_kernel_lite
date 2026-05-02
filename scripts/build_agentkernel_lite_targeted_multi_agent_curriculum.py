#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


PAPERS = [
    {
        "id": "P1",
        "title": "LLM-MARS: Large Language Model for Behavior Tree Generation and NLP-enhanced Dialogue in Multi-Agent Robot Systems",
        "paper_id": "2312.09348",
        "category": "cs.RO",
        "year": "2023",
        "abstract": (
            "This paper introduces LLM-MARS, a Large Language Model based AI system for "
            "Multi-Agent Robot Systems. It enables dynamic dialogues between humans and "
            "robots, behavior-tree generation from operator commands, and informative "
            "answers about robot actions."
        ),
    },
    {
        "id": "P2",
        "title": "DiagGPT: An LLM-based and Multi-agent Dialogue System with Automatic Topic Management for Flexible Task-Oriented Dialogue",
        "paper_id": "2308.08043v1",
        "category": "cs.CL",
        "year": "2023",
        "abstract": (
            "DiagGPT is an LLM-based multi-agent dialogue system for diagnostic "
            "task-oriented dialogue. It manages topics automatically and supports "
            "complex consultation workflows."
        ),
    },
    {
        "id": "P3",
        "title": "Balancing Autonomy and Alignment: A Multi-Dimensional Taxonomy for Autonomous LLM-powered Multi-Agent Architectures",
        "paper_id": "2310.03659v1",
        "category": "cs.AI",
        "year": "2023",
        "abstract": (
            "This paper studies autonomous LLM-powered multi-agent systems and gives a "
            "taxonomy for autonomy, alignment, coordination, risks, and architectural "
            "tradeoffs."
        ),
    },
    {
        "id": "P4",
        "title": "LLM-Coordination: Evaluating and Analyzing Multi-agent Coordination Abilities in Large Language Models",
        "paper_id": "2310.03903v1",
        "category": "cs.CL",
        "year": "2023",
        "abstract": (
            "This paper evaluates and analyzes multi-agent coordination abilities in "
            "large language models, focusing on how LLM agents coordinate under "
            "interactive tasks."
        ),
    },
    {
        "id": "P5",
        "title": "TrainerAgent: Customizable and Efficient Model Training through LLM-Powered Multi-Agent System",
        "paper_id": "2311.06622v1",
        "category": "cs.AI",
        "year": "2023",
        "abstract": (
            "TrainerAgent uses an LLM-powered multi-agent system to customize and "
            "automate model training workflows for users who are not expert algorithm "
            "engineers."
        ),
    },
]


def line(action: str, content: str) -> str:
    return f"Action: {action}\nContent: {content}"


def context(papers: list[dict[str, str]]) -> str:
    blocks = []
    for paper in papers:
        blocks.append(
            "\n".join(
                [
                    f"<AK_EVIDENCE> <AK_EVIDENCE_ID> {paper['id']}",
                    f"<AK_TITLE> {paper['title']}",
                    f"<AK_PAPER_ID> {paper['paper_id']}",
                    f"<AK_CATEGORY> {paper['category']}",
                    f"<AK_YEAR> {paper['year']}",
                    f"<AK_ABSTRACT> {paper['abstract']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def notes(papers: list[dict[str, str]]) -> str:
    blocks = [
        "Recommendation rule: compare evidence, prefer direct topic matches, cite evidence ids, and do not invent titles."
    ]
    for paper in papers:
        blocks.append(f"[{paper['id']}] {paper['title']}\nRelevant reading: {paper['abstract']}")
    return "\n\n".join(blocks)


def example(source_id: str, task_type: str, encoder: str, decoder: str, weight: float) -> dict[str, object]:
    return {
        "source_type": "agentkernel_lite_targeted_multi_agent_curriculum",
        "source_id": source_id,
        "task_type": task_type,
        "encoder_text": encoder,
        "decoder_text": decoder,
        "action": "respond",
        "weight": weight,
    }


def build_examples(repetitions: int) -> list[dict[str, object]]:
    queries = [
        "what the best multi agent llm paper and why?",
        "what is the best multi-agent LLM paper?",
        "best multi-agent llm papers?",
        "which paper should I read first for multi-agent LLM systems?",
    ]
    rec_target = (
        "I would start with [1], \"LLM-MARS: Large Language Model for Behavior Tree Generation and "
        "NLP-enhanced Dialogue in Multi-Agent Robot Systems,\" because it is the top retrieved evidence "
        "and directly combines large language models with multi-agent robot systems. It is best if you "
        "care about LLM agents acting in robotics workflows. If you want a broader architecture survey, "
        "[3] is the better follow-up; if you want coordination evaluation, read [4] next."
    )
    selected_target = (
        "I am using the selected paper [P1]. The paper is \"Balancing Autonomy and Alignment: A "
        "Multi-Dimensional Taxonomy for Autonomous LLM-powered Multi-Agent Architectures.\" Its main "
        "value is that it organizes LLM-powered multi-agent systems by autonomy, alignment, coordination, "
        "risk, and architecture choices, so it is useful as a map of the design space rather than a "
        "single benchmark result."
    )
    direct_target = (
        "Multi-agent intelligence means intelligence that emerges from multiple agents working together, "
        "dividing work, sharing state, negotiating, or checking each other's decisions. In LLM systems, "
        "that usually means specialized agents for planning, retrieval, critique, coding, or execution, "
        "with a controller deciding how their outputs become one final answer."
    )
    examples: list[dict[str, object]] = []
    paper_context = context(PAPERS)
    paper_notes = notes(PAPERS)
    for index in range(repetitions):
        query = queries[index % len(queries)]
        encoder = "\n".join(
            [
                "<AK_CHAT> <AK_RESPOND> <AK_CONTEXT> <AK_READING_NOTES> <AK_ANSWER>",
                "<AK_LOOP> <AK_STATE> mode=chat selected_context=0 retrieval=ranked",
                f"<AK_USER> {query}",
                f"<AK_READING_NOTES>\n{paper_notes}",
                f"<AK_CONTEXT>\n{paper_context}",
            ]
        )
        examples.append(
            example(
                f"targeted_multi_agent_recommend:{index}",
                "targeted_grounded_recommendation_exact_title",
                encoder,
                line("respond", rec_target),
                9.0,
            )
        )

        selected = PAPERS[2]
        selected_encoder = "\n".join(
            [
                "<AK_CHAT> <AK_RESPOND> <AK_ACTIVE_CONTEXT> <AK_SELECTED_PAPER> <AK_USE_CONTEXT>",
                "<AK_LOOP> <AK_STATE> mode=chat selected_context=1 retrieval=active",
                "<AK_HISTORY>\nUser: Add [3] to chat.\nAssistant: Added the selected paper [P1].",
                f"<AK_READING_NOTES>\n[P1] {selected['title']}\nRelevant reading: {selected['abstract']}",
                f"<AK_CONTEXT>\n{context([{**selected, 'id': 'P1'}])}",
                "<AK_USER> tell me more about this paper please",
            ]
        )
        examples.append(
            example(
                f"targeted_selected_followup:{index}",
                "targeted_selected_paper_followup_exact_title",
                selected_encoder,
                line("respond", selected_target),
                8.0,
            )
        )

        direct_encoder = "\n".join(
            [
                "<AK_CHAT> <AK_RESPOND>",
                "<AK_LOOP> <AK_STATE> mode=chat selected_context=0 retrieval=none",
                "<AK_USER> tell me about multi-agent intelligence",
            ]
        )
        examples.append(
            example(
                f"targeted_direct_chat:{index}",
                "targeted_direct_chat_multi_agent",
                direct_encoder,
                line("respond", direct_target),
                6.0,
            )
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repetitions", type=int, default=300)
    parser.add_argument("--eval-fraction", type=float, default=0.04)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = build_examples(max(1, int(args.repetitions)))
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    train_count = 0
    eval_count = 0
    with train_path.open("w", encoding="utf-8") as train, eval_path.open("w", encoding="utf-8") as eval_file:
        for index, row in enumerate(examples):
            split = "eval" if index % max(2, int(1 / max(0.001, float(args.eval_fraction)))) == 0 else "train"
            payload = {**row, "split": split}
            if split == "eval":
                eval_file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                eval_count += 1
            else:
                train.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                train_count += 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": "assistant_intelligence_probe_targeted_multi_agent",
        "decoder_format": "line",
        "manifest_path": str(output_dir / "agentkernel_lite_encdec_dataset_manifest.json"),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(examples),
        "train_examples": train_count,
        "eval_examples": eval_count,
        "source_counts": {"agentkernel_lite_targeted_multi_agent_curriculum": len(examples)},
        "target_action_counts": {"respond": len(examples)},
    }
    Path(manifest["manifest_path"]).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
