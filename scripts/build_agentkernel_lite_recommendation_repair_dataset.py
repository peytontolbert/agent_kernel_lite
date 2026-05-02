#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


CASES = [
    {
        "query": "what's the best multi agent paper?",
        "answer": (
            "For a general multi-agent systems starting point, I would start with [1], "
            "Communicate only when necessary: Cooperative tasking for multi-agent systems. "
            "It directly studies cooperative tasking, communication, and decomposability in multi-agent systems. "
            "If you specifically mean LLM-based multi-agent collaboration, [3], Theory of Mind for Multi-Agent "
            "Collaboration via Large Language Models, is the better match."
        ),
    },
    {
        "query": "what's the best multi-agent llm paper?",
        "answer": (
            "For multi-agent LLMs, the best match in this evidence is [3], Theory of Mind for Multi-Agent "
            "Collaboration via Large Language Models. It directly studies LLM-based agents in a cooperative "
            "multi-agent text game and evaluates Theory of Mind behavior. [1] is useful background for classic "
            "multi-agent systems, but it is not the LLM-specific choice."
        ),
    },
    {
        "query": "best multi agent llm papers?",
        "answer": (
            "The strongest LLM-specific paper here is [3], Theory of Mind for Multi-Agent Collaboration via "
            "Large Language Models. It is about LLM agents collaborating in a multi-agent setting and measuring "
            "Theory of Mind capabilities. I would treat [1] and [2] as broader multi-agent systems background."
        ),
    },
]

GENERAL_CONTEXTS = [
    {
        "query": "what's the best multi agent paper?",
        "intent": "general_multi_agent_systems",
        "selected_id": "1",
        "target": (
            "Selected evidence: [1]. For a general multi-agent systems starting point, I would choose [1], "
            "Communicate only when necessary: Cooperative tasking for multi-agent systems. "
            "It is the best match here because it directly covers communication, decomposability, "
            "task automata, and cooperative control in multi-agent systems. If you specifically mean "
            "LLM-based agents, then [3], Theory of Mind for Multi-Agent Collaboration via Large Language Models, "
            "is the better follow-up."
        ),
    },
    {
        "query": "which paper should I start with for multi-agent systems?",
        "intent": "general_multi_agent_systems",
        "selected_id": "1",
        "target": (
            "Selected evidence: [1]. Start with [1], Communicate only when necessary: Cooperative tasking for multi-agent systems. "
            "The evidence says it directly studies cooperative control, communication rules, and decomposability "
            "in multi-agent systems, which makes it the strongest general starting point."
        ),
    },
    {
        "query": "best multi-agent systems paper",
        "intent": "general_multi_agent_systems",
        "selected_id": "1",
        "target": (
            "Selected evidence: [1]. For multi-agent systems broadly, choose [1], Communicate only when necessary: Cooperative tasking for multi-agent systems. "
            "It is the best match because it directly covers communication, decomposability, task automata, and cooperative control. "
            "[3] is only the better choice if the user specifically asks about LLM-based multi-agent collaboration."
        ),
    },
    {
        "query": "what paper is best for classic multi-agent systems?",
        "intent": "classic_multi_agent_systems",
        "selected_id": "1",
        "target": (
            "Selected evidence: [1]. For classic multi-agent systems, choose [1], Communicate only when necessary: Cooperative tasking for multi-agent systems. "
            "The key evidence is its focus on communication rules, decomposability, task automata, and cooperative control among agents."
        ),
    },
    {
        "query": "best paper about communication in multi-agent systems",
        "intent": "communication_multi_agent_systems",
        "selected_id": "1",
        "target": (
            "Selected evidence: [1]. The best match is [1], Communicate only when necessary: Cooperative tasking for multi-agent systems. "
            "It directly studies communication in multi-agent systems, including over-communications, missing communications, "
            "decomposability, and cooperative tasking."
        ),
    },
]

LLM_CONTEXTS = [
    {
        "query": "what's the best multi-agent llm paper?",
        "intent": "multi_agent_llm",
        "selected_id": "3",
        "target": (
            "Selected evidence: [3]. For multi-agent LLMs, I would choose [3], Theory of Mind for Multi-Agent Collaboration via Large Language Models. "
            "It is the best match because it directly studies LLM-based agents in a cooperative multi-agent text game, "
            "Theory of Mind inference tasks, collaboration, and comparisons with MARL and planning baselines."
        ),
    },
    {
        "query": "recommend multi-agent LLM papers",
        "intent": "multi_agent_llm",
        "selected_id": "3",
        "target": (
            "Selected evidence: [3]. The strongest LLM-specific paper in this evidence is [3], Theory of Mind for Multi-Agent Collaboration via Large Language Models. "
            "It directly covers LLM agents, multi-agent collaboration, Theory of Mind, and cooperative text-game evaluation. "
            "[1] is useful background for classic multi-agent systems, but [3] is the better LLM paper."
        ),
    },
    {
        "query": "best multi agent llm papers?",
        "intent": "multi_agent_llm",
        "selected_id": "3",
        "target": (
            "Selected evidence: [3]. I would start with [3], Theory of Mind for Multi-Agent Collaboration via Large Language Models. "
            "The reason is that it is explicitly about LLM-based agents collaborating in a multi-agent setting and evaluating "
            "Theory of Mind behavior. The other papers are broader multi-agent systems background."
        ),
    },
    {
        "query": "best paper on LLM agents collaborating",
        "intent": "multi_agent_llm_collaboration",
        "selected_id": "3",
        "target": (
            "Selected evidence: [3]. For LLM agents collaborating, choose [3], Theory of Mind for Multi-Agent Collaboration via Large Language Models. "
            "It directly studies LLM-based agents in a cooperative multi-agent text game, Theory of Mind inference, and collaborative behavior."
        ),
    },
    {
        "query": "which paper is best for theory of mind in multi-agent LLMs?",
        "intent": "theory_of_mind_multi_agent_llm",
        "selected_id": "3",
        "target": (
            "Selected evidence: [3]. Choose [3], Theory of Mind for Multi-Agent Collaboration via Large Language Models. "
            "The evidence directly connects multi-agent LLM collaboration with Theory of Mind inference tasks and higher-order Theory of Mind behavior."
        ),
    },
]


CONTEXT = """<AK_READING_NOTES>
Recommendation rule: compare evidence, prefer direct topic matches, cite evidence ids, and copy exact titles from evidence.
[1] Communicate only when necessary: Cooperative tasking for multi-agent systems
Relevant reading: New advances in large scale distributed systems have offered complex functionalities through parallelism of simple components. The key issue in cooperative control of multi-agent systems is synthesis of local control and interaction rules so the entire controlled system achieves desired global behavior.
[2] A new approach of designing Multi-Agent Systems
Relevant reading: Agent technology is a software paradigm for large distributed applications and this work presents a practical method for MAS design with component-oriented architecture and agent-based approach.
[3] Theory of Mind for Multi-Agent Collaboration via Large Language Models
Relevant reading: This study evaluates LLM-based agents in a multi-agent cooperative text game with Theory of Mind inference tasks and observes collaborative behaviors and higher-order Theory of Mind capabilities.
<AK_CONTEXT>
<AK_EVIDENCE> <AK_EVIDENCE_ID> 1
<AK_TITLE> Communicate only when necessary: Cooperative tasking for multi-agent systems
<AK_PAPER_ID> 1106.3134
<AK_CATEGORY> cs.MA
<AK_YEAR> 2011
<AK_ABSTRACT> New advances in large scale distributed systems have offered complex functionalities through parallelism of simple components. The key issue in cooperative control of multi-agent systems is synthesis of local control and interaction rules so the entire controlled system achieves desired global behavior.

<AK_EVIDENCE> <AK_EVIDENCE_ID> 2
<AK_TITLE> A new approach of designing Multi-Agent Systems
<AK_PAPER_ID> 1204.1581
<AK_CATEGORY> cs.MA
<AK_YEAR> 2012
<AK_ABSTRACT> Agent technology is a software paradigm for large distributed applications and this work presents a practical method for MAS design with component-oriented architecture and agent-based approach.

<AK_EVIDENCE> <AK_EVIDENCE_ID> 3
<AK_TITLE> Theory of Mind for Multi-Agent Collaboration via Large Language Models
<AK_PAPER_ID> 2310.10701
<AK_CATEGORY> cs.CL
<AK_YEAR> 2023
<AK_ABSTRACT> This study evaluates LLM-based agents in a multi-agent cooperative text game with Theory of Mind inference tasks, comparing performance with MARL and planning baselines. It observes emergent collaborative behaviors and high-order Theory of Mind capabilities among LLM agents."""


PAPER_META = {
    "1": {
        "title": "Communicate only when necessary: Cooperative tasking for multi-agent systems",
        "category": "cs.MA",
        "year": "2011",
        "abstract": (
            "New advances in large scale distributed systems have offered complex functionalities through "
            "parallelism of simple components. The key issue in cooperative control of multi-agent systems "
            "is synthesis of local control and interaction rules so the entire controlled system achieves "
            "desired global behavior."
        ),
    },
    "3": {
        "title": "Theory of Mind for Multi-Agent Collaboration via Large Language Models",
        "category": "cs.CL",
        "year": "2023",
        "abstract": (
            "This study evaluates LLM-based agents in a multi-agent cooperative text game with Theory of "
            "Mind inference tasks, comparing performance with MARL and planning baselines. It observes "
            "emergent collaborative behaviors and high-order Theory of Mind capabilities among LLM agents."
        ),
    },
}


def selected_context(case: dict[str, str]) -> str:
    ids = ["1", "3"] if case["query"] == "what's the best multi agent paper?" else [str(case["selected_id"])]
    blocks = ["<AK_CONTEXT>"]
    for paper_id in ids:
        paper = PAPER_META[paper_id]
        blocks.append(
            "\n".join(
                [
                    f"<AK_EVIDENCE> <AK_EVIDENCE_ID> {paper_id}",
                    f"<AK_TITLE> {paper['title']}",
                    f"<AK_CATEGORY> {paper['category']}",
                    f"<AK_YEAR> {paper['year']}",
                    f"<AK_ABSTRACT> {paper['abstract']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def selected_answer_target(case: dict[str, str]) -> str:
    query = case["query"]
    selected_id = str(case["selected_id"])
    if query == "what's the best multi agent paper?":
        return (
            "Selected evidence: [1] and [3]. For classic or general multi-agent systems, I would start with "
            "[1], Communicate only when necessary: Cooperative tasking for multi-agent systems, because it "
            "directly covers communication, decomposability, task automata, and cooperative control. If you "
            "mean LLM-based multi-agent collaboration, choose [3], Theory of Mind for Multi-Agent Collaboration "
            "via Large Language Models, because it directly studies LLM agents, Theory of Mind inference, "
            "and cooperative text-game collaboration."
        )
    if selected_id == "1":
        return (
            "Selected evidence: [1]. The best match is [1], Communicate only when necessary: Cooperative tasking "
            "for multi-agent systems. It directly studies communication, decomposability, task automata, and "
            "cooperative control in multi-agent systems."
        )
    return (
        "Selected evidence: [3]. The best match is [3], Theory of Mind for Multi-Agent Collaboration via Large "
        "Language Models. It directly studies LLM-based agents in a cooperative multi-agent text game, Theory "
        "of Mind inference tasks, collaboration, and comparisons with MARL and planning baselines."
    )


def row(case: dict[str, str], repeat: int) -> dict[str, object]:
    return {
        "source_type": "agentkernel_lite_recommendation_repair",
        "source_id": f"recommendation:{repeat}:{case['query']}",
        "task_type": "copy_grounded_recommendation_repair",
        "encoder_text": "\n".join(
            [
                "<AK_CHAT> <AK_RESPOND> <AK_CONTEXT> <AK_READING_NOTES> <AK_ANSWER>",
                "<AK_LOOP> <AK_STATE> mode=chat selected_context=0 retrieval=ranked",
                f"<AK_USER> {case['query']}",
                CONTEXT,
                "Answer directly. Recommend only papers present in evidence. Copy exact titles.",
            ]
        ),
        "decoder_text": "Action: respond\nContent: " + case["answer"],
        "action": "respond",
        "weight": 12.0,
    }


def strong_row(case: dict[str, str], repeat: int) -> dict[str, object]:
    return {
        "source_type": "agentkernel_lite_recommendation_repair",
        "source_id": f"strong_recommendation:{repeat}:{case['query']}",
        "task_type": "strict_copy_grounded_recommendation_repair",
        "encoder_text": "\n".join(
            [
                "<AK_CHAT> <AK_RESPOND> <AK_CONTEXT> <AK_READING_NOTES> <AK_ANSWER>",
                "<AK_LOOP> <AK_STATE> mode=chat selected_context=0 retrieval=ranked",
                f"<AK_USER> {case['query']}",
                CONTEXT,
                "Choose the best matching evidence item. Copy its exact title. Cite the chosen paper id. Do not invent titles.",
            ]
        ),
        "decoder_text": "Action: respond\nContent: " + selected_answer_target(case),
        "action": "respond",
        "weight": 18.0,
    }


def selected_answer_row(case: dict[str, str], repeat: int) -> dict[str, object]:
    return {
        "source_type": "agentkernel_lite_recommendation_repair",
        "source_id": f"selected_answer_recommendation:{repeat}:{case['query']}",
        "task_type": "selected_evidence_recommendation_answer",
        "encoder_text": "\n".join(
            [
                "<AK_CHAT> <AK_RESPOND> <AK_CONTEXT> <AK_READING_NOTES> <AK_ANSWER>",
                "<AK_LOOP> <AK_STATE> mode=chat selected_context=1 retrieval=selected",
                f"<AK_USER> {case['query']}",
                selected_context(case),
                "Answer directly from the selected evidence. Copy exact titles from evidence and cite selected ids.",
            ]
        ),
        "decoder_text": "Action: respond\nContent: " + case["target"],
        "action": "respond",
        "weight": 20.0,
    }


def selection_row(case: dict[str, str], repeat: int) -> dict[str, object]:
    selected_id = str(case["selected_id"])
    intent = str(case["intent"])
    title = (
        "Communicate only when necessary: Cooperative tasking for multi-agent systems"
        if selected_id == "1"
        else "Theory of Mind for Multi-Agent Collaboration via Large Language Models"
    )
    return {
        "source_type": "agentkernel_lite_recommendation_repair",
        "source_id": f"selection_recommendation:{repeat}:{case['query']}",
        "task_type": "intent_conditioned_evidence_selection",
        "encoder_text": "\n".join(
            [
                "<AK_CHAT> <AK_GATHER_CONTEXT> <AK_RERANK> <AK_CONTEXT>",
                "<AK_LOOP> <AK_STATE> mode=chat selected_context=0 retrieval=ranked",
                f"<AK_USER> {case['query']}",
                CONTEXT,
                "Select the best evidence id for the user intent. Do not answer yet.",
            ]
        ),
        "decoder_text": (
            "Action: gather_context\n"
            f"Content: intent={intent}; selected_candidate_id={selected_id}; selected_title={title}"
        ),
        "action": "gather_context",
        "weight": 14.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=160)
    parser.add_argument("--eval-repeats", type=int, default=4)
    parser.add_argument(
        "--include-selector-rows",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Include gather_context/rerank selector rows. Keep this disabled for "
            "single-pass respond training; selector rows require a two-pass runtime "
            "and otherwise collapse the answer distribution."
        ),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    strong_cases = GENERAL_CONTEXTS + LLM_CONTEXTS
    train_rows = [row(case, repeat) for repeat in range(args.repeats) for case in CASES] + [
        strong_row(case, repeat) for repeat in range(args.repeats) for case in strong_cases
    ] + [
        selected_answer_row(case, repeat) for repeat in range(args.repeats) for case in strong_cases
    ]
    eval_rows = [row(case, repeat) for repeat in range(args.eval_repeats) for case in CASES] + [
        strong_row(case, repeat) for repeat in range(args.eval_repeats) for case in strong_cases
    ] + [
        selected_answer_row(case, repeat) for repeat in range(args.eval_repeats) for case in strong_cases
    ]
    if int(args.include_selector_rows):
        train_rows.extend(selection_row(case, repeat) for repeat in range(args.repeats) for case in strong_cases)
        eval_rows.extend(selection_row(case, repeat) for repeat in range(args.eval_repeats) for case in strong_cases)
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    for path, rows in ((train_path, train_rows), (eval_path, eval_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for item in rows:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": (
            "copy_grounded_recommendation_repair_with_selector"
            if int(args.include_selector_rows)
            else "copy_grounded_recommendation_repair_respond_only"
        ),
        "decoder_format": "line",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(train_rows) + len(eval_rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "task_type_counts": {
            "copy_grounded_recommendation_repair": (len(CASES) * (args.repeats + args.eval_repeats)),
            "strict_copy_grounded_recommendation_repair": (len(strong_cases) * (args.repeats + args.eval_repeats)),
            "selected_evidence_recommendation_answer": (len(strong_cases) * (args.repeats + args.eval_repeats)),
            "intent_conditioned_evidence_selection": (
                (len(strong_cases) * (args.repeats + args.eval_repeats)) if int(args.include_selector_rows) else 0
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
