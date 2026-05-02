#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


AK_CHAT = "<AK_CHAT>"
AK_THINK = "<AK_THINK>"
AK_DEEP_RESEARCH = "<AK_DEEP_RESEARCH>"
AK_LOOP = "<AK_LOOP>"
AK_PLAN = "<AK_PLAN>"
AK_STATE = "<AK_STATE>"
AK_CONTEXT = "<AK_CONTEXT>"
AK_HISTORY = "<AK_HISTORY>"
AK_READING_NOTES = "<AK_READING_NOTES>"
AK_ACTIVE_CONTEXT = "<AK_ACTIVE_CONTEXT>"
AK_EVIDENCE = "<AK_EVIDENCE>"
AK_EVIDENCE_ID = "<AK_EVIDENCE_ID>"
AK_TITLE = "<AK_TITLE>"
AK_PAPER_ID = "<AK_PAPER_ID>"
AK_CATEGORY = "<AK_CATEGORY>"
AK_YEAR = "<AK_YEAR>"
AK_ABSTRACT = "<AK_ABSTRACT>"
AK_SELECTED_PAPER = "<AK_SELECTED_PAPER>"
AK_GATHER_CONTEXT = "<AK_GATHER_CONTEXT>"
AK_RESPOND = "<AK_RESPOND>"
AK_USE_CONTEXT = "<AK_USE_CONTEXT>"
AK_NO_RETRIEVAL = "<AK_NO_RETRIEVAL>"
AK_RETRIEVE_NEW = "<AK_RETRIEVE_NEW>"
AK_ANSWER = "<AK_ANSWER>"
AK_USER = "<AK_USER>"


def compact(value: object, limit: int = 900) -> str:
    return " ".join(str(value or "").replace("\r", "\n").split())[:limit].strip()


def line(action: str, content: str) -> str:
    return f"Action: {action}\nContent: {compact(content, 1100)}"


def state(mode: str, selected: bool, retrieval: str) -> str:
    return f"{AK_LOOP} {AK_STATE} mode={mode} selected_context={1 if selected else 0} retrieval={retrieval}"


def hash_split(key: str, eval_fraction: float) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < eval_fraction else "train"


def tagged(text: str, label: str) -> str:
    match = re.search(rf"(?ims)^\s*{re.escape(label)}\s*:\s*(.*?)(?=^\s*[A-Z_ ]+\s*:|\Z)", text or "")
    return compact(match.group(1), 1400) if match else ""


def sentences(text: str, limit: int = 4) -> list[str]:
    text = compact(text, 1800)
    parts = re.findall(r"[^.!?]+[.!?]", text)
    if not parts and text:
        parts = [text]
    return [compact(part, 360) for part in parts[:limit] if compact(part, 360)]


def clean_summary(text: str, *, limit: int = 520) -> str:
    value = compact(text, limit)
    value = re.sub(r"(?i)^(this paper|the paper|we)\s+(is\s+about|presents|introduces|studies|proposes|describes)\s+", "", value).strip()
    return value[:limit].rstrip()


def terms(*values: object, limit: int = 7) -> list[str]:
    skip = {
        "the", "and", "for", "with", "from", "this", "that", "paper", "papers",
        "study", "studies", "using", "based", "toward", "between", "large",
        "model", "models", "method", "approach", "result", "results",
    }
    seen: set[str] = set()
    out: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", " ".join(str(v or "") for v in values).lower()):
        token = token.strip("-")
        if token in skip or token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= limit:
            break
    return out


def paper_row(payload: dict[str, Any], source_file: str, row_index: int) -> dict[str, str] | None:
    tagged_text = str(payload.get("text_a", "") or payload.get("text_b", "") or "")
    title = compact(payload.get("title") or tagged(tagged_text, "TITLE") or tagged(tagged_text, "Title"), 180)
    abstract = compact(payload.get("abstract") or tagged(tagged_text, "ABSTRACT") or tagged(tagged_text, "Abstract"), 1400)
    body = compact(payload.get("text") or payload.get("full_text") or abstract, 1800)
    summary = " ".join(sentences(abstract or body, 3))
    if not title or len(summary) < 80:
        return None
    return {
        "paper_id": compact(payload.get("paper_id") or payload.get("canonical_paper_id") or payload.get("arxiv_id") or payload.get("id"), 80),
        "title": title,
        "summary": summary,
        "categories": compact(payload.get("categories") or payload.get("primary_category"), 120),
        "year": compact(payload.get("year") or payload.get("published_year") or payload.get("update_date"), 40),
        "source_file": source_file,
        "row_index": str(row_index),
    }


def read_rows(path: Path, max_rows: int, max_files: int) -> list[dict[str, str]]:
    import pyarrow.parquet as pq

    paths = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if max_files > 0:
        paths = paths[:max_files]
    columns = [
        "paper_id", "canonical_paper_id", "arxiv_id", "id", "title", "abstract",
        "text", "full_text", "text_a", "text_b", "categories", "primary_category",
        "year", "published_year", "update_date",
    ]
    rows: list[dict[str, str]] = []
    for file_path in paths:
        parquet_file = pq.ParquetFile(file_path)
        available = set(parquet_file.schema_arrow.names)
        read_columns = [column for column in columns if column in available]
        if not ({"title", "text_a", "text_b"} & set(read_columns)):
            continue
        for batch in parquet_file.iter_batches(batch_size=512, columns=read_columns):
            for row_index, payload in enumerate(batch.to_pylist()):
                row = paper_row(payload, file_path.name, row_index)
                if row:
                    rows.append(row)
                    if len(rows) >= max_rows:
                        return rows
    return rows


def evidence(row: dict[str, str], index: int) -> str:
    return "\n".join(
        item for item in [
            f"{AK_EVIDENCE} {AK_EVIDENCE_ID} P{index}",
            f"{AK_TITLE} {row['title']}",
            f"{AK_PAPER_ID} {row['paper_id']}",
            f"{AK_CATEGORY} {row['categories']}",
            f"{AK_YEAR} {row['year']}",
            f"{AK_ABSTRACT} {row['summary']}",
        ]
        if item.strip()
    )


def reading_notes(rows: list[dict[str, str]], query: str, *, mode: str) -> str:
    rule = (
        "Recommendation rule: compare evidence, prefer direct topic matches, cite evidence ids, and do not invent titles."
        if re.search(r"\b(best|top|recommend|which|survey|compare)\b", query.lower())
        else "Synthesis rule: answer the user first, then cite only evidence ids that support the answer."
    )
    blocks = [rule]
    for i, row in enumerate(rows, 1):
        take = " ".join(sentences(row["summary"], 3 if mode == "deep_research" else 2))
        blocks.append(f"[P{i}] {row['title']}\nRelevant reading: {take}")
    return "\n\n".join(blocks)


def selected_reading_notes(row: dict[str, str], query: str) -> str:
    take = " ".join(sentences(row["summary"], 3))
    return "\n\n".join([
        "Selected-context rule: answer from the active selected paper P1, cite [P1], and do not call this fresh retrieval.",
        f"[P1] {row['title']}\nActive paper reading: {take}",
        f"User follow-up intent: {compact(query, 220)}",
    ])


def context(rows: list[dict[str, str]]) -> str:
    return "\n\n".join(evidence(row, index) for index, row in enumerate(rows, 1))


def example(source_id: str, task_type: str, encoder: str, decoder: str, action: str, weight: float) -> dict[str, Any]:
    return {
        "source_type": "agentkernel_lite_assistant_probe_curriculum",
        "source_id": source_id,
        "task_type": task_type,
        "encoder_text": encoder,
        "decoder_text": decoder,
        "action": action,
        "weight": weight,
    }


def make_examples(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seed_chats = [
        (
            "What is multi-agent intelligence?",
            "Multi-agent intelligence is the study and engineering of systems where multiple agents coordinate, communicate, compete, or divide work. A useful answer separates the agents, their shared state, the communication protocol, the decision rule, and how success is measured.",
        ),
        (
            "Can you help me understand a paper if I add it to chat?",
            "Yes. Add the paper to chat and I should answer about that selected paper, use its active context, avoid searching for unrelated papers, and explain the method, evidence, limitations, and takeaway in normal language.",
        ),
        (
            "Do you always need to retrieve papers?",
            "No. For ordinary chat I should answer directly. I should gather context when the user asks for papers, sources, literature, evidence, comparisons, or deeper research.",
        ),
        (
            "Explain the difference between chat, think, and deep research.",
            "Chat should answer directly and briefly. Think should spend more effort synthesizing the available context. Deep research should inspect the evidence more carefully, compare sources, preserve uncertainty, and cite the papers it used.",
        ),
        (
            "If I ask a normal question without mentioning papers, what should you do?",
            "I should answer the question directly. I should not add paper retrieval unless the user asks for papers, sources, citations, literature, comparisons, or a deeper research pass.",
        ),
        (
            "What should you do after I add a paper to chat?",
            "If the next question refers to this paper, the selected paper, it, or the loaded context, I should answer from that active paper context and avoid retrieving unrelated papers unless the user asks for related or newer work.",
        ),
    ]
    for i, (prompt, answer) in enumerate(seed_chats):
        examples.append(example(
            f"seed_chat:{i}",
            "direct_chat_no_retrieval",
            f"{AK_CHAT} {AK_RESPOND}\n{state('chat', False, 'none')}\n{AK_USER} {prompt}",
            line("respond", answer),
            "respond",
            3.0,
        ))

    for i, row in enumerate(rows):
        neg = rows[(i + max(1, len(rows) // 3)) % len(rows)]
        other = rows[(i + max(2, len(rows) // 5)) % len(rows)]
        topic = " ".join(terms(row["title"], row["summary"], limit=5)) or row["title"]

        examples.append(example(
            f"{row['paper_id'] or i}:plan_retrieval",
            "plan_gather_context_for_research_question",
            f"{AK_CHAT} {AK_PLAN} {AK_GATHER_CONTEXT}\n{state('chat', False, 'none')}\n{AK_HISTORY}\nUser: Tell me about {topic}.\nAssistant: I can answer generally or gather papers if you want sources.\n{AK_USER} What are the best papers on {topic}?",
            line("gather_context", f"query={topic}; reason=the user asks for papers/sources, so gather ranked research context before answering"),
            "gather_context",
            3.5,
        ))

        examples.append(example(
            f"{row['paper_id'] or i}:plan_retrieval_sources",
            "plan_gather_context_for_source_request",
            f"{AK_CHAT} {AK_PLAN} {AK_GATHER_CONTEXT}\n{state('chat', False, 'none')}\n{AK_USER} Find sources and papers about {topic}.",
            line("gather_context", f"query={topic}; reason=the user explicitly asks for sources and papers"),
            "gather_context",
            3.2,
        ))

        examples.append(example(
            f"{row['paper_id'] or i}:direct_topic_no_sources",
            "direct_topic_explanation_no_retrieval",
            "\n".join([
                f"{AK_CHAT} {AK_RESPOND}",
                state("chat", False, "none"),
                f"{AK_HISTORY}\nUser: We can use papers when needed.\nAssistant: Yes, but I should also answer normal questions directly.",
                f"{AK_USER} Explain {topic} simply without sources.",
            ]),
            line("respond", f"In plain language, {topic} is a way to describe related systems, methods, or ideas working together on a problem. It matters because it gives us a vocabulary for comparing designs and tradeoffs. A simple example is splitting one complex task into smaller parts that can be handled and checked separately."),
            "respond",
            4.5,
        ))

        examples.append(example(
            f"{row['paper_id'] or i}:direct_general_no_papers",
            "direct_chat_no_retrieval",
            "\n".join([
                f"{AK_CHAT} {AK_RESPOND}",
                state("chat", False, "none"),
                f"{AK_USER} What does {topic} mean at a high level? Do not pull papers.",
            ]),
            line("respond", f"At a high level, {topic} is a concept I can explain directly without paper retrieval. The useful way to understand it is to define the goal, describe the moving parts, and then give one concrete example."),
            "respond",
            4.6,
        ))

        candidates = [row, neg, other]
        query = f"what is the best paper about {topic}?"
        examples.append(example(
            f"{row['paper_id'] or i}:recommend_from_evidence",
            "grounded_recommendation_with_reading_notes",
            "\n".join([
                f"{AK_CHAT} {AK_RESPOND} {AK_CONTEXT} {AK_READING_NOTES} {AK_ANSWER}",
                state("chat", False, "ranked"),
                f"{AK_USER} {query}",
                f"{AK_READING_NOTES}\n{reading_notes(candidates, query, mode='chat')}",
                f"{AK_CONTEXT}\n{context(candidates)}",
            ]),
            line("respond", f"I would start with [1] because it is the closest match in the retrieved evidence. Its main point is: {clean_summary(row['summary'])} I would use [2] and [3] only as comparison papers unless their methods or benchmarks overlap with [1]."),
            "respond",
            5.0,
        ))

        examples.append(example(
            f"{row['paper_id'] or i}:think_synthesis",
            "think_mode_synthesis_across_evidence",
            "\n".join([
                f"{AK_THINK} {AK_RESPOND} {AK_CONTEXT} {AK_READING_NOTES}",
                state("think", False, "ranked"),
                f"{AK_USER} Explain the common idea and difference across these papers.",
                f"{AK_READING_NOTES}\n{reading_notes(candidates, 'compare papers', mode='think')}",
                f"{AK_CONTEXT}\n{context(candidates)}",
            ]),
            line("respond", f"The common thread is that the retrieved papers each address a technical problem from a different angle. For this question, [1] is the strongest evidence: {clean_summary(row['summary'])} I would compare [2] and [3] by checking whether they study the same task, method, benchmark, or limitation."),
            "respond",
            4.0,
        ))

        examples.append(example(
            f"{row['paper_id'] or i}:deep_research",
            "deep_research_evidence_audit",
            "\n".join([
                f"{AK_DEEP_RESEARCH} {AK_RESPOND} {AK_CONTEXT} {AK_READING_NOTES}",
                state("deep_research", False, "ranked"),
                f"{AK_USER} Do a careful research pass on {topic}.",
                f"{AK_READING_NOTES}\n{reading_notes(candidates, topic, mode='deep_research')}",
                f"{AK_CONTEXT}\n{context(candidates)}",
            ]),
            line("respond", f"Based on the retrieved papers, [1] supports this main takeaway: {clean_summary(row['summary'])} The next step is to compare whether [2] or [3] confirm the same claim, contradict it, or only cover adjacent work."),
            "respond",
            4.0,
        ))

        examples.append(example(
            f"{row['paper_id'] or i}:selected_followup_history",
            "selected_paper_followup_with_history",
            "\n".join([
                f"{AK_CHAT} {AK_RESPOND} {AK_ACTIVE_CONTEXT} {AK_SELECTED_PAPER} {AK_USE_CONTEXT}",
                state("chat", True, "active"),
                f"{AK_HISTORY}\nUser: Add [1] to chat.\nAssistant: Added the selected paper [P1].",
                f"{AK_READING_NOTES}\n{selected_reading_notes(row, 'tell me more about this paper')}",
                f"{AK_CONTEXT}\n{context([row])}",
                f"{AK_USER} Tell me more about this paper please.",
            ]),
            line("respond", f"I am using the active selected paper [P1], not retrieving new papers. Its main idea is: {clean_summary(row['summary'])} The most useful follow-up is to inspect the paper's problem statement, method, evidence, and limitations."),
            "respond",
            6.0,
        ))

        followup_variants = [
            "Explain the paper in detail.",
            "Can you explani the paper in detail?",
            "Tell me more about this paper.",
            "What are the main takeaways from it?",
        ]
        for variant_index, followup in enumerate(followup_variants):
            examples.append(example(
                f"{row['paper_id'] or i}:selected_followup_variant:{variant_index}",
                "selected_paper_followup_no_new_retrieval",
                "\n".join([
                    f"{AK_CHAT} {AK_PLAN} {AK_RESPOND} {AK_ACTIVE_CONTEXT} {AK_SELECTED_PAPER} {AK_USE_CONTEXT} {AK_NO_RETRIEVAL}",
                    state("chat", True, "active"),
                    f"{AK_HISTORY}\nUser: Add [1] to chat.\nAssistant: Loaded the selected paper [P1].",
                    f"{AK_READING_NOTES}\n{selected_reading_notes(row, followup)}",
                    f"{AK_CONTEXT}\n{context([row])}",
                    f"{AK_USER} {followup}",
                ]),
                line(
                    "respond",
                    (
                        f"I am using the selected paper [P1], not retrieving new papers. "
                        f"In detail, its core idea is: {clean_summary(row['summary'])} "
                        "A good reading path is to identify the problem, the method or argument, the evidence, and the limitations."
                    ),
                ),
                "respond",
                6.2,
            ))

        related_variants = [
            "Find related papers that build on this work.",
            "What newer work should I read after this paper?",
        ]
        for variant_index, followup in enumerate(related_variants):
            examples.append(example(
                f"{row['paper_id'] or i}:selected_followup_retrieve_new:{variant_index}",
                "selected_paper_followup_retrieve_new",
                "\n".join([
                    f"{AK_CHAT} {AK_PLAN} {AK_GATHER_CONTEXT} {AK_ACTIVE_CONTEXT} {AK_SELECTED_PAPER} {AK_RETRIEVE_NEW}",
                    state("chat", True, "active"),
                    f"{AK_HISTORY}\nUser: Add [1] to chat.\nAssistant: Loaded the selected paper [P1].",
                    f"{AK_CONTEXT}\n{context([row])}",
                    f"{AK_USER} {followup}",
                ]),
                line(
                    "gather_context",
                    f"query=related work {topic}; reason=the user asks for additional literature beyond the selected paper",
                ),
                "gather_context",
                2.8,
            ))

        examples.append(example(
            f"{row['paper_id'] or i}:weak_retrieval_uncertainty",
            "uncertainty_when_evidence_off_topic",
            "\n".join([
                f"{AK_CHAT} {AK_RESPOND} {AK_CONTEXT} {AK_READING_NOTES}",
                state("chat", False, "ranked"),
                f"{AK_USER} What is the best paper about quantum-safe distributed databases?",
                f"{AK_READING_NOTES}\nSynthesis rule: if evidence is off topic, say so and ask for a narrower query.\n[P1] {row['title']}\nRelevant reading: {row['summary']}",
                f"{AK_CONTEXT}\n{context([row])}",
            ]),
            line("respond", "I do not have strong enough retrieved evidence to recommend a best paper for that exact topic. The visible evidence [1] may be useful background, but it does not clearly support the requested claim."),
            "respond",
            1.2,
        ))

        examples.append(example(
            f"{row['paper_id'] or i}:plain_question_after_history",
            "history_aware_direct_chat",
            "\n".join([
                f"{AK_CHAT} {AK_RESPOND}",
                state("chat", False, "none"),
                f"{AK_HISTORY}\nUser: We were discussing {topic}.\nAssistant: I can explain it directly or gather papers.",
                f"{AK_USER} Explain it simply without sources.",
            ]),
            line("respond", f"In simple terms, {topic} is the topic from the prior turn. It matters because it gives us a way to organize the problem and compare possible approaches. One concrete example is using the idea to break a broad system into smaller pieces that can be reasoned about separately."),
            "respond",
            4.5,
        ))
    return examples


def write_dataset(examples: list[dict[str, Any]], output_dir: Path, eval_fraction: float) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for row in examples:
        split = hash_split(str(row["source_id"]), eval_fraction)
        target = eval_rows if split == "eval" else train_rows
        target.append({**row, "split": split})
    for path, rows in ((train_path, train_rows), (eval_path, eval_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    task_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for row in examples:
        task_counts[row["task_type"]] = task_counts.get(row["task_type"], 0) + 1
        action_counts[row["action"]] = action_counts.get(row["action"], 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": "assistant_intelligence_probe",
        "decoder_format": "line",
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(examples),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "source_counts": {"agentkernel_lite_assistant_probe_curriculum": len(examples)},
        "task_type_counts": dict(sorted(task_counts.items())),
        "target_action_counts": dict(sorted(action_counts.items())),
        "schema": {
            "encoder_text": "AgentKernel Lite prompt with history, reading notes, retrieval/selected context state",
            "decoder_text": "line protocol action/content target",
            "weight": "weighted loss multiplier",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-text-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-paper-text-examples", type=int, default=5000)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--eval-fraction", type=float, default=0.03)
    args = parser.parse_args()
    rows = read_rows(Path(args.paper_text_path).expanduser().resolve(), args.max_paper_text_examples, args.max_files)
    examples = make_examples(rows)
    manifest = write_dataset(examples, Path(args.output_dir).expanduser().resolve(), args.eval_fraction)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
