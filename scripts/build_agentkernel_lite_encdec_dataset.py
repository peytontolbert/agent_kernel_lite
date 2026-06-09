#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _compact(value: object, *, limit: int = 4000) -> str:
    text = " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split())
    return text[:limit].rstrip()


def _hash_split(key: str, eval_fraction: float) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < max(0.0, min(0.5, float(eval_fraction))) else "train"


def _is_code_action(action: str) -> bool:
    normalized = str(action or "").strip().lower()
    return normalized in {"code_execute", "propose_code", "propose_artifact_write"} or "code" in normalized


def _chat_decision_text(
    content: str,
    *,
    thought: str = "",
    source_action: str = "",
    extension_capability: str = "",
    retrieval_influenced: bool = False,
) -> str:
    metadata: dict[str, Any] = {}
    if source_action:
        metadata["source_action"] = source_action
    if extension_capability:
        metadata["extension_capability"] = extension_capability
    payload = {
        "thought": _compact(thought, limit=500),
        "action": "respond",
        "content": _compact(content, limit=6000),
        "done": False,
        "selected_retrieval_span_id": None,
        "retrieval_influenced": bool(retrieval_influenced),
        "proposal_metadata": metadata,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _structured_decision_text(
    *,
    action: str,
    content: str,
    thought: str = "",
    retrieval_influenced: bool = False,
    metadata: dict[str, Any] | None = None,
    selected_retrieval_span_id: str | None = None,
) -> str:
    payload = {
        "action": _compact(action or "respond", limit=80),
        "content": _compact(content, limit=7000),
    }
    if selected_retrieval_span_id is not None:
        payload["selected_retrieval_span_id"] = selected_retrieval_span_id
    if retrieval_influenced:
        payload["retrieval_influenced"] = True
    if metadata:
        payload["proposal_metadata"] = metadata
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _chat_target_from_step(
    *,
    prompt: str,
    action: str,
    content: str,
    reason_text: str,
    code_trace_mode: str,
) -> tuple[str, float, str] | None:
    if _is_code_action(action):
        if code_trace_mode == "skip":
            return None
        if code_trace_mode == "raw":
            return (
                _chat_decision_text(
                    content,
                    thought="Raw code/tool action preserved for extension distillation.",
                    source_action=action,
                    extension_capability="code.generate_draft",
                    retrieval_influenced=bool(reason_text),
                ),
                0.35,
                "code.generate_draft",
            )
        command = _compact(content, limit=900)
        feedback = f" Verifier feedback: {reason_text}." if reason_text else ""
        target = (
            "In chat mode I cannot execute commands or edit files directly. "
            "The useful kernel trace points to a coding action that can be routed through the "
            "`code.generate_draft` extension when enabled. "
            f"For this request, the relevant implementation intent is: {command}.{feedback} "
            "I would explain the change, constraints, and next verification step before asking for extension approval."
        )
        return (
            _chat_decision_text(
                target,
                thought="Translate an execution trace into browser-chat guidance.",
                source_action=action,
                extension_capability="code.generate_draft",
                retrieval_influenced=bool(reason_text),
            ),
            0.35,
            "code.generate_draft",
        )
    if content:
        return (
            _chat_decision_text(
                content,
                thought="Answer directly in chat using the available kernel context.",
                source_action=action or "respond",
                retrieval_influenced=bool(reason_text),
            ),
            1.0,
            "",
        )
    fallback = f"I need more context before answering this request: {prompt}"
    return (_chat_decision_text(fallback, source_action=action or "respond"), 0.5, "")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _trajectory_examples(
    trajectory_root: Path,
    *,
    max_files: int,
    objective: str,
    code_trace_mode: str,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    paths = sorted(trajectory_root.rglob("*.json"))
    if max_files > 0:
        paths = paths[-max_files:]
    for path in paths:
        payload = _load_json(path)
        if not payload:
            continue
        prompt = _compact(payload.get("prompt", ""), limit=2500)
        steps = payload.get("steps", [])
        if not prompt or not isinstance(steps, list):
            continue
        metadata = payload.get("task_metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        benchmark_family = _compact(metadata.get("benchmark_family", ""), limit=120)
        difficulty = _compact(metadata.get("difficulty", metadata.get("task_difficulty", "")), limit=120)
        history: list[str] = []
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            content = _compact(step.get("content", ""), limit=6000)
            action = _compact(step.get("action", ""), limit=80) or "respond"
            if not content:
                continue
            verification = step.get("verification", {})
            verification = verification if isinstance(verification, dict) else {}
            reasons = verification.get("reasons", [])
            reason_text = " | ".join(_compact(reason, limit=240) for reason in reasons[:3]) if isinstance(reasons, list) else ""
            if objective == "chat":
                target = _chat_target_from_step(
                    prompt=prompt,
                    action=action,
                    content=content,
                    reason_text=reason_text,
                    code_trace_mode=code_trace_mode,
                )
                if target is None:
                    history.append(f"{index + 1}:{action}:{content[:320]}")
                    continue
                decoder_text, base_weight, extension_capability = target
                encoder_parts = [
                    "AgentKernel Lite chat distillation example.",
                    "Runtime: browser chat, no direct code execution, no file writes.",
                    "If a coding task appears, answer conversationally or route it through an extension capability.",
                    f"User request: {prompt}",
                    f"Benchmark family: {benchmark_family}" if benchmark_family else "",
                    f"Difficulty: {difficulty}" if difficulty else "",
                    f"Recent chat/kernel history: {' || '.join(history[-3:])}" if history else "",
                    f"Prior verifier feedback: {reason_text}" if reason_text else "",
                    f"Observed source action: {action}",
                    f"Available extension: {extension_capability}" if extension_capability else "",
                    "Return a compact JSON chat decision with action=respond.",
                ]
                target_action = "respond"
            elif objective == "text":
                if _is_code_action(action) and code_trace_mode == "skip":
                    history.append(f"{index + 1}:{action}:{content[:320]}")
                    continue
                if _is_code_action(action):
                    content = (
                        "In chat mode I cannot execute commands or edit files directly. "
                        "This request should be routed through a code extension when enabled."
                    )
                decoder_text = content
                base_weight = 1.0
                extension_capability = ""
                encoder_parts = [
                    "AgentKernel Lite plain chat distillation example.",
                    "Runtime: browser chat, no direct code execution, no file writes.",
                    f"User request: {prompt}",
                    f"Benchmark family: {benchmark_family}" if benchmark_family else "",
                    f"Difficulty: {difficulty}" if difficulty else "",
                    f"Recent chat/kernel history: {' || '.join(history[-3:])}" if history else "",
                    "Answer in plain text. Do not emit JSON.",
                ]
                target_action = "respond"
            else:
                decoder_text = content
                base_weight = 1.5 if verification.get("passed", False) else 1.0
                extension_capability = ""
                encoder_parts = [
                    "AgentKernel Lite distillation example.",
                    f"Task: {prompt}",
                    f"Benchmark family: {benchmark_family}" if benchmark_family else "",
                    f"Difficulty: {difficulty}" if difficulty else "",
                    f"Recent history: {' || '.join(history[-3:])}" if history else "",
                    f"Previous verifier feedback: {reason_text}" if reason_text else "",
                    f"Required action surface: {action}",
                ]
                target_action = action
            source_id = f"{path.relative_to(trajectory_root)}:{index}"
            examples.append(
                {
                    "source_type": "trajectory_step",
                    "source_id": source_id,
                    "encoder_text": "\n".join(part for part in encoder_parts if part),
                    "decoder_text": decoder_text,
                    "action": target_action,
                    "source_action": action,
                    "extension_capability": extension_capability,
                    "benchmark_family": benchmark_family,
                    "difficulty": difficulty,
                    "weight": base_weight * (1.5 if verification.get("passed", False) else 1.0),
                }
            )
            history.append(f"{index + 1}:{action}:{content[:320]}")
    return examples


def _format_messages(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = _compact(message.get("role", "unknown"), limit=40) or "unknown"
        content = _compact(message.get("content", ""), limit=2400)
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _sft_examples(
    sft_root: Path,
    *,
    max_files: int,
    max_rows_per_file: int,
    objective: str,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    if not sft_root.exists():
        return examples
    paths = sorted(sft_root.rglob("*.jsonl"))
    if max_files > 0:
        paths = paths[-max_files:]
    for path in paths:
        rows_seen = 0
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError:
            continue
        with handle:
            for line_index, line in enumerate(handle):
                if max_rows_per_file > 0 and rows_seen >= max_rows_per_file:
                    break
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                messages = payload.get("messages", [])
                if not isinstance(messages, list) or len(messages) < 2:
                    continue
                assistant_index = -1
                for index in range(len(messages) - 1, -1, -1):
                    message = messages[index]
                    if isinstance(message, dict) and message.get("role") == "assistant":
                        assistant_index = index
                        break
                if assistant_index <= 0:
                    continue
                target = _compact(messages[assistant_index].get("content", ""), limit=5000)
                context = _format_messages(messages[:assistant_index])
                if not target or not context:
                    continue
                decoder_text = (
                    _chat_decision_text(
                        target,
                        thought="Continue the chat from supervised kernel dialogue.",
                        source_action="respond",
                    )
                    if objective == "chat"
                    else target
                )
                metadata = payload.get("metadata", {})
                metadata = metadata if isinstance(metadata, dict) else {}
                rel = path.relative_to(sft_root)
                source_id = f"{rel}:{line_index}"
                examples.append(
                    {
                        "source_type": "qwen_adapter_sft",
                        "source_id": source_id,
                        "encoder_text": (
                            "AgentKernel Lite chat supervised trace.\n"
                            "Use the observed dialogue and context contract to generate the next browser-chat response.\n"
                            f"{context}"
                        ),
                        "decoder_text": decoder_text,
                        "action": "respond",
                        "source_action": "respond",
                        "extension_capability": "",
                        "benchmark_family": _compact(metadata.get("benchmark_family", ""), limit=120),
                        "difficulty": _compact(payload.get("difficulty", metadata.get("difficulty", "")), limit=120),
                        "weight": float(payload.get("weight", 1.0) or 1.0),
                    }
                )
                rows_seen += 1
    return examples


def _doc_examples(docs_root: Path, *, max_files: int, objective: str) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    paths = sorted(docs_root.glob("*.md"))
    if max_files > 0:
        paths = paths[:max_files]
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        chunks = [chunk.strip() for chunk in text.split("\n## ") if chunk.strip()]
        for index, chunk in enumerate(chunks[:8]):
            target = _compact(chunk, limit=2200)
            if len(target) < 120:
                continue
            source_id = f"{path.name}:{index}"
            decoder_text = (
                _chat_decision_text(
                    target,
                    thought="Explain AgentKernel implementation notes in chat.",
                    source_action="respond",
                )
                if objective == "chat"
                else target
            )
            examples.append(
                {
                    "source_type": "kernel_doc",
                    "source_id": source_id,
                    "encoder_text": (
                        "Answer from AgentKernel implementation notes.\n"
                        f"Document: {path.name}\n"
                        "Task: produce a concise implementation-grounded chat answer."
                    ),
                    "decoder_text": decoder_text,
                    "action": "respond",
                    "source_action": "respond",
                    "extension_capability": "",
                    "benchmark_family": "agentkernel_lite",
                    "difficulty": "documentation",
                    "weight": 0.4,
                }
            )
    return examples


def _paper_rows(paper_chunk_root: Path, *, max_examples: int, max_files: int) -> list[dict[str, Any]]:
    if max_examples <= 0 or not paper_chunk_root.exists():
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("paper-grounded examples require pyarrow") from exc
    paths = sorted(paper_chunk_root.glob("*.parquet"))
    if max_files > 0:
        paths = paths[:max_files]
    columns = ["text", "paper_id", "title", "categories", "year", "chunk_index"]
    rows: list[dict[str, Any]] = []
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        read_columns = [column for column in columns if column in available]
        if "text" not in read_columns:
            continue
        for batch in parquet_file.iter_batches(batch_size=512, columns=read_columns):
            for row in batch.to_pylist():
                text = _compact(row.get("text", ""), limit=2200)
                if len(text) < 240:
                    continue
                rows.append(
                    {
                        "text": text,
                        "paper_id": _compact(row.get("paper_id", ""), limit=80),
                        "title": _compact(row.get("title", ""), limit=220),
                        "categories": _compact(row.get("categories", ""), limit=120),
                        "year": row.get("year", ""),
                        "chunk_index": row.get("chunk_index", ""),
                        "source_file": path.name,
                    }
                )
                if len(rows) >= max_examples:
                    return rows
    return rows


def _first_sentences(text: str, *, limit: int = 420) -> str:
    text = _compact(text, limit=1600)
    if not text:
        return ""
    pieces: list[str] = []
    current: list[str] = []
    for char in text:
        current.append(char)
        if char in ".!?":
            sentence = "".join(current).strip()
            if sentence:
                pieces.append(sentence)
            current = []
            if len(" ".join(pieces)) >= limit:
                break
    if not pieces and current:
        pieces.append("".join(current).strip())
    summary = " ".join(pieces).strip()
    return summary[:limit].rstrip()


def _repetition_score(text: str) -> float:
    words = [word.lower() for word in text.split() if word.strip()]
    if len(words) < 12:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    if not bigrams:
        return 0.0
    return 1.0 - (len(set(bigrams)) / len(bigrams))


def _clean_research_summary(*, title: str, text: str, categories: str, limit: int = 280) -> str:
    cleaned = _compact(text, limit=1600)
    if cleaned.lower().startswith("title:"):
        marker = " abstract:"
        marker_index = cleaned.lower().find(marker)
        if marker_index >= 0:
            cleaned = cleaned[marker_index + len(marker):].strip()
    for marker in (" References ", " ACKNOWLEDG", " Appendix "):
        index = cleaned.find(marker)
        if index > 160:
            cleaned = cleaned[:index].strip()
    summary = _first_sentences(cleaned, limit=limit)
    noisy = (
        len(summary) < 80
        or _repetition_score(summary) > 0.28
        or summary.lower().count("in this paper") > 2
        or summary.count("=") > 4
    )
    if noisy:
        topic = title or "the retrieved paper"
        category_note = f" in {categories}" if categories else ""
        return (
            f"The retrieved excerpt discusses {topic}{category_note}. "
            "The answer should cite [1] and avoid claims beyond the available excerpt."
        )[:limit].rstrip()
    return summary[:limit].rstrip()


def _research_examples(
    paper_chunk_root: Path,
    *,
    max_examples: int,
    max_files: int,
    objective: str,
) -> list[dict[str, Any]]:
    rows = _paper_rows(paper_chunk_root, max_examples=max_examples, max_files=max_files)
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        title = row["title"] or "the retrieved paper"
        paper_id = row["paper_id"]
        categories = row["categories"]
        year = row["year"]
        summary = _clean_research_summary(title=title, text=row["text"], categories=categories, limit=280)
        if len(summary) < 120:
            continue
        mode = ("chat", "think", "deep_research")[index % 3]
        meta = " | ".join(str(item) for item in [paper_id, categories, year] if item not in ("", None))
        evidence = (
            f"[1] {title}\n"
            f"{meta}\n"
            f"{row['text']}"
        )
        if mode == "chat":
            user_request = f"What is the main point of {title}? Cite the retrieved evidence."
            content = f"Based on [1], {title} focuses on this point: {summary}"
            thought = "Answer directly from the retrieved paper chunk."
        elif mode == "think":
            user_request = f"Synthesize the retrieved evidence about {title} and name one uncertainty."
            content = (
                f"The evidence in [1] supports this synthesis: {summary} "
                "The main uncertainty is that only one retrieved chunk is available, so broader claims need more evidence."
            )
            thought = "Synthesize from the evidence while preserving uncertainty."
        else:
            user_request = f"Run a deep research pass over the retrieved evidence for {title}."
            content = (
                f"Evidence review: [1] states or supports the following: {summary} "
                "Synthesis: the answer should stay grounded in that evidence and avoid claims beyond the retrieved chunk. "
                "Gap: more chunks from the same paper or related papers would be needed for a complete review."
            )
            thought = "Review the evidence item before synthesizing."
        if paper_id:
            content = f"{content} Source: arXiv:{paper_id}."
        decoder_text = (
            _chat_decision_text(
                content,
                thought=thought,
                source_action="respond",
                retrieval_influenced=True,
            )
            if objective == "chat"
            else content
        )
        examples.append(
            {
                "source_type": "research_paper_grounded",
                "source_id": f"{row['source_file']}:{row['chunk_index']}:{index}",
                "encoder_text": (
                    "AgentKernel Lite research-grounded chat example.\n"
                    f"Mode: {mode}\n"
                    "Runtime: browser chat, no direct code execution, no file writes.\n"
                    f"User request: {user_request}\n"
                    "Retrieved evidence:\n"
                    f"{evidence}\n"
                    "Return a compact JSON chat decision with action=respond. Cite evidence numbers for supported claims."
                ),
                "decoder_text": decoder_text,
                "action": "respond",
                "source_action": "respond",
                "extension_capability": "",
                "benchmark_family": "research_library_chat",
                "difficulty": mode,
                "research_mode": mode,
                "paper_id": paper_id,
                "weight": 0.8,
            }
        )
    return examples


AK_USER = "<AK_USER>"
AK_CHAT = "<AK_CHAT>"
AK_THINK = "<AK_THINK>"
AK_DEEP_RESEARCH = "<AK_DEEP_RESEARCH>"
AK_LOOP = "<AK_LOOP>"
AK_PLAN = "<AK_PLAN>"
AK_STATE = "<AK_STATE>"
AK_CONTEXT = "<AK_CONTEXT>"
AK_ACTIVE_CONTEXT = "<AK_ACTIVE_CONTEXT>"
AK_EVIDENCE = "<AK_EVIDENCE>"
AK_CANDIDATE = "<AK_CANDIDATE>"
AK_SELECTED_PAPER = "<AK_SELECTED_PAPER>"
AK_CONTEXT_ID = "<AK_CONTEXT_ID>"
AK_TARGET_CONTEXT = "<AK_TARGET_CONTEXT>"
AK_QUERY_REWRITE = "<AK_QUERY_REWRITE>"
AK_RERANK = "<AK_RERANK>"
AK_GATHER_CONTEXT = "<AK_GATHER_CONTEXT>"
AK_RESPOND = "<AK_RESPOND>"
AK_USE_CONTEXT = "<AK_USE_CONTEXT>"
AK_NO_RETRIEVAL = "<AK_NO_RETRIEVAL>"
AK_RETRIEVE_NEW = "<AK_RETRIEVE_NEW>"
AK_SUFFICIENT = "<AK_SUFFICIENT>"
AK_INSUFFICIENT = "<AK_INSUFFICIENT>"
AK_ANSWER = "<AK_ANSWER>"
AK_CITE = "<AK_CITE>"
AK_RENDER = "<AK_RENDER>"
AK_JSON = "<AK_JSON>"
AK_PROFILE = "<AK_PROFILE>"
AK_SLOT = "<AK_SLOT>"
AK_SLOT_NAME = "<AK_SLOT_NAME>"
AK_SLOT_VALUE = "<AK_SLOT_VALUE>"
AK_PREFERENCE = "<AK_PREFERENCE>"
AK_GOAL = "<AK_GOAL>"
AK_DOMAIN = "<AK_DOMAIN>"
AK_TONE = "<AK_TONE>"
AK_CONSTRAINT = "<AK_CONSTRAINT>"
AK_PRIVACY = "<AK_PRIVACY>"
AK_REMEMBER = "<AK_REMEMBER>"
AK_UPDATE_SLOT = "<AK_UPDATE_SLOT>"
AK_DELETE_SLOT = "<AK_DELETE_SLOT>"
AK_EXTENSION = "<AK_EXTENSION>"
AK_CAPABILITY = "<AK_CAPABILITY>"
AK_APPROVAL = "<AK_APPROVAL>"
AK_SAVE_MEMORY = "<AK_SAVE_MEMORY>"
AK_ASK_USER = "<AK_ASK_USER>"
AK_SOURCE_TYPE = "<AK_SOURCE_TYPE>"
AK_EXTENSION_RESULT = "<AK_EXTENSION_RESULT>"
AK_INSTALLED = "<AK_INSTALLED>"
AK_STRUCTURED = "<AK_STRUCTURED>"
AK_ACTION_RESPOND = "<AK_ACTION_RESPOND>"
AK_ACTION_ASK_USER = "<AK_ACTION_ASK_USER>"
AK_ACTION_EXTENSION_REQUEST = "<AK_ACTION_EXTENSION_REQUEST>"
AK_ACTION_SAVE_MEMORY = "<AK_ACTION_SAVE_MEMORY>"
AK_CONTENT = "<AK_CONTENT>"
AK_CONTENT_END = "</AK_CONTENT>"
AK_TASK_TYPE = "<AK_TASK_TYPE>"
AK_INTENT = "<AK_INTENT>"
AK_FIELD = "<AK_FIELD>"
AK_FIELD_NAME = "<AK_FIELD_NAME>"
AK_FIELD_VALUE = "<AK_FIELD_VALUE>"
AK_FIELDS = "<AK_FIELDS>"
AK_FRESHNESS = "<AK_FRESHNESS>"
AK_END = "<AK_END>"
AK_WEB_SEARCH = "<AK_WEB_SEARCH>"
AK_WEB_RESULT = "<AK_WEB_RESULT>"
AK_SOURCE = "<AK_SOURCE>"
AK_MAX_SOURCES = "<AK_MAX_SOURCES>"
AGENTKERNEL_SPECIAL_TOKENS = [
    AK_USER,
    AK_CHAT,
    AK_THINK,
    AK_DEEP_RESEARCH,
    AK_LOOP,
    AK_PLAN,
    AK_STATE,
    AK_CONTEXT,
    AK_ACTIVE_CONTEXT,
    AK_EVIDENCE,
    AK_CANDIDATE,
    AK_SELECTED_PAPER,
    AK_CONTEXT_ID,
    AK_TARGET_CONTEXT,
    AK_QUERY_REWRITE,
    AK_RERANK,
    AK_GATHER_CONTEXT,
    AK_RESPOND,
    AK_USE_CONTEXT,
    AK_NO_RETRIEVAL,
    AK_RETRIEVE_NEW,
    AK_SUFFICIENT,
    AK_INSUFFICIENT,
    AK_ANSWER,
    AK_CITE,
    AK_RENDER,
    AK_JSON,
    AK_PROFILE,
    AK_SLOT,
    AK_SLOT_NAME,
    AK_SLOT_VALUE,
    AK_PREFERENCE,
    AK_GOAL,
    AK_DOMAIN,
    AK_TONE,
    AK_CONSTRAINT,
    AK_PRIVACY,
    AK_REMEMBER,
    AK_UPDATE_SLOT,
    AK_DELETE_SLOT,
    AK_EXTENSION,
    AK_CAPABILITY,
    AK_APPROVAL,
    AK_SAVE_MEMORY,
    AK_ASK_USER,
    AK_SOURCE_TYPE,
    AK_EXTENSION_RESULT,
    AK_INSTALLED,
    AK_STRUCTURED,
    AK_ACTION_RESPOND,
    AK_ACTION_ASK_USER,
    AK_ACTION_EXTENSION_REQUEST,
    AK_ACTION_SAVE_MEMORY,
    AK_CONTENT,
    AK_CONTENT_END,
    AK_TASK_TYPE,
    AK_INTENT,
    AK_FIELD,
    AK_FIELD_NAME,
    AK_FIELD_VALUE,
    AK_FIELDS,
    AK_FRESHNESS,
    AK_END,
    AK_WEB_SEARCH,
    AK_WEB_RESULT,
    AK_SOURCE,
    AK_MAX_SOURCES,
]


USER_SLOT_PROFILES = [
    {
        "name": "busy founder",
        "domain": "product planning",
        "goal": "turn rough ideas into short launch tasks",
        "tone": "direct and practical",
        "preference": "give the next three actions before background detail",
        "constraint": "keep responses under 120 words unless asked",
        "private": "do not reveal calendar details unless the user asks",
        "request": "Help me decide what to do next for the app this afternoon.",
        "answer": (
            "Next: define the smallest useful user setting, write one example flow, and test it on-device. "
            "For the app work, keep the scope to a single configurable behavior before expanding the model."
        ),
        "memory_request": "Remember that PocketPal should feel like a local personal agent, not a paper search app.",
        "memory_content": "PocketPal should feel like a local personal agent, not a paper search app.",
        "missing_slot_request": "Can you set up my preferred daily planning style?",
        "missing_slot_question": "What daily planning style should I use: checklist, time-blocked schedule, or short priority list?",
        "extension_id": "image_generation",
        "capability": "image.generate",
        "extension_request": "Make a small app icon concept for PocketPal.",
        "extension_args": {"prompt": "small friendly local assistant app icon concept for PocketPal"},
        "extension_result": "Generated an icon draft with a compact assistant mark and blue-green app background.",
        "context_type": "project_note",
        "context_text": "PocketPal should prioritize local-first privacy, short practical answers, and user-editable configuration slots.",
        "context_request": "Use my project notes and tell me what matters most for the next build.",
    },
    {
        "name": "student",
        "domain": "study planning",
        "goal": "prepare for exams without long study blocks",
        "tone": "calm and encouraging",
        "preference": "break work into 25 minute sessions",
        "constraint": "avoid dense academic language",
        "private": "never expose grades in a shared summary",
        "request": "Make a quick study plan for tonight.",
        "answer": (
            "Use three 25 minute sessions: review the hardest topic first, do practice questions second, "
            "then make a short mistake list. Stop after the third block and only extend if you still feel clear."
        ),
        "memory_request": "Remember I study better when the first task is the hardest one.",
        "memory_content": "The user studies better when the first task is the hardest one.",
        "missing_slot_request": "Can you tune my homework help style?",
        "missing_slot_question": "Should homework help be step-by-step, hint-first, or answer-checking only?",
        "extension_id": "translator",
        "capability": "translation.text",
        "extension_request": "Translate this note to Spanish: I will study after dinner.",
        "extension_args": {
            "text": "I will study after dinner.",
            "source_language": "English",
            "target_language": "Spanish",
            "modality": "text",
        },
        "extension_result": "Estudiaré después de cenar.",
        "context_type": "study_note",
        "context_text": "The biology exam covers cells, membranes, enzymes, and practice diagrams. The user struggles most with enzymes.",
        "context_request": "Use my study note and pick what I should review first.",
    },
    {
        "name": "caregiver",
        "domain": "household coordination",
        "goal": "remember routines and reduce repeated planning",
        "tone": "brief and warm",
        "preference": "use checklists for errands",
        "constraint": "flag anything that needs another person's confirmation",
        "private": "keep health notes local",
        "request": "What should I pack before leaving?",
        "answer": (
            "Pack keys, wallet, medication, water, and the errand list. Confirm any time-sensitive pickup before leaving. "
            "I will keep health-related notes local to this profile."
        ),
        "memory_request": "Remember that medication reminders are private and should stay local.",
        "memory_content": "Medication reminders are private and should stay local.",
        "missing_slot_request": "Set up how you help with errands.",
        "missing_slot_question": "Should errand help focus on packing lists, route order, or reminders?",
        "extension_id": "translator",
        "capability": "translation.audio",
        "extension_request": "Translate a short spoken errand note into English.",
        "extension_args": {
            "source_language": "Auto",
            "target_language": "English",
            "modality": "audio",
            "surface": "browser",
        },
        "extension_result": "Pick up the prescription before 5 PM.",
        "context_type": "household_note",
        "context_text": "Today includes a pharmacy pickup, school form drop-off, and a grocery stop. The pharmacy closes at 5 PM.",
        "context_request": "Use my household note and tell me the priority.",
    },
    {
        "name": "maker",
        "domain": "hardware projects",
        "goal": "finish small prototypes",
        "tone": "technical but concise",
        "preference": "list parts and risks separately",
        "constraint": "do not assume access to paid cloud tools",
        "private": "do not share supplier notes outside the device",
        "request": "Plan the next prototype step.",
        "answer": (
            "Parts: controller, battery, enclosure, and test leads. Next step: build the smallest powered loop and measure heat. "
            "Risk: battery fit and runtime. Use local tools first."
        ),
        "memory_request": "Remember that I want risks separated from parts in prototype plans.",
        "memory_content": "For prototype plans, separate risks from parts.",
        "missing_slot_request": "Set up how you should help with prototypes.",
        "missing_slot_question": "Should prototype help optimize for speed, low cost, reliability, or learning?",
        "extension_id": "image_generation",
        "capability": "image.generate",
        "extension_request": "Generate a simple enclosure sketch prompt for this prototype.",
        "extension_args": {"prompt": "simple compact electronics enclosure sketch with battery compartment and test ports"},
        "extension_result": "Generated a simple enclosure sketch with labeled battery and test-port areas.",
        "context_type": "project_file",
        "context_text": "The prototype overheated after 12 minutes. Battery fit is tight and the enclosure lacks airflow.",
        "context_request": "Use my project file and tell me the next hardware test.",
    },
]


POCKETPAL_FOCUS_AREAS = [
    "today's priorities",
    "the next app build",
    "a weekly routine",
    "a short decision",
    "a saved project",
    "a private note",
    "a repeat workflow",
    "a deadline",
    "a handoff message",
    "a local-only summary",
    "a troubleshooting pass",
    "a follow-up plan",
]

POCKETPAL_HORIZONS = [
    "this afternoon",
    "tonight",
    "tomorrow morning",
    "this week",
    "before the next meeting",
    "before leaving home",
    "during a 25 minute block",
    "after the current task",
]

POCKETPAL_TONES = [
    "direct and practical",
    "calm and concise",
    "technical but concise",
    "brief and warm",
    "checklist-first",
    "plainspoken and specific",
]

POCKETPAL_RESPONSE_PREFERENCES = [
    "give the next three actions before background detail",
    "start with the highest-impact step",
    "use a checklist when there are multiple tasks",
    "separate risks from actions",
    "ask one question when a key detail is missing",
    "keep the answer short and locally grounded",
    "state what can be done without an extension first",
    "show the decision and then the next step",
]

POCKETPAL_CONSTRAINTS = [
    "keep responses under 120 words unless asked",
    "do not assume cloud access",
    "do not expose private notes in shareable drafts",
    "do not use an extension without approval",
    "avoid claims not supported by local context",
    "prefer reversible changes",
    "flag anything that needs another person's confirmation",
    "keep the first answer actionable",
]

POCKETPAL_PRIVATE_NOTES = [
    "keep health notes local",
    "keep calendar details local unless the user asks",
    "do not include grades in shared summaries",
    "do not share supplier notes outside the device",
    "hide exact addresses from shareable drafts",
    "keep contact details out of generated examples",
    "do not reveal private routines to extensions unless approved",
    "summarize sensitive details instead of quoting them",
]

POCKETPAL_CONTEXT_TYPES = [
    "project_note",
    "study_note",
    "household_note",
    "project_file",
    "calendar_summary",
    "saved_webpage",
    "personal_note",
    "extension_receipt",
]

POCKETPAL_MEMORY_KINDS = [
    "preference",
    "routine",
    "project_fact",
    "privacy_rule",
    "workflow_style",
    "deadline_hint",
]

POCKETPAL_MISSING_SLOTS = [
    ("workflow_style", "Should I use checklist, time-blocked schedule, or short priority list for this?"),
    ("response_length", "Should answers be short, normal, or detailed for this workflow?"),
    ("privacy_level", "Should this be private, shareable, or extension-visible after approval?"),
    ("extension_policy", "Should I ask every time before using this extension, or only for sensitive inputs?"),
    ("priority_rule", "Should I prioritize speed, quality, cost, reliability, or learning?"),
    ("output_format", "Should I format this as bullets, a message draft, or a step-by-step plan?"),
]

POCKETPAL_MISSING_SLOT_REQUESTS = {
    "workflow_style": [
        "Configure how you should plan this workflow.",
        "Set up the planning style for this kind of work.",
        "Decide how you should organize my next task.",
    ],
    "response_length": [
        "Configure how detailed my daily plan should be.",
        "Set how much detail you should use for this workflow.",
        "Choose how long your answers should be when I ask for planning help.",
    ],
    "privacy_level": [
        "Configure how private this workflow should be.",
        "Set whether this kind of note is private or shareable.",
        "Decide what privacy level to use before helping with this.",
    ],
    "extension_policy": [
        "Configure when you should use installed extensions.",
        "Set whether extensions need approval for this workflow.",
        "Decide when you should ask before using an extension.",
    ],
    "priority_rule": [
        "Configure what you should optimize for first.",
        "Set the priority rule for this workflow.",
        "Decide whether speed, quality, cost, reliability, or learning matters most.",
    ],
    "output_format": [
        "Configure the output format for this workflow.",
        "Set how you should format plans for this kind of request.",
        "Choose whether replies should be bullets, drafts, or step-by-step plans.",
    ],
}

POCKETPAL_EXTENSIONS = [
    (
        "image_generation",
        "image.generate",
        "Make a visual draft for {focus}.",
        {"prompt": "local personal assistant visual draft for {focus}"},
        "Generated a visual draft for {focus}.",
    ),
    (
        "translator",
        "translation.text",
        "Translate this note to Spanish: I will handle {focus} {horizon}.",
        {
            "text": "I will handle {focus} {horizon}.",
            "source_language": "English",
            "target_language": "Spanish",
            "modality": "text",
        },
        "Translated the note about {focus} into Spanish.",
    ),
    (
        "translator",
        "translation.audio",
        "Translate a short spoken note about {focus}.",
        {
            "source_language": "Auto",
            "target_language": "English",
            "modality": "audio",
            "surface": "browser",
        },
        "Translated the spoken note about {focus}.",
    ),
    (
        "calendar",
        "calendar.create_event",
        "Put {focus} on the calendar {horizon} at 3pm.",
        {
            "title": "{focus}",
            "when": "{horizon} at 3pm",
            "calendar": "default",
        },
        "Created a calendar event for {focus} {horizon} at 3pm.",
    ),
    (
        "reminders",
        "reminder.create",
        "Remind me about {focus} {horizon}.",
        {
            "title": "{focus}",
            "when": "{horizon}",
            "priority": "normal",
        },
        "Created a reminder for {focus} {horizon}.",
    ),
    (
        "files",
        "file.search",
        "Find my local notes about {focus}.",
        {
            "query": "{focus}",
            "scope": "local_user_files",
        },
        "Found local notes related to {focus}.",
    ),
]


POCKETPAL_WEB_SEARCH_SCENARIOS = [
    {
        "query": "California wildfire insurance market 2025",
        "request": "Search the web for California wildfire insurance market 2025 and summarize the main points.",
        "results": [
            {
                "source": "California FAIR Plan",
                "title": "California FAIR Plan",
                "snippet": "The FAIR Plan is the state-backed fire insurance option when homeowners cannot obtain normal coverage.",
                "url": "https://example.test/fair-plan",
            },
            {
                "source": "Insurance market report",
                "title": "California insurance availability",
                "snippet": "Insurers have reduced exposure in high wildfire-risk areas, increasing pressure on last-resort coverage.",
                "url": "https://example.test/insurance-market",
            },
        ],
        "answer": (
            "The search points to two useful facts: California homeowners in high wildfire-risk areas may rely on the FAIR Plan, "
            "and private insurers have been reducing exposure in those areas. Treat this as a market-availability issue and verify details before making decisions."
        ),
    },
    {
        "query": "Rust async runtime comparison tokio smol",
        "request": "Search online for Rust async runtime comparison tokio smol.",
        "results": [
            {
                "source": "Stack Overflow",
                "title": "Choosing a Rust async runtime",
                "snippet": "Tokio is widely used for production networking; smaller runtimes can be useful for lightweight applications.",
                "url": "https://example.test/rust-runtime",
            },
            {
                "source": "GitHub",
                "title": "smol-rs/smol",
                "snippet": "Smol is a small and fast async runtime for Rust with a compact API surface.",
                "url": "https://example.test/smol",
            },
        ],
        "answer": (
            "The useful distinction is scope: Tokio is the safer default for production networking because of ecosystem support, "
            "while smol is attractive for smaller async programs that value a compact runtime."
        ),
    },
    {
        "query": "solid state battery Toyota production timeline",
        "request": "Do a web search on Toyota solid state battery production timeline.",
        "results": [
            {
                "source": "Automotive report",
                "title": "Toyota solid-state battery plans",
                "snippet": "Toyota has discussed commercialization plans for solid-state batteries, but timelines depend on manufacturing readiness.",
                "url": "https://example.test/toyota-solid-state",
            },
            {
                "source": "Battery research index",
                "title": "Solid-state battery manufacturing",
                "snippet": "Scaling solid-state batteries requires solving cost, durability, and production-yield problems.",
                "url": "https://example.test/solid-state-scale",
            },
        ],
        "answer": (
            "The search suggests Toyota's solid-state battery timeline should be treated as conditional, not guaranteed. "
            "The core blockers are manufacturing scale, cost, durability, and yield."
        ),
    },
]


def _format_template(value: object, **kwargs: str) -> object:
    if isinstance(value, str):
        return value.format(**kwargs)
    if isinstance(value, dict):
        return {key: _format_template(item, **kwargs) for key, item in value.items()}
    if isinstance(value, list):
        return [_format_template(item, **kwargs) for item in value]
    return value


def _profile_instance(base: dict[str, str], index: int) -> dict[str, Any]:
    focus = POCKETPAL_FOCUS_AREAS[index % len(POCKETPAL_FOCUS_AREAS)]
    horizon = POCKETPAL_HORIZONS[(index // len(POCKETPAL_FOCUS_AREAS)) % len(POCKETPAL_HORIZONS)]
    tone = POCKETPAL_TONES[(index // 3) % len(POCKETPAL_TONES)]
    preference = POCKETPAL_RESPONSE_PREFERENCES[(index // 5) % len(POCKETPAL_RESPONSE_PREFERENCES)]
    constraint = POCKETPAL_CONSTRAINTS[(index // 7) % len(POCKETPAL_CONSTRAINTS)]
    private_note = POCKETPAL_PRIVATE_NOTES[(index // 11) % len(POCKETPAL_PRIVATE_NOTES)]
    context_type = POCKETPAL_CONTEXT_TYPES[(index // 13) % len(POCKETPAL_CONTEXT_TYPES)]
    memory_kind = POCKETPAL_MEMORY_KINDS[(index // 17) % len(POCKETPAL_MEMORY_KINDS)]
    missing_slot, missing_question = POCKETPAL_MISSING_SLOTS[(index // 19) % len(POCKETPAL_MISSING_SLOTS)]
    missing_requests = POCKETPAL_MISSING_SLOT_REQUESTS.get(missing_slot, [])
    missing_request = (
        missing_requests[(index // 29) % len(missing_requests)]
        if missing_requests
        else f"Set up how you should help with {focus}."
    )
    extension_id, capability, extension_request, extension_args, extension_result = POCKETPAL_EXTENSIONS[
        (index // 23) % len(POCKETPAL_EXTENSIONS)
    ]
    profile: dict[str, Any] = dict(base)
    profile.update(
        {
            "domain": f"{base['domain']} for {focus}",
            "goal": f"{base['goal']} {horizon}",
            "tone": tone,
            "preference": preference,
            "constraint": constraint,
            "private": private_note,
            "request": f"Help me with {focus} {horizon}.",
            "answer": (
                f"For {focus} {horizon}, start with the highest-impact step, then check the constraint: {constraint}. "
                f"Use a {tone} style and keep the work tied to {base['domain']}."
            ),
            "memory_request": f"Remember that for {focus} I prefer: {preference}.",
            "memory_content": f"For {focus}, the user prefers to {preference}.",
            "memory_kind": memory_kind,
            "missing_slot_request": missing_request,
            "missing_slot": missing_slot,
            "missing_slot_question": missing_question,
            "extension_id": extension_id,
            "capability": capability,
            "extension_request": str(_format_template(extension_request, focus=focus, horizon=horizon)),
            "extension_args": _format_template(extension_args, focus=focus, horizon=horizon),
            "extension_result": str(_format_template(extension_result, focus=focus, horizon=horizon)),
            "context_type": context_type,
            "context_text": (
                f"Local {context_type} for {focus}: the user wants {base['goal']} {horizon}. "
                f"Constraint: {constraint}. Preference: {preference}."
            ),
            "context_request": f"Use my local context and tell me what matters for {focus}.",
        }
    )
    return profile


def _slot_profile_block(profile: dict[str, str]) -> str:
    return "\n".join(
        [
            f"{AK_PROFILE} PocketPal user configuration.",
            f"{AK_SLOT} {AK_SLOT_NAME}=domain {AK_SLOT_VALUE}={profile['domain']} {AK_DOMAIN}",
            f"{AK_SLOT} {AK_SLOT_NAME}=goal {AK_SLOT_VALUE}={profile['goal']} {AK_GOAL}",
            f"{AK_SLOT} {AK_SLOT_NAME}=tone {AK_SLOT_VALUE}={profile['tone']} {AK_TONE}",
            f"{AK_SLOT} {AK_SLOT_NAME}=preference {AK_SLOT_VALUE}={profile['preference']} {AK_PREFERENCE}",
            f"{AK_SLOT} {AK_SLOT_NAME}=constraint {AK_SLOT_VALUE}={profile['constraint']} {AK_CONSTRAINT}",
            f"{AK_SLOT} {AK_SLOT_NAME}=private_note {AK_SLOT_VALUE}={profile['private']} {AK_PRIVACY}",
        ]
    )


def _user_slot_examples(*, max_examples: int, objective: str, seed: int) -> list[dict[str, Any]]:
    if max_examples <= 0:
        return []
    rng = random.Random(int(seed))
    examples: list[dict[str, Any]] = []
    profiles = list(USER_SLOT_PROFILES)
    index = 0
    while len(examples) < max_examples:
        profile = _profile_instance(profiles[index % len(profiles)], index)
        profile_block = _slot_profile_block(profile)
        web_search = POCKETPAL_WEB_SEARCH_SCENARIOS[index % len(POCKETPAL_WEB_SEARCH_SCENARIOS)]
        web_result_lines = "\n".join(
            (
                f"{AK_WEB_RESULT} [W{result_index}] {AK_SOURCE}={result['source']} "
                f"title={result['title']} url={result['url']} snippet={result['snippet']}"
            )
            for result_index, result in enumerate(web_search["results"], start=1)
        )
        variants = [
            {
                "task_type": "slot_conditioned_runtime_minimal",
                "source_id": f"{profile['name']}:runtime_respond:{index}",
                "action": "respond",
                "encoder_text": (
                    f"{AK_CHAT} {AK_RESPOND} {AK_PROFILE} {AK_SLOT}\n"
                    f"{profile_block}\n"
                    f"{AK_USER} {profile['request']}"
                ),
                "content": profile["answer"],
                "metadata": {
                    "task_type": "slot_conditioned_response",
                    "runtime_prompt_shape": "minimal_profile_slots",
                    "used_slots": ["domain", "goal", "tone", "preference", "constraint", "private_note"],
                    "profile_name": profile["name"],
                },
                "weight": 1.6,
            },
            {
                "task_type": "slot_conditioned_response",
                "source_id": f"{profile['name']}:respond:{index}",
                "encoder_text": (
                    f"{AK_CHAT} {AK_RESPOND} {AK_PROFILE} {AK_SLOT}\n"
                    "PocketPal slot-conditioned assistant example.\n"
                    "Use the stable model behavior plus the user-filled slots. Follow privacy and constraint slots. "
                    "Do not mention slot mechanics unless the user asks.\n"
                    f"{profile_block}\n"
                    f"{AK_USER} {profile['request']}\n"
                    "Return a structured chat decision with action=respond."
                ),
                "content": profile["answer"],
                "metadata": {
                    "task_type": "slot_conditioned_response",
                    "used_slots": ["domain", "goal", "tone", "preference", "constraint", "private_note"],
                    "profile_name": profile["name"],
                },
                "weight": 1.3,
            },
            {
                "task_type": "slot_update",
                "source_id": f"{profile['name']}:update:{index}",
                "encoder_text": (
                    f"{AK_CHAT} {AK_RESPOND} {AK_PROFILE} {AK_SLOT} {AK_UPDATE_SLOT} {AK_REMEMBER}\n"
                    "PocketPal user-slot update example.\n"
                    "The user is changing one configurable slot. Acknowledge the change and encode the update in metadata. "
                    "Keep the response short.\n"
                    f"{profile_block}\n"
                    f"{AK_USER} Remember that I prefer {profile['tone']} answers when I am busy.\n"
                    "Return a structured chat decision with action=respond and proposal_metadata.slot_updates."
                ),
                "content": f"Got it. I will use {profile['tone']} answers when you are busy.",
                "metadata": {
                    "task_type": "slot_update",
                    "slot_updates": {"tone": profile["tone"], "busy_mode": True},
                    "profile_name": profile["name"],
                },
                "weight": 1.1,
            },
            {
                "task_type": "slot_privacy_boundary",
                "source_id": f"{profile['name']}:privacy:{index}",
                "encoder_text": (
                    f"{AK_CHAT} {AK_RESPOND} {AK_PROFILE} {AK_PRIVACY} {AK_CONSTRAINT}\n"
                    "PocketPal privacy-boundary example.\n"
                    "Use local profile slots without exposing private slot values to another person. "
                    "Offer a safe summary instead.\n"
                    f"{profile_block}\n"
                    f"{AK_USER} Write a message I can send someone about my plan, but leave out private details.\n"
                    "Return a structured chat decision with action=respond."
                ),
                "content": (
                    "Here is a shareable version: I have a short plan and will handle the next step today. "
                    "I left out private profile details."
                ),
                "metadata": {
                    "task_type": "slot_privacy_boundary",
                    "private_slots_excluded": ["private_note"],
                    "profile_name": profile["name"],
                },
                "weight": 1.2,
            },
            {
                "task_type": "memory_save",
                "source_id": f"{profile['name']}:memory:{index}",
                "action": "save_memory",
                "encoder_text": (
                    f"{AK_CHAT} {AK_SAVE_MEMORY} {AK_PROFILE} {AK_SLOT} {AK_REMEMBER}\n"
                    "PocketPal local memory example.\n"
                    "The user explicitly asks the assistant to remember a preference or fact. "
                    "Save it as local user memory, include a short user-facing acknowledgement, and do not use an extension.\n"
                    f"{profile_block}\n"
                    f"{AK_USER} {profile['memory_request']}\n"
                    "Return a structured decision with action=save_memory and proposal_metadata.memory."
                ),
                "content": "Saved locally.",
                "metadata": {
                    "task_type": "memory_save",
                    "memory": {
                        "scope": "local_user_profile",
                        "kind": profile["memory_kind"],
                        "text": _compact(profile["memory_content"], limit=180),
                    },
                    "profile_name": profile["name"],
                },
                "weight": 3.0,
            },
            {
                "task_type": "memory_save",
                "source_id": f"{profile['name']}:memory_runtime:{index}",
                "action": "save_memory",
                "encoder_text": (
                    f"{AK_CHAT} {AK_SAVE_MEMORY} {AK_PROFILE} {AK_REMEMBER}\n"
                    "PocketPal runtime memory request.\n"
                    f"{AK_SLOT} <AK_SLOT_NAME>=domain <AK_SLOT_VALUE>={profile['domain']} {AK_DOMAIN}\n"
                    f"{AK_SLOT} <AK_SLOT_NAME>=tone <AK_SLOT_VALUE>={profile['tone']} {AK_TONE}\n"
                    f"{AK_USER} {profile['memory_request']}\n"
                    "Return JSON with action=save_memory."
                ),
                "content": "Saved locally.",
                "metadata": {
                    "task_type": "memory_save",
                    "memory": {
                        "scope": "local_user_profile",
                        "kind": profile["memory_kind"],
                        "text": _compact(profile["memory_content"], limit=160),
                    },
                    "runtime_prompt_shape": "minimal_memory_request",
                },
                "weight": 3.5,
            },
            {
                "task_type": "ask_missing_slot",
                "source_id": f"{profile['name']}:ask_slot:{index}",
                "action": "ask_user",
                "encoder_text": (
                    f"{AK_CHAT} {AK_ASK_USER} {AK_PROFILE} {AK_SLOT}\n"
                    "PocketPal missing-slot example.\n"
                    "The user asks to configure behavior, but the requested slot value is not present. "
                    "Ask one concise question instead of guessing.\n"
                    f"{profile_block}\n"
                    f"{AK_USER} {profile['missing_slot_request']}\n"
                    "Return a structured decision with action=ask_user."
                ),
                "content": profile["missing_slot_question"],
                "metadata": {
                    "task_type": "ask_missing_slot",
                    "missing_slot": profile["missing_slot"],
                    "profile_name": profile["name"],
                },
                "weight": 3.0,
            },
            {
                "task_type": "ask_missing_slot",
                "source_id": f"{profile['name']}:ask_slot_runtime:{index}",
                "action": "ask_user",
                "encoder_text": (
                    f"{AK_CHAT} {AK_ASK_USER} {AK_PROFILE} {AK_SLOT}\n"
                    "PocketPal runtime missing-slot request.\n"
                    f"{AK_SLOT} <AK_SLOT_NAME>=domain <AK_SLOT_VALUE>={profile['domain']} {AK_DOMAIN}\n"
                    f"{AK_USER} {profile['missing_slot_request']}\n"
                    "Return JSON with action=ask_user."
                ),
                "content": profile["missing_slot_question"],
                "metadata": {
                    "task_type": "ask_missing_slot",
                    "missing_slot": profile["missing_slot"],
                    "runtime_prompt_shape": "minimal_missing_slot",
                },
                "weight": 3.5,
            },
            {
                "task_type": "extension_request",
                "source_id": f"{profile['name']}:extension:{index}",
                "action": "extension_request",
                "encoder_text": (
                    f"{AK_CHAT} {AK_EXTENSION} {AK_CAPABILITY} {AK_APPROVAL} {AK_INSTALLED}\n"
                    "PocketPal installed-extension routing example.\n"
                    "The requested work needs an installed extension. Emit an extension_request with approval required; "
                    "do not pretend the extension already ran.\n"
                    f"{profile_block}\n"
                    f"{AK_EXTENSION} installed id={profile['extension_id']} {AK_CAPABILITY} {profile['capability']} approval_policy=always_ask\n"
                    f"{AK_USER} {profile['extension_request']}\n"
                    "Return a structured decision with action=extension_request and proposal_metadata.extension_id/capability."
                ),
                "content": f"Requesting approval to use {profile['extension_id']}:{profile['capability']}.",
                "metadata": {
                    "task_type": "extension_request",
                    "extension_id": profile["extension_id"],
                    "capability": profile["capability"],
                    "requires_user_approval": True,
                    "profile_name": profile["name"],
                },
                "weight": 3.0,
            },
            {
                "task_type": "extension_request",
                "source_id": f"{profile['name']}:extension_runtime:{index}",
                "action": "extension_request",
                "encoder_text": (
                    f"{AK_CHAT} {AK_EXTENSION} {AK_CAPABILITY} {AK_APPROVAL} {AK_INSTALLED}\n"
                    "PocketPal runtime extension request.\n"
                    f"{AK_EXTENSION} installed id={profile['extension_id']} {AK_CAPABILITY} {profile['capability']} approval_policy=always_ask\n"
                    f"{AK_USER} {profile['extension_request']}\n"
                    "Return JSON with action=extension_request."
                ),
                "content": f"Requesting approval to use {profile['extension_id']}:{profile['capability']}.",
                "metadata": {
                    "task_type": "extension_request",
                    "extension_id": profile["extension_id"],
                    "capability": profile["capability"],
                    "requires_user_approval": True,
                    "runtime_prompt_shape": "minimal_extension_request",
                },
                "weight": 3.5,
            },
            {
                "task_type": "extension_result_summary",
                "source_id": f"{profile['name']}:extension_result:{index}",
                "action": "respond",
                "encoder_text": (
                    f"{AK_CHAT} {AK_RESPOND} {AK_EXTENSION_RESULT} {AK_EXTENSION}\n"
                    "PocketPal extension-result example.\n"
                    "The extension has already run with user approval. Summarize the result and suggest the next user action.\n"
                    f"{profile_block}\n"
                    f"{AK_EXTENSION_RESULT} id={profile['extension_id']} capability={profile['capability']} status=approved_executed\n"
                    f"{profile['extension_result']}\n"
                    f"{AK_USER} What happened?\n"
                    "Return a structured decision with action=respond."
                ),
                "content": f"{profile['extension_result']} Next, review it and tell me what you want changed.",
                "metadata": {
                    "task_type": "extension_result_summary",
                    "extension_id": profile["extension_id"],
                    "capability": profile["capability"],
                    "profile_name": profile["name"],
                },
                "weight": 1.0,
            },
            {
                "task_type": "web_search_request",
                "source_id": f"{profile['name']}:web_search_request:{index}",
                "action": "extension_request",
                "encoder_text": (
                    f"{AK_CHAT} {AK_WEB_SEARCH} {AK_EXTENSION} {AK_CAPABILITY} {AK_APPROVAL} {AK_MAX_SOURCES}\n"
                    "PocketPal web-search routing example.\n"
                    "The user asks for current or web-backed information. Request the web_search extension with a bounded source count. "
                    "Do not invent search results before the extension runs.\n"
                    f"{profile_block}\n"
                    f"{AK_EXTENSION} installed id=web_search {AK_CAPABILITY} web.search approval_policy=always_ask\n"
                    f"{AK_MAX_SOURCES}=5\n"
                    f"{AK_USER} {web_search['request']}\n"
                    "Return a structured decision with action=extension_request and proposal_metadata.extension_id/capability/query/max_sources."
                ),
                "content": "Requesting approval to search the web.",
                "metadata": {
                    "task_type": "extension_request",
                    "extension_id": "web_search",
                    "capability": "web.search",
                    "query": web_search["query"],
                    "max_sources": 5,
                    "requires_user_approval": True,
                    "profile_name": profile["name"],
                },
                "weight": 3.8,
            },
            {
                "task_type": "web_search_result_synthesis",
                "source_id": f"{profile['name']}:web_search_synthesis:{index}",
                "action": "respond",
                "encoder_text": (
                    f"{AK_CHAT} {AK_RESPOND} {AK_WEB_SEARCH} {AK_WEB_RESULT} {AK_SOURCE} {AK_MAX_SOURCES}\n"
                    "PocketPal web-search synthesis example.\n"
                    "The web_search extension has returned structured sources. Answer only from the provided results, name the source basis, "
                    "and avoid unsupported claims.\n"
                    f"{profile_block}\n"
                    f"{AK_WEB_SEARCH} query={web_search['query']} {AK_MAX_SOURCES}=5\n"
                    f"{web_result_lines}\n"
                    f"{AK_USER} What did you find?\n"
                    "Return a structured decision with action=respond."
                ),
                "content": web_search["answer"],
                "metadata": {
                    "task_type": "web_search_result_synthesis",
                    "query": web_search["query"],
                    "used_context_ids": ["W1", "W2"],
                    "source_count": len(web_search["results"]),
                },
                "weight": 2.2,
            },
            {
                "task_type": "web_search_weak_results",
                "source_id": f"{profile['name']}:web_search_weak:{index}",
                "action": "respond",
                "encoder_text": (
                    f"{AK_CHAT} {AK_RESPOND} {AK_WEB_SEARCH} {AK_INSUFFICIENT}\n"
                    "PocketPal weak web-search result example.\n"
                    "The structured search providers returned no sufficiently relevant results. Do not answer from weak matches; give the user the fallback search link.\n"
                    f"{profile_block}\n"
                    f"{AK_WEB_SEARCH} query=Rust async runtime comparison tokio smol {AK_MAX_SOURCES}=5\n"
                    f"{AK_WEB_RESULT} none coverage=insufficient\n"
                    f"{AK_USER} What did you find?\n"
                    "Return a structured decision with action=respond."
                ),
                "content": (
                    "I could not retrieve enough relevant structured results for that query. "
                    "Use the browser search link and I can summarize the pages you choose to open."
                ),
                "metadata": {
                    "task_type": "web_search_weak_results",
                    "query": "Rust async runtime comparison tokio smol",
                    "source_count": 0,
                    "fallback": "browser_search_link",
                },
                "weight": 2.0,
            },
            {
                "task_type": "user_context_answer",
                "source_id": f"{profile['name']}:context:{index}",
                "action": "respond",
                "encoder_text": (
                    f"{AK_CHAT} {AK_RESPOND} {AK_PROFILE} {AK_CONTEXT} {AK_SOURCE_TYPE}\n"
                    "PocketPal user-context answer example.\n"
                    "Answer from the provided local user context. Do not call it research evidence unless the source type is a paper. "
                    "Respect privacy and profile constraints.\n"
                    f"{profile_block}\n"
                    f"{AK_CONTEXT} {AK_SOURCE_TYPE}={profile['context_type']}\n"
                    f"{AK_EVIDENCE} [U1]: {profile['context_text']}\n"
                    f"{AK_USER} {profile['context_request']}\n"
                    "Return a structured decision with action=respond."
                ),
                "content": f"From [U1], prioritize {profile['context_text'].split(':', 1)[0].lower()}.",
                "metadata": {
                    "task_type": "user_context_answer",
                    "used_context_ids": ["U1"],
                },
                "weight": 1.2,
            },
        ]
        rng.shuffle(variants)
        for variant in variants:
            if len(examples) >= max_examples:
                break
            content = str(variant["content"])
            decoder_text = (
                _structured_decision_text(
                    action=str(variant.get("action", "respond")),
                    content=content,
                    metadata=dict(variant["metadata"]),
                )
                if objective == "chat"
                else content
            )
            examples.append(
                {
                    "source_type": "pocketpal_user_slots",
                    "source_id": variant["source_id"],
                    "task_type": variant["task_type"],
                    "encoder_text": variant["encoder_text"],
                    "decoder_text": decoder_text,
                    "action": str(variant.get("action", "respond")),
                    "source_action": str(variant.get("action", "respond")),
                    "extension_capability": "",
                    "benchmark_family": "pocketpal_user_config",
                    "difficulty": variant["task_type"],
                    "weight": float(variant["weight"]),
                }
            )
        index += 1
    return examples


def _term_tokens(*values: object, limit: int = 8) -> list[str]:
    text = " ".join(str(value or "") for value in values)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower())
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        token = token.strip("-")
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= limit:
            break
    return out


def _paper_text_row(row: dict[str, Any], *, source_file: str = "", row_index: int = 0) -> dict[str, Any] | None:
    title = _compact(row.get("title", ""), limit=240)
    abstract = _compact(row.get("abstract", ""), limit=1400)
    text = _compact(row.get("text", row.get("full_text", "")), limit=2400)
    body = text or abstract
    if not title or len(body) < 160:
        return None
    return {
        "paper_id": _compact(
            row.get("paper_id", row.get("canonical_paper_id", row.get("arxiv_id", row.get("id", "")))),
            limit=90,
        ),
        "canonical_paper_id": _compact(row.get("canonical_paper_id", ""), limit=90),
        "title": title,
        "abstract": abstract,
        "text": body,
        "categories": _compact(row.get("categories", row.get("primary_category", "")), limit=160),
        "year": row.get("year", row.get("published_year", row.get("update_date", ""))),
        "source_file": source_file,
        "row_index": row_index,
    }


def _paper_text_rows_from_jsonl(path: Path, *, max_examples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_examples > 0 and len(rows) >= max_examples:
                break
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            row = _paper_text_row(payload, source_file=path.name, row_index=index)
            if row:
                rows.append(row)
    return rows


def _paper_text_rows_from_parquet(path: Path, *, max_examples: int, max_files: int) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("paper text parquet examples require pyarrow") from exc
    paths = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if max_files > 0:
        paths = paths[:max_files]
    rows: list[dict[str, Any]] = []
    preferred_columns = [
        "paper_id",
        "canonical_paper_id",
        "arxiv_id",
        "id",
        "title",
        "abstract",
        "text",
        "full_text",
        "categories",
        "primary_category",
        "year",
        "published_year",
        "update_date",
    ]
    for file_path in paths:
        parquet_file = pq.ParquetFile(file_path)
        available = set(parquet_file.schema_arrow.names)
        read_columns = [column for column in preferred_columns if column in available]
        if "title" not in read_columns or not ({"text", "full_text", "abstract"} & set(read_columns)):
            continue
        for batch in parquet_file.iter_batches(batch_size=256, columns=read_columns):
            for row_index, payload in enumerate(batch.to_pylist()):
                if max_examples > 0 and len(rows) >= max_examples:
                    return rows
                row = _paper_text_row(payload, source_file=file_path.name, row_index=row_index)
                if row:
                    rows.append(row)
    return rows


def _paper_text_rows_from_path(path: Path, *, max_examples: int, max_files: int) -> list[dict[str, Any]]:
    if not path.exists() or max_examples <= 0:
        return []
    if path.is_dir() or path.suffix.lower() == ".parquet":
        return _paper_text_rows_from_parquet(path, max_examples=max_examples, max_files=max_files)
    if path.suffix.lower() in {".jsonl", ".json"}:
        if path.suffix.lower() == ".jsonl":
            return _paper_text_rows_from_jsonl(path, max_examples=max_examples)
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_rows = payload if isinstance(payload, list) else payload.get("rows", [])
        rows: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_rows):
            if max_examples > 0 and len(rows) >= max_examples:
                break
            if isinstance(raw, dict):
                row = _paper_text_row(raw, source_file=path.name, row_index=index)
                if row:
                    rows.append(row)
        return rows
    return []


def _paper_text_rows_from_dataset(
    dataset_name: str,
    *,
    split: str,
    max_examples: int,
    streaming: bool,
) -> list[dict[str, Any]]:
    if not dataset_name.strip() or max_examples <= 0:
        return []
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("loading Hugging Face paper text datasets requires the 'datasets' package") from exc
    dataset = load_dataset(dataset_name, split=split, streaming=bool(streaming))
    rows: list[dict[str, Any]] = []
    for index, payload in enumerate(dataset):
        if len(rows) >= max_examples:
            break
        if not isinstance(payload, dict):
            continue
        row = _paper_text_row(payload, source_file=dataset_name, row_index=index)
        if row:
            rows.append(row)
    return rows


def _candidate_label(index: int) -> str:
    return f"P{index + 1}"


def _candidate_block(candidates: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for index, candidate in enumerate(candidates):
        label = _candidate_label(index)
        meta = " | ".join(
            str(item)
            for item in [candidate.get("paper_id", ""), candidate.get("categories", ""), candidate.get("year", "")]
            if item not in ("", None)
        )
        parts.append(
            f"{AK_CANDIDATE} {label}: {candidate['title']}\n"
            f"{meta}\n"
            f"{_first_sentences(candidate.get('abstract') or candidate.get('text', ''), limit=360)}"
        )
    return "\n\n".join(parts)


def _active_context_block(row: dict[str, Any], evidence: str, *, label: str = "P1") -> str:
    meta = " | ".join(
        str(item)
        for item in [row.get("paper_id", ""), row.get("categories", ""), row.get("year", "")]
        if item not in ("", None)
    )
    meta_line = f"{meta}\n" if meta else ""
    return (
        f"{AK_CONTEXT_ID} {label}\n"
        f"{AK_SELECTED_PAPER} {label}: {row['title']}\n"
        f"{meta_line}"
        f"{AK_EVIDENCE} [{label}]: {evidence}"
    )


def _selected_context_followup_prompt(index: int) -> str:
    prompts = [
        "Tell me more about the selected paper.",
        "Explain this paper in more detail.",
        "What are the main takeaways from it?",
        "Summarize the loaded paper for me.",
        "What method does this work use?",
        "What should I understand about the paper above?",
        "Give me a clearer explanation of that study.",
        "What are its results and limitations?",
    ]
    return prompts[index % len(prompts)]


def _selected_context_retrieval_prompt(index: int) -> str:
    prompts = [
        "Find related papers that build on this work.",
        "What newer work should I read after this paper?",
        "Compare this selected paper with other research on the topic.",
        "Gather papers that challenge or extend the loaded study.",
        "Search for follow-up work connected to this result.",
        "Find broader literature around the paper above.",
    ]
    return prompts[index % len(prompts)]


def _paper_query(row: dict[str, Any]) -> str:
    terms = _term_tokens(row.get("title", ""), row.get("abstract", ""), limit=6)
    if terms:
        return " ".join(terms)
    return str(row.get("title", ""))


def _make_negative_candidates(rows: list[dict[str, Any]], positive_index: int, *, count: int) -> list[dict[str, Any]]:
    if len(rows) <= 1 or count <= 0:
        return []
    negatives: list[dict[str, Any]] = []
    seen = {positive_index}
    stride = max(1, len(rows) // max(2, count + 1))
    cursor = (positive_index + stride) % len(rows)
    while len(negatives) < count and len(seen) < len(rows):
        if cursor not in seen:
            seen.add(cursor)
            negatives.append(rows[cursor])
        cursor = (cursor + stride + 1) % len(rows)
    return negatives


def _research_multitask_examples_from_rows(
    rows: list[dict[str, Any]],
    *,
    max_examples: int,
    negative_count: int,
    objective: str,
    seed: int,
) -> list[dict[str, Any]]:
    if max_examples <= 0 or not rows:
        return []
    rng = random.Random(int(seed))
    examples: list[dict[str, Any]] = []
    usable_rows = rows[:max_examples]
    for index, row in enumerate(usable_rows):
        query = _paper_query(row)
        if not query:
            continue
        paper_id = row.get("paper_id", "")
        title = row["title"]
        evidence = _first_sentences(row.get("abstract") or row.get("text", ""), limit=520)
        if len(evidence) < 80:
            continue
        negatives = _make_negative_candidates(usable_rows, index, count=negative_count)
        candidates = [row, *negatives]
        rng.shuffle(candidates)
        positive_label = _candidate_label(candidates.index(row))
        candidates_text = _candidate_block(candidates)

        rewrite_query = query.strip()
        examples.append(
            {
                "source_type": "research_retrieval_multitask",
                "source_id": f"{paper_id or index}:query_rewrite",
                "task_type": "query_rewrite",
                "encoder_text": (
                    f"{AK_CHAT} {AK_GATHER_CONTEXT} {AK_QUERY_REWRITE}\n"
                    "AgentKernel Lite retrieval training example.\n"
                    "Rewrite the user request into a compact research-library search query. "
                    "Do not answer yet.\n"
                    f"{AK_USER} Find the best paper about {query}.\n"
                    "Return a structured decision with action=gather_context."
                ),
                "decoder_text": _structured_decision_text(
                    action="gather_context",
                    thought="Rewrite the vague research request into a retrieval query.",
                    content=rewrite_query,
                    retrieval_influenced=False,
                    metadata={"task_type": "query_rewrite", "retrieval_query": rewrite_query},
                ),
                "action": "gather_context",
                "source_action": "gather_context",
                "extension_capability": "",
                "benchmark_family": "research_library_retrieval",
                "difficulty": "query_rewrite",
                "paper_id": paper_id,
                "weight": 0.9,
            }
        )
        examples.append(
            {
                "source_type": "research_retrieval_multitask",
                "source_id": f"{paper_id or index}:rerank",
                "task_type": "rerank_candidates",
                "encoder_text": (
                    f"{AK_CHAT} {AK_GATHER_CONTEXT} {AK_RERANK}\n"
                    "AgentKernel Lite retrieval training example.\n"
                    "Select the candidate that best matches the user request. "
                    "Do not invent a paper outside the candidate list.\n"
                    f"{AK_USER} {query}\n"
                    f"{AK_CONTEXT} Candidates:\n"
                    f"{candidates_text}\n"
                    "Return a structured decision with action=gather_context and the selected candidate id."
                ),
                "decoder_text": _structured_decision_text(
                    action="gather_context",
                    thought="Rank candidate papers by relevance to the research query.",
                    content=f"selected_candidate_id={positive_label}",
                    retrieval_influenced=True,
                    selected_retrieval_span_id=positive_label,
                    metadata={
                        "task_type": "rerank_candidates",
                        "selected_candidate_id": positive_label,
                        "positive_paper_id": paper_id,
                    },
                ),
                "action": "gather_context",
                "source_action": "gather_context",
                "extension_capability": "",
                "benchmark_family": "research_library_retrieval",
                "difficulty": "rerank",
                "paper_id": paper_id,
                "weight": 1.2,
            }
        )
        examples.append(
            {
                "source_type": "research_retrieval_multitask",
                "source_id": f"{paper_id or index}:sufficiency_positive",
                "task_type": "evidence_sufficiency",
                "encoder_text": (
                    f"{AK_CHAT} {AK_RESPOND} {AK_SUFFICIENT}\n"
                    "AgentKernel Lite retrieval training example.\n"
                    "Decide whether the retrieved evidence is sufficient to answer.\n"
                    f"{AK_USER} What does {title} contribute?\n"
                    f"{AK_EVIDENCE} [1]: {title}\n{evidence}\n"
                    "Return a structured chat decision."
                ),
                "decoder_text": _structured_decision_text(
                    action="respond",
                    thought="The retrieved evidence is about the requested paper and is sufficient for a scoped answer.",
                    content="The retrieved evidence is sufficient for a scoped answer. I should answer from [1] and avoid claims beyond the excerpt.",
                    retrieval_influenced=True,
                    selected_retrieval_span_id="P1",
                    metadata={"task_type": "evidence_sufficiency", "evidence_sufficient": True},
                ),
                "action": "respond",
                "source_action": "respond",
                "extension_capability": "",
                "benchmark_family": "research_library_retrieval",
                "difficulty": "sufficiency_positive",
                "paper_id": paper_id,
                "weight": 0.8,
            }
        )
        if negatives:
            negative = negatives[0]
            examples.append(
                {
                    "source_type": "research_retrieval_multitask",
                    "source_id": f"{paper_id or index}:sufficiency_negative",
                    "task_type": "evidence_sufficiency",
                    "encoder_text": (
                        f"{AK_CHAT} {AK_RESPOND} {AK_INSUFFICIENT}\n"
                        "AgentKernel Lite retrieval training example.\n"
                        "Decide whether the retrieved evidence is sufficient to answer. "
                        "If it is off-topic, say retrieval is weak instead of answering.\n"
                        f"{AK_USER} {query}\n"
                        f"{AK_EVIDENCE} [1]: {negative['title']}\n"
                        f"{_first_sentences(negative.get('abstract') or negative.get('text', ''), limit=520)}\n"
                        "Return a structured chat decision."
                    ),
                    "decoder_text": _structured_decision_text(
                        action="respond",
                        thought="The retrieved evidence does not sufficiently support the user request.",
                        content=(
                            "I do not have enough relevant retrieved evidence to answer this well. "
                            "I should gather better context or ask for a narrower research query."
                        ),
                        retrieval_influenced=True,
                        selected_retrieval_span_id=None,
                        metadata={"task_type": "evidence_sufficiency", "evidence_sufficient": False},
                    ),
                    "action": "respond",
                    "source_action": "respond",
                    "extension_capability": "",
                    "benchmark_family": "research_library_retrieval",
                    "difficulty": "sufficiency_negative",
                    "paper_id": paper_id,
                    "weight": 1.1,
                }
            )
        answer = (
            f"Based on [1], {title} contributes the following: {evidence} "
            "This answer is scoped to the retrieved evidence."
        )
        examples.append(
            {
                "source_type": "research_retrieval_multitask",
                "source_id": f"{paper_id or index}:answer_synthesis",
                "task_type": "answer_synthesis",
                "encoder_text": (
                    f"{AK_CHAT} {AK_RESPOND} {AK_ANSWER}\n"
                    "AgentKernel Lite research answer training example.\n"
                    f"{AK_USER} What is the main contribution of {title}?\n"
                    f"{AK_EVIDENCE} [1]: {title}\n{evidence}\n"
                    "Answer directly, cite [1] for supported claims, and do not list unrelated papers."
                ),
                "decoder_text": (
                    _structured_decision_text(
                        action="respond",
                        thought="Synthesize the answer from the selected retrieved evidence.",
                        content=answer,
                        retrieval_influenced=True,
                        selected_retrieval_span_id="P1",
                        metadata={"task_type": "answer_synthesis", "positive_paper_id": paper_id},
                    )
                    if objective == "chat"
                    else answer
                ),
                "action": "respond",
                "source_action": "respond",
                "extension_capability": "",
                "benchmark_family": "research_library_retrieval",
                "difficulty": "answer_synthesis",
                "paper_id": paper_id,
                "weight": 1.0,
            }
        )
        active_context = _active_context_block(row, evidence, label="P1")
        followup_prompt = _selected_context_followup_prompt(index)
        examples.append(
            {
                "source_type": "research_retrieval_multitask",
                "source_id": f"{paper_id or index}:selected_context_plan",
                "task_type": "selected_context_followup_plan",
                "encoder_text": (
                    f"{AK_CHAT} {AK_LOOP} {AK_PLAN} {AK_STATE} {AK_ACTIVE_CONTEXT} {AK_SELECTED_PAPER}\n"
                    "AgentKernel Lite loop training example.\n"
                    "The user has already added a paper to chat. Choose whether to answer from that active context "
                    "or gather new papers. If the active selected paper is the target, do not retrieve new context.\n"
                    f"{AK_CONTEXT} Active context:\n"
                    f"{active_context}\n"
                    f"{AK_USER} {followup_prompt}\n"
                    "Return a structured decision with action=respond, target_context_id=P1, and retrieval=none."
                ),
                "decoder_text": _structured_decision_text(
                    action="respond",
                    content=f"{AK_USE_CONTEXT} {AK_TARGET_CONTEXT}=P1 {AK_NO_RETRIEVAL}",
                    retrieval_influenced=True,
                    selected_retrieval_span_id="P1",
                    metadata={
                        "task_type": "selected_context_followup_plan",
                        "loop_action": "respond_from_active_context",
                        "target_context_id": "P1",
                        "needs_new_retrieval": False,
                    },
                ),
                "action": "respond",
                "source_action": "respond",
                "extension_capability": "",
                "benchmark_family": "research_library_retrieval",
                "difficulty": "selected_context_plan",
                "paper_id": paper_id,
                "weight": 1.3,
            }
        )
        selected_answer = (
            f"The selected paper [P1], {title}, is about this: {evidence} "
            "A useful way to read it is to separate the core problem, the method or argument, "
            "and the supported takeaway. This explanation is limited to the active paper context."
        )
        examples.append(
            {
                "source_type": "research_retrieval_multitask",
                "source_id": f"{paper_id or index}:selected_context_answer",
                "task_type": "selected_context_answer",
                "encoder_text": (
                    f"{AK_CHAT} {AK_LOOP} {AK_RESPOND} {AK_STATE} {AK_ACTIVE_CONTEXT} {AK_SELECTED_PAPER} {AK_USE_CONTEXT}\n"
                    "AgentKernel Lite selected-context answer training example.\n"
                    "Answer the user's follow-up from the paper that is already loaded in chat. "
                    "Do not say retrieval failed and do not introduce unrelated papers.\n"
                    f"{AK_CONTEXT} Active context:\n"
                    f"{active_context}\n"
                    f"{AK_USER} {followup_prompt}\n"
                    f"Answer directly, cite {AK_CITE} [P1] for supported claims, and keep the response conversational."
                ),
                "decoder_text": _structured_decision_text(
                    action="respond",
                    content=selected_answer,
                    retrieval_influenced=True,
                    selected_retrieval_span_id="P1",
                    metadata={
                        "task_type": "selected_context_answer",
                        "loop_action": "answer_from_active_context",
                        "target_context_id": "P1",
                        "needs_new_retrieval": False,
                    },
                ),
                "action": "respond",
                "source_action": "respond",
                "extension_capability": "",
                "benchmark_family": "research_library_retrieval",
                "difficulty": "selected_context_answer",
                "paper_id": paper_id,
                "weight": 1.4,
            }
        )
        retrieval_followup = _selected_context_retrieval_prompt(index)
        related_query_terms = _term_tokens(title, row.get("abstract", ""), limit=7)
        related_query = " ".join(["related work", *related_query_terms]).strip()
        examples.append(
            {
                "source_type": "research_retrieval_multitask",
                "source_id": f"{paper_id or index}:selected_context_retrieve_new",
                "task_type": "selected_context_retrieve_new_plan",
                "encoder_text": (
                    f"{AK_CHAT} {AK_LOOP} {AK_PLAN} {AK_STATE} {AK_ACTIVE_CONTEXT} {AK_SELECTED_PAPER}\n"
                    "AgentKernel Lite loop training example.\n"
                    "The selected paper is useful context, but the user is asking for additional literature. "
                    "Choose gather_context and rewrite the search query without answering yet.\n"
                    f"{AK_CONTEXT} Active context:\n"
                    f"{active_context}\n"
                    f"{AK_USER} {retrieval_followup}\n"
                    "Return a structured decision with action=gather_context and a compact retrieval query."
                ),
                "decoder_text": _structured_decision_text(
                    action="gather_context",
                    content=f"{AK_RETRIEVE_NEW} {related_query}",
                    retrieval_influenced=True,
                    selected_retrieval_span_id="P1",
                    metadata={
                        "task_type": "selected_context_retrieve_new_plan",
                        "loop_action": "gather_new_context_from_active_context",
                        "target_context_id": "P1",
                        "needs_new_retrieval": True,
                        "retrieval_query": related_query,
                    },
                ),
                "action": "gather_context",
                "source_action": "gather_context",
                "extension_capability": "",
                "benchmark_family": "research_library_retrieval",
                "difficulty": "selected_context_retrieve_new",
                "paper_id": paper_id,
                "weight": 1.2,
            }
        )
    return examples


def _paper_text_multitask_examples(
    *,
    paper_text_dataset: str,
    paper_text_split: str,
    paper_text_path: Path,
    max_examples: int,
    max_files: int,
    negative_count: int,
    objective: str,
    seed: int,
    streaming: bool,
) -> list[dict[str, Any]]:
    if max_examples <= 0:
        return []
    rows = (
        _paper_text_rows_from_path(paper_text_path, max_examples=max_examples, max_files=max_files)
        if str(paper_text_path).strip() and paper_text_path != Path("")
        else []
    )
    if not rows and paper_text_dataset.strip():
        rows = _paper_text_rows_from_dataset(
            paper_text_dataset,
            split=paper_text_split,
            max_examples=max_examples,
            streaming=streaming,
        )
    return _research_multitask_examples_from_rows(
        rows,
        max_examples=max_examples,
        negative_count=negative_count,
        objective=objective,
        seed=seed,
    )


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        encoder_text = _compact(row.get("encoder_text", ""), limit=8000)
        decoder_text = _compact(row.get("decoder_text", ""), limit=8000)
        if not encoder_text or not decoder_text:
            continue
        key = hashlib.sha256(f"{encoder_text}\n-->\n{decoder_text}".encode("utf-8")).hexdigest()
        clean = dict(row)
        clean["encoder_text"] = encoder_text
        clean["decoder_text"] = decoder_text
        clean["example_id"] = key
        deduped[key] = clean
    return sorted(deduped.values(), key=lambda item: (str(item.get("source_type", "")), str(item.get("source_id", ""))))


def _limit_mixed(rows: list[dict[str, Any]], max_examples: int) -> list[dict[str, Any]]:
    if max_examples <= 0 or len(rows) <= max_examples:
        return rows
    groups: dict[str, list[dict[str, Any]]] = {}
    discovered: list[str] = []
    for row in rows:
        source_type = str(row.get("source_type", "unknown") or "unknown")
        if source_type not in groups:
            groups[source_type] = []
            discovered.append(source_type)
        groups[source_type].append(row)
    preferred = ["trajectory_step", "qwen_adapter_sft", "research_paper_grounded", "kernel_doc"]
    order = [source_type for source_type in preferred if source_type in groups]
    order.extend(source_type for source_type in discovered if source_type not in order)
    selected: list[dict[str, Any]] = []
    index = 0
    while len(selected) < max_examples:
        emitted = False
        for source_type in order:
            group = groups[source_type]
            if index >= len(group):
                continue
            selected.append(group[index])
            emitted = True
            if len(selected) >= max_examples:
                break
        if not emitted:
            break
        index += 1
    return sorted(selected, key=lambda item: (str(item.get("source_type", "")), str(item.get("source_id", ""))))


def build_dataset(
    *,
    repo_root: Path,
    output_dir: Path,
    trajectory_root: Path,
    docs_root: Path,
    sft_root: Path,
    paper_chunk_root: Path,
    paper_text_dataset: str,
    paper_text_split: str,
    paper_text_path: Path,
    max_trajectory_files: int,
    max_doc_files: int,
    max_sft_files: int,
    max_sft_rows_per_file: int,
    max_research_examples: int,
    max_research_files: int,
    max_paper_text_examples: int,
    paper_text_negative_count: int,
    paper_text_streaming: bool,
    eval_fraction: float,
    max_examples: int,
    objective: str,
    code_trace_mode: str,
    seed: int,
    max_user_slot_examples: int = 0,
) -> dict[str, Any]:
    objective_value = str(objective).strip().lower()
    objective = objective_value if objective_value in {"chat", "text"} else "action"
    code_trace_mode = str(code_trace_mode).strip().lower()
    rows = [
        *_trajectory_examples(
            trajectory_root,
            max_files=max_trajectory_files,
            objective=objective,
            code_trace_mode=code_trace_mode,
        ),
        *_sft_examples(
            sft_root,
            max_files=max_sft_files,
            max_rows_per_file=max_sft_rows_per_file,
            objective=objective,
        ),
        *_doc_examples(docs_root, max_files=max_doc_files, objective=objective),
        *_research_examples(
            paper_chunk_root,
            max_examples=max_research_examples,
            max_files=max_research_files,
            objective=objective,
        ),
        *_paper_text_multitask_examples(
            paper_text_dataset=paper_text_dataset,
            paper_text_split=paper_text_split,
            paper_text_path=paper_text_path,
            max_examples=max_paper_text_examples,
            max_files=max_research_files,
            negative_count=paper_text_negative_count,
            objective=objective,
            seed=seed,
            streaming=paper_text_streaming,
        ),
        *_user_slot_examples(max_examples=max_user_slot_examples, objective=objective, seed=seed),
    ]
    rows = _dedupe(rows)
    rows = _limit_mixed(rows, max_examples)
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for row in rows:
        split = _hash_split(str(row.get("example_id", "")), eval_fraction)
        row["split"] = split
        if split == "eval":
            eval_rows.append(row)
        else:
            train_rows.append(row)
    if not train_rows and eval_rows:
        train_rows.append(eval_rows.pop())
    if not eval_rows and len(train_rows) > 1:
        eval_rows.append(train_rows.pop())
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    source_counts: dict[str, int] = {}
    task_type_counts: dict[str, int] = {}
    target_action_counts: dict[str, int] = {}
    source_action_counts: dict[str, int] = {}
    extension_counts: dict[str, int] = {}
    for row in rows:
        source_type = str(row.get("source_type", "unknown"))
        source_counts[source_type] = source_counts.get(source_type, 0) + 1
        task_type = str(row.get("task_type", "") or "")
        if task_type:
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
        target_action = str(row.get("action", "unknown"))
        target_action_counts[target_action] = target_action_counts.get(target_action, 0) + 1
        source_action = str(row.get("source_action", "unknown"))
        source_action_counts[source_action] = source_action_counts.get(source_action, 0) + 1
        extension = str(row.get("extension_capability", "") or "")
        if extension:
            extension_counts[extension] = extension_counts.get(extension, 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": objective,
        "code_trace_mode": code_trace_mode,
        "paper_text_dataset": paper_text_dataset,
        "paper_text_split": paper_text_split,
        "paper_text_path": str(paper_text_path) if str(paper_text_path) != "." else "",
        "paper_text_negative_count": int(paper_text_negative_count),
        "paper_text_streaming": bool(paper_text_streaming),
        "max_user_slot_examples": int(max_user_slot_examples),
        "agentkernel_special_tokens": list(AGENTKERNEL_SPECIAL_TOKENS),
        "seed": int(seed),
        "repo_root": str(repo_root),
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "task_type_counts": dict(sorted(task_type_counts.items())),
        "target_action_counts": dict(sorted(target_action_counts.items())),
        "source_action_counts": dict(sorted(source_action_counts.items())),
        "extension_counts": dict(sorted(extension_counts.items())),
        "schema": {
            "encoder_text": "context/compiler input for the encoder",
            "decoder_text": "target structured chat decision for the decoder when objective=chat",
            "task_type": "optional multitask label such as query_rewrite, rerank_candidates, evidence_sufficiency, answer_synthesis",
            "pocketpal_user_slots": "synthetic examples that teach user-filled profile slots, slot updates, and privacy boundaries",
            "weight": "loss weight; trainer may ignore until weighted loss is enabled",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--output-dir", default="artifacts/agentkernel_lite_encdec/dataset")
    parser.add_argument("--trajectory-root", default="trajectories/episodes")
    parser.add_argument("--docs-root", default="docs")
    parser.add_argument("--sft-root", default="trajectories/improvement/candidates/qwen_adapter")
    parser.add_argument("--paper-chunk-root", default="")
    parser.add_argument("--paper-text-dataset", default="")
    parser.add_argument("--paper-text-split", default="train")
    parser.add_argument("--paper-text-path", default="")
    parser.add_argument("--max-trajectory-files", type=int, default=5000)
    parser.add_argument("--max-doc-files", type=int, default=48)
    parser.add_argument("--max-sft-files", type=int, default=64)
    parser.add_argument("--max-sft-rows-per-file", type=int, default=2000)
    parser.add_argument("--max-research-examples", type=int, default=0)
    parser.add_argument("--max-research-files", type=int, default=0)
    parser.add_argument("--max-paper-text-examples", type=int, default=0)
    parser.add_argument("--paper-text-negative-count", type=int, default=3)
    parser.add_argument("--paper-text-streaming", choices=("0", "1"), default="1")
    parser.add_argument("--max-user-slot-examples", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--objective", choices=("chat", "text", "action"), default="chat")
    parser.add_argument("--code-trace-mode", choices=("skip", "explain", "raw"), default="explain")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manifest = build_dataset(
        repo_root=repo_root,
        output_dir=(repo_root / args.output_dir).resolve(),
        trajectory_root=(repo_root / args.trajectory_root).resolve(),
        docs_root=(repo_root / args.docs_root).resolve(),
        sft_root=(repo_root / args.sft_root).resolve(),
        paper_chunk_root=Path(args.paper_chunk_root).resolve() if str(args.paper_chunk_root).strip() else Path(""),
        paper_text_dataset=str(args.paper_text_dataset),
        paper_text_split=str(args.paper_text_split),
        paper_text_path=Path(args.paper_text_path).expanduser().resolve() if str(args.paper_text_path).strip() else Path(""),
        max_trajectory_files=int(args.max_trajectory_files),
        max_doc_files=int(args.max_doc_files),
        max_sft_files=int(args.max_sft_files),
        max_sft_rows_per_file=int(args.max_sft_rows_per_file),
        max_research_examples=int(args.max_research_examples),
        max_research_files=int(args.max_research_files),
        max_paper_text_examples=int(args.max_paper_text_examples),
        paper_text_negative_count=int(args.paper_text_negative_count),
        paper_text_streaming=str(args.paper_text_streaming) == "1",
        max_user_slot_examples=int(args.max_user_slot_examples),
        eval_fraction=float(args.eval_fraction),
        max_examples=int(args.max_examples),
        objective=str(args.objective),
        code_trace_mode=str(args.code_trace_mode),
        seed=int(args.seed),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
