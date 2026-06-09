#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _slot_block(slots: dict[str, str]) -> str:
    if not slots:
        return "<AK_PROFILE> User text slots: none"
    lines = ["<AK_PROFILE> User text slots:"]
    for name, value in slots.items():
        lines.append(f"<AK_SLOT> <AK_SLOT_NAME>={name} <AK_SLOT_VALUE>={value}")
    lines.append(
        "When preserving exact user-provided details, write the matching [[SOURCE_TEXT]] or [[DATA_CONTEXT]] placeholder in the JSON content."
    )
    return "\n".join(lines)


def _prompt(
    *,
    agent_name: str,
    instruction: str,
    user_text: str,
    data_context: str = "No saved user data sources.",
    stale_context: str = "Selected paper [P1]: unrelated research context.",
    slots: dict[str, str] | None = None,
    tool_policy: str = "ask_before_extensions",
    action_policy: str = "allow_extension_requests",
    web_results: list[dict[str, str]] | None = None,
) -> str:
    lines = [
        "<AK_CHAT> <AK_RESPOND> <AK_EXTENSION> <AK_WEB_SEARCH> PocketPal user-configured agent example.",
        "<AK_AGENT_ACTIVE>",
        f"Agent name: {agent_name}",
        f"Agent instruction: {instruction}",
        "Retrieval policy: auto",
        f"Tool policy: {tool_policy}",
        f"Action policy: {action_policy}",
        "The active agent instruction is the primary task contract for this turn.",
        "</AK_AGENT_ACTIVE>",
        "<AK_PROFILE> PocketPal installed tools:",
        "<AK_EXTENSION> installed id=web_search <AK_CAPABILITY> web.search approval_policy=always_ask",
        "<AK_MAX_SOURCES>=5",
        "When the active agent instruction or user request needs current, recent, online, or web-backed information, return action=extension_request with proposal_metadata.extension_id=web_search, capability=web.search, query, max_sources, and requires_user_approval=true. Do not invent web results before the extension runs.",
        "<AK_CONTEXT> User data pointers:",
        data_context,
        _slot_block(slots or {}),
        f"<AK_CONTEXT> Stale selected paper context: {stale_context}",
        "Use stale paper context only when the current user request asks about that paper or research evidence.",
    ]
    if web_results:
        lines.append("<AK_WEB_RESULT> web_search returned these sources:")
        for index, result in enumerate(web_results, start=1):
            title = result["title"]
            url = result["url"]
            snippet = result["snippet"]
            lines.append(f"[W{index}] {title} ({url}): {snippet}")
        lines.append("The extension already ran. Return action=respond and synthesize only these web results.")
    else:
        lines.append("If web search is needed, return action=extension_request and include proposal_metadata.extension_id/capability/query/max_sources/requires_user_approval.")
    lines.extend(
        [
            f"<AK_USER> {user_text}",
            "Return compact JSON with the correct action and content for the active agent.",
        ]
    )
    return "\n".join(lines)


def _row(
    *,
    source_id: str,
    action: str,
    content: str,
    task_type: str,
    metadata: dict[str, Any] | None,
    weight: float,
    **prompt_kwargs: Any,
) -> dict[str, Any]:
    decoder = {"action": action, "content": content}
    if metadata is not None:
        proposal_metadata = {"task_type": task_type}
        proposal_metadata.update(metadata)
        decoder["proposal_metadata"] = proposal_metadata
    encoder_text = _prompt(**prompt_kwargs)
    return {
        "action": action,
        "decoder_text": json.dumps(decoder, ensure_ascii=False, sort_keys=True),
        "encoder_text": encoder_text,
        "example_id": hashlib.sha256(f"{source_id}\n{encoder_text}\n{content}".encode()).hexdigest(),
        "source_id": source_id,
        "source_type": "pocketpal_stage3_tools",
        "task_type": task_type,
        "weight": float(weight),
    }


WEB_REQUESTS = [
    {
        "user": "search the web for the current SwiftUI navigation documentation",
        "query": "current SwiftUI navigation documentation",
    },
    {
        "user": "look up recent reviews for the Anker 737 power bank",
        "query": "recent reviews Anker 737 power bank",
    },
    {
        "user": "find the latest TestFlight upload limits",
        "query": "latest TestFlight upload limits",
    },
    {
        "user": "use the web to check whether WKWebView supports find interactions",
        "query": "WKWebView supports find interactions",
    },
    {
        "user": "search for current App Store screenshot size requirements",
        "query": "current App Store screenshot size requirements",
    },
    {
        "user": "can you web search the newest BitNet quantization notes",
        "query": "newest BitNet quantization notes",
    },
]


WEB_SYNTHESIS = [
    {
        "user": "What did you find about the release window?",
        "query": "PocketPal beta release window",
        "results": [
            {
                "title": "PocketPal Beta Notes",
                "url": "https://example.com/beta",
                "snippet": "PocketPal beta build 42 is planned for June 3 with web search enabled for up to five sources.",
            },
            {
                "title": "PocketPal TestFlight Checklist",
                "url": "https://example.com/checklist",
                "snippet": "The TestFlight checklist says links must be clickable and agent actions require local approval.",
            },
        ],
        "answer": "The sources say PocketPal beta build 42 is planned for June 3, with web search capped at five sources. They also say links should be clickable and agent actions require local approval. Sources: PocketPal Beta Notes and PocketPal TestFlight Checklist.",
    },
    {
        "user": "Summarize the web results for me.",
        "query": "local agent browser search safety",
        "results": [
            {
                "title": "Local Agent Search Safety",
                "url": "https://example.com/safety",
                "snippet": "Local agents should ask before opening external tools and should separate retrieved snippets from model-generated claims.",
            },
            {
                "title": "Browser Retrieval Design",
                "url": "https://example.com/retrieval",
                "snippet": "A five source cap keeps mobile web retrieval fast while still allowing cross-source comparison.",
            },
        ],
        "answer": "The results recommend asking before external tool use, separating retrieved snippets from generated claims, and keeping mobile search bounded. The retrieval design source says a five-source cap helps keep searches fast while allowing comparison.",
    },
]


def _web_results_context(results: list[dict[str, str]]) -> str:
    return " ".join(
        f"[W{index}] {result['title']}: {result['snippet']} URL: {result['url']}"
        for index, result in enumerate(results, start=1)
    )


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    search_instruction = (
        "When the user asks for current, recent, online, or web-backed information, request the web_search extension. "
        "Do not invent results before the extension runs."
    )
    for repeat in range(90):
        for item in WEB_REQUESTS:
            rows.append(
                _row(
                    source_id=f"stage3_web_request_{repeat:03d}_{len(rows):05d}",
                    agent_name="Web Search Assistant",
                    instruction=search_instruction,
                    user_text=item["user"],
                    slots={"SOURCE_TEXT": item["user"]},
                    action="extension_request",
                    content="Requesting approval to search the web.",
                    task_type="web_search_request",
                    metadata={
                        "extension_id": "web_search",
                        "capability": "web.search",
                        "query": "[[SOURCE_TEXT]]",
                        "max_sources": 5,
                        "requires_user_approval": True,
                    },
                    weight=7.0,
                )
            )

    synthesis_instruction = (
        "Use returned web_search results to answer the user. Answer only from the provided sources, preserve source facts, "
        "and say when the sources do not provide enough information."
    )
    for repeat in range(90):
        for item in WEB_SYNTHESIS:
            data_context = _web_results_context(item["results"])
            rows.append(
                _row(
                    source_id=f"stage3_web_synthesis_{repeat:03d}_{len(rows):05d}",
                    agent_name="Web Search Assistant",
                    instruction=synthesis_instruction,
                    user_text=item["user"],
                    web_results=item["results"],
                    slots={"SOURCE_TEXT": item["user"], "DATA_CONTEXT": data_context},
                    action="respond",
                    content="Based on the web results: [[DATA_CONTEXT]]",
                    task_type="web_search_result_synthesis",
                    metadata=None,
                    weight=5.0,
                )
            )

    no_search_instruction = "Answer from the user's message or saved data when available. Only request web search when the user explicitly needs online information."
    no_search_cases = [
        ("How's it going?", "It's going well. What would you like help with?"),
        ("Rewrite this professionally: hey sam please send the deck today", "Please send the deck today, Sam."),
        ("what is my launch code", "I found this in your saved data: [[DATA_CONTEXT]]"),
    ]
    for repeat in range(90):
        for user_text, content in no_search_cases:
            data_context = (
                "[D1] saved note (note, chunk 1): Launch code is ORBIT-42 for the May TestFlight build."
                if "launch code" in user_text
                else "No saved user data sources."
            )
            slots = {"SOURCE_TEXT": user_text}
            if "launch code" in user_text:
                slots["DATA_CONTEXT"] = data_context
            rows.append(
                _row(
                    source_id=f"stage3_no_search_{repeat:03d}_{len(rows):05d}",
                    agent_name="General Assistant",
                    instruction=no_search_instruction,
                    user_text=user_text,
                    data_context=data_context,
                    slots=slots,
                    action="respond",
                    content=content,
                    task_type="no_search_when_not_needed",
                    metadata={},
                    weight=4.0,
                )
            )
    return rows


def _read_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for key in ("train_dataset_path", "eval_dataset_path"):
        path_value = str(manifest.get(key, "") or "")
        if not path_value:
            continue
        path = Path(path_value)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def write_dataset(rows: list[dict[str, Any]], output_dir: Path, eval_fraction: float, name: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"{name}_train.jsonl"
    eval_path = output_dir / f"{name}_eval.jsonl"
    eval_every = max(2, int(round(1.0 / max(0.01, min(0.5, float(eval_fraction))))))
    train_rows = [row for index, row in enumerate(rows) if index % eval_every != 0]
    eval_rows = [row for index, row in enumerate(rows) if index % eval_every == 0]
    for path, split_rows in ((train_path, train_rows), (eval_path, eval_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    action_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for row in rows:
        action_counts[str(row["action"])] = action_counts.get(str(row["action"]), 0) + 1
        task_counts[str(row["task_type"])] = task_counts.get(str(row["task_type"]), 0) + 1
        source_counts[str(row.get("source_type", "unknown"))] = source_counts.get(str(row.get("source_type", "unknown")), 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path.resolve()),
        "eval_examples": len(eval_rows),
        "manifest_path": str((output_dir / f"{name}_manifest.json").resolve()),
        "objective": "pocketpal_stage3_tool_and_web_search_control",
        "schema": {"encoder_text": "PocketPal active agent, tools, optional web results, and user request", "decoder_text": "compact JSON action decision"},
        "source_counts": source_counts,
        "target_action_counts": action_counts,
        "task_type_counts": task_counts,
        "total_examples": len(rows),
        "train_dataset_path": str(train_path.resolve()),
        "train_examples": len(train_rows),
    }
    (output_dir / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage3_tool_dataset")
    parser.add_argument("--mixed-output-dir", default="")
    parser.add_argument("--include-manifest", action="append", default=[])
    parser.add_argument("--stage3-repeat", type=int, default=1)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    args = parser.parse_args()
    stage3_rows = build_rows()
    repeated_stage3_rows: list[dict[str, Any]] = []
    for repeat in range(max(1, int(args.stage3_repeat))):
        for row in stage3_rows:
            next_row = dict(row)
            next_row["source_id"] = f"{row['source_id']}:repeat_{repeat:02d}"
            next_row["example_id"] = hashlib.sha256(f"{next_row['source_id']}\n{row['encoder_text']}\n{row['decoder_text']}".encode()).hexdigest()
            repeated_stage3_rows.append(next_row)
    manifest = write_dataset(stage3_rows, Path(args.output_dir), float(args.eval_fraction), "pocketpal_stage3_tool")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if str(args.mixed_output_dir).strip():
        mixed_rows: list[dict[str, Any]] = []
        for manifest_arg in args.include_manifest:
            mixed_rows.extend(_read_manifest_rows(Path(manifest_arg).expanduser().resolve()))
        mixed_rows.extend(repeated_stage3_rows)
        mixed_manifest = write_dataset(mixed_rows, Path(args.mixed_output_dir), float(args.eval_fraction), "pocketpal_stage3_mixed")
        print(json.dumps(mixed_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
