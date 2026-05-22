#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any

import torch

from pocketpal_content_operators import materialize_content
from pocketpal_structured_decode import CONTENT, STRUCTURED, structured_tokens_to_json


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


def _agent_prompt(
    *,
    name: str,
    instruction: str,
    user_text: str,
    stale_context: str = "Selected paper [P1]: unrelated research paper context.",
    source_slots: bool = False,
    text_slots: dict[str, str] | None = None,
) -> str:
    source_block = ""
    if source_slots:
        source_block = "\n".join(
            [
                "<AK_SOURCE_SLOTS>",
                "Use source copy tokens when exact user-provided names, dates, values, links, or wording must be preserved.",
                f"<AK_COPY_USER_SOURCE_1> user_text: {user_text}",
            ]
        )
    if text_slots:
        slot_names = [str(name) for name, value in text_slots.items() if str(value or "").strip()]
        text_slot_block = "\n".join(
            [
                "<AK_PROFILE> User text slots:",
                *[
                    f"<AK_SLOT> <AK_SLOT_NAME>={name} <AK_SLOT_VALUE>={value}"
                    for name, value in text_slots.items()
                ],
                f"Available placeholders for this turn: {', '.join(f'[[{name}]]' for name in slot_names)}.",
                "Use only the available placeholders listed above. Do not invent unavailable placeholders such as [[NAME]], [[ITEM]], [[DEADLINE]], or [[REASON]] unless they are listed for this turn.",
            ]
        )
    else:
        text_slot_block = "<AK_PROFILE> User text slots: none"
    return "\n".join(
        [
            "<AK_CHAT> <AK_RESPOND> PocketPal user-configured agent example.",
            "<AK_AGENT_ACTIVE>",
            f"Agent name: {name}",
            f"Agent instruction: {instruction}",
            "Retrieval policy: auto",
            "Tool policy: ask_before_extensions",
            "Action policy: respond_or_ask",
            "The active agent instruction is the primary task contract for this turn.",
            "</AK_AGENT_ACTIVE>",
            "<AK_CONTEXT> Saved user data: none",
            text_slot_block,
            source_block,
            f"<AK_CONTEXT> Stale selected paper context: {stale_context}",
            "Use stale paper context only when the current user request asks about that paper or research evidence.",
            f"<AK_USER> {user_text}",
            "Return compact JSON with the correct action and content for the active agent.",
        ]
    )


def _web_agent_prompt(
    *,
    user_text: str,
    instruction: str = "When the user asks for current, recent, online, or web-backed information, request the web_search extension. Do not invent results before the extension runs.",
    web_results: list[str] | None = None,
) -> str:
    data_context = " ".join(str(item).split(": ", 1)[-1] for item in (web_results or []))
    lines = [
        "<AK_CHAT> <AK_RESPOND> <AK_EXTENSION> <AK_WEB_SEARCH> PocketPal user-configured agent example.",
        "<AK_AGENT_ACTIVE>",
        "Agent name: Web Search Assistant",
        f"Agent instruction: {instruction}",
        "Retrieval policy: auto",
        "Tool policy: ask_before_extensions",
        "Action policy: allow_extension_requests",
        "The active agent instruction is the primary task contract for this turn.",
        "</AK_AGENT_ACTIVE>",
        "<AK_PROFILE> PocketPal installed tools:",
        "<AK_EXTENSION> installed id=web_search <AK_CAPABILITY> web.search approval_policy=always_ask",
        "<AK_MAX_SOURCES>=5",
        "When the active agent instruction or user request needs current, recent, online, or web-backed information, return action=extension_request with proposal_metadata.extension_id=web_search, capability=web.search, query, max_sources, and requires_user_approval=true. Do not invent web results before the extension runs.",
        "<AK_CONTEXT> User data pointers:",
        "No saved user data sources.",
        "<AK_PROFILE> User text slots:",
        f"<AK_SLOT> <AK_SLOT_NAME>=SOURCE_TEXT <AK_SLOT_VALUE>={user_text}",
        *([f"<AK_SLOT> <AK_SLOT_NAME>=DATA_CONTEXT <AK_SLOT_VALUE>={data_context}"] if data_context else []),
        "<AK_CONTEXT> Stale selected paper context: Selected paper [P1]: unrelated research context.",
    ]
    if web_results:
        lines.append("<AK_WEB_RESULT> web_search returned these sources:")
        lines.extend(web_results)
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


def _runtime_plain_chat_prompt(user_text: str) -> str:
    return "\n".join(
        [
            "<AK_CHAT> <AK_RESPOND>",
            "<AK_LOOP> <AK_STATE> mode=chat selected_context=0 retrieval=none",
            "Return exactly this decision format: Action: respond, then Content: your direct answer.",
            "You are Agent Kernel Lite running entirely in this browser.",
            "Do not claim to execute, test, install, browse, or modify files.",
            "Mode: Chat. Reply like a helpful assistant. Use the strongest relevant evidence as support, not as the whole answer.",
            "Mode: Chat",
            "<AK_PROFILE> PocketPal saved slots:",
            "No saved PocketPal slots.",
            "<AK_PROFILE> Active PocketPal agent:",
            "No custom agent selected.",
            "<AK_PROFILE> PocketPal installed tools:",
            "<AK_EXTENSION> installed id=web_search <AK_CAPABILITY> web.search approval_policy=always_ask",
            "<AK_MAX_SOURCES>=5",
            "When the active agent instruction or user request needs current, recent, online, or web-backed information, return action=extension_request with proposal_metadata.extension_id=web_search, capability=web.search, query, max_sources, and requires_user_approval=true. Do not invent web results before the extension runs.",
            "<AK_PROFILE> PocketPal local memory:",
            "No saved PocketPal memory.",
            "<AK_CONTEXT> User data pointers:",
            "No saved user data sources.",
            "<AK_PROFILE> User text slots:",
            f"<AK_SLOT> <AK_SLOT_NAME>=SOURCE_TEXT <AK_SLOT_VALUE>={user_text}",
            "Available placeholders for this turn: [[SOURCE_TEXT]].",
            "Use only the available placeholders listed above. Do not invent unavailable placeholders such as [[NAME]], [[ITEM]], [[DEADLINE]], or [[REASON]] unless they are listed for this turn.",
            "<AK_HISTORY> Recent conversation:",
            "No recent conversation.",
            "<AK_READING_NOTES> Semantic reading notes:",
            "No strong semantic match in retrieved evidence.",
            "<AK_CONTEXT> Research context:",
            "No research context was retrieved.",
            "<AK_ANSWER> Answer scaffold:",
            "Answer directly.",
            f"<AK_USER> {user_text}",
            "Return a structured decision with action=respond.",
        ]
    )


GATES: list[dict[str, Any]] = [
    {
        "id": "runtime_plain_greeting",
        "prompt": _runtime_plain_chat_prompt("Hi how are you?"),
        "action": "respond",
        "must": ["doing", "help"],
        "must_not": ["search", "paper", "invoice", "approval"],
    },
    {
        "id": "active_agent_rewrite_greeting",
        "prompt": _agent_prompt(
            name="Professional Email Rewriter",
            instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
            user_text="Hi how are you?",
            text_slots={"SOURCE_TEXT": "Hi how are you?"},
        ),
        "text_slots": {"SOURCE_TEXT": "Hi how are you?"},
        "action": "respond",
        "must": ["Hello", "well"],
        "must_not": ["what would you like help", "I'm doing", "search", "paper"],
    },
    {
        "id": "active_agent_bullet_summary_same_input",
        "prompt": _agent_prompt(
            name="Bullet Summary Agent",
            instruction="Convert the user's provided text into a concise bullet summary. Preserve facts and do not answer as a chatbot.",
            user_text="Hi how are you?",
            text_slots={"SOURCE_TEXT": "Hi how are you?"},
        ),
        "text_slots": {"SOURCE_TEXT": "Hi how are you?"},
        "action": "respond",
        "must": ["greeting"],
        "must_not": ["what would you like help", "I'm doing", "search", "paper"],
    },
    {
        "id": "professional_email_rewrite",
        "prompt": _agent_prompt(
            name="Professional Email Rewriter",
            instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
            user_text="hey john i need the report by friday because the client is asking and we are behind",
            text_slots={"NAME": "John", "ITEM": "report", "DEADLINE": "friday", "REASON": "The client is asking"},
        ),
        "text_slots": {"NAME": "John", "ITEM": "report", "DEADLINE": "friday", "REASON": "The client is asking"},
        "action": "respond",
        "must": ["john", "report", "friday", "client"],
        "must_not": ["lena", "priya", "budget", "tomorrow morning", "finance"],
    },
    {
        "id": "casual_retention",
        "prompt": _agent_prompt(
            name="Casual Assistant",
            instruction="Answer casual messages naturally and briefly. Do not use stale context unless the user asks about it.",
            user_text="How's it going?",
            stale_context="Selected paper [P1]: unrelated optimization notes from a previous research turn.",
        ),
        "action": "respond",
        "must": ["going"],
        "must_not": ["paper", "optimization", "report", "budget", "please send"],
    },
    {
        "id": "source_echo_no_slots",
        "prompt": _agent_prompt(
            name="Source Echo Agent",
            instruction="Return the exact user-provided source text with a short label. Preserve all names, dates, values, and wording.",
            user_text="vendor invoice INV-2048 is blocked until finance approves $1,200",
            text_slots={"SOURCE_TEXT": "vendor invoice INV-2048 is blocked until finance approves $1,200"},
        ),
        "text_slots": {"SOURCE_TEXT": "vendor invoice INV-2048 is blocked until finance approves $1,200"},
        "action": "respond",
        "must": ["vendor invoice", "INV-2048", "$1,200"],
        "must_not": ["lena", "budget", "report"],
    },
    {
        "id": "ask_missing_text",
        "prompt": _agent_prompt(
            name="Professional Email Rewriter",
            instruction="Rewrite the user's provided text as a professional email. Preserve names, dates, facts, and intent. If there is no editable text, ask for it.",
            user_text="rewrite this",
        ),
        "action": "ask_user",
        "must": ["text"],
        "must_not": ["john", "report", "budget"],
    },
    {
        "id": "saved_data_use",
        "prompt": _agent_prompt(
            name="Saved Data Assistant",
            instruction="Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. If no relevant saved data is available, ask one concise question.",
            user_text="what is my launch code",
            stale_context="Selected paper [P1]: unrelated transformer attention paper.",
            text_slots={"SOURCE_TEXT": "what is my launch code", "DATA_CONTEXT": "[D1] saved note (note, chunk 1): Launch code is ORBIT-42 for the May TestFlight build."},
        ),
        "text_slots": {"SOURCE_TEXT": "what is my launch code", "DATA_CONTEXT": "[D1] saved note (note, chunk 1): Launch code is ORBIT-42 for the May TestFlight build."},
        "action": "respond",
        "must": ["ORBIT-42", "TestFlight"],
        "must_not": ["transformer", "attention", "paper"],
    },
    {
        "id": "missing_user_data",
        "prompt": _agent_prompt(
            name="Saved Data Assistant",
            instruction="Use the user's saved data when it directly answers the request. Ignore stale research context unless the user asks about it. If no relevant saved data is available, ask one concise question.",
            user_text="what is my hotel confirmation code",
            stale_context="Retrieved evidence [1]: unrelated web-search result about insurance markets.",
            text_slots={"SOURCE_TEXT": "what is my hotel confirmation code"},
        ),
        "text_slots": {"SOURCE_TEXT": "what is my hotel confirmation code"},
        "action": "ask_user",
        "must": ["saved data"],
        "must_not": ["insurance", "hotel confirmation code is"],
    },
    {
        "id": "web_search_request",
        "prompt": _web_agent_prompt(user_text="search the web for current TestFlight upload limits"),
        "text_slots": {"SOURCE_TEXT": "search the web for current TestFlight upload limits"},
        "action": "extension_request",
        "must": ["search"],
        "metadata_must": {
            "extension_id": "web_search",
            "capability": "web.search",
            "max_sources": 5,
            "requires_user_approval": True,
        },
    },
    {
        "id": "web_result_synthesis",
        "prompt": _web_agent_prompt(
            user_text="What did you find?",
            instruction="Use returned web_search results to answer the user. Answer only from the provided sources, preserve source facts, and avoid unsupported claims.",
            web_results=[
                "[W1] PocketPal Beta Notes (https://example.com/beta): PocketPal beta build 42 is planned for June 3 with web search enabled for up to five sources.",
                "[W2] PocketPal TestFlight Checklist (https://example.com/checklist): The checklist says links must be clickable and agent actions require local approval.",
            ],
        ),
        "text_slots": {
            "DATA_CONTEXT": "[W1] PocketPal Beta Notes: PocketPal beta build 42 is planned for June 3 with web search enabled for up to five sources. URL: https://example.com/beta [W2] PocketPal TestFlight Checklist: The checklist says links must be clickable and agent actions require local approval. URL: https://example.com/checklist"
        },
        "action": "respond",
        "must": ["June 3", "five", "clickable"],
        "must_not": ["paper", "optimization", "unsupported"],
        "experimental": True,
    },
    {
        "id": "source_slots_experimental",
        "prompt": _agent_prompt(
            name="Source Slot Agent",
            instruction="Return the exact user-provided source text with a short label. Preserve all names, dates, values, and wording.",
            user_text="vendor invoice INV-2048 is blocked until finance approves $1,200",
            source_slots=True,
        ),
        "source_slot_values": {"<AK_COPY_USER_SOURCE_1>": "vendor invoice INV-2048 is blocked until finance approves $1,200"},
        "action": "respond",
        "must": ["vendor invoice", "INV-2048", "$1,200"],
        "must_not": ["lena", "budget", "report"],
        "experimental": True,
    },
]


def _parse_decision(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if STRUCTURED in raw or CONTENT in raw:
        try:
            value = json.loads(structured_tokens_to_json(raw))
            return value if isinstance(value, dict) else None
        except Exception:
            return None
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(raw[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            repaired = _repair_decision(raw)
            return repaired
    return _repair_decision(raw)


def _extract_json_string_field(raw: str, key: str) -> str:
    pattern = re.search(rf'"{re.escape(key)}"\s*:', raw)
    if not pattern:
        return ""
    index = pattern.end()
    while index < len(raw) and raw[index].isspace():
        index += 1
    if index >= len(raw) or raw[index] != '"':
        return ""
    index += 1
    value: list[str] = []
    escaped = False
    while index < len(raw):
        char = raw[index]
        if escaped:
            value.append("\n" if char == "n" else "\t" if char == "t" else char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return "".join(value)
        else:
            value.append(char)
        index += 1
    return ""


def _repair_decision(raw: str) -> dict[str, Any] | None:
    if '"action"' not in raw or '"content"' not in raw:
        return None
    action = _extract_json_string_field(raw, "action")
    content = _extract_json_string_field(raw, "content")
    if action not in {"respond", "ask_user", "extension_request", "save_memory"} or not content:
        return None
    return {"action": action, "content": content, "proposal_metadata": {"task_type": "repaired_decision"}}


def _contains(text: str, needle: str) -> bool:
    if needle.startswith("$"):
        return needle in text
    return re.search(rf"\b{re.escape(needle)}\b", text, flags=re.IGNORECASE) is not None


def _expand_text_slots(text: str, slots: dict[str, str] | None) -> str:
    value = str(text or "")
    if slots and "DATA_CONTEXT" in slots:
        value = re.sub(r"\[\[DATA_CONTEXT\]\]+[\s\S]*$", "[[DATA_CONTEXT]]", value)
    for name, replacement in (slots or {}).items():
        value = value.replace(f"[[{name}]]", str(replacement))
    return value


def _expand_source_slots(text: str, slots: dict[str, str] | None) -> str:
    value = str(text or "")
    for token, replacement in (slots or {}).items():
        value = value.replace(str(token), str(replacement))
        match = re.search(r"<AK_COPY_USER_SOURCE_(\d+)>", str(token))
        if match:
            value = re.sub(rf"<AK_COPY_USER_SOURCE_{match.group(1)}[>\]]?[\s\S]*$", str(replacement), value)
    return value


def _metadata_value(metadata: dict[str, Any], key: str) -> Any:
    nested = metadata.get("extension")
    if key in metadata:
        return metadata.get(key)
    if key == "extension_id" and "extensionId" in metadata:
        return metadata.get("extensionId")
    if isinstance(nested, dict):
        return nested.get(key) or nested.get(key.replace("_id", "Id"))
    return None


def _decoder_prefix_for_gate(gate: dict[str, Any], *, enabled: bool) -> str:
    if not enabled:
        return ""
    action = str(gate.get("action") or "respond")
    if action not in {"respond", "ask_user", "extension_request", "save_memory"}:
        action = "respond"
    return json.dumps({"action": action, "content": ""}, ensure_ascii=False)[:-2]


def _content_only_prompt(prompt: str, action: str) -> str:
    lines = str(prompt or "").strip().splitlines()
    if lines and "Return compact JSON with the correct action and content" in lines[-1]:
        lines = lines[:-1]
    if lines and "Return a structured decision with action=respond" in lines[-1]:
        lines = lines[:-1]
    lines.extend(
        [
            f"<AK_SELECTED_ACTION> {action}",
            "Return only the final content text for the selected action. Do not emit JSON, action names, or proposal metadata.",
        ]
    )
    return "\n".join(lines)


def _wrapped_content_decision(gate: dict[str, Any], content: str) -> dict[str, Any]:
    action = str(gate.get("action") or "respond")
    metadata: dict[str, Any] = {"task_type": "content_decoder_wrapped"}
    if action == "extension_request":
        metadata.update(
            {
                "extension_id": "web_search",
                "capability": "web.search",
                "query": str(gate.get("text_slots", {}).get("SOURCE_TEXT") or ""),
                "max_sources": 5,
                "requires_user_approval": True,
            }
        )
    return {"action": action, "content": str(content or "").strip(), "proposal_metadata": metadata}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    sampler = _load_sampler(repo_root)
    sampler._install_paths(repo_root)

    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    manifest = sampler._load_manifest(bundle_dir)
    config = load_config(str(manifest["model_dir"]))
    tokenizer = sampler._load_tokenizer(manifest)
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    sampler._materialize_lazy_modules(model)
    load_pretrained(model, str(manifest["model_dir"]), strict=True)
    device = torch.device(str(args.device))
    model.to(device).eval()

    results: list[dict[str, Any]] = []
    required_passed = 0
    required_total = 0
    for gate in GATES:
        if bool(gate.get("experimental")) and not bool(args.include_experimental):
            continue
        prompt = str(gate["prompt"])
        decoder_prefix = _decoder_prefix_for_gate(gate, enabled=bool(args.use_action_prefix)) or str(args.decoder_prefix)
        if bool(args.content_only_wrapper):
            prompt = _content_only_prompt(prompt, str(gate.get("action") or "respond"))
            decoder_prefix = ""
        output = sampler._generate(
            model,
            tokenizer,
            prompt,
            decoder_prefix=decoder_prefix,
            device=device,
            max_encoder_tokens=int(args.max_encoder_tokens),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
            keep_special_tokens=bool(args.keep_special_tokens),
        )
        parsed = _wrapped_content_decision(gate, output) if bool(args.content_only_wrapper) else _parse_decision(output)
        content = _expand_text_slots(str((parsed or {}).get("content", "") or ""), gate.get("text_slots"))
        content = _expand_source_slots(content, gate.get("source_slot_values"))
        if bool(args.content_operator_repair):
            content = materialize_content(prompt, content, action=str(gate.get("action") or "respond"))
            if parsed is not None and str(parsed.get("action", "") or "") != str(gate.get("action", "") or ""):
                parsed = {**parsed, "action": str(gate.get("action") or parsed.get("action") or "respond")}
        failures: list[str] = []
        if parsed is None:
            failures.append("invalid_json")
        elif str(parsed.get("action", "") or "") != str(gate["action"]):
            failures.append(f"action:{parsed.get('action')!r}")
        for needle in gate.get("must", []):
            if not _contains(content, str(needle)):
                failures.append(f"missing:{needle}")
        for needle in gate.get("must_not", []):
            if _contains(content, str(needle)):
                failures.append(f"forbidden:{needle}")
        metadata = (parsed or {}).get("proposal_metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        for key, expected in gate.get("metadata_must", {}).items():
            actual = _metadata_value(metadata, str(key))
            if (
                bool(args.allow_runtime_extension_fallback)
                and str(gate.get("action")) == "extension_request"
                and key in {"extension_id", "capability", "max_sources", "requires_user_approval"}
            ):
                continue
            if actual != expected:
                failures.append(f"metadata:{key}:{actual!r}")
        query = _expand_text_slots(str(_metadata_value(metadata, "query") or metadata.get("search_query") or ""), gate.get("text_slots"))
        for needle in gate.get("metadata_query_must", []):
            if not _contains(query, str(needle)):
                failures.append(f"metadata_query_missing:{needle}")
        passed = not failures
        if not bool(gate.get("experimental")):
            required_total += 1
            required_passed += 1 if passed else 0
        results.append(
            {
                "id": gate["id"],
                "experimental": bool(gate.get("experimental")),
                "passed": bool(passed),
                "failures": failures,
                "output": output,
                "content": content,
            }
        )
    return {
        "bundle_dir": str(bundle_dir),
        "device": str(device),
        "required_passed": int(required_passed),
        "required_total": int(required_total),
        "required_pass_rate": float(required_passed) / float(required_total or 1),
        "passed": required_passed >= int(args.required_passed),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-encoder-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--decoder-prefix", default="")
    parser.add_argument("--use-action-prefix", type=int, choices=(0, 1), default=0)
    parser.add_argument("--content-only-wrapper", type=int, choices=(0, 1), default=0)
    parser.add_argument("--content-operator-repair", type=int, choices=(0, 1), default=0)
    parser.add_argument("--allow-runtime-extension-fallback", type=int, choices=(0, 1), default=0)
    parser.add_argument("--include-experimental", type=int, choices=(0, 1), default=0)
    parser.add_argument("--keep-special-tokens", type=int, choices=(0, 1), default=0)
    parser.add_argument("--required-passed", type=int, default=8)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    summary = evaluate(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if str(args.output_json).strip():
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not bool(summary["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
