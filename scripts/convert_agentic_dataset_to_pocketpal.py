#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator


AK_USER = "<AK_USER>"
AK_CHAT = "<AK_CHAT>"
AK_CONTEXT = "<AK_CONTEXT>"
AK_PROFILE = "<AK_PROFILE>"
AK_SLOT = "<AK_SLOT>"
AK_SLOT_NAME = "<AK_SLOT_NAME>"
AK_SLOT_VALUE = "<AK_SLOT_VALUE>"
AK_EXTENSION = "<AK_EXTENSION>"
AK_CAPABILITY = "<AK_CAPABILITY>"
AK_APPROVAL = "<AK_APPROVAL>"
AK_EXTENSION_RESULT = "<AK_EXTENSION_RESULT>"
AK_RESPOND = "<AK_RESPOND>"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _compact(value: object, *, limit: int = 5000) -> str:
    text = " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split())
    return text[:limit].rstrip()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_serialized(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return value


def _iter_local_rows(path: Path, *, max_rows: int) -> Iterator[dict[str, Any]]:
    emitted = 0
    if path.is_dir():
        paths = sorted([*path.glob("*.jsonl"), *path.glob("*.json")])
    else:
        paths = [path]
    for item in paths:
        if max_rows > 0 and emitted >= max_rows:
            break
        if not item.exists():
            continue
        if item.suffix.lower() == ".jsonl":
            with item.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if max_rows > 0 and emitted >= max_rows:
                        break
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        emitted += 1
                        yield row
            continue
        payload = _load_json(item)
        rows = payload
        if isinstance(payload, dict):
            rows = payload.get("rows") or payload.get("data") or payload.get("examples") or []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if max_rows > 0 and emitted >= max_rows:
                break
            if isinstance(row, dict):
                emitted += 1
                yield row


def _iter_hf_rows(dataset_name: str, *, config: str, split: str, streaming: bool, max_rows: int) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Hugging Face dataset conversion requires the 'datasets' package") from exc
    dataset = load_dataset(dataset_name, config or None, split=split, streaming=bool(streaming))
    for index, row in enumerate(dataset):
        if max_rows > 0 and index >= max_rows:
            break
        if isinstance(row, dict):
            yield row


def _as_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("messages", "conversation", "conversations", "dialogue", "turns"):
        value = row.get(key)
        value = _parse_serialized(value)
        if isinstance(value, list):
            messages: list[dict[str, Any]] = []
            for item in value:
                if not isinstance(item, dict):
                    continue
                role = item.get("role", item.get("from", item.get("speaker", "")))
                content = item.get("content", item.get("value", item.get("text", "")))
                normalized_role = str(role).lower().replace("_", " ").replace("-", " ").strip()
                if normalized_role in {"human", "user"}:
                    role = "user"
                elif normalized_role in {"gpt", "assistant", "model"}:
                    role = "assistant"
                elif normalized_role in {"tool call", "toolcall", "function call"}:
                    role = "tool_call"
                elif normalized_role in {"tool", "function"}:
                    role = "tool"
                messages.append({**item, "role": str(role or "unknown"), "content": content})
            if messages:
                return messages
    chat_text = str(row.get("chat", "") or "")
    if chat_text:
        messages = []
        for match in re.finditer(
            r"(USER|ASSISTANT)\s*:\s*(.*?)(?=\n\s*(?:USER|ASSISTANT)\s*:|\s*<\|endoftext\|>\s*$|\Z)",
            chat_text,
            flags=re.DOTALL | re.IGNORECASE,
        ):
            role = "user" if match.group(1).lower() == "user" else "assistant"
            content = _compact(match.group(2).replace("<|endoftext|>", ""), limit=5000)
            if content:
                messages.append({"role": role, "content": content})
        if messages:
            return messages
    instruction = _compact(row.get("instruction", row.get("prompt", row.get("query", row.get("question", "")))), limit=3000)
    user_input = _compact(row.get("input", row.get("context", "")), limit=3000)
    output = _compact(row.get("output", row.get("answer", row.get("response", row.get("completion", "")))), limit=5000)
    user_text = "\n".join(part for part in [instruction, user_input] if part)
    messages = []
    if user_text:
        messages.append({"role": "user", "content": user_text})
    if output:
        messages.append({"role": "assistant", "content": output})
    return messages


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return _compact(content, limit=5000)


def _last_user_and_assistant(messages: list[dict[str, Any]]) -> tuple[str, str, list[dict[str, Any]]]:
    assistant_index = -1
    for index in range(len(messages) - 1, -1, -1):
        if str(messages[index].get("role", "")).lower() == "assistant":
            assistant_index = index
            break
    if assistant_index <= 0:
        return "", "", []
    user_text = ""
    for index in range(assistant_index - 1, -1, -1):
        if str(messages[index].get("role", "")).lower() == "user":
            user_text = _message_text(messages[index])
            break
    assistant_text = _message_text(messages[assistant_index])
    history = messages[:assistant_index]
    return user_text, assistant_text, history


def _tool_calls_from_value(value: Any) -> list[dict[str, Any]]:
    if value is None or value == "":
        return []
    parsed = _parse_serialized(value)
    if isinstance(parsed, dict):
        if "function" in parsed or "name" in parsed or "arguments" in parsed:
            return [parsed]
        for key in ("tool_calls", "function_calls", "calls"):
            calls = parsed.get(key)
            if isinstance(calls, list):
                return [item for item in calls if isinstance(item, dict)]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _tool_calls_from_text(value: object) -> list[dict[str, Any]]:
    text = str(value or "")
    if not text:
        return []
    calls: list[dict[str, Any]] = []
    for match in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL | re.IGNORECASE):
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            calls.append(parsed)
    for match in re.finditer(
        r"<functioncall>\s*(\{.*?\})(?=\s*(?:FUNCTION RESPONSE|<\|endoftext\|>|\Z))",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            calls.append(parsed)
    if calls:
        return calls
    for match in re.finditer(r"\[([A-Za-z][A-Za-z0-9 _.-]{1,120})\((.*?)\)\]", text, flags=re.DOTALL):
        name = _compact(match.group(1), limit=120)
        arg_text = match.group(2).strip()
        args: dict[str, Any] = {}
        for arg_match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\".*?\"|'.*?'|[^,]+)", arg_text, flags=re.DOTALL):
            raw_value = arg_match.group(2).strip()
            parsed_value = _parse_serialized(raw_value)
            args[arg_match.group(1)] = parsed_value
        calls.append({"name": name, "arguments": args})
    if calls:
        return calls
    stripped = text.strip()
    if stripped.startswith("{") and ("function" in stripped or "arguments" in stripped or "name" in stripped):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return []
        return [parsed] if isinstance(parsed, dict) else []
    return []


def _tool_calls_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    role = str(message.get("role", "")).lower()
    if role == "tool_call":
        content = message.get("content", "")
        if isinstance(content, dict):
            return [content]
        calls = _tool_calls_from_value(content)
        if calls:
            return calls
        return _tool_calls_from_text(content)
    for key in ("tool_calls", "function_call", "function_calls"):
        calls = _tool_calls_from_value(message.get(key))
        if calls:
            return calls
    return _tool_calls_from_text(message.get("content", message.get("value", "")))


def _row_tool_calls(row: dict[str, Any], assistant_message: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    for key in ("tool_calls", "function_calls", "calls", "target_calls"):
        calls = _tool_calls_from_value(row.get(key))
        if calls:
            return calls
    if assistant_message:
        for key in ("tool_calls", "function_call", "function_calls"):
            calls = _tool_calls_from_value(assistant_message.get(key))
            if calls:
                return calls
        calls = _tool_calls_from_text(assistant_message.get("content", assistant_message.get("value", "")))
        if calls:
            return calls
    return []


def _first_tool_request(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    for index, message in enumerate(messages):
        calls = _tool_calls_from_message(message)
        if not calls:
            continue
        user_text = ""
        for prior_index in range(index - 1, -1, -1):
            if str(messages[prior_index].get("role", "")).lower() == "user":
                user_text = _message_text(messages[prior_index])
                break
        return user_text, calls, messages[:index]
    return "", [], []


def _tool_name_and_args(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    function = call.get("function")
    if isinstance(function, str):
        try:
            function = json.loads(function)
        except json.JSONDecodeError:
            function = {"name": function}
    if isinstance(function, dict):
        name = str(function.get("name", call.get("name", "")) or "")
        args = function.get("arguments", call.get("arguments", {}))
    else:
        name = str(call.get("name", call.get("tool_name", call.get("api_name", ""))) or "")
        args = call.get("arguments", call.get("args", call.get("parameters", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"raw": args}
    if not isinstance(args, dict):
        args = {"value": args}
    return _compact(name, limit=120), args


def _extension_id_for_tool(name: str) -> str:
    normalized = name.lower().replace(".", "_").replace("-", "_").replace(" ", "_")
    if any(token in normalized for token in ("calendar", "event", "schedule")):
        return "calendar"
    if any(token in normalized for token in ("email", "mail", "gmail", "outlook")):
        return "email"
    if any(token in normalized for token in ("translate", "translation")):
        return "translator"
    if any(token in normalized for token in ("image", "draw", "photo")):
        return "image_generation"
    if any(token in normalized for token in ("file", "document", "drive", "note")):
        return "files"
    return normalized.split("_")[0] if normalized else "external_tool"


def _decision_text(action: str, content: str, metadata: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {
        "action": action,
        "content": _compact(content, limit=5000),
    }
    if metadata:
        payload["proposal_metadata"] = metadata
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _history_text(history: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in history[-6:]:
        role = _compact(message.get("role", "unknown"), limit=40) or "unknown"
        text = _message_text(message)
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _convert_row(
    row: dict[str, Any],
    *,
    row_index: int,
    source_name: str,
    function_weight: float,
    respond_weight: float,
) -> dict[str, Any] | None:
    messages = _as_messages(row)
    tool_user_text, message_calls, tool_history = _first_tool_request(messages)
    user_text, assistant_text, history = _last_user_and_assistant(messages)
    if tool_user_text and message_calls:
        user_text = tool_user_text
        history = tool_history
    if not user_text:
        return None
    assistant_message = next((message for message in reversed(messages) if str(message.get("role", "")).lower() == "assistant"), None)
    calls = message_calls or _row_tool_calls(row, assistant_message=assistant_message)
    profile_block = (
        f"{AK_PROFILE} PocketPal conversion profile.\n"
        f"{AK_SLOT} {AK_SLOT_NAME}=privacy {AK_SLOT_VALUE}=ask before extensions that use user data\n"
        f"{AK_SLOT} {AK_SLOT_NAME}=style {AK_SLOT_VALUE}=local-first and concise"
    )
    history_block = _history_text(history)
    source_id = _compact(row.get("id", row.get("example_id", f"{source_name}:{row_index}")), limit=200)
    if calls:
        name, args = _tool_name_and_args(calls[0])
        extension_id = _extension_id_for_tool(name)
        capability = name or f"{extension_id}.run"
        encoder_text = (
            f"{AK_CHAT} {AK_EXTENSION} {AK_CAPABILITY} {AK_APPROVAL}\n"
            "PocketPal external agentic conversion example.\n"
            "Map the dataset tool/function call to an installed extension request. "
            "Always request user approval before using an extension with user data.\n"
            f"{profile_block}\n"
            f"{AK_EXTENSION} installed id={extension_id} {AK_CAPABILITY} {capability} approval_policy=always_ask\n"
            f"{AK_CONTEXT} Recent conversation:\n{history_block}\n"
            f"{AK_USER} {user_text}\n"
            "Return a structured decision with action=extension_request."
        )
        decoder_text = _decision_text(
            "extension_request",
            f"Requesting approval to use {extension_id}:{capability}.",
            {
                "task_type": "converted_function_call",
                "source_dataset": source_name,
                "extension_id": extension_id,
                "capability": capability,
                "requires_user_approval": True,
                "tool_args_present": bool(args),
            },
        )
        action = "extension_request"
        task_type = "converted_function_call"
        weight = float(function_weight)
    else:
        if not assistant_text:
            return None
        encoder_text = (
            f"{AK_CHAT} {AK_RESPOND}\n"
            "PocketPal external agentic instruction example.\n"
            "Answer directly when no installed extension or local retrieval is required. "
            "Keep the response concise and compatible with local-first assistant behavior.\n"
            f"{profile_block}\n"
            f"{AK_CONTEXT} Recent conversation:\n{history_block}\n"
            f"{AK_USER} {user_text}\n"
            "Return a structured decision with action=respond."
        )
        decoder_text = _decision_text(
            "respond",
            assistant_text,
            {"task_type": "converted_agentic_instruction", "source_dataset": source_name},
        )
        action = "respond"
        task_type = "converted_agentic_instruction"
        weight = float(respond_weight)
    example_id = hashlib.sha256(f"{encoder_text}\n-->\n{decoder_text}".encode("utf-8")).hexdigest()
    return {
        "example_id": example_id,
        "source_type": "external_agentic_sft",
        "source_id": source_id,
        "task_type": task_type,
        "encoder_text": encoder_text,
        "decoder_text": decoder_text,
        "action": action,
        "source_action": action,
        "extension_capability": "",
        "benchmark_family": "pocketpal_external_agentic",
        "difficulty": task_type,
        "weight": weight,
    }


def _dedupe(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("example_id", "") or "")
        if not key:
            key = hashlib.sha256(
                f"{row.get('encoder_text', '')}\n-->\n{row.get('decoder_text', '')}".encode("utf-8")
            ).hexdigest()
            row["example_id"] = key
        deduped[key] = row
    return sorted(deduped.values(), key=lambda item: str(item.get("source_id", "")))


def _hash_split(key: str, eval_fraction: float) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < max(0.0, min(0.5, float(eval_fraction))) else "train"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def convert_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows_iter: Iterator[dict[str, Any]]
    source_name = str(args.source_name or args.hf_dataset or args.input_path or "external_agentic")
    if str(args.hf_dataset).strip():
        rows_iter = _iter_hf_rows(
            str(args.hf_dataset),
            config=str(args.hf_config),
            split=str(args.hf_split),
            streaming=str(args.hf_streaming) == "1",
            max_rows=int(args.max_rows),
        )
    else:
        rows_iter = _iter_local_rows(Path(args.input_path).expanduser().resolve(), max_rows=int(args.max_rows))
    converted: list[dict[str, Any]] = []
    function_weight = float(getattr(args, "function_weight", 1.1))
    respond_weight = float(getattr(args, "respond_weight", 0.6))
    for index, row in enumerate(rows_iter):
        example = _convert_row(
            row,
            row_index=index,
            source_name=source_name,
            function_weight=function_weight,
            respond_weight=respond_weight,
        )
        if example is not None:
            converted.append(example)
    rows = _dedupe(converted)
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for row in rows:
        split = _hash_split(str(row.get("example_id", "")), float(args.eval_fraction))
        row["split"] = split
        if split == "eval":
            eval_rows.append(row)
        else:
            train_rows.append(row)
    if not train_rows and eval_rows:
        train_rows.append(eval_rows.pop())
    if not eval_rows and len(train_rows) > 1:
        eval_rows.append(train_rows.pop())
    train_path = output_dir / "agentkernel_lite_external_agentic_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_external_agentic_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_external_agentic_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    task_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for row in rows:
        task = str(row.get("task_type", "") or "unknown")
        action = str(row.get("action", "") or "unknown")
        task_counts[task] = task_counts.get(task, 0) + 1
        action_counts[action] = action_counts.get(action, 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": "chat",
        "source_name": source_name,
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "source_counts": {"external_agentic_sft": len(rows)},
        "task_type_counts": dict(sorted(task_counts.items())),
        "target_action_counts": dict(sorted(action_counts.items())),
        "source_action_counts": dict(sorted(action_counts.items())),
        "extension_counts": {},
        "agentkernel_special_tokens": [
            AK_USER,
            AK_CHAT,
            AK_CONTEXT,
            AK_PROFILE,
            AK_SLOT,
            AK_SLOT_NAME,
            AK_SLOT_VALUE,
            AK_EXTENSION,
            AK_CAPABILITY,
            AK_APPROVAL,
            AK_EXTENSION_RESULT,
            AK_RESPOND,
        ],
        "schema": {
            "encoder_text": "PocketPal context/compiler input",
            "decoder_text": "structured decision target",
            "task_type": "converted_function_call or converted_agentic_instruction",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default="", help="Local JSON/JSONL file or directory.")
    parser.add_argument("--hf-dataset", default="", help="Optional Hugging Face dataset name.")
    parser.add_argument("--hf-config", default="", help="Optional Hugging Face dataset config/subset name.")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-streaming", choices=("0", "1"), default="1")
    parser.add_argument("--source-name", default="")
    parser.add_argument("--output-dir", default="artifacts/pocketpal_external_agentic_dataset")
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--function-weight", type=float, default=1.1)
    parser.add_argument("--respond-weight", type=float, default=0.6)
    args = parser.parse_args()
    if not str(args.input_path).strip() and not str(args.hf_dataset).strip():
        raise SystemExit("provide --input-path or --hf-dataset")
    manifest = convert_dataset(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
