#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _compact(value: object, limit: int = 900) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split())[:limit].strip()


def _prompt(user_text: str, instruction: str, agent_name: str = "User Configured Instruction Agent") -> str:
    return "\n".join(
        [
            "<AK_CHAT> <AK_RESPOND>",
            "<AK_LOOP> <AK_STATE> mode=chat selected_context=0 retrieval=none",
            "Return exactly this decision format: Action: respond, then Content: your direct answer.",
            "You are Agent Kernel Lite running entirely in this browser.",
            "Mode: Chat. Reply like a helpful assistant, but an active agent instruction overrides base chat behavior.",
            "<AK_AGENT_ACTIVE>",
            f"Agent name: {agent_name}",
            f"Agent instruction: {instruction}",
            "Retrieval policy: auto",
            "Tool policy: ask_before_extensions",
            "Action policy: respond_or_ask",
            "The active agent instruction is the primary task contract for this turn. Apply it directly to the user request. Do not answer as the base assistant when an active agent is selected unless the active agent instruction asks for normal assistant chat.",
            "</AK_AGENT_ACTIVE>",
            f"Active agent selected: {agent_name}.",
            f"Follow this agent instruction unless it conflicts with user safety or the current user request: {instruction}",
            "Agent policies: retrieval=auto tools=ask_before_extensions actions=respond_or_ask.",
            "<AK_PROFILE> Active PocketPal agent:",
            f"<AK_SLOT> <AK_SLOT_NAME>=agent_name <AK_SLOT_VALUE>={agent_name}",
            f"<AK_SLOT> <AK_SLOT_NAME>=agent_instruction <AK_SLOT_VALUE>={instruction}",
            "<AK_SLOT> <AK_SLOT_NAME>=agent_retrieval_policy <AK_SLOT_VALUE>=auto",
            "<AK_SLOT> <AK_SLOT_NAME>=agent_tool_policy <AK_SLOT_VALUE>=ask_before_extensions",
            "<AK_SLOT> <AK_SLOT_NAME>=agent_action_policy <AK_SLOT_VALUE>=respond_or_ask",
            "<AK_PROFILE> PocketPal installed tools:",
            "<AK_EXTENSION> installed id=web_search <AK_CAPABILITY> web.search approval_policy=always_ask",
            "<AK_MAX_SOURCES>=5",
            "<AK_CONTEXT> User data pointers:",
            "No saved user data sources.",
            "<AK_PROFILE> User text slots:",
            f"<AK_SLOT> <AK_SLOT_NAME>=SOURCE_TEXT <AK_SLOT_VALUE>={user_text}",
            "<AK_HISTORY> Recent conversation:",
            "No recent conversation.",
            "<AK_CONTEXT> Research context:",
            "No research context was retrieved.",
            "<AK_USER> " + user_text,
            "Return a structured decision that follows the active agent instruction.",
        ]
    )


def _row(source_id: str, instruction: str, user_text: str, assistant_text: str, weight: float = 6.0) -> dict[str, Any]:
    content = _compact(assistant_text, 700)
    decoder = {"action": "respond", "content": content, "proposal_metadata": {"task_type": "external_active_instruction"}}
    encoder_text = _prompt(_compact(user_text, 700), _compact(instruction, 700))
    return {
        "action": "respond",
        "decoder_text": json.dumps(decoder, ensure_ascii=False, sort_keys=True),
        "encoder_text": encoder_text,
        "example_id": hashlib.sha256(f"{source_id}\n{encoder_text}\n{content}".encode()).hexdigest(),
        "source_id": source_id,
        "source_type": "pocketpal_stage7_external_instruction",
        "task_type": "external_active_instruction",
        "weight": float(weight),
    }


def _hf_rows(dataset_name: str, config: str, split: str, max_rows: int) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, config or None, split=split, streaming=True)
    for index, row in enumerate(dataset):
        if max_rows > 0 and index >= max_rows:
            break
        if isinstance(row, dict):
            yield row


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("messages") or row.get("conversations") or row.get("conversation")
    return value if isinstance(value, list) else []


def build_rows(max_rows_per_config: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs = ["smol-rewrite", "smol-summarize", "smol-constraints"]
    for config in configs:
        for index, row in enumerate(_hf_rows("HuggingFaceTB/smoltalk", config, "train", max_rows_per_config)):
            messages = _messages(row)
            system = next((_compact(item.get("content"), 700) for item in messages if item.get("role") == "system"), "")
            if not system:
                continue
            assistant_index = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "assistant"), -1)
            if assistant_index <= 0:
                continue
            user_text = ""
            for item in reversed(messages[:assistant_index]):
                if item.get("role") == "user":
                    user_text = _compact(item.get("content"), 700)
                    break
            assistant_text = _compact(messages[assistant_index].get("content"), 700)
            if len(user_text) < 3 or len(assistant_text) < 3:
                continue
            rows.append(_row(f"stage7_{config}_{index:05d}", system, user_text, assistant_text))
    return rows


def _read_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for key in ("train_dataset_path", "eval_dataset_path"):
        path = Path(str(manifest.get(key, "") or ""))
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
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
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path.resolve()),
        "eval_examples": len(eval_rows),
        "manifest_path": str((output_dir / f"{name}_manifest.json").resolve()),
        "objective": "pocketpal_stage7_external_active_instruction",
        "source_counts": {"pocketpal_stage7_external_instruction": sum(1 for row in rows if row.get("source_type") == "pocketpal_stage7_external_instruction")},
        "total_examples": len(rows),
        "train_dataset_path": str(train_path.resolve()),
        "train_examples": len(train_rows),
    }
    (output_dir / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage7_external_instruction_dataset")
    parser.add_argument("--mixed-output-dir", default="")
    parser.add_argument("--include-manifest", action="append", default=[])
    parser.add_argument("--max-rows-per-config", type=int, default=2000)
    parser.add_argument("--stage7-repeat", type=int, default=1)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    args = parser.parse_args()
    stage7_rows = build_rows(int(args.max_rows_per_config))
    print(json.dumps(write_dataset(stage7_rows, Path(args.output_dir), float(args.eval_fraction), "pocketpal_stage7_external_instruction"), indent=2, sort_keys=True))
    if str(args.mixed_output_dir).strip():
        mixed_rows: list[dict[str, Any]] = []
        for manifest_arg in args.include_manifest:
            mixed_rows.extend(_read_manifest_rows(Path(manifest_arg).expanduser().resolve()))
        repeated: list[dict[str, Any]] = []
        for repeat in range(max(1, int(args.stage7_repeat))):
            for row in stage7_rows:
                next_row = dict(row)
                next_row["source_id"] = f"{row['source_id']}:repeat_{repeat:02d}"
                next_row["example_id"] = hashlib.sha256(f"{next_row['source_id']}\n{row['encoder_text']}\n{row['decoder_text']}".encode()).hexdigest()
                repeated.append(next_row)
        mixed_rows.extend(repeated)
        print(json.dumps(write_dataset(mixed_rows, Path(args.mixed_output_dir), float(args.eval_fraction), "pocketpal_stage7_mixed"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
