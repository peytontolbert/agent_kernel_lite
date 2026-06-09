#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_TEXTS = [
    "vendor invoice INV-2048 is blocked until finance approves $1,200",
    "meeting moved to 11 and Priya needs the updated slides",
    "Maya needs the design review follow up tomorrow at 11",
    "legal needs one more day on the contract draft for Omar",
    "the prototype link is ready but battery usage is higher than expected",
    "client ACME asked for the Friday report before noon",
    "server deploy C-17 failed because the staging token expired",
    "Nina moved the launch review to Monday at 10",
]


def _prompt(user_text: str, source_texts: list[str]) -> str:
    lines = [
        "<AK_CHAT> <AK_RESPOND> PocketPal source-copy-token example.",
        "<AK_AGENT_ACTIVE>",
        "Agent name: Source Slot Agent",
        "Agent instruction: Return the exact user-provided source text with a short label. Preserve all names, dates, values, and wording.",
        "Retrieval policy: auto",
        "Tool policy: ask_before_extensions",
        "Action policy: respond_or_ask",
        "The active agent instruction is the primary task contract for this turn.",
        "</AK_AGENT_ACTIVE>",
        "<AK_CONTEXT> Saved user data: none",
        "<AK_PROFILE> User text slots: none",
        "<AK_SOURCE_SLOTS>",
        "Use source copy tokens when exact user-provided names, dates, values, links, or wording must be preserved.",
    ]
    for index, text in enumerate(source_texts, start=1):
        kind = "user_text" if index == 1 else "user_span"
        lines.append(f"<AK_COPY_USER_SOURCE_{index}> {kind}: {text}")
    lines.extend(
        [
            "<AK_CONTEXT> Stale selected paper context: Selected paper [P1]: unrelated research paper context.",
            "Use stale paper context only when the current user request asks about that paper or research evidence.",
            f"<AK_USER> {user_text}",
            "Return compact JSON with the correct action and content for the active agent. For exact copying, write the matching <AK_COPY_USER_SOURCE_N> token in content.",
        ]
    )
    return "\n".join(lines)


def _row(source_id: str, user_text: str, source_texts: list[str], content: str, weight: float = 8.0) -> dict[str, Any]:
    decoder = {
        "action": "respond",
        "content": content,
        "proposal_metadata": {"task_type": "source_copy_token"},
    }
    encoder_text = _prompt(user_text, source_texts)
    return {
        "action": "respond",
        "decoder_text": json.dumps(decoder, ensure_ascii=False, sort_keys=True),
        "encoder_text": encoder_text,
        "example_id": hashlib.sha256(f"{source_id}\n{encoder_text}\n{content}".encode()).hexdigest(),
        "source_id": source_id,
        "source_type": "pocketpal_stage4_source_copy",
        "task_type": "source_copy_token",
        "weight": float(weight),
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat in range(220):
        for index, text in enumerate(SOURCE_TEXTS):
            rotated = SOURCE_TEXTS[index:] + SOURCE_TEXTS[:index]
            rows.append(
                _row(
                    source_id=f"stage4_source_copy_primary_{repeat:03d}_{index:02d}",
                    user_text=text,
                    source_texts=rotated[:4],
                    content="Source text: <AK_COPY_USER_SOURCE_1>",
                )
            )
            rows.append(
                _row(
                    source_id=f"stage4_source_copy_label_{repeat:03d}_{index:02d}",
                    user_text=f"echo this exactly: {text}",
                    source_texts=[f"echo this exactly: {text}", text],
                    content="Source text: <AK_COPY_USER_SOURCE_2>",
                    weight=7.0,
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
        "objective": "pocketpal_stage4_source_copy_tokens",
        "schema": {"encoder_text": "PocketPal active agent with source copy tokens", "decoder_text": "compact JSON action decision"},
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
    parser.add_argument("--output-dir", default="tmp/pocketpal_stage4_source_copy_dataset")
    parser.add_argument("--mixed-output-dir", default="")
    parser.add_argument("--include-manifest", action="append", default=[])
    parser.add_argument("--stage4-repeat", type=int, default=1)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    args = parser.parse_args()
    stage4_rows = build_rows()
    manifest = write_dataset(stage4_rows, Path(args.output_dir), float(args.eval_fraction), "pocketpal_stage4_source_copy")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if str(args.mixed_output_dir).strip():
        mixed_rows: list[dict[str, Any]] = []
        for manifest_arg in args.include_manifest:
            mixed_rows.extend(_read_manifest_rows(Path(manifest_arg).expanduser().resolve()))
        repeated_rows: list[dict[str, Any]] = []
        for repeat in range(max(1, int(args.stage4_repeat))):
            for row in stage4_rows:
                next_row = dict(row)
                next_row["source_id"] = f"{row['source_id']}:repeat_{repeat:02d}"
                next_row["example_id"] = hashlib.sha256(f"{next_row['source_id']}\n{row['encoder_text']}\n{row['decoder_text']}".encode()).hexdigest()
                repeated_rows.append(next_row)
        mixed_rows.extend(repeated_rows)
        mixed_manifest = write_dataset(mixed_rows, Path(args.mixed_output_dir), float(args.eval_fraction), "pocketpal_stage4_mixed")
        print(json.dumps(mixed_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
