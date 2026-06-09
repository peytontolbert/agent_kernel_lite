#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_pocketpal_teacher_agent_curriculum import build_agent_specs
from scripts.generate_pocketpal_teacher_agent_curriculum import _encoder
from scripts.generate_pocketpal_teacher_agent_curriculum import _parse_teacher
from scripts.generate_pocketpal_teacher_agent_curriculum import _teacher_prompt
from scripts.generate_pocketpal_teacher_agent_curriculum import USER_TEXTS
from scripts.pocketpal_source_slots import compile_source_slots, pointerize_exact_text, source_slot_metadata


EXTRA_USER_TEXTS = [
    "please make this shorter: the launch is still blocked because design has not sent the updated screenshots",
    "summarize this: support saw four login failures after the latest build and two users recovered after reinstalling",
    "extract the owner deadline and blocker from this: Priya owns onboarding copy by Tuesday but legal approval is missing",
    "classify this ticket: user cannot reset password because the recovery email never arrives",
    "turn this into a plan: finish the TestFlight build and verify the active agent flow",
    "translate to Spanish: the meeting moved to tomorrow morning",
    "what should I do next if the app crashes after enabling the rewrite agent?",
    "make this professional: hey alex send the mockups today because engineering is waiting",
    "how are you doing today?",
    "rewrite it",
]


def _chat_with_curl(endpoint: str, model: str, prompt: str, *, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return only valid JSON with keys action and content. "
                    "Use action respond or ask_user. Follow the active agent instruction exactly. "
                    "Do not include markdown or chain of thought."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.15,
        "max_tokens": 260,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    command = [
        "curl",
        "-sS",
        "--max-time",
        str(max(1, int(timeout))),
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(payload),
        endpoint.rstrip("/") + "/chat/completions",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=max(2, int(timeout) + 5))
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or f"curl exited {completed.returncode}").strip())
    parsed = json.loads(completed.stdout)
    return str(parsed["choices"][0]["message"].get("content") or "")


def generate(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(int(args.seed))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_agent_specs()
    user_texts = USER_TEXTS + EXTRA_USER_TEXTS
    rows: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = int(args.max_examples) * 4
    while len(rows) < int(args.max_examples) and attempts < max_attempts:
        attempts += 1
        agent_name, instruction = rng.choice(specs)
        user_text = rng.choice(user_texts)
        prompt = _teacher_prompt(agent_name, instruction, user_text)
        try:
            parsed = _parse_teacher(_chat_with_curl(str(args.endpoint), str(args.model), prompt, timeout=int(args.timeout)))
        except Exception as exc:
            print(json.dumps({"event": "teacher_error", "attempt": attempts, "error": str(exc)[:500]}), flush=True)
            continue
        if parsed is None:
            continue
        source_slots = compile_source_slots(user_text=user_text, max_slots=8)
        content = pointerize_exact_text(parsed["content"], source_slots)
        encoder_text = _encoder(agent_name, instruction, user_text)
        decoder_text = json.dumps(
            {
                "action": parsed["action"],
                "content": content,
                "proposal_metadata": {"task_type": "teacher_user_agent", **source_slot_metadata(source_slots)},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        example_id = hashlib.sha256(f"{encoder_text}\n-->\n{decoder_text}".encode("utf-8")).hexdigest()
        if any(row["example_id"] == example_id for row in rows):
            continue
        rows.append(
            {
                "example_id": example_id,
                "source_id": f"teacher_user_agent_curl:{len(rows):06d}",
                "source_type": "pocketpal_teacher_user_agent_curl",
                "task_type": "teacher_user_agent",
                "action": parsed["action"],
                "encoder_text": encoder_text,
                "decoder_text": decoder_text,
                "weight": float(args.weight),
            }
        )
        if len(rows) % 25 == 0:
            print(json.dumps({"event": "generated", "rows": len(rows), "attempts": attempts}), flush=True)

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for row in rows:
        bucket = int(hashlib.sha256(row["example_id"].encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        row["split"] = "eval" if bucket < float(args.eval_fraction) else "train"
        (eval_rows if row["split"] == "eval" else train_rows).append(row)
    if not eval_rows and len(train_rows) > 1:
        eval_rows.append(train_rows.pop())

    train_path = output_dir / "pocketpal_teacher_user_agent_train.jsonl"
    eval_path = output_dir / "pocketpal_teacher_user_agent_eval.jsonl"
    for path, split_rows in ((train_path, train_rows), (eval_path, eval_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    action_counts: dict[str, int] = {}
    for row in rows:
        action_counts[str(row["action"])] = action_counts.get(str(row["action"]), 0) + 1
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": "pocketpal_teacher_user_agent_curl",
        "manifest_path": str(output_dir / "pocketpal_teacher_user_agent_manifest.json"),
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "source_counts": {"pocketpal_teacher_user_agent_curl": len(rows)},
        "target_action_counts": dict(sorted(action_counts.items())),
        "task_type_counts": {"teacher_user_agent": len(rows)},
    }
    Path(manifest["manifest_path"]).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=120)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--weight", type=float, default=3.0)
    print(json.dumps(generate(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
