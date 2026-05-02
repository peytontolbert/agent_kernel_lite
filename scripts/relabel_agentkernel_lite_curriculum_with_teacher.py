#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


def compact(value: object, limit: int = 4000) -> str:
    return " ".join(str(value or "").replace("\r", "\n").split())[:limit].strip()


def parse_action_content(text: str) -> tuple[str, str]:
    match = re.search(r"(?is)\bAction:\s*([a-z_]+)\s*Content:\s*(.*)\Z", text or "")
    if not match:
        return "", compact(text, 1200)
    return match.group(1).strip(), compact(match.group(2), 1400)


def line(action: str, content: str) -> str:
    return f"Action: {action}\nContent: {compact(content, 1200)}"


def post_ollama_chat(
    *,
    host: str,
    model: str,
    system: str,
    user: str,
    timeout: int,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": int(max_tokens),
        },
    }
    request = Request(
        host.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    message = body.get("message", {})
    if not isinstance(message, dict):
        return ""
    return compact(message.get("content", ""), 1400)


def post_vllm_chat(
    *,
    host: str,
    model: str,
    system: str,
    user: str,
    timeout: int,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": int(max_tokens),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = Request(
        host.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    choices = body.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return ""
    return compact(message.get("content", ""), 1400)


def build_prompt(row: dict[str, Any]) -> str:
    action, old_content = parse_action_content(str(row.get("decoder_text", "")))
    return "\n".join(
        [
            "Rewrite the target assistant response for this AgentKernel Lite training example.",
            "",
            "Rules:",
            "- Return only the final answer content, not Action/Content labels.",
            "- Do not include hidden reasoning, analysis, or training commentary.",
            "- Answer like a helpful chat assistant.",
            "- Use only evidence ids that exist in the prompt, such as [1], [2], [3], or [P1].",
            "- Use [P1] only when the prompt has selected_context=1 or an active selected paper.",
            "- For normal chat with retrieval=none, answer directly and do not cite evidence.",
            "- For paper recommendations, cite the best retrieved evidence id and explain why.",
            "- For selected-paper follow-up, explain the selected paper rather than retrieving new papers.",
            "- Do not invent titles or paper claims.",
            "",
            f"Expected action: {action or row.get('action', 'respond')}",
            f"Task type: {row.get('task_type', '')}",
            "",
            "AgentKernel prompt:",
            str(row.get("encoder_text", "")),
            "",
            "Old weak target:",
            old_content,
            "",
            "Final answer content:",
        ]
    )


def valid_teacher_content(row: dict[str, Any], content: str) -> bool:
    value = compact(content, 1400)
    if len(value) < 24:
        return False
    lowered = value.lower()
    forbidden = (
        "action:",
        "content:",
        "thinking process",
        "hidden reasoning",
        "training example",
        "old weak target",
        "i should ",
        "the user asked for no sources",
    )
    if any(item in lowered for item in forbidden):
        return False
    encoder = str(row.get("encoder_text", ""))
    if "[P1]" in value and "selected_context=1" not in encoder and "<AK_SELECTED_PAPER>" not in encoder:
        return False
    return True


def relabel_row(
    row: dict[str, Any],
    *,
    provider: str,
    host: str,
    model: str,
    timeout: int,
    max_tokens: int,
    retries: int,
) -> dict[str, Any]:
    if str(row.get("action", "")).strip() != "respond":
        return {**row, "teacher_provider": "deterministic_keep"}
    system = "You are a precise teacher generating concise supervised targets for a small on-device research assistant."
    user = build_prompt(row)
    last_error = ""
    for attempt in range(max(1, retries + 1)):
        try:
            if provider == "ollama":
                content = post_ollama_chat(
                    host=host,
                    model=model,
                    system=system,
                    user=user,
                    timeout=timeout,
                    max_tokens=max_tokens,
                )
            else:
                content = post_vllm_chat(
                    host=host,
                    model=model,
                    system=system,
                    user=user,
                    timeout=timeout,
                    max_tokens=max_tokens,
                )
            if valid_teacher_content(row, content):
                return {
                    **row,
                    "decoder_text": line("respond", content),
                    "teacher_provider": provider,
                    "teacher_model": model,
                }
            last_error = "invalid teacher content"
        except (OSError, URLError, json.JSONDecodeError, TimeoutError) as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    return {
        **row,
        "teacher_provider": f"{provider}_fallback_original",
        "teacher_model": model,
        "teacher_error": last_error,
    }


def process_file(
    *,
    input_path: Path,
    output_path: Path,
    provider: str,
    host: str,
    model: str,
    timeout: int,
    max_tokens: int,
    retries: int,
    limit: int,
    respond_only: bool,
    concurrency: int,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"rows": 0, "teacher": 0, "fallback": 0, "kept": 0}
    pending: list[dict[str, Any]] = []

    def flush(rows: list[dict[str, Any]], dst: Any) -> None:
        if not rows:
            return
        with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
            futures = [
                executor.submit(
                    relabel_row,
                    row,
                    provider=provider,
                    host=host,
                    model=model,
                    timeout=timeout,
                    max_tokens=max_tokens,
                    retries=retries,
                )
                for row in rows
            ]
            for future in futures:
                new_row = future.result()
                provider_name = str(new_row.get("teacher_provider", ""))
                if provider_name == provider:
                    counts["teacher"] += 1
                elif "fallback" in provider_name:
                    counts["fallback"] += 1
                else:
                    counts["kept"] += 1
                counts["rows"] += 1
                dst.write(json.dumps(new_row, ensure_ascii=False, sort_keys=True) + "\n")
                if counts["rows"] % 25 == 0:
                    print(json.dumps(counts, sort_keys=True), flush=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_text in src:
            if limit > 0 and counts["rows"] >= limit:
                break
            row = json.loads(line_text)
            if respond_only and str(row.get("action", "")) != "respond":
                continue
            pending.append(row)
            if len(pending) >= max(1, int(concurrency)):
                flush(pending, dst)
                pending = []
        flush(pending, dst)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel AgentKernel Lite curriculum targets with a stronger teacher model.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("ollama", "vllm"), default="ollama")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--respond-only", type=int, choices=(0, 1), default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_in = Path(manifest["train_dataset_path"])
    eval_in = Path(manifest["eval_dataset_path"])
    train_out = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_out = output_dir / "agentkernel_lite_encdec_eval.jsonl"

    train_counts = process_file(
        input_path=train_in,
        output_path=train_out,
        provider=args.provider,
        host=args.host,
        model=args.model,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        retries=args.retries,
        limit=int(args.limit),
        respond_only=bool(args.respond_only),
        concurrency=int(args.concurrency),
    )
    eval_limit = max(0, min(int(args.limit), 512)) if int(args.limit) > 0 else 0
    eval_counts = process_file(
        input_path=eval_in,
        output_path=eval_out,
        provider=args.provider,
        host=args.host,
        model=args.model,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        retries=args.retries,
        limit=eval_limit,
        respond_only=bool(args.respond_only),
        concurrency=int(args.concurrency),
    )

    new_manifest = {
        **manifest,
        "manifest_path": str(output_dir / "agentkernel_lite_encdec_dataset_manifest.json"),
        "train_dataset_path": str(train_out),
        "eval_dataset_path": str(eval_out),
        "teacher_provider": args.provider,
        "teacher_model": args.model,
        "teacher_train_counts": train_counts,
        "teacher_eval_counts": eval_counts,
        "total_examples": train_counts["rows"] + eval_counts["rows"],
        "train_examples": train_counts["rows"],
        "eval_examples": eval_counts["rows"],
    }
    Path(new_manifest["manifest_path"]).write_text(json.dumps(new_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(new_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
