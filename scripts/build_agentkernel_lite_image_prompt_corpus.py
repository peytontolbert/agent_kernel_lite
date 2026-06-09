#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_DATASETS = {
    "journeydb": {
        "dataset": "bitmind/JourneyDB",
        "split": "train",
        "columns": ("prompt", "caption", "text"),
    },
    "diffusiondb": {
        "dataset": "poloclub/diffusiondb",
        "split": "train",
        "columns": ("prompt", "text"),
    },
}


DEFAULT_MIXES = {
    "general-hq-v0": [
        {
            "name": "diffusiondb",
            "dataset": "poloclub/diffusiondb",
            "config": "2m_text_only",
            "split": "train",
            "columns": ("prompt", "text"),
            "weight": 4,
        },
        {
            "name": "fal_prompts",
            "dataset": "fal/image-generation-prompts",
            "split": "train",
            "columns": ("prompt", "tags", "text"),
            "weight": 2,
        },
        {
            "name": "diffusion_prompt_styles",
            "dataset": "Pratofeitoo/Images-Diffusion-Prompt-Style",
            "split": "train",
            "columns": ("prompt_text", "prompt", "tags"),
            "weight": 1,
        },
    ],
    "diffusiondb-journeydb-v0": [
        {
            "name": "diffusiondb",
            "dataset": "poloclub/diffusiondb",
            "config": "2m_text_only",
            "split": "train",
            "columns": ("prompt", "text"),
            "weight": 1,
        },
        {
            "name": "journeydb",
            "dataset": "JourneyDB/JourneyDB",
            "split": "train",
            "columns": ("prompt", "caption", "text"),
            "weight": 1,
        },
    ],
}


BLOCKED_TERMS = (
    "nsfw",
    "nude",
    "naked",
    "porn",
    "gore",
    "blood",
    "sexy",
    "explicit",
    "suggestive",
    "cleavage",
    "lingerie",
    "onlyfans",
    "underage",
)


NON_VISUAL_PHRASES = (
    "youre doing it wrong",
    "you're doing it wrong",
    "what are you doing",
    "i don't know",
    "i dont know",
)


def clean_prompt(value: Any, *, min_words: int, max_chars: int) -> str:
    text = str(value or "")
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s--(?:ar|aspect|v|version|seed|model|style|stylize|s|q|quality|chaos|niji)\s+\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s--(?:hd|uplight|upbeta|testp?)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("\u0000", "")
    if not text:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    if args_max_words := getattr(clean_prompt, "max_words", 0):
        words = text.split()
        if len(words) > args_max_words:
            text = " ".join(words[:args_max_words]).strip()
    if len(text.split()) < min_words:
        return ""
    lowered = text.lower()
    if any(term in lowered for term in BLOCKED_TERMS):
        return ""
    if lowered in NON_VISUAL_PHRASES:
        return ""
    if lowered.startswith(("http ", "www.")):
        return ""
    return text


def prompt_from_row(row: dict[str, Any], columns: tuple[str, ...], *, min_words: int, max_chars: int) -> str:
    for column in columns:
        if column in row:
            prompt = clean_prompt(row[column], min_words=min_words, max_chars=max_chars)
            if prompt:
                return prompt
    return ""


def load_hf_dataset(name: str, split: str, streaming: bool, *, config: str = "", trust_remote_code: bool = False):
    try:
        from datasets import load_dataset
    except Exception as error:
        raise RuntimeError("This script requires the Hugging Face datasets package") from error
    kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    if config:
        return load_dataset(name, config, **kwargs)
    return load_dataset(name, **kwargs)


def source_iter(source: dict[str, Any], args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    dataset = load_hf_dataset(
        source["dataset"],
        source.get("split", "train"),
        args.streaming,
        config=source.get("config", ""),
        trust_remote_code=bool(source.get("trust_remote_code", False)),
    )
    shuffle_buffer = int(source.get("shuffle_buffer", args.shuffle_buffer))
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=args.seed + int(source.get("seed_offset", 0)), buffer_size=shuffle_buffer)
    columns = tuple(source.get("columns", ("prompt", "caption", "text")))
    scanned = 0
    for row in dataset:
        scanned += 1
        prompt = prompt_from_row(row, columns, min_words=args.min_words, max_chars=args.max_chars)
        if not prompt:
            continue
        yield {
            "prompt": prompt,
            "source_dataset": source["dataset"],
            "source_name": source.get("name", source["dataset"]),
            "source_index": scanned - 1,
        }


def parse_sources(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sources_json:
        data = json.loads(Path(args.sources_json).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("--sources-json must contain a JSON array")
        return data
    if args.mix:
        if args.mix not in DEFAULT_MIXES:
            raise ValueError(f"unknown --mix {args.mix}; available: {', '.join(sorted(DEFAULT_MIXES))}")
        return [dict(source) for source in DEFAULT_MIXES[args.mix]]
    preset = DEFAULT_DATASETS.get(args.preset, {})
    dataset_name = args.dataset or preset.get("dataset")
    if not dataset_name:
        raise ValueError("--dataset is required when --preset custom")
    return [
        {
            "name": args.preset,
            "dataset": dataset_name,
            "config": args.config,
            "split": args.split or preset.get("split") or "train",
            "columns": tuple(args.columns.split(",")) if args.columns else tuple(preset.get("columns", ("prompt", "caption", "text"))),
            "weight": 1,
        }
    ]


def build_prompts(args: argparse.Namespace) -> None:
    clean_prompt.max_words = max(0, int(args.max_words))
    sources = parse_sources(args)
    expanded_sources: list[dict[str, Any]] = []
    for source in sources:
        expanded_sources.extend([source] * max(1, int(source.get("weight", 1))))
    iterators = [iter(source_iter(source, args)) for source in expanded_sources]
    seen: set[str] = set()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "artifact_kind": "agentkernel_lite_image_prompt_corpus",
        "mix": args.mix,
        "sources": sources,
        "limit": args.limit,
        "streaming": args.streaming,
        "shuffle_buffer": args.shuffle_buffer,
        "seed": args.seed,
        "min_words": args.min_words,
        "max_chars": args.max_chars,
    }
    count = 0
    scanned = 0
    exhausted = [False for _ in iterators]
    with output_path.open("w", encoding="utf-8") as out:
        while not all(exhausted):
            made_progress = False
            for index, iterator in enumerate(iterators):
                if exhausted[index]:
                    continue
                try:
                    record = next(iterator)
                except StopIteration:
                    exhausted[index] = True
                    continue
                except Exception as error:
                    exhausted[index] = True
                    print(
                        json.dumps(
                            {
                                "source_error": {
                                    "source": expanded_sources[index].get("name", expanded_sources[index].get("dataset", "")),
                                    "error": f"{type(error).__name__}: {str(error)[:500]}",
                                }
                            }
                        ),
                        flush=True,
                    )
                    continue
                scanned += 1
                made_progress = True
                prompt = record["prompt"]
                key = prompt.lower()
                if args.dedupe and key in seen:
                    continue
                seen.add(key)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                if count % args.log_every == 0:
                    print(json.dumps({"prompts": count, "scanned": scanned}), flush=True)
                if args.limit and count >= args.limit:
                    exhausted = [True for _ in iterators]
                    break
            if not made_progress:
                continue
    metadata["rows"] = count
    metadata["scanned"] = scanned
    output_path.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "rows": count, "scanned": scanned}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean prompt corpus for Agent Kernel Lite image teacher distillation.")
    parser.add_argument("--preset", choices=("journeydb", "diffusiondb", "custom"), default="journeydb")
    parser.add_argument("--mix", choices=tuple(sorted(DEFAULT_MIXES)), default="")
    parser.add_argument("--sources-json", default="", help="JSON file containing a list of HF source configs.")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--columns", default="")
    parser.add_argument("--output", default="data/vision/prompts/journeydb_prompts_100k.jsonl")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--dedupe", action="store_true", default=True)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=420)
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()
    build_prompts(args)


if __name__ == "__main__":
    main()
