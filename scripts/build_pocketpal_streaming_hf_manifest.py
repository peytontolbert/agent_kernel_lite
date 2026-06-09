#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_TRAIN_SPECS = [
    "HuggingFaceTB/smol-smoltalk:default:train:summary:1500",
    "allenai/tulu-3-sft-mixture:default:train:summary:1200",
    "microsoft/orca-agentinstruct-1M-v1::train:plan:1500",
    "nvidia/HelpSteer2::train:ranking:1000",
    "nvidia/OpenMathInstruct-2::train:summary:1000",
    "nvidia/When2Call::mcq:plan:800",
]

DEFAULT_EVAL_SPECS = [
    "HuggingFaceTB/smol-smoltalk:default:train:summary:120",
    "allenai/tulu-3-sft-mixture:default:train:summary:80",
    "microsoft/orca-agentinstruct-1M-v1::train:plan:120",
    "nvidia/HelpSteer2::train:ranking:80",
    "nvidia/When2Call::mcq:plan:80",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "tmp/pocketpal_stage94_streaming_quality_manifest"))
    parser.add_argument("--train-stream-spec", action="append", default=[])
    parser.add_argument("--eval-stream-spec", action="append", default=[])
    parser.add_argument("--stream-weight", type=float, default=2.5)
    parser.add_argument("--objective", default="pocketpal_stage94_streaming_quality_curriculum")
    args = parser.parse_args()

    train_specs = args.train_stream_spec or DEFAULT_TRAIN_SPECS
    eval_specs = args.eval_stream_spec or DEFAULT_EVAL_SPECS
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_streaming_manifest",
        "dataset_format": "hf_stream",
        "objective": str(args.objective),
        "manifest_path": str(manifest_path),
        "stream_weight": float(args.stream_weight),
        "train_stream_specs": train_specs,
        "eval_stream_specs": eval_specs,
        "train_examples_estimate": sum(_spec_limit(spec) for spec in train_specs),
        "eval_examples_estimate": sum(_spec_limit(spec) for spec in eval_specs),
        "notes": "Rows are streamed directly from Hugging Face during training; no dataset JSONL snapshot is written.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _spec_limit(spec: str) -> int:
    parts = str(spec).split(":")
    while len(parts) < 5:
        parts.append("")
    try:
        return max(0, int(parts[4] or 1000))
    except ValueError:
        return 1000


if __name__ == "__main__":
    main()
