#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--output-bundle", required=True)
    parser.add_argument("--brainstorm-bias-delta", type=float, default=0.0)
    parser.add_argument("--json-bias-delta", type=float, default=0.0)
    args = parser.parse_args()

    source = Path(args.source_bundle).resolve()
    output = Path(args.output_bundle).resolve()
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(source, output, ignore=shutil.ignore_patterns("eval_*.json"))

    model_path = output / "model" / "model.safetensors"
    state = torch.load(model_path, map_location="cpu")
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    bias = state_dict.get("agent_intent_head.bias")
    if bias is None:
        raise SystemExit("missing agent_intent_head.bias")
    # Current intent ids from the corpus manifest.
    bias[13] += float(args.json_bias_delta)
    bias[17] += float(args.brainstorm_bias_delta)
    torch.save(state, model_path)

    manifest_path = output / "agentkernel_lite_encdec_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_path"] = str(manifest_path.resolve())
    manifest["model_dir"] = str((output / "model").resolve())
    manifest["tokenizer_dir"] = str((output / "tokenizer").resolve())
    manifest.setdefault("training_summary", {})["intent_bias_calibration"] = {
        "source_bundle": str(source),
        "json_bias_delta": float(args.json_bias_delta),
        "brainstorm_bias_delta": float(args.brainstorm_bias_delta),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_bundle": str(output), "brainstorm_bias_delta": float(args.brainstorm_bias_delta), "json_bias_delta": float(args.json_bias_delta)}, indent=2))


if __name__ == "__main__":
    main()
