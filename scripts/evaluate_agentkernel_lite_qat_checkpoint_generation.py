#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from evaluate_agentkernel_lite_generation import DEFAULT_PROBES, _load_probes, _score_probe  # noqa: E402
from sample_agentkernel_lite_encdec import (  # noqa: E402
    _generate,
    _install_paths,
    _load_manifest,
    _load_tokenizer,
    _materialize_lazy_modules,
)


def _csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw or "").split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an AgentKernel Lite QAT training checkpoint before export conversion."
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--manifest-bundle-dir", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-encoder-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--decoder-prefix", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--bitnet-qat-include", default="")
    parser.add_argument("--bitnet-qat-exclude", default="")
    parser.add_argument("--convert", choices=("none", "dense", "quantized"), default="none")
    parser.add_argument("--probes-json", default="")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    _install_paths(repo_root)

    from compress.apply import apply_compression
    from runtime.seq2seq import EncoderDecoderLM

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    manifest_bundle_dir = (
        Path(args.manifest_bundle_dir).expanduser().resolve()
        if str(args.manifest_bundle_dir).strip()
        else bundle_dir
    )
    manifest = _load_manifest(manifest_bundle_dir)
    tokenizer = _load_tokenizer(manifest)
    config_payload = dict(manifest["model_config"])
    from specs.config import ModelConfig

    config = ModelConfig(**config_payload)
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    _materialize_lazy_modules(model)
    apply_compression(
        model,
        quant={
            "scheme": "bitnet_qat",
            "include": _csv(args.bitnet_qat_include) or None,
            "exclude": _csv(args.bitnet_qat_exclude) or None,
        },
    )
    device = torch.device(str(args.device))
    payload = torch.load(Path(args.checkpoint).expanduser().resolve(), map_location=device)
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    model.to(device)
    if str(args.convert) == "dense":
        from train_agentkernel_lite_encdec import _convert_trainable_bitnet_to_dense

        _convert_trainable_bitnet_to_dense(model)
    elif str(args.convert) == "quantized":
        from train_agentkernel_lite_encdec import _convert_trainable_bitnet_to_quantized

        _convert_trainable_bitnet_to_quantized(model)
    model.eval()

    probes_path = Path(args.probes_json).expanduser().resolve() if str(args.probes_json).strip() else None
    probes = _load_probes(probes_path) if probes_path else list(DEFAULT_PROBES)
    results: list[dict[str, Any]] = []
    for probe in probes:
        output = _generate(
            model,
            tokenizer,
            str(probe["prompt"]),
            decoder_prefix=str(args.decoder_prefix),
            device=device,
            max_encoder_tokens=int(args.max_encoder_tokens),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
        )
        results.append(_score_probe(probe, output))
    passed = sum(1 for item in results if item.get("passed"))
    summary = {
        "bundle_dir": str(bundle_dir),
        "manifest_bundle_dir": str(manifest_bundle_dir),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "convert": str(args.convert),
        "device": str(device),
        "probe_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / max(1, len(results)),
        "results": results,
    }
    if str(args.output_json).strip():
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
