#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _install_model_stack_path(repo_root: Path) -> None:
    model_stack_root = repo_root / "other_repos" / "model-stack"
    if str(model_stack_root) not in sys.path:
        sys.path.insert(0, str(model_stack_root))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_patterns(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _materialize_lazy_modules(model: torch.nn.Module) -> None:
    for module in model.modules():
        ensure_self_attn = getattr(module, "_ensure_self_attn", None)
        if callable(ensure_self_attn):
            ensure_self_attn()


def _tokenizer_info(tokenizer_dir: Path) -> dict[str, Any]:
    config_path = tokenizer_dir / "tokenizer_config.json"
    config = _read_json(config_path) if config_path.exists() else {}
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    source_kind = str(config.get("tokenizer_kind", "agentkernel-bpe") or "agentkernel-bpe")
    browser_kind = "agentkernel-bpe" if "bpe" in source_kind.lower() else source_kind
    return {
        "kind": browser_kind,
        "source_kind": source_kind,
        "path": "tokenizer/tokenizer.json" if tokenizer_path.exists() else "tokenizer/tokenizer_config.json",
        "config_path": "tokenizer/tokenizer_config.json",
        "pad_token_id": int(config.get("pad_token_id", 0) or 0),
        "bos_token_id": int(config.get("bos_token_id", 1) or 1),
        "eos_token_id": int(config.get("eos_token_id", 2) or 2),
        "unk_token_id": int(config.get("unk_token_id", 3) or 3),
        "vocab_size": int(config.get("vocab_size", 0) or 0),
    }


def _copy_tokenizer(tokenizer_dir: Path, browser_dir: Path) -> dict[str, Any]:
    target_dir = browser_dir / "tokenizer"
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in tokenizer_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, target_dir / path.name)
    return _tokenizer_info(tokenizer_dir)


def _write_readme(*, browser_dir: Path, source_bundle: Path, model_manifest: dict[str, Any]) -> None:
    parameter_count = int(model_manifest.get("parameter_count", 0) or 0)
    training = model_manifest.get("training_summary", {})
    eval_history = training.get("eval_history", []) if isinstance(training, dict) else []
    final_eval = None
    if eval_history:
        final_eval = eval_history[-1].get("eval_loss")
    readme = [
        "---",
        "library_name: model-stack",
        "tags:",
        "- agentkernel-lite",
        "- bitnet",
        "- webgpu",
        "- encoder-decoder",
        "---",
        "",
        "# AgentKernel Lite Encoder-Decoder Browser BitNet",
        "",
        "Self-contained browser BitNet export for the AgentKernel Lite chat model.",
        "",
        f"- Source bundle: `{source_bundle}`",
        f"- Parameters before BitNet packing: `{parameter_count}`",
        f"- Final eval loss: `{final_eval}`",
        "- Browser entrypoint: `manifest.json`",
        "- Runtime: Model Stack browser BitNet WebGPU encoder-decoder with packed BitNet WASM fallback",
        "- Tokenizer: AgentKernel byte-level BPE attached under `tokenizer/`",
        "",
        "Web app route after uploading this directory to Hugging Face:",
        "",
        "```text",
        "?modelStackManifest=https://huggingface.co/<org>/<repo>/resolve/main/manifest.json",
        "```",
        "",
        "Serving notes: WebGPU is used when available; Safari or other no-WebGPU browsers use the packed BitNet WASM fallback. Large model files are fetched by the browser and cached by the app.",
        "",
    ]
    (browser_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")


def _refresh_modelcard(browser_dir: Path, manifest_path: Path, browser_manifest: dict[str, Any]) -> None:
    modelcard_path = browser_dir / "modelcard.json"
    modelcard = _read_json(modelcard_path) if modelcard_path.exists() else {}
    modelcard.update(
        {
            "format": "browser-bitnet",
            "manifest": manifest_path.name,
            "layer_count": len(browser_manifest.get("layers", []) or []),
            "dense_tensor_count": len(browser_manifest.get("dense_tensors", {}) or {}),
            "sha256": _sha256(manifest_path),
            "tokenizer": browser_manifest.get("tokenizer", {}),
        }
    )
    _write_json(modelcard_path, modelcard)


def export_browser_bitnet(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    _install_model_stack_path(repo_root)

    from compress.apply import apply_compression
    from export.browser_bitnet import export_browser_bitnet_bundle
    from runtime.checkpoint import load_pretrained
    from runtime.seq2seq import EncoderDecoderLM
    from specs.config import ModelConfig

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    bundle_manifest_path = Path(args.bundle_manifest).expanduser().resolve() if args.bundle_manifest else (
        bundle_dir / "agentkernel_lite_encdec_manifest.json"
    )
    bundle_manifest = _read_json(bundle_manifest_path)
    model_dir = Path(str(bundle_manifest.get("model_dir", "") or bundle_dir / "model")).expanduser().resolve()
    tokenizer_dir = Path(str(bundle_manifest.get("tokenizer_dir", "") or bundle_dir / "tokenizer")).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else bundle_dir / "browser_bitnet"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig(**dict(bundle_manifest["model_config"]))
    model = EncoderDecoderLM(cfg, tie_embeddings=True, vocab_size=int(cfg.vocab_size))
    _materialize_lazy_modules(model)
    load_pretrained(model, str(model_dir), strict=True)
    model.eval()
    device = torch.device(str(args.device))
    model.to(device)

    quant_include = _csv_patterns(str(args.quant_include))
    quant_exclude = _csv_patterns(str(args.quant_exclude))
    dense_float32_include = _csv_patterns(str(args.dense_float32_include))
    quant_summary: dict[str, Any] = {"quant": {"scheme": "none", "num": 0, "modules": []}}
    if bool(args.quantize_bitnet):
        quant_summary = apply_compression(
            model,
            quant={
                "scheme": "bitnet",
                "include": quant_include or None,
                "exclude": quant_exclude or None,
                "weight_opt": "none",
                "activation_quant": "none",
                "spin": False,
                "spin_random": True,
                "spin_seed": int(args.spin_seed),
            },
        )
    manifest_path = export_browser_bitnet_bundle(
        model.cpu().eval(),
        output_dir,
        model_cfg=cfg,
        max_seq_len=int(args.max_seq_len),
        dense_dtype=str(args.dense_dtype),
        dense_float32_patterns=dense_float32_include,
    )

    browser_manifest = _read_json(manifest_path)
    browser_manifest["tokenizer"] = _copy_tokenizer(tokenizer_dir, output_dir)
    browser_manifest["compression"] = {
        "quantization": quant_summary.get("quant", {}),
        "quantize_bitnet": bool(args.quantize_bitnet),
        "quant_include": quant_include,
        "quant_exclude": quant_exclude,
        "dense_dtype": str(args.dense_dtype),
        "dense_float32_include": dense_float32_include,
    }
    browser_manifest["agentkernel_lite"] = {
        "source_bundle_manifest_path": str(bundle_manifest_path),
        "source_model_dir": str(model_dir),
        "source_tokenizer_dir": str(tokenizer_dir),
        "model_family": str(bundle_manifest.get("model_family", "agentkernel_lite_encdec_v1")),
        "parameter_count": int(bundle_manifest.get("parameter_count", 0) or 0),
        "chat_contract": bundle_manifest.get("chat_contract", {}),
    }
    _write_json(manifest_path, browser_manifest)
    _write_readme(browser_dir=output_dir, source_bundle=bundle_dir, model_manifest=bundle_manifest)
    _refresh_modelcard(output_dir, manifest_path, browser_manifest)

    export_summary = {
        "artifact_kind": "agentkernel_lite_browser_bitnet_export",
        "format": browser_manifest.get("format"),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "source_bundle_manifest_path": str(bundle_manifest_path),
        "source_model_dir": str(model_dir),
        "source_tokenizer_dir": str(tokenizer_dir),
        "device": str(device),
        "max_seq_len": int(args.max_seq_len),
        "dense_dtype": str(args.dense_dtype),
        "dense_float32_include": dense_float32_include,
        "dense_tensor_count": len(browser_manifest.get("dense_tensors", {}) or {}),
        "layer_count": len(browser_manifest.get("layers", []) or []),
        "quantization": quant_summary.get("quant", {}),
        "quantize_bitnet": bool(args.quantize_bitnet),
        "quant_include": quant_include,
        "quant_exclude": quant_exclude,
        "size_bytes": _directory_size(output_dir),
        "model": asdict(cfg),
        "tokenizer": browser_manifest.get("tokenizer", {}),
    }
    _write_json(output_dir / "agentkernel_lite_browser_bitnet_export.json", export_summary)
    export_summary["size_bytes"] = _directory_size(output_dir)
    _write_json(output_dir / "agentkernel_lite_browser_bitnet_export.json", export_summary)

    bundle_manifest["browser_bitnet_manifest_path"] = str(manifest_path)
    bundle_manifest["browser_bitnet"] = export_summary
    training = bundle_manifest.setdefault("training_summary", {})
    if isinstance(training, dict):
        training["browser_bitnet_exported"] = True
    _write_json(bundle_manifest_path, bundle_manifest)

    webapp_model_dir = Path(str(args.webapp_model_dir)).expanduser().resolve() if args.webapp_model_dir else None
    if webapp_model_dir is not None:
        webapp_model_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(output_dir, webapp_model_dir, dirs_exist_ok=True)
        export_summary["webapp_model_dir"] = str(webapp_model_dir)
        export_summary["webapp_manifest_path"] = str(webapp_model_dir / "manifest.json")
        _write_json(output_dir / "agentkernel_lite_browser_bitnet_export.json", export_summary)
        _write_json(webapp_model_dir / "agentkernel_lite_browser_bitnet_export.json", export_summary)

    return export_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an AgentKernel Lite bundle as browser BitNet.")
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--bundle-manifest", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--webapp-model-dir", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--dense-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--dense-float32-include", default="")
    parser.add_argument("--spin-seed", type=int, default=0)
    parser.add_argument("--quantize-bitnet", type=int, choices=(0, 1), default=1)
    parser.add_argument("--quant-include", default="")
    parser.add_argument("--quant-exclude", default="")
    args = parser.parse_args()
    summary = export_browser_bitnet(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
