#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_STACK = ROOT / "other_repos" / "model-stack"
if str(MODEL_STACK) not in sys.path:
    sys.path.insert(0, str(MODEL_STACK))

from scripts.evaluate_pocketpal_agent_gates import GATES, _decoder_prefix_for_gate
from scripts.sample_agentkernel_lite_encdec import _load_manifest, _load_tokenizer, _materialize_lazy_modules


def _load_model(bundle_dir: Path, device: torch.device):
    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    manifest = _load_manifest(bundle_dir)
    model_dir = Path(str(manifest["model_dir"]))
    config = load_config(str(model_dir))
    tokenizer = _load_tokenizer(manifest)
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    _materialize_lazy_modules(model)
    load_pretrained(model, str(model_dir), strict=True)
    model.to(device).eval()
    return model, tokenizer


def _prefix_ids(tokenizer: Any, prefix: str) -> list[int]:
    ids = tokenizer.encode(prefix, max_length=128)
    special = {int(tokenizer.bos_token_id), int(tokenizer.eos_token_id), int(tokenizer.pad_token_id)}
    return [int(token_id) for token_id in ids if int(token_id) not in special]


def _top_next_tokens(model: torch.nn.Module, tokenizer: Any, prompt: str, prefix: str, device: torch.device, *, top_k: int) -> list[dict[str, Any]]:
    enc_ids = tokenizer.encode(prompt, max_length=1024)
    dec_ids = [int(tokenizer.bos_token_id), *_prefix_ids(tokenizer, prefix)]
    enc = torch.tensor([enc_ids], dtype=torch.long, device=device)
    dec = torch.tensor([dec_ids], dtype=torch.long, device=device)
    enc_attention_mask = torch.ones_like(enc, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(enc, dec, enc_attention_mask, None)[0, -1].float()
        logits[int(tokenizer.pad_token_id)] = -float("inf")
        logits[int(tokenizer.unk_token_id)] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        values, indices = torch.topk(probs, k=max(1, int(top_k)))
    rows: list[dict[str, Any]] = []
    for prob, token_id in zip(values.tolist(), indices.tolist(), strict=False):
        text = tokenizer.decode([int(token_id)])
        rows.append({"token_id": int(token_id), "token": text, "prob": float(prob)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    device = torch.device(str(args.device))
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    model, tokenizer = _load_model(bundle_dir, device)
    reports: list[dict[str, Any]] = []
    for gate in GATES:
        if bool(gate.get("experimental")):
            continue
        prefix = _decoder_prefix_for_gate(gate, enabled=True)
        reports.append(
            {
                "id": str(gate.get("id") or ""),
                "decoder_prefix": prefix,
                "top_next_tokens": _top_next_tokens(
                    model,
                    tokenizer,
                    str(gate["prompt"]),
                    prefix,
                    device,
                    top_k=int(args.top_k),
                ),
            }
        )
    output = {
        "bundle_dir": str(bundle_dir),
        "device": str(device),
        "top_k": int(args.top_k),
        "reports": reports,
    }
    Path(args.output_json).write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_json": str(Path(args.output_json).resolve()), "gates": len(reports)}, sort_keys=True))


if __name__ == "__main__":
    main()
