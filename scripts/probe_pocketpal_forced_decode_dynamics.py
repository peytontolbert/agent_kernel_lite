#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_STACK = ROOT / "other_repos" / "model-stack"
if str(MODEL_STACK) not in sys.path:
    sys.path.insert(0, str(MODEL_STACK))

from scripts.build_pocketpal_agent_gate_repair_dataset import TARGETS
from scripts.evaluate_pocketpal_agent_gates import GATES, _decoder_prefix_for_gate
from scripts.sample_agentkernel_lite_encdec import _load_manifest, _load_tokenizer, _materialize_lazy_modules


ATTRACTOR_CONTENTS = [
    "Hello, I hope you are well.",
    "Could you please send those documents as soon as possible?",
    "- Maria: own launch slides",
    "Follow-Up on Friday Contract Review",
    "Design reviewed the search flow and agreed links should be clickable.",
    "Your launch code is ORBIT-t May TestFlight build.",
]


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


def _content_payload(action: str, content: str, metadata: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {"action": action, "content": content}
    if metadata:
        payload["proposal_metadata"] = metadata
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _continuation_ids(tokenizer: Any, full_text: str, prefix: str) -> list[int]:
    special = {int(tokenizer.bos_token_id), int(tokenizer.eos_token_id), int(tokenizer.pad_token_id)}
    full_ids = [int(x) for x in tokenizer.encode(full_text, max_length=512) if int(x) not in special]
    prefix_ids = [int(x) for x in tokenizer.encode(prefix, max_length=256) if int(x) not in special]
    if prefix_ids and full_ids[: len(prefix_ids)] == prefix_ids:
        return full_ids[len(prefix_ids) :]
    return full_ids


def _trace_sequence(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    prefix: str,
    continuation: str,
    device: torch.device,
    *,
    max_positions: int,
) -> dict[str, Any]:
    enc_ids = tokenizer.encode(prompt, max_length=1024)
    prefix_special = {int(tokenizer.bos_token_id), int(tokenizer.eos_token_id), int(tokenizer.pad_token_id)}
    prefix_ids = [int(x) for x in tokenizer.encode(prefix, max_length=256) if int(x) not in prefix_special]
    target_ids = _continuation_ids(tokenizer, continuation, prefix)
    dec_ids = [int(tokenizer.bos_token_id), *prefix_ids]
    enc = torch.tensor([enc_ids], dtype=torch.long, device=device)
    enc_attention_mask = torch.ones_like(enc, dtype=torch.long, device=device)
    token_reports: list[dict[str, Any]] = []
    nll = 0.0
    top1_matches = 0
    with torch.no_grad():
        for position, target_id in enumerate(target_ids[: max(1, int(max_positions))]):
            dec = torch.tensor([dec_ids], dtype=torch.long, device=device)
            logits = model(enc, dec, enc_attention_mask, None)[0, -1].float()
            log_probs = F.log_softmax(logits, dim=-1)
            prob = float(log_probs[int(target_id)].exp().item())
            top_values, top_indices = torch.topk(torch.softmax(logits, dim=-1), k=5)
            top_tokens = [
                {
                    "token_id": int(token_id),
                    "token": tokenizer.decode([int(token_id)]),
                    "prob": float(value),
                }
                for value, token_id in zip(top_values.tolist(), top_indices.tolist(), strict=False)
            ]
            top1 = int(top_indices[0].item())
            if top1 == int(target_id):
                top1_matches += 1
            token_reports.append(
                {
                    "position": int(position),
                    "target_id": int(target_id),
                    "target_token": tokenizer.decode([int(target_id)]),
                    "target_prob": prob,
                    "target_nll": float(-log_probs[int(target_id)].item()),
                    "top1_match": bool(top1 == int(target_id)),
                    "top_tokens": top_tokens,
                }
            )
            nll += float(-log_probs[int(target_id)].item())
            dec_ids.append(int(target_id))
    denom = max(1, len(token_reports))
    return {
        "tokens_traced": len(token_reports),
        "mean_nll": nll / denom,
        "top1_accuracy": top1_matches / denom,
        "worst_positions": sorted(token_reports, key=lambda item: item["target_nll"], reverse=True)[:8],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-positions", type=int, default=96)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    device = torch.device(str(args.device))
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    model, tokenizer = _load_model(bundle_dir, device)
    reports: list[dict[str, Any]] = []
    for gate in GATES:
        if bool(gate.get("experimental")):
            continue
        gate_id = str(gate.get("id") or "")
        if gate_id not in TARGETS:
            continue
        target = TARGETS[gate_id]
        prefix = _decoder_prefix_for_gate(gate, enabled=True)
        target_payload = _content_payload(str(target["action"]), str(target["content"]), dict(target.get("metadata") or {}))
        target_trace = _trace_sequence(
            model,
            tokenizer,
            str(gate["prompt"]),
            prefix,
            target_payload,
            device,
            max_positions=int(args.max_positions),
        )
        attractor_scores: list[dict[str, Any]] = []
        for content in ATTRACTOR_CONTENTS:
            payload = _content_payload(str(target["action"]), content, dict(target.get("metadata") or {}))
            trace = _trace_sequence(
                model,
                tokenizer,
                str(gate["prompt"]),
                prefix,
                payload,
                device,
                max_positions=int(args.max_positions),
            )
            attractor_scores.append(
                {
                    "content": content,
                    "mean_nll": trace["mean_nll"],
                    "top1_accuracy": trace["top1_accuracy"],
                }
            )
        reports.append(
            {
                "id": gate_id,
                "target_content": str(target["content"]),
                "decoder_prefix": prefix,
                "target_trace": target_trace,
                "attractor_scores": sorted(attractor_scores, key=lambda item: item["mean_nll"])[:5],
            }
        )

    output = {
        "bundle_dir": str(bundle_dir),
        "device": str(device),
        "max_positions": int(args.max_positions),
        "reports": reports,
    }
    Path(args.output_json).write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_json": str(Path(args.output_json).resolve()), "gates": len(reports)}, sort_keys=True))


if __name__ == "__main__":
    main()
