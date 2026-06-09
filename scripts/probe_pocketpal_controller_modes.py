#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any

import torch
import torch.nn.functional as F

class CaptureSpec:
    def __init__(self, move_to_cpu: bool = True, dtype: torch.dtype | None = None) -> None:
        self.move_to_cpu = move_to_cpu
        self.dtype = dtype


class _ActivationCache:
    def __init__(self, spec: CaptureSpec) -> None:
        self.spec = spec
        self.store: dict[str, torch.Tensor] = {}

    def put(self, key: str, value: torch.Tensor) -> None:
        value = value.detach().clone()
        if self.spec.dtype is not None:
            value = value.to(self.spec.dtype)
        if self.spec.move_to_cpu:
            value = value.cpu()
        self.store[key] = value

    def items(self):
        return self.store.items()


class ActivationTracer:
    def __init__(self, model: torch.nn.Module, *, spec: CaptureSpec) -> None:
        self.model = model
        self.spec = spec
        self.names: list[str] = []
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.cache = _ActivationCache(spec)

    def add_modules(self, names: list[str]) -> None:
        self.names.extend(names)

    def trace(self):
        tracer = self

        class _Context:
            def __enter__(self):
                modules = dict(tracer.model.named_modules())
                for name in tracer.names:
                    module = modules.get(name)
                    if module is None:
                        continue

                    def hook(_module, _inputs, output, *, key=name):
                        if isinstance(output, torch.Tensor):
                            tracer.cache.put(key, output)
                        elif isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
                            tracer.cache.put(key, output[0])

                    tracer.handles.append(module.register_forward_hook(hook))
                return tracer.cache

            def __exit__(self, _exc_type, _exc, _tb):
                for handle in tracer.handles:
                    handle.remove()
                tracer.handles.clear()

        return _Context()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_cases(gates: Any) -> list[dict[str, str]]:
    return [
        {
            "id": "quick_dinner_planner",
            "expected": "plan",
            "prompt": gates._agent_prompt(
                name="Quick Dinner Planner",
                instruction="Turn the user's ingredients and constraints into a concise dinner plan. Do not ask unless a required constraint is missing.",
                user_text="I have chicken, rice, broccoli, and 30 minutes. Make a simple dinner plan.",
                text_slots={"SOURCE_TEXT": "I have chicken, rice, broccoli, and 30 minutes. Make a simple dinner plan."},
            ),
        },
        {
            "id": "meeting_action_extractor",
            "expected": "action_items",
            "prompt": gates._agent_prompt(
                name="Action Item Extractor",
                instruction="Extract concrete action items from the user's text as concise bullets. Preserve owners and deadlines.",
                user_text="Maria owns the launch slides, Devin fixes the login bug by Thursday, and I will send the customer email tomorrow.",
                text_slots={
                    "SOURCE_TEXT": "Maria owns the launch slides, Devin fixes the login bug by Thursday, and I will send the customer email tomorrow."
                },
            ),
        },
        {
            "id": "friendly_tone_rewriter",
            "expected": "rewrite",
            "prompt": gates._agent_prompt(
                name="Friendly Tone Agent",
                instruction="Rewrite the user's text in a friendly, warm tone while preserving the meaning.",
                user_text="Your update is late and this is blocking my release.",
                text_slots={"SOURCE_TEXT": "Your update is late and this is blocking my release."},
            ),
        },
        {
            "id": "spanish_translator",
            "expected": "translation",
            "prompt": gates._agent_prompt(
                name="Spanish Translator",
                instruction="Translate the user's English text into Spanish. Do not answer the text.",
                user_text="Please send me the invoice tomorrow morning.",
                text_slots={"SOURCE_TEXT": "Please send me the invoice tomorrow morning."},
            ),
        },
        {
            "id": "web_search_demand",
            "expected": "web_search",
            "prompt": gates._web_agent_prompt(user_text="search the web for current iPhone 17 release rumors"),
        },
    ]


def _classify(parsed: dict[str, Any] | None, raw: str) -> str:
    action = str((parsed or {}).get("action", "") or "")
    content = str((parsed or {}).get("content", "") or raw or "")
    metadata = (parsed or {}).get("proposal_metadata", {})
    if action == "extension_request":
        return "web_search"
    if action == "ask_user":
        return "ask_user"
    lower = content.lower()
    if "what would you like help" in lower or "it's going well" in lower or "i'm doing well" in lower:
        return "casual_collapse"
    if lower.startswith("-") or "\n-" in lower:
        return "bullet"
    if "[[source_text]]" in lower or lower.startswith("source text:"):
        return "source_echo"
    if "[[data_context]]" in lower or "saved data" in lower:
        return "saved_data"
    if "could you please" in lower or "thank you" in lower or "hope you are well" in lower:
        return "rewrite"
    if any(word in lower for word in ("hola", "por favor", "factura", "manana", "mañana")):
        return "translation"
    if isinstance(metadata, dict) and "task_type" in metadata:
        return f"other:{metadata.get('task_type')}"
    return "other"


def _encode_prompt(model: torch.nn.Module, tokenizer: Any, prompt: str, max_tokens: int, device: torch.device) -> torch.Tensor:
    ids = tokenizer.encode(prompt, max_length=max_tokens)
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    padding = (tensor != int(getattr(tokenizer, "pad_token_id", 0))).to(dtype=torch.float32, device=device)
    with torch.inference_mode():
        return model.encode_pooled(tensor, padding).detach().float().cpu()


def _trace_norms(model: torch.nn.Module, tokenizer: Any, prompt: str, max_tokens: int, device: torch.device) -> dict[str, float]:
    if ActivationTracer is None or CaptureSpec is None:
        return {}
    ids = tokenizer.encode(prompt, max_length=max_tokens)
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    padding = (tensor != int(getattr(tokenizer, "pad_token_id", 0))).to(dtype=torch.float32, device=device)
    tracer = ActivationTracer(model, spec=CaptureSpec(move_to_cpu=True, dtype=torch.float32))
    names = [f"encoder.{index}" for index in range(len(getattr(model, "encoder", [])))]
    tracer.add_modules(names)
    with torch.inference_mode():
        with tracer.trace() as cache:
            _ = model.encode(tensor, padding)
    norms: dict[str, float] = {}
    for key, value in cache.items():
        if isinstance(value, torch.Tensor):
            norms[f"{key}.norm"] = float(value.float().norm(dim=-1).mean().item())
            norms[f"{key}.std"] = float(value.float().std(unbiased=False).item())
    return norms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-encoder-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    args = parser.parse_args()

    repo = _repo_root()
    sampler = _load_module(repo / "scripts" / "sample_agentkernel_lite_encdec.py", "sample_agentkernel_lite_encdec")
    gates = _load_module(repo / "scripts" / "evaluate_pocketpal_agent_gates.py", "evaluate_pocketpal_agent_gates")
    sampler._install_paths(repo)

    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    manifest = sampler._load_manifest(bundle_dir)
    config = load_config(str(manifest["model_dir"]))
    tokenizer = sampler._load_tokenizer(manifest)
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    sampler._materialize_lazy_modules(model)
    load_pretrained(model, str(manifest["model_dir"]), strict=True)
    device = torch.device(str(args.device))
    model.to(device).eval()

    cases = _default_cases(gates)
    pooled: list[torch.Tensor] = []
    results: list[dict[str, Any]] = []
    for case in cases:
        raw = sampler._generate(
            model,
            tokenizer,
            case["prompt"],
            decoder_prefix="",
            device=device,
            max_encoder_tokens=int(args.max_encoder_tokens),
            max_new_tokens=int(args.max_new_tokens),
            temperature=0.0,
            top_p=0.9,
            repetition_penalty=1.0,
        )
        parsed = gates._parse_decision(raw)
        mode = _classify(parsed, raw)
        pooled.append(_encode_prompt(model, tokenizer, case["prompt"], int(args.max_encoder_tokens), device))
        results.append(
            {
                "id": case["id"],
                "expected": case["expected"],
                "mode": mode,
                "passed": mode == case["expected"] or (case["expected"] == "action_items" and mode == "bullet"),
                "output": raw,
                "parsed": parsed,
                "trace": _trace_norms(model, tokenizer, case["prompt"], int(args.max_encoder_tokens), device),
            }
        )

    matrix: list[list[float]] = []
    if pooled:
        features = torch.cat(pooled, dim=0)
        sims = F.cosine_similarity(features[:, None, :], features[None, :, :], dim=-1)
        matrix = [[round(float(value), 4) for value in row] for row in sims.tolist()]
    summary = {
        "bundle_dir": str(bundle_dir),
        "passed": sum(1 for row in results if row["passed"]),
        "total": len(results),
        "mode_counts": {mode: sum(1 for row in results if row["mode"] == mode) for mode in sorted({row["mode"] for row in results})},
        "ids": [row["id"] for row in results],
        "encoder_pooled_cosine": matrix,
        "results": results,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if str(args.output_json).strip():
        Path(args.output_json).expanduser().resolve().write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
