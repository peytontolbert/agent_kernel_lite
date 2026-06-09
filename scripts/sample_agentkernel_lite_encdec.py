#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch


DEFAULT_PROMPTS = [
    (
        "Answer from AgentKernel implementation notes. Document: "
        "context_retrieval_public_datasets_audit.md Task: produce a concise "
        "implementation-grounded chat answer."
    ),
    (
        "Answer from AgentKernel implementation notes. Document: "
        "agentkernel_lite_wasm_core.md Task: explain what the browser lite core can do today."
    ),
    (
        "User asks: explain how retrieval evidence should improve a small on-device chat model. "
        "Mode: chat. Use concise grounded answer."
    ),
    (
        "User asks: summarize why browser BitNet export is useful for AgentKernel Lite. "
        "Mode: think. Use concise answer."
    ),
    (
        "Research-grounded answer. Topic: transformer attention and retrieval augmented generation. "
        "Task: answer like a helpful assistant using provided evidence."
    ),
    (
        "<AK_CHAT> <AK_RESPOND> <AK_PROFILE> <AK_SLOT>\n"
        "PocketPal user configuration.\n"
        "<AK_SLOT> <AK_SLOT_NAME>=domain <AK_SLOT_VALUE>=product planning <AK_DOMAIN>\n"
        "<AK_SLOT> <AK_SLOT_NAME>=goal <AK_SLOT_VALUE>=turn rough ideas into short launch tasks <AK_GOAL>\n"
        "<AK_SLOT> <AK_SLOT_NAME>=tone <AK_SLOT_VALUE>=direct and practical <AK_TONE>\n"
        "<AK_SLOT> <AK_SLOT_NAME>=constraint <AK_SLOT_VALUE>=keep responses under 120 words unless asked <AK_CONSTRAINT>\n"
        "<AK_USER> Help me decide what to do next for the app this afternoon."
    ),
]


class ByteTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    vocab_size = 260

    def encode(self, text: str, *, max_length: int) -> list[int]:
        ids = [self.bos_token_id]
        for byte in str(text).encode("utf-8", errors="replace"):
            if len(ids) >= max_length - 1:
                break
            ids.append(int(byte) + 4)
        ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        data: list[int] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id == self.eos_token_id:
                break
            if 4 <= token_id <= 259:
                data.append(token_id - 4)
        return bytes(data).decode("utf-8", errors="replace")


class TokenizersBpe:
    def __init__(self, tokenizer_path: Path) -> None:
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.pad_token_id = int(self.tokenizer.token_to_id("<pad>") or 0)
        self.bos_token_id = int(self.tokenizer.token_to_id("<s>") or 1)
        self.eos_token_id = int(self.tokenizer.token_to_id("</s>") or 2)
        self.unk_token_id = int(self.tokenizer.token_to_id("<unk>") or 3)
        self.vocab_size = int(self.tokenizer.get_vocab_size())

    def encode(self, text: str, *, max_length: int) -> list[int]:
        ids = list(self.tokenizer.encode(str(text), add_special_tokens=True).ids)
        return ids[:max_length]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        clean: list[int] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id == self.eos_token_id:
                break
            clean.append(token_id)
        return self.tokenizer.decode(clean, skip_special_tokens=bool(skip_special_tokens))


class HuggingFaceTokenizer:
    def __init__(self, tokenizer_dir: Path, tokenizer_name: str) -> None:
        from transformers import AutoTokenizer

        source = str(tokenizer_dir) if tokenizer_dir.exists() else tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.bos_token
        self.pad_token_id = int(self.tokenizer.pad_token_id or 0)
        self.bos_token_id = int(self.tokenizer.bos_token_id or self.tokenizer.eos_token_id or self.pad_token_id)
        self.eos_token_id = int(self.tokenizer.eos_token_id or self.bos_token_id)
        self.unk_token_id = int(self.tokenizer.unk_token_id or self.pad_token_id)
        self.vocab_size = int(len(self.tokenizer))

    def encode(self, text: str, *, max_length: int) -> list[int]:
        ids = self.tokenizer(
            str(text),
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
        )["input_ids"]
        return [int(token_id) for token_id in ids]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        clean: list[int] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id == self.eos_token_id:
                break
            clean.append(token_id)
        return str(self.tokenizer.decode(clean, skip_special_tokens=bool(skip_special_tokens)))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _install_paths(repo_root: Path) -> None:
    model_stack = repo_root / "other_repos" / "model-stack"
    for path in (repo_root, model_stack):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _materialize_lazy_modules(model: torch.nn.Module) -> None:
    for module in model.modules():
        ensure_self_attn = getattr(module, "_ensure_self_attn", None)
        if callable(ensure_self_attn):
            ensure_self_attn()


def _load_manifest(bundle_dir: Path) -> dict[str, Any]:
    path = bundle_dir / "agentkernel_lite_encdec_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tokenizer(manifest: dict[str, Any]):
    tokenizer_kind = str(manifest.get("tokenizer_kind", "byte") or "byte").lower()
    tokenizer_dir = Path(str(manifest.get("tokenizer_dir", "") or ""))
    if tokenizer_kind == "byte":
        return ByteTokenizer()
    if tokenizer_kind == "agentkernel-bpe":
        return TokenizersBpe(tokenizer_dir / "tokenizer.json")
    return HuggingFaceTokenizer(tokenizer_dir, str(manifest.get("tokenizer_name", "")))


def _load_prompts(path: Path | None, inline_prompts: list[str] | None = None) -> list[str]:
    prompts = [prompt for prompt in (inline_prompts or []) if str(prompt).strip()]
    if prompts:
        return prompts
    if path is None:
        return DEFAULT_PROMPTS
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            prompts.append(text)
    return prompts or DEFAULT_PROMPTS


def _select_token(
    logits: torch.Tensor,
    generated: list[int],
    *,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    forbidden_ids: set[int],
) -> int:
    logits = logits.float().clone()
    for token_id in forbidden_ids:
        if 0 <= token_id < logits.numel():
            logits[token_id] = -float("inf")
    if repetition_penalty > 1.0:
        for token_id in set(generated):
            if 0 <= token_id < logits.numel():
                logits[token_id] = logits[token_id] / repetition_penalty if logits[token_id] > 0 else logits[token_id] * repetition_penalty
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / max(float(temperature), 1e-4), dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    keep = cdf <= max(0.01, min(1.0, float(top_p)))
    keep[0] = True
    filtered_probs = sorted_probs[keep]
    filtered_idx = sorted_idx[keep]
    filtered_probs = filtered_probs / filtered_probs.sum().clamp_min(1e-12)
    choice = torch.multinomial(filtered_probs, num_samples=1)
    return int(filtered_idx[choice].item())


def _single_token_id(tokenizer, text: str) -> int | None:
    try:
        ids = [
            int(token_id)
            for token_id in tokenizer.encode(str(text), max_length=8)
            if int(token_id)
            not in {
                int(tokenizer.bos_token_id),
                int(tokenizer.eos_token_id),
                int(tokenizer.pad_token_id),
            }
        ]
    except Exception:
        return None
    return ids[0] if len(ids) == 1 else None


def _ak_content_control_ids(tokenizer) -> set[int]:
    control_tokens = [
        "<AK_STRUCTURED>",
        "<AK_ACTION_RESPOND>",
        "<AK_ACTION_ASK_USER>",
        "<AK_ACTION_EXTENSION_REQUEST>",
        "<AK_ACTION_SAVE_MEMORY>",
        "<AK_CONTENT>",
        "<AK_TASK_TYPE>",
        "<AK_INTENT>",
        "<AK_FIELD>",
        "<AK_FIELD_NAME>",
        "<AK_FIELD_VALUE>",
        "<AK_FIELDS>",
        "<AK_FRESHNESS>",
        *[f"<AK_COPY_USER_SOURCE_{index}>" for index in range(1, 25)],
    ]
    ids: set[int] = set()
    for token in control_tokens:
        token_id = _single_token_id(tokenizer, token)
        if token_id is not None:
            ids.add(int(token_id))
    return ids


def _is_complete_json_object(text: str) -> bool:
    raw = str(text or "").strip()
    if not (raw.startswith("{") and raw.endswith("}")):
        return False
    try:
        return isinstance(json.loads(raw), dict)
    except json.JSONDecodeError:
        return False


def _is_complete_ak_decision(text: str) -> bool:
    raw = str(text or "").strip()
    if "<AK_STRUCTURED>" not in raw:
        return False
    has_action = any(
        token in raw
        for token in (
            "<AK_ACTION_RESPOND>",
            "<AK_ACTION_ASK_USER>",
            "<AK_ACTION_EXTENSION_REQUEST>",
            "<AK_ACTION_SAVE_MEMORY>",
        )
    )
    if not has_action:
        return False
    return "<AK_END>" in raw or ("<AK_CONTENT>" in raw and "</AK_CONTENT>" in raw)


def _is_complete_ak_content(text: str) -> bool:
    raw = str(text or "").strip()
    if "<AK_CONTENT>" not in raw:
        return False
    return "<AK_END>" in raw or "</AK_CONTENT>" in raw


def _generate(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    *,
    decoder_prefix: str = "",
    device: torch.device,
    max_encoder_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    keep_special_tokens: bool = False,
    forbid_ak_content_control: bool = False,
) -> str:
    enc_ids = tokenizer.encode(prompt, max_length=max_encoder_tokens)
    prefix_ids: list[int] = []
    if str(decoder_prefix).strip():
        prefix_ids = [
            int(token_id)
            for token_id in tokenizer.encode(str(decoder_prefix), max_length=128)
            if int(token_id)
            not in {
                int(tokenizer.bos_token_id),
                int(tokenizer.eos_token_id),
                int(tokenizer.pad_token_id),
            }
        ]
    dec_ids = [int(tokenizer.bos_token_id), *prefix_ids]
    forbidden = {int(tokenizer.pad_token_id), int(tokenizer.unk_token_id)}
    if bool(forbid_ak_content_control):
        forbidden |= _ak_content_control_ids(tokenizer)
    with torch.no_grad():
        enc = torch.tensor([enc_ids], dtype=torch.long, device=device)
        enc_attention_mask = torch.ones_like(enc, dtype=torch.long, device=device)
        for _ in range(int(max_new_tokens)):
            dec = torch.tensor([dec_ids], dtype=torch.long, device=device)
            logits = model(enc, dec, enc_attention_mask, None)[0, -1]
            next_id = _select_token(
                logits,
                dec_ids[1:],
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                forbidden_ids=forbidden,
            )
            if next_id == int(tokenizer.eos_token_id):
                break
            dec_ids.append(next_id)
            decoded = tokenizer.decode(dec_ids[1:], skip_special_tokens=False)
            if _is_complete_json_object(decoded) or _is_complete_ak_decision(decoded) or _is_complete_ak_content(decoded):
                break
    return tokenizer.decode(dec_ids[1:], skip_special_tokens=not bool(keep_special_tokens))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--prompts-file", default="")
    parser.add_argument("--prompt", action="append", default=[], help="Inline prompt to decode. May be repeated.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-encoder-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--decoder-prefix", default="")
    parser.add_argument("--keep-special-tokens", type=int, choices=(0, 1), default=0)
    parser.add_argument("--forbid-ak-content-control", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    _install_paths(repo_root)

    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    torch.manual_seed(int(args.seed))
    bundle_dir = Path(args.bundle_dir).resolve()
    manifest = _load_manifest(bundle_dir)
    model_dir = Path(str(manifest["model_dir"]))
    config = load_config(str(model_dir))
    tokenizer = _load_tokenizer(manifest)
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    _materialize_lazy_modules(model)
    load_pretrained(model, str(model_dir), strict=True)
    device = torch.device(str(args.device))
    model.to(device).eval()

    prompts_path = Path(args.prompts_file).resolve() if str(args.prompts_file).strip() else None
    prompts = _load_prompts(prompts_path, [str(item) for item in args.prompt])
    print(
        json.dumps(
            {
                "bundle_dir": str(bundle_dir),
                "device": str(device),
                "parameter_count": sum(int(param.numel()) for param in model.parameters()),
                "tokenizer_kind": manifest.get("tokenizer_kind"),
                "tokenizer_vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "repetition_penalty": float(args.repetition_penalty),
            },
            sort_keys=True,
        )
    )
    for index, prompt in enumerate(prompts, start=1):
        output = _generate(
            model,
            tokenizer,
            prompt,
            decoder_prefix=str(args.decoder_prefix),
            device=device,
            max_encoder_tokens=int(args.max_encoder_tokens),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
            keep_special_tokens=bool(args.keep_special_tokens),
            forbid_ak_content_control=bool(args.forbid_ak_content_control),
        )
        print(f"\n=== SAMPLE {index} ===")
        print(f"PROMPT: {prompt}")
        print(output)


if __name__ == "__main__":
    main()
