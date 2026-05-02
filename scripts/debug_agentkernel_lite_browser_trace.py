#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent_kernel.modeling.model_python import maybe_reexec_under_model_python

maybe_reexec_under_model_python(require_full_torch=True)

import torch
import torch.nn.functional as F

from sample_agentkernel_lite_encdec import (
    _install_paths,
    _load_manifest,
    _load_tokenizer,
    _materialize_lazy_modules,
)


NODE_HELPER = r"""
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";

function contentType(url) {
  if (String(url).endsWith(".wasm")) return "application/wasm";
  if (String(url).endsWith(".json")) return "application/json";
  if (String(url).endsWith(".js") || String(url).endsWith(".mjs")) return "text/javascript";
  return "application/octet-stream";
}

const nativeFetch = globalThis.fetch.bind(globalThis);
globalThis.fetch = async function debugFetch(resource, init) {
  const raw = resource instanceof URL ? resource.href : String(resource);
  if (raw.startsWith("file:")) {
    const data = await readFile(fileURLToPath(raw));
    return new Response(data, { status: 200, headers: { "content-type": contentType(raw) } });
  }
  return nativeFetch(resource, init);
};

const inputPath = process.argv[2];
const outputPath = process.argv[3];
if (!inputPath || !outputPath) throw new Error("usage: node trace_node.mjs input.json output.json");
const input = JSON.parse(await readFile(inputPath, "utf8"));
const runtimeModule = await import(pathToFileURL(input.runtimeJs).href);
const runtime = await runtimeModule.BitNetEncoderDecoderWASM.fromManifestUrl(pathToFileURL(input.browserManifest).href, {
  layerConcurrency: 1,
  progress: () => {},
});
const trace = await runtime.debugTrace(input.encIds, input.decIds);
await writeFile(outputPath, JSON.stringify(trace));
"""


def _compare(left: list[float], right: list[float]) -> dict[str, float | bool]:
    if len(left) != len(right):
        return {"shape_match": False, "max_abs": math.inf, "mean_abs": math.inf, "rmse": math.inf, "cosine": 0.0}
    max_abs = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for a, b in zip(left, right):
        da = float(a)
        db = float(b)
        diff = da - db
        abs_diff = abs(diff)
        max_abs = max(max_abs, abs_diff)
        sum_abs += abs_diff
        sum_sq += diff * diff
        dot += da * db
        left_sq += da * da
        right_sq += db * db
    count = max(len(left), 1)
    cosine = dot / math.sqrt(max(left_sq * right_sq, 1e-24))
    return {
        "shape_match": True,
        "max_abs": max_abs,
        "mean_abs": sum_abs / count,
        "rmse": math.sqrt(sum_sq / count),
        "cosine": cosine,
    }


def _record(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().float().cpu().reshape(-1).tolist()
    return {
        "name": name,
        "shape": list(tensor.shape),
        "len": len(values),
        "values": values,
    }


def _python_trace(model: torch.nn.Module, enc_ids: list[int], dec_ids: list[int], device: torch.device) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    decoder_outputs: dict[str, torch.Tensor] = {}
    hooks = []
    for index, block in enumerate(model.decoder):
        hooks.append(block.register_forward_hook(lambda _m, _i, out, index=index: decoder_outputs.__setitem__(f"decoder.{index}", out)))
    try:
        from runtime.blocks import apply_native_norm, prepare_encoder_attention_mask

        with torch.no_grad():
            enc = torch.tensor([enc_ids], dtype=torch.long, device=device)
            dec = torch.tensor([dec_ids], dtype=torch.long, device=device)
            enc_attention_mask = torch.ones_like(enc, dtype=torch.long, device=device)
            x = F.embedding(enc, model.enc_embed.weight, padding_idx=model.enc_embed.padding_idx)
            if model.enc_pos_embed is not None:
                positions = torch.arange(enc.shape[1], device=device).unsqueeze(0).expand(enc.shape[0], enc.shape[1])
                x = x + F.embedding(positions, model.enc_pos_embed.weight, padding_idx=model.enc_pos_embed.padding_idx)
            traces.append(_record("enc_embed", x[0]))
            for index, block in enumerate(model.encoder):
                attn_mask = prepare_encoder_attention_mask(
                    x,
                    enc_attention_mask,
                    num_heads=model.cfg.n_heads,
                    rpb_table=getattr(block, "rpb_table", None),
                    rpb_max_distance=(
                        int(block.bc.rpb_max_distance) if getattr(block, "rpb_table", None) is not None else None
                    ),
                )
                n1 = apply_native_norm(x, block.n1)
                traces.append(_record(f"encoder.{index}.n1", n1[0]))
                attn_out = block.attn.forward(n1, None, None, attn_mask, None)
                traces.append(_record(f"encoder.{index}.attn", attn_out[0]))
                x = x + attn_out
                traces.append(_record(f"encoder.{index}.attn_resid", x[0]))
                n2 = apply_native_norm(x, block.n2)
                traces.append(_record(f"encoder.{index}.n2", n2[0]))
                mlp_out = block.mlp(n2)
                traces.append(_record(f"encoder.{index}.mlp", mlp_out[0]))
                x = x + mlp_out
                traces.append(_record(f"encoder.{index}", x[0]))
            memory = apply_native_norm(x, model.enc_norm)
            traces.append(_record("enc_norm", memory[0]))
            dec_embed = F.embedding(dec, model.dec_embed.weight, padding_idx=model.dec_embed.padding_idx)
            traces.append(_record("dec_embed", dec_embed[0]))
            hidden = model.decode(dec, memory, None, enc_attention_mask)
            for index in range(len(model.decoder)):
                traces.append(_record(f"decoder.{index}", decoder_outputs[f"decoder.{index}"][0]))
            traces.append(_record("dec_norm", hidden[0]))
            logits = model.lm_head(hidden)
            traces.append(_record("logits", logits[0]))
    finally:
        for hook in hooks:
            hook.remove()
    return traces


def _run_js_trace(runtime_js: Path, browser_manifest: Path, enc_ids: list[int], dec_ids: list[int]) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="agentkernel_trace_") as temp_raw:
        temp = Path(temp_raw)
        helper = temp / "trace_node.mjs"
        inp = temp / "input.json"
        out = temp / "output.json"
        helper.write_text(NODE_HELPER, encoding="utf-8")
        inp.write_text(
            json.dumps(
                {
                    "runtimeJs": str(runtime_js.resolve()),
                    "browserManifest": str(browser_manifest.resolve()),
                    "encIds": enc_ids,
                    "decIds": dec_ids,
                }
            ),
            encoding="utf-8",
        )
        subprocess.run(["node", str(helper), str(inp), str(out)], check=True)
        payload = json.loads(out.read_text(encoding="utf-8"))
    return list(payload["traces"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Python model-stack and browser runtime layer traces.")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--browser-manifest", required=True)
    parser.add_argument("--runtime-js", default="/data/repository_library/exports/agent_kernel/vendor/model-stack-bitnet/encdec_runtime.js")
    parser.add_argument("--prompt", default="whats the best multi agent llm paper?")
    parser.add_argument("--decoder-prefix", default="Action: respond\nCon")
    parser.add_argument("--max-encoder-tokens", type=int, default=768)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _install_paths(repo_root)
    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    bundle_dir = Path(args.bundle_dir).resolve()
    manifest = _load_manifest(bundle_dir)
    tokenizer = _load_tokenizer(manifest)
    model_dir = Path(str(manifest["model_dir"]))
    config = load_config(str(model_dir))
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    _materialize_lazy_modules(model)
    load_pretrained(model, str(model_dir), strict=True)
    device = torch.device(str(args.device))
    model.to(device).eval()

    enc_ids = tokenizer.encode(str(args.prompt), max_length=int(args.max_encoder_tokens))
    prefix_ids = [
        int(token_id)
        for token_id in tokenizer.encode(str(args.decoder_prefix), max_length=128)
        if int(token_id)
        not in {
            int(tokenizer.bos_token_id),
            int(tokenizer.eos_token_id),
            int(tokenizer.pad_token_id),
        }
    ]
    dec_ids = [int(tokenizer.bos_token_id), *prefix_ids]
    py_traces = _python_trace(model, enc_ids, dec_ids, device)
    js_traces = _run_js_trace(Path(args.runtime_js), Path(args.browser_manifest), enc_ids, dec_ids)
    js_by_name = {str(item["name"]): item for item in js_traces}
    comparisons = []
    for py_item in py_traces:
        name = str(py_item["name"])
        js_item = js_by_name[name]
        cmp = _compare(py_item["values"], js_item["values"])
        comparisons.append({"name": name, "shape": py_item["shape"], **cmp})
    payload = {
        "prompt": args.prompt,
        "decoder_prefix": args.decoder_prefix,
        "enc_len": len(enc_ids),
        "dec_len": len(dec_ids),
        "comparisons": comparisons,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for item in comparisons:
        print(
            f"{item['name']}: max_abs={float(item['max_abs']):.6g} "
            f"mean_abs={float(item['mean_abs']):.6g} cosine={float(item['cosine']):.8f}"
        )


if __name__ == "__main__":
    main()
