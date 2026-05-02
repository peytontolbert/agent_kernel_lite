#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent_kernel.modeling.model_python import maybe_reexec_under_model_python

maybe_reexec_under_model_python(require_full_torch=True)

import torch

from sample_agentkernel_lite_encdec import (
    _install_paths,
    _load_manifest,
    _load_tokenizer,
    _materialize_lazy_modules,
)


DEFAULT_BUNDLE_DIR = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "agentkernel_lite_encdec"
    / "chatfirst_retrieval_special_compact_100m_from16000_train_17000"
)
DEFAULT_RUNTIME_JS = (
    Path("/data/repository_library/exports/agent_kernel/vendor/model-stack-bitnet/encdec_runtime.js")
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
globalThis.fetch = async function parityFetch(resource, init) {
  const raw = resource instanceof URL ? resource.href : String(resource);
  if (raw.startsWith("file:")) {
    const data = await readFile(fileURLToPath(raw));
    return new Response(data, {
      status: 200,
      headers: { "content-type": contentType(raw) },
    });
  }
  return nativeFetch(resource, init);
};

function byteToUnicodeMap() {
  const bs = [];
  for (let i = 33; i <= 126; i += 1) bs.push(i);
  for (let i = 161; i <= 172; i += 1) bs.push(i);
  for (let i = 174; i <= 255; i += 1) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b += 1) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }
  const byteEncoder = new Map();
  const byteDecoder = new Map();
  for (let i = 0; i < bs.length; i += 1) {
    const ch = String.fromCodePoint(cs[i]);
    byteEncoder.set(bs[i], ch);
    byteDecoder.set(ch, bs[i]);
  }
  return { byteEncoder, byteDecoder };
}

const BYTE_UNICODE = byteToUnicodeMap();
const BPE_PAIR_SEP = "\u0001";
const GPT2_PRETOKEN_PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function createBpeTokenizer(tokenizerJson) {
  const model = tokenizerJson?.model || {};
  const vocab = model.vocab || {};
  const idToToken = [];
  for (const [token, id] of Object.entries(vocab)) idToToken[Number(id)] = token;
  const ranks = new Map();
  for (const [rank, merge] of (model.merges || []).entries()) {
    const pair = Array.isArray(merge) ? merge : String(merge).split(/\s+/);
    if (pair.length >= 2) ranks.set(`${pair[0]}${BPE_PAIR_SEP}${pair[1]}`, rank);
  }
  const added = tokenizerJson?.added_tokens || [];
  const specialIds = new Set(
    added.filter((item) => item?.special).map((item) => Number(item.id)),
  );
  const specialTokenEntries = added
    .filter((item) => item?.special && typeof item.content === "string" && item.content.length > 0)
    .map((item) => ({ token: item.content, id: Number(item.id) }))
    .filter((item) => Number.isFinite(item.id))
    .sort((a, b) => b.token.length - a.token.length);
  const padTokenId = Number(vocab["<pad>"] ?? 0);
  const bosTokenId = Number(vocab["<s>"] ?? 1);
  const eosTokenId = Number(vocab["</s>"] ?? 2);
  const unkTokenId = Number(vocab["<unk>"] ?? 3);

  function bpeToken(byteLevelToken) {
    let word = Array.from(byteLevelToken);
    while (word.length > 1) {
      let bestIndex = -1;
      let bestRank = Infinity;
      for (let i = 0; i < word.length - 1; i += 1) {
        const rank = ranks.get(`${word[i]}${BPE_PAIR_SEP}${word[i + 1]}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }
      if (bestIndex < 0) break;
      const merged = `${word[bestIndex]}${word[bestIndex + 1]}`;
      const next = [];
      for (let i = 0; i < word.length; i += 1) {
        if (i === bestIndex) {
          next.push(merged);
          i += 1;
        } else {
          next.push(word[i]);
        }
      }
      word = next;
    }
    return word;
  }

  function splitSpecialSegments(text) {
    const source = String(text || "");
    if (!specialTokenEntries.length) return [{ text: source, specialId: null }];
    const segments = [];
    let cursor = 0;
    while (cursor < source.length) {
      let match = null;
      let matchIndex = -1;
      for (const entry of specialTokenEntries) {
        const index = source.indexOf(entry.token, cursor);
        if (index >= 0 && (matchIndex < 0 || index < matchIndex)) {
          match = entry;
          matchIndex = index;
        }
      }
      if (!match) {
        segments.push({ text: source.slice(cursor), specialId: null });
        break;
      }
      if (matchIndex > cursor) segments.push({ text: source.slice(cursor, matchIndex), specialId: null });
      segments.push({ text: match.token, specialId: match.id });
      cursor = matchIndex + match.token.length;
    }
    return segments;
  }

  function encodePlainSegment(text, ids, maxLength) {
    const pieces = String(text || "").match(GPT2_PRETOKEN_PATTERN) || [];
    for (const piece of pieces) {
      const bytes = new TextEncoder().encode(piece);
      let byteLevel = "";
      for (const byte of bytes) byteLevel += BYTE_UNICODE.byteEncoder.get(byte);
      for (const token of bpeToken(byteLevel)) {
        if (ids.length >= maxLength - 1) break;
        ids.push(Number(vocab[token] ?? unkTokenId));
      }
      if (ids.length >= maxLength - 1) break;
    }
  }

  function encode(text, maxLength = 1024) {
    const ids = [bosTokenId];
    for (const segment of splitSpecialSegments(text)) {
      if (ids.length >= maxLength - 1) break;
      if (segment.specialId !== null) ids.push(segment.specialId);
      else encodePlainSegment(segment.text, ids, maxLength);
    }
    ids.push(eosTokenId);
    return ids;
  }

  function decode(ids) {
    let byteLevel = "";
    for (const rawId of ids) {
      const id = Number(rawId);
      if (id === eosTokenId) break;
      if (id === padTokenId || specialIds.has(id)) continue;
      byteLevel += idToToken[id] || "";
    }
    const bytes = [];
    for (const ch of Array.from(byteLevel)) {
      const byte = BYTE_UNICODE.byteDecoder.get(ch);
      if (byte !== undefined) bytes.push(byte);
    }
    return new TextDecoder().decode(new Uint8Array(bytes));
  }

  function piece(id) {
    return idToToken[Number(id)] || "";
  }

  return { bosTokenId, eosTokenId, padTokenId, unkTokenId, specialIds, encode, decode, piece };
}

function topK(logits, k, tokenizer) {
  const items = [];
  for (let i = 0; i < logits.length; i += 1) items.push({ id: i, logit: Number(logits[i]) });
  items.sort((a, b) => b.logit - a.logit);
  return items.slice(0, k).map((item) => ({
    id: item.id,
    logit: item.logit,
    piece: tokenizer.piece(item.id),
    text: tokenizer.decode([item.id]),
  }));
}

async function cachedLogits(runtime, encIds, prefix) {
  const session = runtime.createGenerationSession(encIds);
  let logits = null;
  for (const tokenId of prefix) {
    logits = await session.next(Number(tokenId));
  }
  return logits || new Float32Array();
}

const inputPath = process.argv[2];
const outputPath = process.argv[3];
if (!inputPath || !outputPath) {
  throw new Error("usage: node parity_node.mjs input.json output.json");
}
const input = JSON.parse(await readFile(inputPath, "utf8"));
const runtimeModule = await import(input.runtimeUrl);
const runtime = await runtimeModule.BitNetEncoderDecoderWASM.fromManifestUrl(input.manifestUrl, {
  layerConcurrency: Number(input.layerConcurrency || 2),
  progress: (progress) => {
    if (progress?.phase === "ready" || progress?.phase === "wasm_ready") {
      console.error(progress.message || "runtime ready");
    }
  },
});
const manifest = await fetch(input.manifestUrl).then((response) => response.json());
const tokenizerUrl = new URL(manifest.tokenizer.path, new URL(".", input.manifestUrl)).href;
const tokenizerJson = await fetch(tokenizerUrl).then((response) => response.json());
const tokenizer = createBpeTokenizer(tokenizerJson);
const jsEncIds = tokenizer.encode(input.prompt, Number(input.maxEncoderTokens || 1024));
const steps = [];
for (const prefix of input.prefixes) {
  const full = await runtime.forward(input.encIds, prefix);
  const vocabSize = Number(runtime.graph?.vocab_size || manifest.model?.vocab_size || 0);
  const offset = (prefix.length - 1) * vocabSize;
  const fullLast = full.slice(offset, offset + vocabSize);
  const cachedLast = await cachedLogits(runtime, input.encIds, prefix);
  steps.push({
    prefix,
    fullTopK: topK(fullLast, Number(input.topK || 8), tokenizer),
    cachedTopK: topK(cachedLast, Number(input.topK || 8), tokenizer),
    fullLogits: Array.from(fullLast),
    cachedLogits: Array.from(cachedLast),
  });
}
await writeFile(outputPath, JSON.stringify({
  tokenizer: {
    jsEncIds,
    pyEncIds: input.encIds,
    exactMatch: JSON.stringify(jsEncIds) === JSON.stringify(input.encIds),
    decodedPrompt: tokenizer.decode(jsEncIds),
  },
  steps,
}, null, 2));
"""


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return


@contextmanager
def serve_directory(root: Path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    handler = partial(QuietHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def token_piece(tokenizer: Any, token_id: int) -> str:
    token_id = int(token_id)
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None:
        if hasattr(inner, "id_to_token"):
            value = inner.id_to_token(token_id)
            if value is not None:
                return str(value)
        if hasattr(inner, "convert_ids_to_tokens"):
            return str(inner.convert_ids_to_tokens([token_id])[0])
    if 4 <= token_id <= 259:
        return repr(bytes([token_id - 4]).decode("utf-8", errors="replace"))
    specials = {
        int(getattr(tokenizer, "pad_token_id", 0)): "<pad>",
        int(getattr(tokenizer, "bos_token_id", 1)): "<s>",
        int(getattr(tokenizer, "eos_token_id", 2)): "</s>",
        int(getattr(tokenizer, "unk_token_id", 3)): "<unk>",
    }
    return specials.get(token_id, str(token_id))


def topk_from_logits(logits: torch.Tensor, tokenizer: Any, k: int) -> list[dict[str, Any]]:
    values, indices = torch.topk(logits.detach().float().cpu(), k=min(int(k), int(logits.numel())))
    items: list[dict[str, Any]] = []
    for value, index in zip(values.tolist(), indices.tolist()):
        token_id = int(index)
        try:
            text = tokenizer.decode([token_id])
        except Exception:
            text = ""
        items.append(
            {
                "id": token_id,
                "logit": float(value),
                "piece": token_piece(tokenizer, token_id),
                "text": str(text),
            }
        )
    return items


def compare_vectors(left: torch.Tensor, right_values: list[float], *, top_k: int) -> dict[str, Any]:
    right = torch.tensor(right_values, dtype=torch.float32)
    left = left.detach().float().cpu()
    if left.numel() != right.numel():
        return {
            "shape_match": False,
            "left_size": int(left.numel()),
            "right_size": int(right.numel()),
        }
    diff = (left - right).abs()
    left_top = set(torch.topk(left, k=min(top_k, left.numel())).indices.tolist())
    right_top = set(torch.topk(right, k=min(top_k, right.numel())).indices.tolist())
    denom = float(left.norm().item() * right.norm().item())
    cosine = float(torch.dot(left, right).item() / denom) if denom else 0.0
    return {
        "shape_match": True,
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((left - right) ** 2)).item()),
        "cosine": cosine,
        "top1_match": int(torch.argmax(left).item()) == int(torch.argmax(right).item()),
        "topk_overlap": len(left_top & right_top),
        "topk": int(top_k),
    }


def load_model(
    *,
    repo_root: Path,
    bundle_manifest: dict[str, Any],
    device: torch.device,
    bitnet: bool,
    spin_seed: int,
    bitnet_include: list[str] | None = None,
    bitnet_exclude: list[str] | None = None,
):
    _install_paths(repo_root)
    from compress.apply import apply_compression
    from runtime.checkpoint import load_config, load_pretrained
    from runtime.seq2seq import EncoderDecoderLM

    model_dir = Path(str(bundle_manifest["model_dir"]))
    config = load_config(str(model_dir))
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=int(config.vocab_size))
    _materialize_lazy_modules(model)
    load_pretrained(model, str(model_dir), strict=True)
    model.to(device).eval()
    if bitnet:
        apply_compression(
            model,
            quant={
                "scheme": "bitnet",
                "include": bitnet_include or None,
                "exclude": bitnet_exclude or None,
                "weight_opt": "none",
                "activation_quant": "none",
                "spin": False,
                "spin_random": True,
                "spin_seed": int(spin_seed),
            },
        )
        model.to(device).eval()
    return model


def model_last_logits(
    model: torch.nn.Module,
    enc_ids: list[int],
    dec_ids: list[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        enc = torch.tensor([enc_ids], dtype=torch.long, device=device)
        dec = torch.tensor([dec_ids], dtype=torch.long, device=device)
        enc_attention_mask = torch.ones_like(enc, dtype=torch.long, device=device)
        return model(enc, dec, enc_attention_mask, None)[0, -1].detach().float().cpu()


def repeats_tail(generated: list[int], token_id: int, max_tail: int = 3) -> bool:
    count = 0
    for existing in reversed(generated):
        if int(existing) != int(token_id):
            break
        count += 1
        if count >= max_tail:
            return True
    return False


def would_repeat_ngram(generated: list[int], token_id: int, ngram_size: int = 4) -> bool:
    if ngram_size <= 1 or len(generated) < ngram_size - 1:
        return False
    prefix = [int(item) for item in generated[-ngram_size + 1 :]]
    for index in range(0, len(generated) - ngram_size + 1):
        if [int(item) for item in generated[index : index + len(prefix)]] == prefix:
            if int(generated[index + len(prefix)]) == int(token_id):
                return True
    return False


def select_next_for_prefix(
    logits: torch.Tensor,
    generated: list[int],
    tokenizer: Any,
    *,
    repetition_penalty: float,
) -> int:
    adjusted = logits.detach().float().cpu().clone()
    blocked = {
        int(getattr(tokenizer, "pad_token_id", 0)),
        int(getattr(tokenizer, "bos_token_id", 1)),
        int(getattr(tokenizer, "unk_token_id", 3)),
    }
    for token_id in blocked:
        if 0 <= token_id < adjusted.numel():
            adjusted[token_id] = -float("inf")
    if repetition_penalty > 1.0:
        for token_id in set(int(item) for item in generated):
            if 0 <= token_id < adjusted.numel():
                adjusted[token_id] = (
                    adjusted[token_id] / repetition_penalty
                    if adjusted[token_id] > 0
                    else adjusted[token_id] * repetition_penalty
                )
    for token_id in range(int(adjusted.numel())):
        if repeats_tail(generated, token_id) or would_repeat_ngram(generated, token_id):
            adjusted[token_id] = -float("inf")
    return int(torch.argmax(adjusted).item())


def browser_manifest_url(path_or_url: str, *, server_root: str | None) -> str:
    raw = str(path_or_url).strip()
    if raw.startswith(("http://", "https://")):
        return raw
    path = Path(raw).resolve()
    if server_root is None:
        return path_to_file_url(path)
    return f"{server_root}/{path.name}"


def path_to_file_url(path: Path) -> str:
    return path.resolve().as_uri()


def default_browser_manifest(bundle_dir: Path, manifest: dict[str, Any]) -> Path:
    raw = str(manifest.get("browser_bitnet_manifest_path", "") or "").strip()
    if raw:
        path = Path(raw).expanduser()
        if path.exists():
            return path.resolve()
    return (bundle_dir / "browser_bitnet" / "manifest.json").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare AgentKernel Lite Python and browser BitNet runtimes.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    parser.add_argument("--browser-manifest", default="")
    parser.add_argument("--runtime-js", default=str(DEFAULT_RUNTIME_JS))
    parser.add_argument("--prompt", default="whats the best multi agent llm paper?")
    parser.add_argument("--decoder-prefix", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-encoder-tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--layer-concurrency", type=int, default=2)
    parser.add_argument("--spin-seed", type=int, default=0)
    parser.add_argument("--bitnet-include", default="")
    parser.add_argument("--bitnet-exclude", default="")
    parser.add_argument("--repetition-penalty", type=float, default=1.16)
    parser.add_argument("--skip-source-fp", action="store_true")
    parser.add_argument("--skip-python-bitnet", action="store_true")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--fail-on-top1-mismatch", action="store_true")
    return parser.parse_args()


def csv_patterns(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    bundle_dir = Path(args.bundle_dir).resolve()
    bundle_manifest = _load_manifest(bundle_dir)
    browser_manifest_path = (
        Path(args.browser_manifest).expanduser().resolve()
        if str(args.browser_manifest).strip()
        else default_browser_manifest(bundle_dir, bundle_manifest)
    )
    runtime_js = Path(args.runtime_js).expanduser().resolve()
    if not runtime_js.exists():
        raise FileNotFoundError(f"runtime JS does not exist: {runtime_js}")
    if not browser_manifest_path.exists():
        raise FileNotFoundError(f"browser manifest does not exist: {browser_manifest_path}")

    _install_paths(repo_root)
    tokenizer = _load_tokenizer(bundle_manifest)
    prompt = str(args.prompt)
    enc_ids = tokenizer.encode(prompt, max_length=int(args.max_encoder_tokens))
    decoder_prefix_ids: list[int] = []
    if str(args.decoder_prefix).strip():
        encoded_prefix = tokenizer.encode(str(args.decoder_prefix), max_length=128)
        decoder_prefix_ids = [
            int(token_id)
            for token_id in encoded_prefix
            if int(token_id)
            not in {
                int(getattr(tokenizer, "bos_token_id", 1)),
                int(getattr(tokenizer, "eos_token_id", 2)),
                int(getattr(tokenizer, "pad_token_id", 0)),
            }
        ]
    device = torch.device(str(args.device))
    bitnet_include = csv_patterns(str(args.bitnet_include))
    bitnet_exclude = csv_patterns(str(args.bitnet_exclude))

    source_model = None
    bitnet_model = None
    if not args.skip_source_fp:
        source_model = load_model(
            repo_root=repo_root,
            bundle_manifest=bundle_manifest,
            device=device,
            bitnet=False,
            spin_seed=int(args.spin_seed),
        )
    if not args.skip_python_bitnet:
        bitnet_model = load_model(
            repo_root=repo_root,
            bundle_manifest=bundle_manifest,
            device=device,
            bitnet=True,
            spin_seed=int(args.spin_seed),
            bitnet_include=bitnet_include,
            bitnet_exclude=bitnet_exclude,
        )
    prefix_model = bitnet_model or source_model
    if prefix_model is None:
        raise RuntimeError("at least one Python model must be enabled to build decoder prefixes")

    prefixes: list[list[int]] = []
    py_steps: list[dict[str, Any]] = []
    dec_ids = [int(getattr(tokenizer, "bos_token_id", 1)), *decoder_prefix_ids]
    for step_index in range(max(1, int(args.steps))):
        prefix = list(dec_ids)
        prefixes.append(prefix)
        step: dict[str, Any] = {
            "step": step_index,
            "prefix": prefix,
            "prefix_text": tokenizer.decode(prefix[1:]),
        }
        source_logits = None
        bitnet_logits = None
        if source_model is not None:
            source_logits = model_last_logits(source_model, enc_ids, prefix, device=device)
            step["pythonSourceTopK"] = topk_from_logits(source_logits, tokenizer, int(args.top_k))
        if bitnet_model is not None:
            bitnet_logits = model_last_logits(bitnet_model, enc_ids, prefix, device=device)
            step["pythonBitnetTopK"] = topk_from_logits(bitnet_logits, tokenizer, int(args.top_k))
        if source_logits is not None and bitnet_logits is not None:
            step["sourceVsPythonBitnet"] = compare_vectors(source_logits, bitnet_logits.tolist(), top_k=int(args.top_k))
        chooser_logits = bitnet_logits if bitnet_logits is not None else source_logits
        assert chooser_logits is not None
        next_id = select_next_for_prefix(
            chooser_logits,
            dec_ids[1:],
            tokenizer,
            repetition_penalty=float(args.repetition_penalty),
        )
        step["selectedNextId"] = int(next_id)
        step["selectedNextPiece"] = token_piece(tokenizer, int(next_id))
        try:
            step["selectedNextText"] = tokenizer.decode([int(next_id)])
        except Exception:
            step["selectedNextText"] = ""
        py_steps.append(step)
        if int(next_id) == int(getattr(tokenizer, "eos_token_id", 2)):
            break
        dec_ids.append(int(next_id))

    result: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": str(bundle_dir),
        "browser_manifest_path": str(browser_manifest_path),
        "runtime_js": str(runtime_js),
        "device": str(device),
        "bitnet_include": bitnet_include,
        "bitnet_exclude": bitnet_exclude,
        "prompt": prompt,
        "enc_ids": enc_ids,
        "prefixes": prefixes,
        "python_steps": py_steps,
        "summary": {},
    }

    with tempfile.TemporaryDirectory(prefix="agentkernel-lite-parity-") as tmp_raw:
        tmp = Path(tmp_raw)
        node_helper_path = tmp / "browser_parity_node.mjs"
        node_input_path = tmp / "input.json"
        node_output_path = tmp / "browser_output.json"
        node_helper_path.write_text(NODE_HELPER, encoding="utf-8")
        server_context = serve_directory(browser_manifest_path.parent)
        with server_context as server_root:
            manifest_url = browser_manifest_url(str(browser_manifest_path), server_root=server_root)
            node_input_path.write_text(
                json.dumps(
                    {
                        "manifestUrl": manifest_url,
                        "runtimeUrl": path_to_file_url(runtime_js),
                        "prompt": prompt,
                        "encIds": enc_ids,
                        "prefixes": prefixes,
                        "topK": int(args.top_k),
                        "maxEncoderTokens": int(args.max_encoder_tokens),
                        "layerConcurrency": int(args.layer_concurrency),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            subprocess.run(
                ["node", str(node_helper_path), str(node_input_path), str(node_output_path)],
                cwd=str(repo_root),
                check=True,
            )
        browser = json.loads(node_output_path.read_text(encoding="utf-8"))

    result["browser"] = browser
    python_bitnet_full_mismatches = 0
    python_bitnet_cached_mismatches = 0
    browser_cache_mismatches = 0
    source_bitnet_mismatches = 0
    source_browser_full_mismatches = 0
    source_browser_cached_mismatches = 0
    for index, (py_step, browser_step) in enumerate(zip(py_steps, browser.get("steps", []))):
        full_logits = browser_step.get("fullLogits", [])
        cached_logits = browser_step.get("cachedLogits", [])
        if source_model is not None:
            source_logits = model_last_logits(source_model, enc_ids, py_step["prefix"], device=device)
            py_step["pythonSourceVsBrowserFull"] = compare_vectors(
                source_logits,
                full_logits,
                top_k=int(args.top_k),
            )
            if not py_step["pythonSourceVsBrowserFull"].get("top1_match", False):
                source_browser_full_mismatches += 1
            py_step["pythonSourceVsBrowserCached"] = compare_vectors(
                source_logits,
                cached_logits,
                top_k=int(args.top_k),
            )
            if not py_step["pythonSourceVsBrowserCached"].get("top1_match", False):
                source_browser_cached_mismatches += 1
        if "pythonBitnetTopK" in py_step:
            py_bitnet_logits = model_last_logits(bitnet_model, enc_ids, py_step["prefix"], device=device)  # type: ignore[arg-type]
            py_step["pythonBitnetVsBrowserFull"] = compare_vectors(
                py_bitnet_logits,
                full_logits,
                top_k=int(args.top_k),
            )
            if not py_step["pythonBitnetVsBrowserFull"].get("top1_match", False):
                python_bitnet_full_mismatches += 1
            py_step["pythonBitnetVsBrowserCached"] = compare_vectors(
                py_bitnet_logits,
                cached_logits,
                top_k=int(args.top_k),
            )
            if not py_step["pythonBitnetVsBrowserCached"].get("top1_match", False):
                python_bitnet_cached_mismatches += 1
        if "sourceVsPythonBitnet" in py_step and not py_step["sourceVsPythonBitnet"].get("top1_match", False):
            source_bitnet_mismatches += 1
        cache_cmp = compare_vectors(torch.tensor(full_logits), cached_logits, top_k=int(args.top_k))
        browser_step["browserFullVsCached"] = cache_cmp
        if not cache_cmp.get("top1_match", False):
            browser_cache_mismatches += 1
        py_step["browserFullTopK"] = browser_step.get("fullTopK", [])
        py_step["browserCachedTopK"] = browser_step.get("cachedTopK", [])
        py_step["browserFullVsCached"] = cache_cmp
        py_step["step"] = index

    result["summary"] = {
        "tokenizer_exact_match": bool(browser.get("tokenizer", {}).get("exactMatch")),
        "python_bitnet_vs_browser_full_top1_mismatches": python_bitnet_full_mismatches,
        "python_bitnet_vs_browser_cached_top1_mismatches": python_bitnet_cached_mismatches,
        "browser_full_vs_cached_top1_mismatches": browser_cache_mismatches,
        "source_fp_vs_python_bitnet_top1_mismatches": source_bitnet_mismatches,
        "source_fp_vs_browser_full_top1_mismatches": source_browser_full_mismatches,
        "source_fp_vs_browser_cached_top1_mismatches": source_browser_cached_mismatches,
        "step_count": len(py_steps),
    }

    output_path = Path(args.json_out).expanduser().resolve() if str(args.json_out).strip() else None
    if output_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = bundle_dir / "parity" / f"browser_parity_{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"parity_json={output_path}")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    for step in result["python_steps"]:
        py_top = (step.get("pythonBitnetTopK") or step.get("pythonSourceTopK") or [{}])[0]
        browser_full = (step.get("browserFullTopK") or [{}])[0]
        browser_cached = (step.get("browserCachedTopK") or [{}])[0]
        print(
            "step={step} prefix_len={prefix_len} "
            "py={py_id}:{py_piece!r} browser_full={bf_id}:{bf_piece!r} browser_cached={bc_id}:{bc_piece!r}".format(
                step=step["step"],
                prefix_len=len(step["prefix"]),
                py_id=py_top.get("id"),
                py_piece=py_top.get("piece"),
                bf_id=browser_full.get("id"),
                bf_piece=browser_full.get("piece"),
                bc_id=browser_cached.get("id"),
                bc_piece=browser_cached.get("piece"),
            )
        )

    if args.fail_on_top1_mismatch:
        reference_mismatches = (
            python_bitnet_cached_mismatches
            if bitnet_model is not None
            else source_browser_cached_mismatches
        )
        if reference_mismatches or browser_cache_mismatches or not result["summary"]["tokenizer_exact_match"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
