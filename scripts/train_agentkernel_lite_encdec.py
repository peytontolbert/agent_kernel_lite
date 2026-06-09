#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _model_stack_root(repo_root: Path) -> Path:
    return repo_root / "other_repos" / "model-stack"


def _install_model_stack_path(repo_root: Path) -> None:
    root = str(_model_stack_root(repo_root))
    if root not in sys.path:
        sys.path.insert(0, root)


class ByteTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    vocab_size = 260

    def __call__(
        self,
        text: str,
        *,
        max_length: int,
        padding: str = "max_length",
        truncation: bool = True,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
        del padding, truncation
        ids = [byte + 4 for byte in str(text).encode("utf-8", errors="replace")]
        if add_special_tokens:
            ids = [self.bos_token_id, *ids, self.eos_token_id]
        ids = ids[:max_length]
        attention = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(self.pad_token_id)
            attention.append(0)
        return {"input_ids": ids, "attention_mask": attention}

    def save_pretrained(self, output_dir: str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_kind": "byte_fallback_v1",
                    "pad_token_id": self.pad_token_id,
                    "bos_token_id": self.bos_token_id,
                    "eos_token_id": self.eos_token_id,
                    "unk_token_id": self.unk_token_id,
                    "vocab_size": self.vocab_size,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


class LocalBpeTokenizer:
    pad_token = "<pad>"
    bos_token = "<s>"
    eos_token = "</s>"
    unk_token = "<unk>"
    default_agentkernel_special_tokens = [
        "<AK_USER>",
        "<AK_CHAT>",
        "<AK_THINK>",
        "<AK_DEEP_RESEARCH>",
        "<AK_LOOP>",
        "<AK_PLAN>",
        "<AK_STATE>",
        "<AK_CONTEXT>",
        "<AK_ACTIVE_CONTEXT>",
        "<AK_EVIDENCE>",
        "<AK_EVIDENCE_ID>",
        "<AK_TITLE>",
        "<AK_PAPER_ID>",
        "<AK_CATEGORY>",
        "<AK_YEAR>",
        "<AK_ABSTRACT>",
        "<AK_CANDIDATE>",
        "<AK_SELECTED_PAPER>",
        "<AK_CONTEXT_ID>",
        "<AK_TARGET_CONTEXT>",
        "<AK_QUERY_REWRITE>",
        "<AK_RERANK>",
        "<AK_ABSTRACT_RETRIEVAL>",
        "<AK_GATHER_CONTEXT>",
        "<AK_RESPOND>",
        "<AK_USE_CONTEXT>",
        "<AK_FULL_TEXT>",
        "<AK_LOAD_FULL_TEXT>",
        "<AK_NO_RETRIEVAL>",
        "<AK_RETRIEVE_NEW>",
        "<AK_RETRIEVAL_PAIR>",
        "<AK_SUFFICIENT>",
        "<AK_INSUFFICIENT>",
        "<AK_ANSWER>",
        "<AK_CITE>",
        "<AK_RENDER>",
        "<AK_JSON>",
        "<AK_PROFILE>",
        "<AK_SLOT>",
        "<AK_SLOT_NAME>",
        "<AK_SLOT_VALUE>",
        "<AK_PREFERENCE>",
        "<AK_GOAL>",
        "<AK_DOMAIN>",
        "<AK_TONE>",
        "<AK_CONSTRAINT>",
        "<AK_PRIVACY>",
        "<AK_REMEMBER>",
        "<AK_UPDATE_SLOT>",
        "<AK_DELETE_SLOT>",
        "<AK_EXTENSION>",
        "<AK_CAPABILITY>",
        "<AK_APPROVAL>",
        "<AK_SAVE_MEMORY>",
        "<AK_ASK_USER>",
        "<AK_SOURCE_TYPE>",
        "<AK_EXTENSION_RESULT>",
        "<AK_INSTALLED>",
        "<AK_DECISION>",
        "<AK_NEED_MEMORY>",
        "<AK_MEMORY_TYPE>",
        "<AK_MEMORY_QUERY>",
        "<AK_MEMORY_SLOT>",
        "<AK_MEMORY_RESULT>",
        "<AK_USE_MEMORY>",
        "<AK_UNCERTAIN>",
        "<AK_SUFFICIENT_CONTEXT>",
        "<AK_REASON>",
        "<AK_VERIFY>",
        "<AK_RETRIEVE>",
        "<AK_REFLECT>",
        "<AK_ASK>",
        "<AK_ABSTAIN>",
        "<AK_RET_PAPERS>",
        "<AK_RET_MEMORY>",
        "<AK_RET_GRAPH>",
        "<AK_RET_EXACT>",
        "<AK_RET_SEMANTIC>",
        "<AK_RET_HYBRID>",
        "<AK_CONF_HIGH>",
        "<AK_CONF_MEDIUM>",
        "<AK_CONF_LOW>",
        "<AK_OOD_QUERY>",
        "<AK_OOD_EVIDENCE>",
        "<AK_OOD_RESPONSE>",
        "<AK_INSUFFICIENT_EVIDENCE>",
        "<AK_NEEDS_VERIFICATION>",
        "<AK_MEM_READ>",
        "<AK_MEM_WRITE>",
        "<AK_MEM_UPDATE>",
        "<AK_MEM_EPISODIC>",
        "<AK_MEM_SEMANTIC>",
        "<AK_CHECK_FACT>",
        "<AK_CHECK_CONSISTENCY>",
        "<AK_CHECK_EVIDENCE>",
        "<AK_RETRIEVE_AGAIN>",
        "<AK_ACTION_SPACE_CODE>",
        "<AK_ACTION_SPACE_ARTIFACT>",
        "<AK_ACTION_SPACE_RETRIEVAL>",
        "<AK_ACTION_SPACE_RESPOND>",
        "<AK_STRUCTURED>",
        "<AK_ACTION_RESPOND>",
        "<AK_ACTION_ASK_USER>",
        "<AK_ACTION_EXTENSION_REQUEST>",
        "<AK_ACTION_SAVE_MEMORY>",
        "<AK_CONTENT>",
        "</AK_CONTENT>",
        "<AK_TASK_TYPE>",
        "<AK_INTENT>",
        "<AK_FIELD>",
        "<AK_FIELD_NAME>",
        "<AK_FIELD_VALUE>",
        "<AK_FIELDS>",
        "<AK_FRESHNESS>",
        "<AK_END>",
        "<AK_RET_CODE>",
        "<AK_OOD>",
        "<AK_ARTIFACT_REPAIR>",
        "<AK_SOURCE_INSPECT>",
        "<AK_PATCH_BUILD>",
        "<AK_SAFE_STOP>",
        "<AK_SOURCE_SLOTS>",
        *[f"<AK_COPY_USER_SOURCE_{index}>" for index in range(1, 25)],
    ]

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = int(tokenizer.token_to_id(self.pad_token))
        self.bos_token_id = int(tokenizer.token_to_id(self.bos_token))
        self.eos_token_id = int(tokenizer.token_to_id(self.eos_token))
        self.unk_token_id = int(tokenizer.token_to_id(self.unk_token))
        self.vocab_size = int(tokenizer.get_vocab_size())
        self.agentkernel_special_tokens = [
            token
            for token in self.default_agentkernel_special_tokens
            if tokenizer.token_to_id(token) is not None
        ]

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text: str,
        *,
        max_length: int,
        padding: str = "max_length",
        truncation: bool = True,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
        encoding = self.tokenizer.encode(str(text), add_special_tokens=add_special_tokens)
        ids = list(encoding.ids)
        if truncation:
            ids = ids[:max_length]
        attention = [1] * len(ids)
        if padding == "max_length":
            while len(ids) < max_length:
                ids.append(self.pad_token_id)
                attention.append(0)
        return {"input_ids": ids, "attention_mask": attention}

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode([int(token_id) for token_id in ids], skip_special_tokens=True)

    def save_pretrained(self, output_dir: str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path / "tokenizer.json"))
        (path / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_kind": "agentkernel_bytelevel_bpe_v1",
                    "pad_token": self.pad_token,
                    "bos_token": self.bos_token,
                    "eos_token": self.eos_token,
                    "unk_token": self.unk_token,
                    "pad_token_id": self.pad_token_id,
                    "bos_token_id": self.bos_token_id,
                    "eos_token_id": self.eos_token_id,
                    "unk_token_id": self.unk_token_id,
                    "agentkernel_special_tokens": self.agentkernel_special_tokens,
                    "vocab_size": self.vocab_size,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def _iter_jsonl_texts(path: Path) -> Iterator[str]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("encoder_text", "decoder_text"):
                text = str(row.get(key, "") or "").strip()
                if text:
                    yield text


def _tokenizer_training_texts(
    dataset_manifest: dict[str, Any],
    *,
    max_texts: int,
    special_tokens: list[str] | tuple[str, ...] = (),
) -> Iterator[str]:
    emitted = 0
    if special_tokens:
        yield " ".join(str(token) for token in special_tokens)
        emitted += 1
        if max_texts > 0 and emitted >= max_texts:
            return
    for key in ("train_dataset_path", "eval_dataset_path"):
        path_value = str(dataset_manifest.get(key, "") or "")
        if not path_value:
            continue
        for text in _iter_jsonl_texts(Path(path_value)):
            yield text
            emitted += 1
            if max_texts > 0 and emitted >= max_texts:
                return


def _train_agentkernel_bpe(
    dataset_manifest: dict[str, Any],
    *,
    vocab_size: int,
    max_texts: int,
    use_agentkernel_special_tokens: bool,
) -> LocalBpeTokenizer:
    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.processors import TemplateProcessing
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:
        raise RuntimeError(
            "AgentKernel BPE training requires the 'tokenizers' package. "
            "Install tokenizers or run with --tokenizer-kind byte."
        ) from exc

    special_tokens = [
        LocalBpeTokenizer.pad_token,
        LocalBpeTokenizer.bos_token,
        LocalBpeTokenizer.eos_token,
        LocalBpeTokenizer.unk_token,
    ]
    if use_agentkernel_special_tokens:
        special_tokens.extend(LocalBpeTokenizer.default_agentkernel_special_tokens)
    tokenizer = Tokenizer(BPE(unk_token=LocalBpeTokenizer.unk_token))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=max(int(vocab_size), len(special_tokens)),
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        _tokenizer_training_texts(
            dataset_manifest,
            max_texts=int(max_texts),
            special_tokens=LocalBpeTokenizer.default_agentkernel_special_tokens if use_agentkernel_special_tokens else (),
        ),
        trainer=trainer,
    )
    bos_id = tokenizer.token_to_id(LocalBpeTokenizer.bos_token)
    eos_id = tokenizer.token_to_id(LocalBpeTokenizer.eos_token)
    if bos_id is None or eos_id is None:
        raise RuntimeError("trained AgentKernel BPE tokenizer is missing BOS/EOS special tokens")
    tokenizer.post_processor = TemplateProcessing(
        single=f"{LocalBpeTokenizer.bos_token} $A {LocalBpeTokenizer.eos_token}",
        pair=(
            f"{LocalBpeTokenizer.bos_token} $A {LocalBpeTokenizer.eos_token} "
            f"$B:1 {LocalBpeTokenizer.eos_token}:1"
        ),
        special_tokens=[
            (LocalBpeTokenizer.bos_token, int(bos_id)),
            (LocalBpeTokenizer.eos_token, int(eos_id)),
        ],
    )
    return LocalBpeTokenizer(tokenizer)


def _ensure_agentkernel_special_tokens(tokenizer) -> None:
    missing = [
        token
        for token in LocalBpeTokenizer.default_agentkernel_special_tokens
        if tokenizer.token_to_id(token) is None
    ]
    if missing:
        tokenizer.add_special_tokens(missing)


def _load_tokenizer(args: argparse.Namespace, *, dataset_manifest: dict[str, Any]):
    if bool(getattr(args, "byte_tokenizer", 0)):
        args.tokenizer_kind = "byte"
    tokenizer_kind = str(args.tokenizer_kind).strip().lower()
    if tokenizer_kind == "byte":
        return ByteTokenizer()
    if tokenizer_kind == "agentkernel-bpe":
        source_dir = str(getattr(args, "tokenizer_source_dir", "") or "").strip()
        source_tokenizer = Path(source_dir).expanduser().resolve() / "tokenizer.json" if source_dir else None
        existing_tokenizer = Path(str(args.output_dir)).expanduser().resolve() / "tokenizer" / "tokenizer.json"
        tokenizer_path = existing_tokenizer if existing_tokenizer.exists() else source_tokenizer
        if tokenizer_path is not None and tokenizer_path.exists():
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Loading an existing AgentKernel BPE tokenizer requires the 'tokenizers' package."
                ) from exc
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            checkpoint_continuation = bool(
                str(getattr(args, "init_from_checkpoint", "") or "").strip()
                or str(getattr(args, "resume_from", "") or "").strip()
                or bool(getattr(args, "resume_latest", 0))
            )
            allow_vocab_expand = str(getattr(args, "checkpoint_vocab_mismatch", "") or "").strip().lower() == "expand"
            if bool(args.agentkernel_special_tokens) and (not checkpoint_continuation or allow_vocab_expand):
                _ensure_agentkernel_special_tokens(tokenizer)
            return LocalBpeTokenizer(tokenizer)
        return _train_agentkernel_bpe(
            dataset_manifest,
            vocab_size=int(args.tokenizer_vocab_size),
            max_texts=int(args.tokenizer_max_texts),
            use_agentkernel_special_tokens=bool(args.agentkernel_special_tokens),
        )
    if tokenizer_kind != "hf":
        raise ValueError(f"unknown tokenizer kind: {args.tokenizer_kind}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_name), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    return tokenizer


class EncDecJsonlDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        *,
        max_encoder_tokens: int,
        max_decoder_tokens: int,
        max_retrieval_query_tokens: int = 96,
        max_retrieval_doc_tokens: int = 256,
        max_retrieval_negatives: int = 0,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("encoder_text", "")).strip() and str(row.get("decoder_text", "")).strip():
                    self.rows.append(row)
        self.tokenizer = tokenizer
        self.max_encoder_tokens = int(max_encoder_tokens)
        self.max_decoder_tokens = int(max_decoder_tokens)
        self.max_retrieval_query_tokens = int(max_retrieval_query_tokens)
        self.max_retrieval_doc_tokens = int(max_retrieval_doc_tokens)
        self.max_retrieval_negatives = max(0, int(max_retrieval_negatives))
        self.pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        self.decoder_start_token_id = int(
            getattr(tokenizer, "bos_token_id", None)
            or getattr(tokenizer, "eos_token_id", None)
            or self.pad_token_id
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return _encode_encdec_row(
            self.rows[index],
            tokenizer=self.tokenizer,
            max_encoder_tokens=self.max_encoder_tokens,
            max_decoder_tokens=self.max_decoder_tokens,
            max_retrieval_query_tokens=self.max_retrieval_query_tokens,
            max_retrieval_doc_tokens=self.max_retrieval_doc_tokens,
            max_retrieval_negatives=self.max_retrieval_negatives,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )


def _parse_retrieval_negative_doc_texts(raw: Any) -> list[str]:
    if raw is None:
        return []
    value: Any = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = [part.strip() for part in text.splitlines() if part.strip()]
    if isinstance(value, dict):
        value = value.get("docs") or value.get("negatives") or value.get("retrieval_negative_doc_texts") or []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _parse_positive_ints(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value > 0:
            values.append(value)
    return tuple(sorted(set(values)))


def _encode_decoder_sequence(
    tokenizer,
    text: str,
    *,
    prefix_text: str = "",
    max_decoder_tokens: int,
    pad_token_id: int,
    decoder_start_token_id: int,
) -> tuple[list[int], list[int], list[float]]:
    prefix_ids: list[int] = []
    if str(prefix_text or "").strip():
        prefix = tokenizer(
            str(prefix_text),
            max_length=int(max_decoder_tokens),
            padding=False,
            truncation=True,
            add_special_tokens=False,
        )
        prefix_ids = [
            int(token_id)
            for token_id in list(prefix["input_ids"])
            if int(token_id) not in {int(pad_token_id), int(decoder_start_token_id)}
        ]
    target = tokenizer(
        str(text),
        max_length=int(max_decoder_tokens),
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
    )
    labels = list(target["input_ids"])
    if labels and int(labels[0]) == int(decoder_start_token_id):
        labels = labels[1:]
    labels = [*prefix_ids, *labels]
    labels = labels[: int(max_decoder_tokens)]
    label_weights: list[float] = []
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    for index, token in enumerate(labels):
        if int(token) == int(pad_token_id):
            label_weights.append(0.0)
            continue
        if index < len(prefix_ids):
            label_weights.append(0.0)
            continue
        piece = str(tokenizer.decode([int(token)]) or "").strip()
        normalized = piece.lower()
        if not piece:
            weight = 0.6
        elif any(ch.isdigit() for ch in piece):
            weight = 2.2
        elif piece in {"{", "}", "[", "]", ":", ",", "-", "\n"} or any(ch in piece for ch in "{}[]:"):
            weight = 1.8
        elif re.search(r"[A-Z]", piece) and re.search(r"[a-z]", piece):
            weight = 2.0
        elif re.search(r"[A-Za-z0-9]", piece) and len(normalized) >= 4 and normalized not in stopwords:
            weight = 1.7
        elif normalized in stopwords:
            weight = 0.7
        else:
            weight = 1.0
        label_weights.append(float(weight))
    while len(labels) < int(max_decoder_tokens):
        labels.append(int(pad_token_id))
        label_weights.append(0.0)
    decoder_input_ids = [int(decoder_start_token_id), *labels[:-1]]
    label_ids = [
        token if token != int(pad_token_id) and index >= len(prefix_ids) else -100
        for index, token in enumerate(labels)
    ]
    return decoder_input_ids, label_ids, label_weights


def _encode_encdec_row(
    row: dict[str, Any],
    *,
    tokenizer,
    max_encoder_tokens: int,
    max_decoder_tokens: int,
    max_retrieval_query_tokens: int,
    max_retrieval_doc_tokens: int,
    max_retrieval_negatives: int,
    pad_token_id: int,
    decoder_start_token_id: int,
) -> dict[str, torch.Tensor]:
    enc = tokenizer(
        str(row["encoder_text"]),
        max_length=int(max_encoder_tokens),
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
    )
    decoder_input_ids, label_ids, label_token_weights = _encode_decoder_sequence(
        tokenizer,
        str(row["decoder_text"]),
        prefix_text=str(row.get("decoder_train_prefix", "") or ""),
        max_decoder_tokens=int(max_decoder_tokens),
        pad_token_id=int(pad_token_id),
        decoder_start_token_id=int(decoder_start_token_id),
    )
    negative_decoder_text = str(row.get("negative_decoder_text", "") or "").strip()
    negative_prefix_text = str(
        row.get("negative_decoder_train_prefix", row.get("decoder_train_prefix", "")) or ""
    )
    negative_decoder_input_ids, negative_label_ids, _negative_label_token_weights = _encode_decoder_sequence(
        tokenizer,
        negative_decoder_text if negative_decoder_text else "",
        prefix_text=negative_prefix_text if negative_decoder_text else "",
        max_decoder_tokens=int(max_decoder_tokens),
        pad_token_id=int(pad_token_id),
        decoder_start_token_id=int(decoder_start_token_id),
    )
    if not negative_decoder_text:
        negative_label_ids = [-100 for _ in negative_label_ids]
    retrieval_query_text = str(row.get("retrieval_query_text", "") or "").strip()
    retrieval_doc_text = str(row.get("retrieval_doc_text", "") or "").strip()
    has_retrieval_pair = bool(retrieval_query_text and retrieval_doc_text)
    retrieval_query = tokenizer(
        retrieval_query_text,
        max_length=int(max_retrieval_query_tokens),
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
    )
    retrieval_doc = tokenizer(
        retrieval_doc_text,
        max_length=int(max_retrieval_doc_tokens),
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
    )
    state_text = str(row.get("state_text", "") or row.get("expected_content", "") or "").strip()
    if not state_text:
        state_text = str(row.get("retrieval_doc_text", "") or "").strip()
    state_target = tokenizer(
        state_text,
        max_length=min(128, int(max_decoder_tokens)),
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
    )
    state_target_token_weights = []
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    for token in state_target["input_ids"]:
        token = int(token)
        if token == int(pad_token_id) or token <= 3:
            state_target_token_weights.append(0.0)
            continue
        piece = str(tokenizer.decode([token]) or "").strip()
        normalized = piece.lower()
        if not piece:
            weight = 0.0
        elif any(ch.isdigit() for ch in piece):
            weight = 2.2
        elif piece in {"{", "}", "[", "]", ":", ",", "-", "\n"} or any(ch in piece for ch in "{}[]:"):
            weight = 0.2
        elif re.search(r"[A-Z]", piece) and re.search(r"[a-z]", piece):
            weight = 2.0
        elif re.search(r"[A-Za-z0-9]", piece) and len(normalized) >= 4 and normalized not in stopwords:
            weight = 1.7
        elif normalized in stopwords:
            weight = 0.3
        else:
            weight = 0.8
        state_target_token_weights.append(float(weight))
    negative_texts = _parse_retrieval_negative_doc_texts(row.get("retrieval_negative_doc_texts"))
    negative_texts = negative_texts[: max(0, int(max_retrieval_negatives))]
    negative_doc_ids: list[list[int]] = []
    negative_doc_masks: list[list[int]] = []
    negative_valid_mask: list[int] = []
    for index in range(max(0, int(max_retrieval_negatives))):
        text = negative_texts[index] if index < len(negative_texts) else ""
        negative_doc = tokenizer(
            text,
            max_length=int(max_retrieval_doc_tokens),
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )
        negative_doc_ids.append([int(item) for item in negative_doc["input_ids"]])
        negative_doc_masks.append([int(item) for item in negative_doc["attention_mask"]])
        negative_valid_mask.append(1 if text else 0)
    if max(0, int(max_retrieval_negatives)) == 0:
        negative_doc_ids_tensor = torch.empty((0, int(max_retrieval_doc_tokens)), dtype=torch.long)
        negative_doc_masks_tensor = torch.empty((0, int(max_retrieval_doc_tokens)), dtype=torch.long)
        negative_valid_mask_tensor = torch.empty((0,), dtype=torch.bool)
    else:
        negative_doc_ids_tensor = torch.tensor(negative_doc_ids, dtype=torch.long)
        negative_doc_masks_tensor = torch.tensor(negative_doc_masks, dtype=torch.long)
        negative_valid_mask_tensor = torch.tensor(negative_valid_mask, dtype=torch.bool)

    def target_value(name: str) -> torch.Tensor:
        raw = row.get(name, None)
        if raw is None or raw == "":
            return torch.tensor(float("nan"), dtype=torch.float32)
        try:
            return torch.tensor(float(raw), dtype=torch.float32)
        except (TypeError, ValueError):
            return torch.tensor(float("nan"), dtype=torch.float32)

    def intent_label_id() -> torch.Tensor:
        raw = row.get("intent_label_id", row.get("intent_id", None))
        if raw is None or raw == "":
            return torch.tensor(-1, dtype=torch.long)
        try:
            return torch.tensor(int(raw), dtype=torch.long)
        except (TypeError, ValueError):
            return torch.tensor(-1, dtype=torch.long)

    def contrastive_label_id() -> torch.Tensor:
        raw = row.get("contrastive_label_id", None)
        if raw is None or raw == "":
            return torch.tensor(-1, dtype=torch.long)
        try:
            return torch.tensor(int(raw), dtype=torch.long)
        except (TypeError, ValueError):
            return torch.tensor(-1, dtype=torch.long)

    return {
        "enc_input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
        "enc_attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        "dec_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
        "labels": torch.tensor(label_ids, dtype=torch.long),
        "label_token_weights": torch.tensor(label_token_weights, dtype=torch.float32),
        "negative_dec_input_ids": torch.tensor(negative_decoder_input_ids, dtype=torch.long),
        "negative_labels": torch.tensor(negative_label_ids, dtype=torch.long),
        "negative_loss_weight": torch.tensor(float(row.get("negative_loss_weight", 0.0) or 0.0), dtype=torch.float32),
        "loss_weight": torch.tensor(float(row.get("weight", 1.0) or 1.0), dtype=torch.float32),
        "retrieval_query_ids": torch.tensor(retrieval_query["input_ids"], dtype=torch.long),
        "retrieval_query_attention_mask": torch.tensor(retrieval_query["attention_mask"], dtype=torch.long),
        "retrieval_doc_ids": torch.tensor(retrieval_doc["input_ids"], dtype=torch.long),
        "retrieval_doc_attention_mask": torch.tensor(retrieval_doc["attention_mask"], dtype=torch.long),
        "state_target_ids": torch.tensor(state_target["input_ids"], dtype=torch.long),
        "state_target_attention_mask": torch.tensor(state_target["attention_mask"], dtype=torch.long),
        "state_target_token_weights": torch.tensor(state_target_token_weights, dtype=torch.float32),
        "retrieval_negative_doc_ids": negative_doc_ids_tensor,
        "retrieval_negative_doc_attention_mask": negative_doc_masks_tensor,
        "retrieval_negative_doc_mask": negative_valid_mask_tensor,
        "retrieval_pair_mask": torch.tensor(1 if has_retrieval_pair else 0, dtype=torch.bool),
        "retrieval_loss_weight": torch.tensor(float(row.get("retrieval_loss_weight", 1.0) or 1.0), dtype=torch.float32),
        "query_confidence_target": target_value("query_confidence_target"),
        "retrieval_coverage_target": target_value("retrieval_coverage_target"),
        "ood_query_target": target_value("ood_query_target"),
        "ood_evidence_target": target_value("ood_evidence_target"),
        "answer_confidence_target": target_value("answer_confidence_target"),
        "needs_verification_target": target_value("needs_verification_target"),
        "paper_action_validity_target": target_value("paper_action_validity_target"),
        "intent_label_id": intent_label_id(),
        "contrastive_label_id": contrastive_label_id(),
    }


class EncDecParquetIterableDataset(IterableDataset):
    columns = [
        "encoder_text",
        "decoder_text",
        "expected_content",
        "state_text",
        "negative_decoder_text",
        "negative_loss_weight",
        "action",
        "task_type",
        "weight",
        "retrieval_query_text",
        "retrieval_doc_text",
        "retrieval_negative_doc_texts",
        "retrieval_loss_weight",
        "query_confidence_target",
        "retrieval_coverage_target",
        "ood_query_target",
        "ood_evidence_target",
        "answer_confidence_target",
        "needs_verification_target",
        "paper_action_validity_target",
        "intent_label_id",
        "contrastive_label_id",
    ]

    def __init__(
        self,
        path: Path,
        tokenizer,
        *,
        max_encoder_tokens: int,
        max_decoder_tokens: int,
        max_retrieval_query_tokens: int = 96,
        max_retrieval_doc_tokens: int = 256,
        max_retrieval_negatives: int = 0,
        batch_read_size: int = 512,
        shuffle: bool = False,
        shuffle_buffer_size: int = 0,
        require_retrieval_pair: bool = False,
        action_include: tuple[str, ...] = (),
        task_type_include: tuple[str, ...] = (),
        seed: int = 1,
    ) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("parquet training data requires pyarrow") from exc
        self.paths = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
        self.paths = [item for item in self.paths if item.exists() and item.suffix == ".parquet"]
        if not self.paths:
            raise FileNotFoundError(f"no parquet training shards found at {path}")
        self.row_count = sum(int(pq.ParquetFile(item).metadata.num_rows) for item in self.paths)
        self.tokenizer = tokenizer
        self.max_encoder_tokens = int(max_encoder_tokens)
        self.max_decoder_tokens = int(max_decoder_tokens)
        self.max_retrieval_query_tokens = int(max_retrieval_query_tokens)
        self.max_retrieval_doc_tokens = int(max_retrieval_doc_tokens)
        self.max_retrieval_negatives = max(0, int(max_retrieval_negatives))
        self.batch_read_size = max(1, int(batch_read_size))
        self.shuffle = bool(shuffle)
        self.shuffle_buffer_size = max(0, int(shuffle_buffer_size))
        self.require_retrieval_pair = bool(require_retrieval_pair)
        self.action_include = tuple(str(item).strip() for item in action_include if str(item).strip())
        self.task_type_include = tuple(str(item).strip() for item in task_type_include if str(item).strip())
        self.seed = int(seed)
        self.pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        self.decoder_start_token_id = int(
            getattr(tokenizer, "bos_token_id", None)
            or getattr(tokenizer, "eos_token_id", None)
            or self.pad_token_id
        )

    def __len__(self) -> int:
        return self.row_count

    def __iter__(self):
        import pyarrow.parquet as pq

        rng = random.Random(self.seed)
        worker = get_worker_info()
        if worker is None:
            paths = list(self.paths)
        else:
            paths = list(self.paths[worker.id :: worker.num_workers])
        if self.shuffle:
            rng.shuffle(paths)
        buffer: list[dict[str, torch.Tensor]] = []

        def emit(item: dict[str, torch.Tensor]):
            if self.shuffle and self.shuffle_buffer_size > 1:
                buffer.append(item)
                if len(buffer) >= self.shuffle_buffer_size:
                    index = rng.randrange(len(buffer))
                    return buffer.pop(index)
                return None
            return item

        for path in paths:
            parquet_file = pq.ParquetFile(path)
            available = set(parquet_file.schema_arrow.names)
            read_columns = [column for column in self.columns if column in available]
            for batch in parquet_file.iter_batches(batch_size=self.batch_read_size, columns=read_columns):
                for row in batch.to_pylist():
                    if not str(row.get("encoder_text", "") or "").strip():
                        continue
                    if not str(row.get("decoder_text", "") or "").strip():
                        continue
                    if self.require_retrieval_pair and not (
                        str(row.get("retrieval_query_text", "") or "").strip()
                        and str(row.get("retrieval_doc_text", "") or "").strip()
                    ):
                        continue
                    if self.action_include and str(row.get("action", "") or "") not in self.action_include:
                        continue
                    if self.task_type_include and str(row.get("task_type", "") or "") not in self.task_type_include:
                        continue
                    item = _encode_encdec_row(
                        row,
                        tokenizer=self.tokenizer,
                        max_encoder_tokens=self.max_encoder_tokens,
                        max_decoder_tokens=self.max_decoder_tokens,
                        max_retrieval_query_tokens=self.max_retrieval_query_tokens,
                        max_retrieval_doc_tokens=self.max_retrieval_doc_tokens,
                        max_retrieval_negatives=self.max_retrieval_negatives,
                        pad_token_id=self.pad_token_id,
                        decoder_start_token_id=self.decoder_start_token_id,
                    )
                    emitted = emit(item)
                    if emitted is not None:
                        yield emitted
        if self.shuffle and self.shuffle_buffer_size > 1:
            while buffer:
                index = rng.randrange(len(buffer))
                yield buffer.pop(index)


class EncDecHfStreamIterableDataset(IterableDataset):
    """Streaming HF curriculum rows converted on the fly.

    The manifest supplies compact dataset specs; the trainer stores only model
    progress/checkpoints, not dataset snapshots.
    """

    def __init__(
        self,
        specs: list[str],
        tokenizer,
        *,
        max_encoder_tokens: int,
        max_decoder_tokens: int,
        max_retrieval_query_tokens: int = 96,
        max_retrieval_doc_tokens: int = 256,
        max_retrieval_negatives: int = 0,
        weight: float = 3.0,
        shuffle: bool = False,
        shuffle_buffer_size: int = 0,
        seed: int = 1,
    ) -> None:
        if not specs:
            raise ValueError("streaming dataset requires at least one dataset spec")
        self.specs = [str(item) for item in specs if str(item).strip()]
        self.tokenizer = tokenizer
        self.max_encoder_tokens = int(max_encoder_tokens)
        self.max_decoder_tokens = int(max_decoder_tokens)
        self.max_retrieval_query_tokens = int(max_retrieval_query_tokens)
        self.max_retrieval_doc_tokens = int(max_retrieval_doc_tokens)
        self.max_retrieval_negatives = max(0, int(max_retrieval_negatives))
        self.weight = float(weight)
        self.shuffle = bool(shuffle)
        self.shuffle_buffer_size = max(0, int(shuffle_buffer_size))
        self.seed = int(seed)
        self.pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        self.decoder_start_token_id = int(
            getattr(tokenizer, "bos_token_id", None)
            or getattr(tokenizer, "eos_token_id", None)
            or self.pad_token_id
        )
        self.row_count = sum(self._spec_limit(spec) for spec in self.specs)

    @staticmethod
    def _passes_quality_filter(prompt: str, answer: str) -> bool:
        prompt_text = " ".join(str(prompt or "").split())
        answer_text = " ".join(str(answer or "").split())
        if len(prompt_text) < 12 or len(answer_text) < 18:
            return False
        lower = answer_text.lower()
        blocked = (
            "bfcl v2",
            "live multiple",
            "function calling leaderboard",
            "<tool_call>",
            "<functioncall>",
        )
        if any(item in lower for item in blocked):
            return False
        words = re.findall(r"[A-Za-z0-9_'-]+", answer_text)
        if len(words) < 5:
            return False
        unique_ratio = len(set(word.lower() for word in words)) / max(1, len(words))
        if unique_ratio < 0.35:
            return False
        return True

    @staticmethod
    def _spec_limit(spec: str) -> int:
        parts = str(spec).split(":")
        while len(parts) < 5:
            parts.append("")
        try:
            return max(0, int(parts[4] or 1000))
        except ValueError:
            return 1000

    def __len__(self) -> int:
        return self.row_count

    def __iter__(self):
        try:
            from datasets import load_dataset
            from build_pocketpal_nvidia_hf_curriculum import _extract_pair, _parse_spec, _row
        except ImportError as exc:
            raise RuntimeError("HF streaming curriculum requires datasets and the NVIDIA curriculum builder") from exc

        rng = random.Random(self.seed)
        worker = get_worker_info()
        specs = list(self.specs)
        if self.shuffle:
            rng.shuffle(specs)
        if worker is not None:
            specs = specs[worker.id :: worker.num_workers]

        buffer: list[dict[str, torch.Tensor]] = []

        def emit(item: dict[str, torch.Tensor]):
            if self.shuffle and self.shuffle_buffer_size > 1:
                buffer.append(item)
                if len(buffer) >= self.shuffle_buffer_size:
                    index = rng.randrange(len(buffer))
                    return buffer.pop(index)
                return None
            return item

        for spec in specs:
            name, config, split, intent, limit = _parse_spec(spec)
            kwargs: dict[str, Any] = {"split": split, "streaming": True}
            try:
                stream = load_dataset(name, config, **kwargs) if config else load_dataset(name, **kwargs)
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "event": "hf_stream_dataset_skipped",
                            "dataset": name,
                            "config": config,
                            "split": split,
                            "intent": intent,
                            "error": repr(exc),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            scanned = 0
            for row_index, source_row in enumerate(stream):
                if scanned >= int(limit):
                    break
                if not isinstance(source_row, dict):
                    continue
                prompt, answer, context, retrieval_doc = _extract_pair(source_row, fallback_intent=intent)
                if not self._passes_quality_filter(prompt, answer):
                    scanned += 1
                    continue
                built = _row(
                    dataset_name=name if not config else f"{name}/{config}",
                    source_id=f"{name}:{config or 'default'}:{split}:{row_index}",
                    intent=intent,
                    user=prompt,
                    content=answer,
                    context=context,
                    retrieval_doc=retrieval_doc,
                    weight=self.weight,
                )
                scanned += 1
                if built is None:
                    continue
                item = _encode_encdec_row(
                    built,
                    tokenizer=self.tokenizer,
                    max_encoder_tokens=self.max_encoder_tokens,
                    max_decoder_tokens=self.max_decoder_tokens,
                    max_retrieval_query_tokens=self.max_retrieval_query_tokens,
                    max_retrieval_doc_tokens=self.max_retrieval_doc_tokens,
                    max_retrieval_negatives=self.max_retrieval_negatives,
                    pad_token_id=self.pad_token_id,
                    decoder_start_token_id=self.decoder_start_token_id,
                )
                emitted = emit(item)
                if emitted is not None:
                    yield emitted
        if self.shuffle and self.shuffle_buffer_size > 1:
            while buffer:
                index = rng.randrange(len(buffer))
                yield buffer.pop(index)


def _is_parquet_dataset_path(path: Path) -> bool:
    return path.suffix == ".parquet" or (path.is_dir() and any(path.glob("*.parquet")))


def _load_encdec_dataset(
    path: Path,
    tokenizer,
    *,
    max_encoder_tokens: int,
    max_decoder_tokens: int,
    max_retrieval_query_tokens: int,
    max_retrieval_doc_tokens: int,
    max_retrieval_negatives: int,
    shuffle: bool = False,
    shuffle_buffer_size: int = 0,
    require_retrieval_pair: bool = False,
    action_include: tuple[str, ...] = (),
    task_type_include: tuple[str, ...] = (),
    seed: int = 1,
):
    if _is_parquet_dataset_path(path):
        return EncDecParquetIterableDataset(
            path,
            tokenizer,
            max_encoder_tokens=max_encoder_tokens,
            max_decoder_tokens=max_decoder_tokens,
            max_retrieval_query_tokens=max_retrieval_query_tokens,
            max_retrieval_doc_tokens=max_retrieval_doc_tokens,
            max_retrieval_negatives=max_retrieval_negatives,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            require_retrieval_pair=require_retrieval_pair,
            action_include=action_include,
            task_type_include=task_type_include,
            seed=seed,
        )
    return EncDecJsonlDataset(
        path,
        tokenizer,
        max_encoder_tokens=max_encoder_tokens,
        max_decoder_tokens=max_decoder_tokens,
        max_retrieval_query_tokens=max_retrieval_query_tokens,
        max_retrieval_doc_tokens=max_retrieval_doc_tokens,
        max_retrieval_negatives=max_retrieval_negatives,
    )


def _collate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([row[key] for row in rows], dim=0) for key in rows[0]}


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _retrieval_contrastive_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    temperature: float,
) -> torch.Tensor:
    pair_mask = batch.get("retrieval_pair_mask")
    if pair_mask is None or int(pair_mask.sum().detach().cpu().item()) <= 1:
        return next(model.parameters()).new_tensor(0.0)
    pair_mask = pair_mask.to(dtype=torch.bool)
    query_ids = batch["retrieval_query_ids"][pair_mask]
    query_mask = batch["retrieval_query_attention_mask"][pair_mask]
    doc_ids = batch["retrieval_doc_ids"][pair_mask]
    doc_mask = batch["retrieval_doc_attention_mask"][pair_mask]
    if hasattr(model, "retrieval_query_embedding") and hasattr(model, "retrieval_doc_embedding"):
        query_embeddings = model.retrieval_query_embedding(query_ids, query_mask)
        doc_embeddings = model.retrieval_doc_embedding(doc_ids, doc_mask)
    else:
        query_hidden = model.encode(query_ids, query_mask)
        doc_hidden = model.encode(doc_ids, doc_mask)
        query_embeddings = F.normalize(_mean_pool(query_hidden, query_mask), dim=-1)
        doc_embeddings = F.normalize(_mean_pool(doc_hidden, doc_mask), dim=-1)
    logits = query_embeddings @ doc_embeddings.transpose(0, 1)
    logits = logits / max(float(temperature), 1e-6)
    labels = torch.arange(logits.shape[0], device=logits.device)
    query_loss = F.cross_entropy(logits, labels, reduction="none")
    doc_loss = F.cross_entropy(logits.transpose(0, 1), labels, reduction="none")
    weights = batch["retrieval_loss_weight"][pair_mask].to(device=logits.device, dtype=logits.dtype).clamp_min(0.0)
    return ((query_loss + doc_loss) * 0.5 * weights).sum() / weights.sum().clamp_min(1e-6)


def _ternary_grouped_signed_residual_ste(
    embeddings: torch.Tensor,
    *,
    threshold_ratio: float,
    group_size: int,
    residual_dims: int,
) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError("retrieval embeddings must be rank-2")
    group_size = max(1, int(group_size))
    residual_dims = max(0, int(residual_dims))
    chunks: list[torch.Tensor] = []
    for start in range(0, embeddings.shape[1], group_size):
        block = embeddings[:, start : start + group_size]
        max_abs = block.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
        threshold = max_abs * max(0.0, float(threshold_ratio))
        code = torch.where(block > threshold, torch.ones_like(block), torch.where(block < -threshold, -torch.ones_like(block), torch.zeros_like(block)))
        pos = (code > 0).to(dtype=block.dtype)
        neg = (code < 0).to(dtype=block.dtype)
        pos_scale = (block * pos).sum(dim=1, keepdim=True) / pos.sum(dim=1, keepdim=True).clamp_min(1.0)
        neg_scale = ((-block) * neg).sum(dim=1, keepdim=True) / neg.sum(dim=1, keepdim=True).clamp_min(1.0)
        chunks.append(torch.where(code > 0, pos_scale, torch.where(code < 0, -neg_scale, torch.zeros_like(block))))
    quantized = torch.cat(chunks, dim=1)
    if residual_dims > 0:
        residual = embeddings - quantized
        keep = min(residual_dims, residual.shape[1])
        if keep > 0:
            indices = residual.abs().topk(k=keep, dim=1).indices
            correction = torch.zeros_like(residual).scatter(1, indices, residual.gather(1, indices))
            quantized = quantized + correction
    return embeddings + (quantized - embeddings).detach()


def _retrieval_ternary_aware_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    temperature: float,
    threshold_ratio: float,
    group_size: int,
    residual_dims: int,
) -> torch.Tensor:
    pair_mask = batch.get("retrieval_pair_mask")
    if pair_mask is None or int(pair_mask.sum().detach().cpu().item()) <= 1:
        return next(model.parameters()).new_tensor(0.0)
    pair_mask = pair_mask.to(dtype=torch.bool)
    query_ids = batch["retrieval_query_ids"][pair_mask]
    query_mask = batch["retrieval_query_attention_mask"][pair_mask]
    doc_ids = batch["retrieval_doc_ids"][pair_mask]
    doc_mask = batch["retrieval_doc_attention_mask"][pair_mask]
    if hasattr(model, "retrieval_query_embedding") and hasattr(model, "retrieval_doc_embedding"):
        query_embeddings = model.retrieval_query_embedding(query_ids, query_mask)
        doc_embeddings = model.retrieval_doc_embedding(doc_ids, doc_mask)
    else:
        query_hidden = model.encode(query_ids, query_mask)
        doc_hidden = model.encode(doc_ids, doc_mask)
        query_embeddings = F.normalize(_mean_pool(query_hidden, query_mask), dim=-1)
        doc_embeddings = F.normalize(_mean_pool(doc_hidden, doc_mask), dim=-1)
    quantized_docs = _ternary_grouped_signed_residual_ste(
        doc_embeddings,
        threshold_ratio=float(threshold_ratio),
        group_size=int(group_size),
        residual_dims=int(residual_dims),
    )
    quantized_docs = F.normalize(quantized_docs.float(), dim=-1).to(dtype=query_embeddings.dtype)
    logits = query_embeddings @ quantized_docs.transpose(0, 1)
    logits = logits / max(float(temperature), 1e-6)
    labels = torch.arange(logits.shape[0], device=logits.device)
    query_loss = F.cross_entropy(logits, labels, reduction="none")
    doc_loss = F.cross_entropy(logits.transpose(0, 1), labels, reduction="none")
    weights = batch["retrieval_loss_weight"][pair_mask].to(device=logits.device, dtype=logits.dtype).clamp_min(0.0)
    return ((query_loss + doc_loss) * 0.5 * weights).sum() / weights.sum().clamp_min(1e-6)


def _retrieval_ternary_teacher_distill_loss(
    model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    temperature: float,
    threshold_ratio: float,
    group_size: int,
    residual_dims: int,
) -> torch.Tensor:
    pair_mask = batch.get("retrieval_pair_mask")
    if pair_mask is None or int(pair_mask.sum().detach().cpu().item()) <= 1:
        return next(model.parameters()).new_tensor(0.0)
    pair_mask = pair_mask.to(dtype=torch.bool)
    query_ids = batch["retrieval_query_ids"][pair_mask]
    query_mask = batch["retrieval_query_attention_mask"][pair_mask]
    doc_ids = batch["retrieval_doc_ids"][pair_mask]
    doc_mask = batch["retrieval_doc_attention_mask"][pair_mask]
    if hasattr(model, "retrieval_query_embedding") and hasattr(model, "retrieval_doc_embedding"):
        query_embeddings = model.retrieval_query_embedding(query_ids, query_mask)
        doc_embeddings = model.retrieval_doc_embedding(doc_ids, doc_mask)
    else:
        query_hidden = model.encode(query_ids, query_mask)
        doc_hidden = model.encode(doc_ids, doc_mask)
        query_embeddings = F.normalize(_mean_pool(query_hidden, query_mask), dim=-1)
        doc_embeddings = F.normalize(_mean_pool(doc_hidden, doc_mask), dim=-1)
    quantized_docs = _ternary_grouped_signed_residual_ste(
        doc_embeddings,
        threshold_ratio=float(threshold_ratio),
        group_size=int(group_size),
        residual_dims=int(residual_dims),
    )
    quantized_docs = F.normalize(quantized_docs.float(), dim=-1).to(dtype=query_embeddings.dtype)
    student_logits = query_embeddings @ quantized_docs.transpose(0, 1)

    with torch.no_grad():
        if hasattr(teacher_model, "retrieval_query_embedding") and hasattr(teacher_model, "retrieval_doc_embedding"):
            teacher_query = teacher_model.retrieval_query_embedding(query_ids, query_mask)
            teacher_doc = teacher_model.retrieval_doc_embedding(doc_ids, doc_mask)
        else:
            teacher_query_hidden = teacher_model.encode(query_ids, query_mask)
            teacher_doc_hidden = teacher_model.encode(doc_ids, doc_mask)
            teacher_query = F.normalize(_mean_pool(teacher_query_hidden, query_mask), dim=-1)
            teacher_doc = F.normalize(_mean_pool(teacher_doc_hidden, doc_mask), dim=-1)
        teacher_logits = teacher_query @ teacher_doc.transpose(0, 1)

    temp = max(float(temperature), 1e-6)
    student_log_probs = F.log_softmax(student_logits.float() / temp, dim=1)
    teacher_probs = F.softmax(teacher_logits.float() / temp, dim=1)
    row_loss = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1) * (temp * temp)
    student_log_probs_t = F.log_softmax(student_logits.float().transpose(0, 1) / temp, dim=1)
    teacher_probs_t = F.softmax(teacher_logits.float().transpose(0, 1) / temp, dim=1)
    col_loss = F.kl_div(student_log_probs_t, teacher_probs_t, reduction="none").sum(dim=1) * (temp * temp)
    weights = batch["retrieval_loss_weight"][pair_mask].to(device=student_logits.device, dtype=student_logits.dtype).clamp_min(0.0)
    row_weighted = (row_loss * weights).sum() / weights.sum().clamp_min(1e-6)
    col_weighted = (col_loss * weights).sum() / weights.sum().clamp_min(1e-6)
    return (row_weighted + col_weighted) * 0.5


def _retrieval_ternary_reconstruction_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    threshold_ratio: float,
    group_size: int,
    residual_dims: int,
) -> torch.Tensor:
    pair_mask = batch.get("retrieval_pair_mask")
    if pair_mask is None or int(pair_mask.sum().detach().cpu().item()) <= 0:
        return next(model.parameters()).new_tensor(0.0)
    pair_mask = pair_mask.to(dtype=torch.bool)
    doc_ids = batch["retrieval_doc_ids"][pair_mask]
    doc_mask = batch["retrieval_doc_attention_mask"][pair_mask]
    if hasattr(model, "retrieval_doc_embedding"):
        doc_embeddings = model.retrieval_doc_embedding(doc_ids, doc_mask)
    else:
        doc_hidden = model.encode(doc_ids, doc_mask)
        doc_embeddings = F.normalize(_mean_pool(doc_hidden, doc_mask), dim=-1)
    quantized_docs = _ternary_grouped_signed_residual_ste(
        doc_embeddings,
        threshold_ratio=float(threshold_ratio),
        group_size=int(group_size),
        residual_dims=int(residual_dims),
    )
    target = quantized_docs.detach()
    per_example = F.mse_loss(doc_embeddings.float(), target.float(), reduction="none").mean(dim=1)
    weights = batch["retrieval_loss_weight"][pair_mask].to(device=per_example.device, dtype=per_example.dtype).clamp_min(0.0)
    return (per_example * weights).sum() / weights.sum().clamp_min(1e-6)


def _retrieval_hard_negative_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    temperature: float,
    use_ternary_docs: bool,
    threshold_ratio: float,
    group_size: int,
    residual_dims: int,
) -> torch.Tensor:
    pair_mask = batch.get("retrieval_pair_mask")
    if pair_mask is None or int(pair_mask.sum().detach().cpu().item()) <= 1:
        return next(model.parameters()).new_tensor(0.0)
    pair_mask = pair_mask.to(dtype=torch.bool)
    query_ids = batch["retrieval_query_ids"][pair_mask]
    query_mask = batch["retrieval_query_attention_mask"][pair_mask]
    doc_ids = batch["retrieval_doc_ids"][pair_mask]
    doc_mask = batch["retrieval_doc_attention_mask"][pair_mask]
    if hasattr(model, "retrieval_query_embedding") and hasattr(model, "retrieval_doc_embedding"):
        query_embeddings = model.retrieval_query_embedding(query_ids, query_mask)
        positive_doc_embeddings = model.retrieval_doc_embedding(doc_ids, doc_mask)
    else:
        query_hidden = model.encode(query_ids, query_mask)
        doc_hidden = model.encode(doc_ids, doc_mask)
        query_embeddings = F.normalize(_mean_pool(query_hidden, query_mask), dim=-1)
        positive_doc_embeddings = F.normalize(_mean_pool(doc_hidden, doc_mask), dim=-1)

    candidate_embeddings = [positive_doc_embeddings]
    negative_doc_ids = batch.get("retrieval_negative_doc_ids")
    negative_doc_masks = batch.get("retrieval_negative_doc_attention_mask")
    negative_valid_mask = batch.get("retrieval_negative_doc_mask")
    if (
        negative_doc_ids is not None
        and negative_doc_masks is not None
        and negative_valid_mask is not None
        and negative_doc_ids.ndim == 3
        and negative_doc_ids.shape[1] > 0
    ):
        selected_negative_ids = negative_doc_ids[pair_mask]
        selected_negative_masks = negative_doc_masks[pair_mask]
        selected_negative_valid = negative_valid_mask[pair_mask].to(dtype=torch.bool)
        if int(selected_negative_valid.sum().detach().cpu().item()) > 0:
            flat_negative_ids = selected_negative_ids[selected_negative_valid]
            flat_negative_masks = selected_negative_masks[selected_negative_valid]
            if hasattr(model, "retrieval_doc_embedding"):
                negative_embeddings = model.retrieval_doc_embedding(flat_negative_ids, flat_negative_masks)
            else:
                negative_hidden = model.encode(flat_negative_ids, flat_negative_masks)
                negative_embeddings = F.normalize(_mean_pool(negative_hidden, flat_negative_masks), dim=-1)
            candidate_embeddings.append(negative_embeddings)
    docs = torch.cat(candidate_embeddings, dim=0)
    if bool(use_ternary_docs):
        docs = _ternary_grouped_signed_residual_ste(
            docs,
            threshold_ratio=float(threshold_ratio),
            group_size=int(group_size),
            residual_dims=int(residual_dims),
        )
        docs = F.normalize(docs.float(), dim=-1).to(dtype=query_embeddings.dtype)
    logits = query_embeddings @ docs.transpose(0, 1)
    logits = logits / max(float(temperature), 1e-6)
    labels = torch.arange(query_embeddings.shape[0], device=logits.device)
    row_loss = F.cross_entropy(logits, labels, reduction="none")
    weights = batch["retrieval_loss_weight"][pair_mask].to(device=logits.device, dtype=logits.dtype).clamp_min(0.0)
    return (row_loss * weights).sum() / weights.sum().clamp_min(1e-6)


def _agent_policy_head_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if not hasattr(model, "agent_policy_logits"):
        return next(model.parameters()).new_tensor(0.0)
    logits = model.agent_policy_logits(batch["enc_input_ids"], batch["enc_attention_mask"])
    if not logits:
        return next(model.parameters()).new_tensor(0.0)
    target_names = {
        "query_confidence": "query_confidence_target",
        "retrieval_coverage": "retrieval_coverage_target",
        "ood_query": "ood_query_target",
        "ood_evidence": "ood_evidence_target",
        "answer_confidence": "answer_confidence_target",
        "needs_verification": "needs_verification_target",
        "paper_action_validity": "paper_action_validity_target",
    }
    losses = []
    for head_name, target_name in target_names.items():
        if head_name not in logits or target_name not in batch:
            continue
        targets = batch[target_name].to(device=logits[head_name].device, dtype=logits[head_name].dtype)
        mask = torch.isfinite(targets)
        if int(mask.sum().detach().cpu().item()) <= 0:
            continue
        predictions = torch.sigmoid(logits[head_name][mask])
        losses.append(F.mse_loss(predictions, targets[mask].clamp(0.0, 1.0), reduction="mean"))
    if not losses:
        return next(model.parameters()).new_tensor(0.0)
    return torch.stack(losses).mean()


def _agent_intent_head_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if not hasattr(model, "agent_intent_logits"):
        return next(model.parameters()).new_tensor(0.0)
    logits = model.agent_intent_logits(batch["enc_input_ids"], batch["enc_attention_mask"])
    if logits is None:
        return next(model.parameters()).new_tensor(0.0)
    labels = batch.get("intent_label_id")
    if labels is None:
        return next(model.parameters()).new_tensor(0.0)
    labels = labels.to(device=logits.device, dtype=torch.long)
    mask = (labels >= 0) & (labels < int(logits.shape[-1]))
    if int(mask.sum().detach().cpu().item()) <= 0:
        return next(model.parameters()).new_tensor(0.0)
    losses = F.cross_entropy(logits[mask].float(), labels[mask], reduction="none")
    weights = batch.get("loss_weight")
    if weights is None:
        return losses.mean()
    weights = weights.to(device=losses.device, dtype=losses.dtype)[mask].clamp_min(0.0)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def _pooled_controller_state(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    encode_controller_state = getattr(model, "encode_controller_state", None)
    if callable(encode_controller_state):
        return encode_controller_state(input_ids, attention_mask).float()
    hidden = model.encode(input_ids, attention_mask)
    return _mean_pool(hidden, attention_mask).float()


def _intent_contrastive_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    temperature: float,
) -> torch.Tensor:
    labels = batch.get("contrastive_label_id")
    if labels is None or int((labels >= 0).sum().detach().cpu().item()) <= 1:
        labels = batch.get("intent_label_id")
    if labels is None:
        return next(model.parameters()).new_tensor(0.0)
    labels = labels.to(device=batch["enc_input_ids"].device, dtype=torch.long)
    valid = labels >= 0
    if int(valid.sum().detach().cpu().item()) <= 1:
        return next(model.parameters()).new_tensor(0.0)
    labels = labels[valid]
    if int(torch.unique(labels).shape[0]) <= 1:
        return next(model.parameters()).new_tensor(0.0)
    pooled = _pooled_controller_state(model, batch["enc_input_ids"][valid], batch["enc_attention_mask"][valid])
    pooled = F.normalize(pooled.float(), dim=-1)
    logits = pooled @ pooled.transpose(0, 1)
    logits = logits / max(float(temperature), 1e-6)
    eye = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    positives = same_label & ~eye
    anchor_mask = positives.any(dim=1)
    if int(anchor_mask.sum().detach().cpu().item()) <= 0:
        return next(model.parameters()).new_tensor(0.0)
    logits = logits - logits.detach().amax(dim=1, keepdim=True)
    exp_logits = torch.exp(logits).masked_fill(eye, 0.0)
    positive_sum = (exp_logits * positives.to(dtype=exp_logits.dtype)).sum(dim=1)
    denominator = exp_logits.sum(dim=1).clamp_min(1e-12)
    losses = -torch.log((positive_sum / denominator).clamp_min(1e-12))
    return losses[anchor_mask].mean()


def _future_bow_aux_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    buckets: int,
) -> torch.Tensor:
    head = getattr(model, "future_bow_aux_head", None)
    if head is None or int(buckets) <= 0:
        return next(model.parameters()).new_tensor(0.0)
    labels = batch["labels"]
    valid = labels >= 0
    token_weights = batch.get("label_token_weights")
    if token_weights is not None:
        valid = valid & (token_weights.to(device=labels.device, dtype=torch.float32) > 1.05)
    if not bool(valid.any().item()):
        return next(model.parameters()).new_tensor(0.0)
    pooled = _pooled_controller_state(model, batch["enc_input_ids"], batch["enc_attention_mask"])
    logits = head(pooled)
    targets = torch.zeros((labels.shape[0], int(buckets)), device=logits.device, dtype=logits.dtype)
    bucket_ids = labels.clamp_min(0).remainder(int(buckets)).to(device=logits.device, dtype=torch.long)
    valid_on_device = valid.to(device=logits.device)
    targets.scatter_(1, bucket_ids.masked_fill(~valid_on_device, 0), valid_on_device.to(dtype=logits.dtype))
    per_example = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=1)
    weights = batch["loss_weight"].to(device=logits.device, dtype=logits.dtype).clamp_min(0.0)
    return (per_example * weights).sum() / weights.sum().clamp_min(1e-6)


def _future_sketch_aux_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    buckets: int,
    min_token_weight: float,
    topk: int,
    windows: tuple[int, ...] = (),
) -> torch.Tensor:
    head = getattr(model, "future_sketch_aux_head", None)
    if head is None or int(buckets) <= 0:
        return next(model.parameters()).new_tensor(0.0)
    labels = batch["labels"]
    valid = labels >= 0
    token_weights = batch.get("label_token_weights")
    if token_weights is None:
        return next(model.parameters()).new_tensor(0.0)
    info_weights = token_weights.to(device=labels.device, dtype=torch.float32)
    valid = valid & (info_weights >= float(min_token_weight))
    if not bool(valid.any().item()):
        return next(model.parameters()).new_tensor(0.0)
    pooled = _pooled_controller_state(model, batch["enc_input_ids"], batch["enc_attention_mask"])
    logits = head(pooled)
    horizon_windows = tuple(sorted({max(1, int(item)) for item in windows if int(item) > 0}))
    if horizon_windows:
        expected_dim = int(buckets) * len(horizon_windows)
        if logits.shape[-1] != expected_dim:
            raise RuntimeError(
                f"future sketch head dim {logits.shape[-1]} does not match "
                f"buckets*windows {expected_dim}"
            )
        logits_by_window = logits.reshape(labels.shape[0], len(horizon_windows), int(buckets))
    else:
        horizon_windows = (labels.shape[1],)
        logits_by_window = logits.reshape(labels.shape[0], 1, int(buckets))

    per_window_losses: list[torch.Tensor] = []
    per_window_valid: list[torch.Tensor] = []
    bucket_ids_all = labels.clamp_min(0).remainder(int(buckets)).to(device=logits.device, dtype=torch.long)
    values_all = (info_weights * valid.to(dtype=info_weights.dtype)).to(device=logits.device, dtype=logits.dtype)
    for window_index, window in enumerate(horizon_windows):
        window_len = min(int(window), labels.shape[1])
        window_valid = valid[:, :window_len]
        if not bool(window_valid.any().item()):
            continue
        targets = torch.zeros((labels.shape[0], int(buckets)), device=logits.device, dtype=logits.dtype)
        bucket_ids = bucket_ids_all[:, :window_len]
        values = values_all[:, :window_len]
        targets.scatter_add_(1, bucket_ids, values)
        if int(topk) > 0 and int(topk) < int(buckets):
            keep = min(int(topk), int(buckets))
            top_values, top_indices = targets.topk(k=keep, dim=1)
            sparse = torch.zeros_like(targets)
            sparse.scatter_(1, top_indices, top_values)
            targets = sparse
        row_sums = targets.sum(dim=1, keepdim=True)
        has_target = row_sums.squeeze(1) > 0
        if not bool(has_target.any().item()):
            continue
        targets = targets / row_sums.clamp_min(1e-6)
        log_probs = F.log_softmax(logits_by_window[:, window_index, :], dim=-1)
        per_window_losses.append(-(targets * log_probs).sum(dim=1))
        per_window_valid.append(has_target)
    if not per_window_losses:
        return next(model.parameters()).new_tensor(0.0)
    stacked_losses = torch.stack(per_window_losses, dim=1)
    stacked_valid = torch.stack(per_window_valid, dim=1).to(device=logits.device, dtype=logits.dtype)
    per_example = (stacked_losses * stacked_valid).sum(dim=1) / stacked_valid.sum(dim=1).clamp_min(1.0)
    has_target = stacked_valid.any(dim=1)
    weights = batch["loss_weight"].to(device=logits.device, dtype=logits.dtype).clamp_min(0.0)
    weights = weights * has_target.to(dtype=weights.dtype)
    return (per_example * weights).sum() / weights.sum().clamp_min(1e-6)


def _state_sketch_aux_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    buckets: int,
    topk: int,
    min_token_weight: float,
) -> torch.Tensor:
    head = getattr(model, "state_sketch_aux_head", None)
    if head is None or int(buckets) <= 0:
        return next(model.parameters()).new_tensor(0.0)
    state_ids = batch.get("state_target_ids")
    state_mask = batch.get("state_target_attention_mask")
    state_weights = batch.get("state_target_token_weights")
    if state_ids is None or state_mask is None or state_weights is None:
        return next(model.parameters()).new_tensor(0.0)
    info_weights = state_weights.to(device=state_ids.device, dtype=torch.float32)
    valid = (
        state_mask.to(device=state_ids.device, dtype=torch.bool)
        & (state_ids > 3)
        & (info_weights >= float(min_token_weight))
    )
    if not bool(valid.any().item()):
        return next(model.parameters()).new_tensor(0.0)
    pooled = _pooled_controller_state(model, batch["enc_input_ids"], batch["enc_attention_mask"])
    logits = head(pooled)
    targets = torch.zeros((state_ids.shape[0], int(buckets)), device=logits.device, dtype=logits.dtype)
    bucket_ids = state_ids.clamp_min(0).remainder(int(buckets)).to(device=logits.device, dtype=torch.long)
    valid_on_device = valid.to(device=logits.device)
    values = (info_weights * valid.to(dtype=info_weights.dtype)).to(device=logits.device, dtype=logits.dtype)
    targets.scatter_add_(1, bucket_ids, values)
    if int(topk) > 0 and int(topk) < int(buckets):
        keep = min(int(topk), int(buckets))
        top_values, top_indices = targets.topk(k=keep, dim=1)
        sparse = torch.zeros_like(targets)
        sparse.scatter_(1, top_indices, top_values)
        targets = sparse
    row_sums = targets.sum(dim=1, keepdim=True)
    has_target = row_sums.squeeze(1) > 0
    if not bool(has_target.any().item()):
        return next(model.parameters()).new_tensor(0.0)
    targets = targets / row_sums.clamp_min(1e-6)
    log_probs = F.log_softmax(logits, dim=-1)
    per_example = -(targets * log_probs).sum(dim=1)
    weights = batch["loss_weight"].to(device=logits.device, dtype=logits.dtype).clamp_min(0.0)
    weights = weights * has_target.to(dtype=weights.dtype)
    return (per_example * weights).sum() / weights.sum().clamp_min(1e-6)


def _target_sketch_aux_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    buckets: int,
    topk: int,
    min_token_weight: float,
) -> torch.Tensor:
    head = getattr(model, "target_sketch_aux_head", None)
    if head is None or int(buckets) <= 0:
        return next(model.parameters()).new_tensor(0.0)
    labels = batch["labels"]
    token_weights = batch.get("label_token_weights")
    if token_weights is None:
        return next(model.parameters()).new_tensor(0.0)
    info_weights = token_weights.to(device=labels.device, dtype=torch.float32)
    valid = (labels >= 0) & (labels > 3) & (info_weights >= float(min_token_weight))
    if not bool(valid.any().item()):
        return next(model.parameters()).new_tensor(0.0)
    pooled = _pooled_controller_state(model, batch["enc_input_ids"], batch["enc_attention_mask"])
    logits = head(pooled)
    targets = torch.zeros((labels.shape[0], int(buckets)), device=logits.device, dtype=logits.dtype)
    bucket_ids = labels.clamp_min(0).remainder(int(buckets)).to(device=logits.device, dtype=torch.long)
    values = (info_weights * valid.to(dtype=info_weights.dtype)).to(device=logits.device, dtype=logits.dtype)
    targets.scatter_add_(1, bucket_ids, values)
    if int(topk) > 0 and int(topk) < int(buckets):
        keep = min(int(topk), int(buckets))
        top_values, top_indices = targets.topk(k=keep, dim=1)
        sparse = torch.zeros_like(targets)
        sparse.scatter_(1, top_indices, top_values)
        targets = sparse
    row_sums = targets.sum(dim=1, keepdim=True)
    has_target = row_sums.squeeze(1) > 0
    if not bool(has_target.any().item()):
        return next(model.parameters()).new_tensor(0.0)
    targets = targets / row_sums.clamp_min(1e-6)
    log_probs = F.log_softmax(logits, dim=-1)
    per_example = -(targets * log_probs).sum(dim=1)
    weights = batch["loss_weight"].to(device=logits.device, dtype=logits.dtype).clamp_min(0.0)
    weights = weights * has_target.to(dtype=weights.dtype)
    return (per_example * weights).sum() / weights.sum().clamp_min(1e-6)


def _decoder_future_sketch_aux_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    buckets: int,
    min_token_weight: float,
    topk: int,
    windows: tuple[int, ...],
) -> torch.Tensor:
    head = getattr(model, "decoder_future_sketch_aux_head", None)
    if head is None or int(buckets) <= 0:
        return next(model.parameters()).new_tensor(0.0)
    labels = batch["labels"]
    token_weights = batch.get("label_token_weights")
    if token_weights is None:
        return next(model.parameters()).new_tensor(0.0)
    horizon_windows = tuple(sorted({max(1, int(item)) for item in windows if int(item) > 0}))
    if not horizon_windows:
        horizon_windows = (8, 32)
    info_weights = token_weights.to(device=labels.device, dtype=torch.float32)
    valid_future = (labels >= 0) & (labels > 3) & (info_weights >= float(min_token_weight))
    if not bool(valid_future.any().item()):
        return next(model.parameters()).new_tensor(0.0)
    memory = model.encode(batch["enc_input_ids"], batch["enc_attention_mask"])
    hidden = model.decode(
        batch["dec_input_ids"],
        memory,
        None,
        batch["enc_attention_mask"],
        None,
    ).float()
    logits = head(hidden)
    expected_dim = int(buckets) * len(horizon_windows)
    if logits.shape[-1] != expected_dim:
        raise RuntimeError(
            f"decoder future sketch head dim {logits.shape[-1]} does not match "
            f"buckets*windows {expected_dim}"
        )
    logits_by_window = logits.reshape(labels.shape[0], labels.shape[1], len(horizon_windows), int(buckets))
    bucket_ids_all = labels.clamp_min(0).remainder(int(buckets)).to(device=logits.device, dtype=torch.long)
    values_all = (info_weights * valid_future.to(dtype=info_weights.dtype)).to(
        device=logits.device,
        dtype=logits.dtype,
    )

    per_window_losses: list[torch.Tensor] = []
    per_window_valid: list[torch.Tensor] = []
    batch_size, seq_len = labels.shape
    for window_index, window in enumerate(horizon_windows):
        window = min(int(window), max(1, seq_len - 1))
        targets = torch.zeros((batch_size, seq_len, int(buckets)), device=logits.device, dtype=logits.dtype)
        for offset in range(1, window + 1):
            if offset >= seq_len:
                break
            target_slice = targets[:, :-offset, :]
            ids = bucket_ids_all[:, offset:]
            values = values_all[:, offset:]
            target_slice.scatter_add_(2, ids.unsqueeze(-1), values.unsqueeze(-1))
        if int(topk) > 0 and int(topk) < int(buckets):
            keep = min(int(topk), int(buckets))
            flat_targets = targets.reshape(-1, int(buckets))
            top_values, top_indices = flat_targets.topk(k=keep, dim=1)
            sparse = torch.zeros_like(flat_targets)
            sparse.scatter_(1, top_indices, top_values)
            targets = sparse.reshape_as(targets)
        row_sums = targets.sum(dim=-1, keepdim=True)
        has_target = row_sums.squeeze(-1) > 0
        if not bool(has_target.any().item()):
            continue
        targets = targets / row_sums.clamp_min(1e-6)
        log_probs = F.log_softmax(logits_by_window[:, :, window_index, :], dim=-1)
        per_window_losses.append(-(targets * log_probs).sum(dim=-1))
        per_window_valid.append(has_target)
    if not per_window_losses:
        return next(model.parameters()).new_tensor(0.0)
    stacked_losses = torch.stack(per_window_losses, dim=2)
    stacked_valid = torch.stack(per_window_valid, dim=2).to(device=logits.device, dtype=logits.dtype)
    per_token = (stacked_losses * stacked_valid).sum(dim=2) / stacked_valid.sum(dim=2).clamp_min(1.0)
    token_mask = stacked_valid.any(dim=2)
    per_example = (per_token * token_mask.to(dtype=per_token.dtype)).sum(dim=1) / token_mask.to(dtype=per_token.dtype).sum(dim=1).clamp_min(1.0)
    has_example = token_mask.any(dim=1)
    weights = batch["loss_weight"].to(device=logits.device, dtype=logits.dtype).clamp_min(0.0)
    weights = weights * has_example.to(dtype=weights.dtype)
    return (per_example * weights).sum() / weights.sum().clamp_min(1e-6)


def _decoder_source_copy_anchor_loss(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    min_token_weight: float,
) -> torch.Tensor:
    labels = batch["labels"].to(device=logits.device, dtype=torch.long)
    enc_ids = batch["enc_input_ids"].to(device=logits.device, dtype=torch.long)
    valid_labels = labels >= 0
    source_present = (labels.unsqueeze(-1) == enc_ids.unsqueeze(1)).any(dim=-1)
    mask = valid_labels & source_present
    if "label_token_weights" in batch:
        token_weights = batch["label_token_weights"].to(device=logits.device, dtype=logits.dtype)
        mask = mask & (token_weights >= float(min_token_weight))
    else:
        token_weights = mask.to(dtype=logits.dtype)
    if int(mask.sum().detach().cpu().item()) <= 0:
        return logits.new_tensor(0.0)
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    weights = token_weights * mask.to(dtype=token_weights.dtype)
    per_example = (token_losses * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    example_weights = batch["loss_weight"].to(device=logits.device, dtype=per_example.dtype).clamp_min(0.0)
    active = weights.sum(dim=1) > 0
    example_weights = example_weights * active.to(dtype=example_weights.dtype)
    return (per_example * example_weights).sum() / example_weights.sum().clamp_min(1e-6)


def _decoder_multi_token_ce_loss(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    horizons: tuple[int, ...],
    min_token_weight: float,
) -> torch.Tensor:
    labels = batch["labels"].to(device=logits.device, dtype=torch.long)
    horizon_values = tuple(sorted({int(item) for item in horizons if int(item) > 1}))
    if not horizon_values:
        return logits.new_tensor(0.0)
    token_weights = batch.get("label_token_weights")
    if token_weights is not None:
        token_weights = token_weights.to(device=logits.device, dtype=logits.dtype)
    else:
        token_weights = (labels != -100).to(device=logits.device, dtype=logits.dtype)
    example_weights = batch["loss_weight"].to(device=logits.device, dtype=logits.dtype).clamp_min(0.0)
    horizon_losses: list[torch.Tensor] = []
    horizon_weights: list[float] = []
    seq_len = int(labels.shape[1])
    for horizon in horizon_values:
        if horizon >= seq_len:
            continue
        source_logits = logits[:, :-horizon, :].contiguous()
        target_labels = labels[:, horizon:].contiguous()
        target_weights = token_weights[:, horizon:].contiguous()
        mask = target_labels != -100
        if float(min_token_weight) > 0.0:
            mask = mask & (target_weights >= float(min_token_weight))
        if int(mask.sum().detach().cpu().item()) <= 0:
            continue
        token_losses = F.cross_entropy(
            source_logits.reshape(-1, source_logits.shape[-1]),
            target_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(target_labels.shape)
        weights = target_weights * mask.to(dtype=target_weights.dtype)
        per_example = (token_losses * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        active = weights.sum(dim=1) > 0
        weighted = (per_example * example_weights * active.to(dtype=example_weights.dtype)).sum()
        denom = (example_weights * active.to(dtype=example_weights.dtype)).sum().clamp_min(1e-6)
        horizon_losses.append(weighted / denom)
        horizon_weights.append(float(horizon) ** -0.5)
    if not horizon_losses:
        return logits.new_tensor(0.0)
    weight_tensor = logits.new_tensor(horizon_weights)
    stacked = torch.stack(horizon_losses)
    return (stacked * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-6)


def _decoder_drift_alignment_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    min_token_weight: float,
    contrastive_temperature: float,
) -> torch.Tensor:
    state_ids = batch.get("state_target_ids")
    state_mask = batch.get("state_target_attention_mask")
    state_weights = batch.get("state_target_token_weights")
    if state_ids is None or state_mask is None or state_weights is None:
        return next(model.parameters()).new_tensor(0.0)
    labels = batch["labels"]
    label_weights = batch.get("label_token_weights")
    if label_weights is None:
        label_weights = (labels != -100).to(device=labels.device, dtype=torch.float32)
    info_weights = label_weights.to(device=labels.device, dtype=torch.float32)
    dec_mask = (labels != -100) & (info_weights >= float(min_token_weight))
    if not bool(dec_mask.any().item()):
        return next(model.parameters()).new_tensor(0.0)
    state_info_weights = state_weights.to(device=state_ids.device, dtype=torch.float32)
    target_mask = (
        state_mask.to(device=state_ids.device, dtype=torch.bool)
        & (state_ids > 3)
        & (state_info_weights >= float(min_token_weight))
    )
    has_decoder = dec_mask.any(dim=1)
    has_target = target_mask.any(dim=1)
    active = has_decoder & has_target
    if not bool(active.any().item()):
        return next(model.parameters()).new_tensor(0.0)

    memory = model.encode(batch["enc_input_ids"], batch["enc_attention_mask"])
    decoder_hidden = model.decode(
        batch["dec_input_ids"],
        memory,
        None,
        batch["enc_attention_mask"],
        None,
    ).float()
    dec_values = (info_weights * dec_mask.to(dtype=info_weights.dtype)).to(
        device=decoder_hidden.device,
        dtype=decoder_hidden.dtype,
    )
    decoder_pooled = (decoder_hidden * dec_values.unsqueeze(-1)).sum(dim=1)
    decoder_pooled = decoder_pooled / dec_values.sum(dim=1, keepdim=True).clamp_min(1.0)

    with torch.no_grad():
        target_hidden = model.encode(state_ids, state_mask).float()
        target_values = (state_info_weights * target_mask.to(dtype=state_info_weights.dtype)).to(
            device=target_hidden.device,
            dtype=target_hidden.dtype,
        )
        target_pooled = (target_hidden * target_values.unsqueeze(-1)).sum(dim=1)
        target_pooled = target_pooled / target_values.sum(dim=1, keepdim=True).clamp_min(1.0)

    decoder_repr = F.normalize(decoder_pooled[active], dim=-1)
    target_repr = F.normalize(target_pooled[active].to(device=decoder_repr.device), dim=-1)
    cosine_losses = 1.0 - (decoder_repr * target_repr).sum(dim=-1)
    example_weights = batch["loss_weight"].to(device=decoder_repr.device, dtype=decoder_repr.dtype).clamp_min(0.0)
    active_weights = example_weights[active]
    cosine_loss = (cosine_losses * active_weights).sum() / active_weights.sum().clamp_min(1e-6)

    temperature = float(contrastive_temperature)
    if temperature <= 0.0 or int(decoder_repr.shape[0]) < 2:
        return cosine_loss
    logits = decoder_repr @ target_repr.transpose(0, 1)
    logits = logits / max(temperature, 1e-4)
    targets = torch.arange(logits.shape[0], device=logits.device)
    contrastive = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.transpose(0, 1), targets))
    return 0.5 * cosine_loss + 0.5 * contrastive


def _scheduled_aux_weight(
    base_weight: float,
    *,
    step: int,
    total_steps: int,
    schedule: str,
) -> float:
    base = float(base_weight)
    if base <= 0.0:
        return 0.0
    name = str(schedule or "constant").strip().lower()
    if name == "constant":
        return base
    if name != "warmup_cosine_decay":
        raise ValueError(f"unknown auxiliary weight schedule: {schedule}")
    total = max(1, int(total_steps))
    progress = min(1.0, max(0.0, float(step) / float(total)))
    warmup_end = 0.20
    decay_start = 0.70
    final_ratio = 0.20
    if progress < warmup_end:
        return base * (progress / warmup_end)
    if progress <= decay_start:
        return base
    decay_progress = (progress - decay_start) / max(1e-6, 1.0 - decay_start)
    cosine = 0.5 * (1.0 + torch.cos(torch.tensor(decay_progress * 3.141592653589793)).item())
    return base * (final_ratio + (1.0 - final_ratio) * cosine)


def _weighted_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    decoder_loss_weight: float = 1.0,
    decoder_eos_loss_weight: float = 0.0,
    decoder_json_structure_loss_weight: float = 0.0,
    decoder_token_weight_alpha: float = 0.0,
    decoder_multi_token_ce_weight: float = 0.0,
    decoder_multi_token_horizons: tuple[int, ...] = (),
    decoder_multi_token_min_token_weight: float = 0.0,
    decoder_seq_eos_token_weight: float = 1.0,
    decoder_stop_token_loss_weight: float = 1.0,
    decoder_eos_token_id: int = 2,
    decoder_stop_token_ids: tuple[int, ...] = (),
    decoder_json_structure_token_ids: tuple[int, ...] = (),
    decoder_control_token_suppression_weight: float = 0.0,
    decoder_control_token_ids: tuple[int, ...] = (),
    negative_decoder_loss_weight: float = 0.0,
    negative_divergent_decoder_loss_weight: float = 0.0,
    negative_first_token_margin_weight: float = 0.0,
    negative_first_token_margin: float = 1.0,
    retrieval_contrastive_weight: float = 0.0,
    retrieval_temperature: float = 0.05,
    retrieval_ternary_aware_weight: float = 0.0,
    retrieval_ternary_threshold_ratio: float = 0.20,
    retrieval_ternary_group_size: int = 16,
    retrieval_ternary_residual_dims: int = 64,
    retrieval_ternary_teacher_distill_weight: float = 0.0,
    retrieval_ternary_teacher_temperature: float = 0.05,
    retrieval_ternary_reconstruction_weight: float = 0.0,
    retrieval_hard_negative_weight: float = 0.0,
    retrieval_hard_negative_ternary: bool = True,
    teacher_model: torch.nn.Module | None = None,
    teacher_distill_weight: float = 0.0,
    teacher_distill_temperature: float = 1.0,
    policy_head_loss_weight: float = 0.0,
    intent_head_loss_weight: float = 0.0,
    intent_contrastive_weight: float = 0.0,
    intent_contrastive_temperature: float = 0.10,
    encoder_rep_distill_weight: float = 0.0,
    future_bow_aux_weight: float = 0.0,
    future_bow_buckets: int = 512,
    future_sketch_aux_weight: float = 0.0,
    future_sketch_buckets: int = 256,
    future_sketch_min_token_weight: float = 1.2,
    future_sketch_topk: int = 8,
    future_sketch_windows: tuple[int, ...] = (),
    state_sketch_aux_weight: float = 0.0,
    state_sketch_buckets: int = 256,
    state_sketch_topk: int = 12,
    state_sketch_min_token_weight: float = 1.2,
    target_sketch_aux_weight: float = 0.0,
    target_sketch_buckets: int = 256,
    target_sketch_topk: int = 12,
    target_sketch_min_token_weight: float = 1.2,
    decoder_future_sketch_aux_weight: float = 0.0,
    decoder_future_sketch_buckets: int = 256,
    decoder_future_sketch_min_token_weight: float = 1.2,
    decoder_future_sketch_topk: int = 8,
    decoder_future_sketch_windows: tuple[int, ...] = (),
    decoder_drift_alignment_weight: float = 0.0,
    decoder_drift_alignment_min_token_weight: float = 1.2,
    decoder_drift_alignment_temperature: float = 0.0,
    decoder_source_copy_anchor_weight: float = 0.0,
    decoder_source_copy_min_token_weight: float = 1.2,
    aux_grad_budget: float = 0.0,
) -> torch.Tensor:
    logits = None
    needs_decoder_logits = (
        float(decoder_loss_weight) > 0.0
        or float(decoder_multi_token_ce_weight) > 0.0
        or float(decoder_source_copy_anchor_weight) > 0.0
        or float(negative_first_token_margin_weight) > 0.0
        or (
        teacher_model is not None and float(teacher_distill_weight) > 0.0
    )
    )
    if needs_decoder_logits:
        logits = model(
            batch["enc_input_ids"],
            batch["dec_input_ids"],
            batch["enc_attention_mask"],
            None,
        )
    if float(decoder_loss_weight) > 0.0:
        assert logits is not None
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch["labels"].reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        token_losses = token_losses.reshape(batch["labels"].shape)
        token_weights = (batch["labels"] != -100).to(device=token_losses.device, dtype=token_losses.dtype)
        if float(decoder_token_weight_alpha) > 0.0 and "label_token_weights" in batch:
            token_weights = batch["label_token_weights"].to(device=token_losses.device, dtype=token_losses.dtype)
            token_weights = 1.0 + float(decoder_token_weight_alpha) * (token_weights - 1.0)
            token_weights = token_weights.clamp_min(0.0)
        if float(decoder_seq_eos_token_weight) != 1.0:
            eos_mask = batch["labels"] == int(decoder_eos_token_id)
            token_weights = torch.where(
                eos_mask,
                token_weights * float(decoder_seq_eos_token_weight),
                token_weights,
            )
        if float(decoder_stop_token_loss_weight) != 1.0 and decoder_stop_token_ids:
            stop_ids = torch.tensor(
                [int(token_id) for token_id in decoder_stop_token_ids],
                device=batch["labels"].device,
                dtype=batch["labels"].dtype,
            )
            stop_mask = torch.isin(batch["labels"], stop_ids)
            token_weights = torch.where(
                stop_mask,
                token_weights * float(decoder_stop_token_loss_weight),
                token_weights,
            )
        token_weights = token_weights * (batch["labels"] != -100).to(dtype=token_weights.dtype)
        per_example = (token_losses * token_weights).sum(dim=1) / token_weights.sum(dim=1).clamp_min(1.0)
        weights = batch["loss_weight"].to(dtype=per_example.dtype).clamp_min(0.0)
        seq_loss = (per_example * weights).sum() / weights.sum().clamp_min(1e-6)
    else:
        seq_loss = next(model.parameters()).new_tensor(0.0)
    multi_token_ce_loss = next(model.parameters()).new_tensor(0.0)
    if float(decoder_multi_token_ce_weight) > 0.0:
        assert logits is not None
        multi_token_ce_loss = _decoder_multi_token_ce_loss(
            logits,
            batch,
            horizons=tuple(decoder_multi_token_horizons),
            min_token_weight=float(decoder_multi_token_min_token_weight),
        )
    source_copy_anchor_loss = next(model.parameters()).new_tensor(0.0)
    if float(decoder_source_copy_anchor_weight) > 0.0:
        assert logits is not None
        source_copy_anchor_loss = _decoder_source_copy_anchor_loss(
            logits,
            batch,
            min_token_weight=float(decoder_source_copy_min_token_weight),
        )
    eos_loss = next(model.parameters()).new_tensor(0.0)
    if float(decoder_eos_loss_weight) > 0.0:
        assert logits is not None
        labels = batch["labels"]
        eos_mask = labels == int(decoder_eos_token_id)
        if bool(eos_mask.any().item()):
            token_losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape(labels.shape)
            per_example_eos = (token_losses * eos_mask.to(dtype=token_losses.dtype)).sum(dim=1) / eos_mask.to(dtype=token_losses.dtype).sum(dim=1).clamp_min(1.0)
            has_eos = eos_mask.any(dim=1)
            weights = batch["loss_weight"].to(device=per_example_eos.device, dtype=per_example_eos.dtype).clamp_min(0.0)
            eos_loss = (per_example_eos * weights * has_eos.to(dtype=weights.dtype)).sum() / (weights * has_eos.to(dtype=weights.dtype)).sum().clamp_min(1e-6)
    json_structure_loss = next(model.parameters()).new_tensor(0.0)
    if float(decoder_json_structure_loss_weight) > 0.0 and decoder_json_structure_token_ids:
        assert logits is not None
        labels = batch["labels"]
        structural_ids = torch.tensor(
            [int(token_id) for token_id in decoder_json_structure_token_ids],
            device=labels.device,
            dtype=labels.dtype,
        )
        structure_mask = torch.isin(labels, structural_ids)
        if bool(structure_mask.any().item()):
            token_losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape(labels.shape)
            per_example_structure = (token_losses * structure_mask.to(dtype=token_losses.dtype)).sum(dim=1) / structure_mask.to(dtype=token_losses.dtype).sum(dim=1).clamp_min(1.0)
            has_structure = structure_mask.any(dim=1)
            weights = batch["loss_weight"].to(device=per_example_structure.device, dtype=per_example_structure.dtype).clamp_min(0.0)
            json_structure_loss = (per_example_structure * weights * has_structure.to(dtype=weights.dtype)).sum() / (weights * has_structure.to(dtype=weights.dtype)).sum().clamp_min(1e-6)
    control_suppression_loss = next(model.parameters()).new_tensor(0.0)
    if float(decoder_control_token_suppression_weight) > 0.0 and decoder_control_token_ids:
        assert logits is not None
        labels = batch["labels"]
        control_ids = torch.tensor(
            [int(token_id) for token_id in decoder_control_token_ids],
            device=labels.device,
            dtype=torch.long,
        )
        valid_content = (labels != -100) & ~torch.isin(labels.to(dtype=torch.long), control_ids)
        if bool(valid_content.any().item()):
            probs = torch.softmax(logits.float(), dim=-1)
            control_mass = probs.index_select(dim=-1, index=control_ids).sum(dim=-1).clamp(max=1.0 - 1e-5)
            token_losses = -torch.log1p(-control_mass)
            token_mask = valid_content.to(device=token_losses.device, dtype=token_losses.dtype)
            per_example_control = (token_losses * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0)
            has_content = valid_content.any(dim=1)
            weights = batch["loss_weight"].to(device=per_example_control.device, dtype=per_example_control.dtype).clamp_min(0.0)
            control_suppression_loss = (
                per_example_control * weights * has_content.to(dtype=weights.dtype)
            ).sum() / (weights * has_content.to(dtype=weights.dtype)).sum().clamp_min(1e-6)
    negative_decoder_loss = next(model.parameters()).new_tensor(0.0)
    negative_divergent_decoder_loss = next(model.parameters()).new_tensor(0.0)
    negative_first_token_margin_loss = next(model.parameters()).new_tensor(0.0)
    if (
        (float(negative_decoder_loss_weight) > 0.0 or float(negative_divergent_decoder_loss_weight) > 0.0)
        and "negative_labels" in batch
    ):
        negative_labels = batch["negative_labels"]
        negative_valid = negative_labels != -100
        negative_weights = batch.get("negative_loss_weight")
        if negative_weights is not None and negative_valid.any() and bool((negative_weights > 0).any().item()):
            negative_logits = model(
                batch["enc_input_ids"],
                batch["negative_dec_input_ids"],
                batch["enc_attention_mask"],
                None,
            )
            negative_probs = torch.softmax(negative_logits.float(), dim=-1)
            safe_labels = negative_labels.clamp_min(0)
            token_probs = negative_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            token_losses = -torch.log1p(-token_probs.clamp(max=1.0 - 1e-5))
            if float(negative_decoder_loss_weight) > 0.0:
                full_losses = token_losses * negative_valid.to(dtype=token_losses.dtype)
                per_example_negative = full_losses.sum(dim=1) / negative_valid.to(dtype=token_losses.dtype).sum(dim=1).clamp_min(1.0)
                weights = negative_weights.to(device=per_example_negative.device, dtype=per_example_negative.dtype).clamp_min(0.0)
                negative_decoder_loss = (per_example_negative * weights).sum() / weights.sum().clamp_min(1e-6)
            if float(negative_divergent_decoder_loss_weight) > 0.0:
                labels = batch["labels"]
                divergent_mask = negative_valid & (labels != -100) & (negative_labels != labels)
                if bool(divergent_mask.any().item()):
                    divergent_losses = token_losses * divergent_mask.to(dtype=token_losses.dtype)
                    divergent_counts = divergent_mask.to(dtype=token_losses.dtype).sum(dim=1)
                    per_example_divergent = divergent_losses.sum(dim=1) / divergent_counts.clamp_min(1.0)
                    has_divergent = divergent_counts > 0
                    weights = negative_weights.to(device=per_example_divergent.device, dtype=per_example_divergent.dtype).clamp_min(0.0)
                    negative_divergent_decoder_loss = (
                        per_example_divergent * weights * has_divergent.to(dtype=weights.dtype)
                    ).sum() / (weights * has_divergent.to(dtype=weights.dtype)).sum().clamp_min(1e-6)
    if float(negative_first_token_margin_weight) > 0.0 and "negative_labels" in batch:
        assert logits is not None
        labels = batch["labels"]
        negative_labels = batch["negative_labels"]
        negative_weights = batch.get("negative_loss_weight")
        if negative_weights is not None:
            valid_labels = (labels != -100) & (negative_labels != -100)
            divergent = valid_labels & (labels != negative_labels)
            has_divergence = divergent.any(dim=1)
            first_divergence_positions = divergent.float().argmax(dim=1).long()
            valid = has_divergence & (negative_weights > 0)
            if bool(valid.any().item()):
                batch_positions = torch.arange(labels.shape[0], device=labels.device)
                target_positions = first_divergence_positions.to(device=labels.device)
                positive_ids = labels[batch_positions, target_positions].clamp_min(0)
                negative_ids = negative_labels[batch_positions, target_positions].clamp_min(0)
                step_logits = logits[batch_positions, target_positions, :].float()
                positive_logits = step_logits.gather(-1, positive_ids.unsqueeze(-1)).squeeze(-1)
                negative_logits_for_positive_prefix = step_logits.gather(-1, negative_ids.unsqueeze(-1)).squeeze(-1)
                margin_losses = F.relu(
                    float(negative_first_token_margin)
                    - (positive_logits - negative_logits_for_positive_prefix)
                )
                weights = negative_weights.to(device=margin_losses.device, dtype=margin_losses.dtype).clamp_min(0.0)
                margin_losses = margin_losses * valid.to(dtype=margin_losses.dtype)
                negative_first_token_margin_loss = (margin_losses * weights).sum() / (weights * valid.to(dtype=weights.dtype)).sum().clamp_min(1e-6)
    distill_loss = next(model.parameters()).new_tensor(0.0)
    if teacher_model is not None and float(teacher_distill_weight) > 0.0:
        assert logits is not None
        temperature = max(float(teacher_distill_temperature), 1e-4)
        with torch.no_grad():
            teacher_logits = teacher_model(
                batch["enc_input_ids"],
                batch["dec_input_ids"],
                batch["enc_attention_mask"],
                None,
            )
        valid_tokens = (batch["labels"] != -100).to(dtype=logits.dtype)
        student_log_probs = F.log_softmax(logits.float() / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
        token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        per_example_kl = (token_kl * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp_min(1.0)
        weights = batch["loss_weight"].to(device=per_example_kl.device, dtype=per_example_kl.dtype).clamp_min(0.0)
        distill_loss = (per_example_kl * weights).sum() / weights.sum().clamp_min(1e-6)
        distill_loss = distill_loss * (temperature * temperature)
    encoder_rep_distill_loss = next(model.parameters()).new_tensor(0.0)
    if teacher_model is not None and float(encoder_rep_distill_weight) > 0.0:
        student_hidden = model.encode(batch["enc_input_ids"], batch["enc_attention_mask"])
        student_pooled = _mean_pool(student_hidden, batch["enc_attention_mask"]).float()
        with torch.no_grad():
            teacher_hidden = teacher_model.encode(batch["enc_input_ids"], batch["enc_attention_mask"])
            teacher_pooled = _mean_pool(teacher_hidden, batch["enc_attention_mask"]).float()
        encoder_rep_distill_loss = F.mse_loss(student_pooled, teacher_pooled)
    retrieval_loss = next(model.parameters()).new_tensor(0.0)
    if float(retrieval_contrastive_weight) > 0.0:
        retrieval_loss = _retrieval_contrastive_loss(
            model,
            batch,
            temperature=float(retrieval_temperature),
        )
    ternary_loss = next(model.parameters()).new_tensor(0.0)
    if float(retrieval_ternary_aware_weight) > 0.0:
        ternary_loss = _retrieval_ternary_aware_loss(
            model,
            batch,
            temperature=float(retrieval_temperature),
            threshold_ratio=float(retrieval_ternary_threshold_ratio),
            group_size=int(retrieval_ternary_group_size),
            residual_dims=int(retrieval_ternary_residual_dims),
        )
    ternary_teacher_loss = next(model.parameters()).new_tensor(0.0)
    if teacher_model is not None and float(retrieval_ternary_teacher_distill_weight) > 0.0:
        ternary_teacher_loss = _retrieval_ternary_teacher_distill_loss(
            model,
            teacher_model,
            batch,
            temperature=float(retrieval_ternary_teacher_temperature),
            threshold_ratio=float(retrieval_ternary_threshold_ratio),
            group_size=int(retrieval_ternary_group_size),
            residual_dims=int(retrieval_ternary_residual_dims),
        )
    ternary_reconstruction_loss = next(model.parameters()).new_tensor(0.0)
    if float(retrieval_ternary_reconstruction_weight) > 0.0:
        ternary_reconstruction_loss = _retrieval_ternary_reconstruction_loss(
            model,
            batch,
            threshold_ratio=float(retrieval_ternary_threshold_ratio),
            group_size=int(retrieval_ternary_group_size),
            residual_dims=int(retrieval_ternary_residual_dims),
        )
    hard_negative_loss = next(model.parameters()).new_tensor(0.0)
    if float(retrieval_hard_negative_weight) > 0.0:
        hard_negative_loss = _retrieval_hard_negative_loss(
            model,
            batch,
            temperature=float(retrieval_temperature),
            use_ternary_docs=bool(retrieval_hard_negative_ternary),
            threshold_ratio=float(retrieval_ternary_threshold_ratio),
            group_size=int(retrieval_ternary_group_size),
            residual_dims=int(retrieval_ternary_residual_dims),
        )
    policy_loss = next(model.parameters()).new_tensor(0.0)
    if float(policy_head_loss_weight) > 0.0:
        policy_loss = _agent_policy_head_loss(model, batch)
    intent_head_loss = next(model.parameters()).new_tensor(0.0)
    if float(intent_head_loss_weight) > 0.0:
        intent_head_loss = _agent_intent_head_loss(model, batch)
    intent_loss = next(model.parameters()).new_tensor(0.0)
    if float(intent_contrastive_weight) > 0.0:
        intent_loss = _intent_contrastive_loss(
            model,
            batch,
            temperature=float(intent_contrastive_temperature),
        )
    future_bow_loss = next(model.parameters()).new_tensor(0.0)
    if float(future_bow_aux_weight) > 0.0:
        future_bow_loss = _future_bow_aux_loss(model, batch, buckets=int(future_bow_buckets))
    future_sketch_loss = next(model.parameters()).new_tensor(0.0)
    if float(future_sketch_aux_weight) > 0.0:
        future_sketch_loss = _future_sketch_aux_loss(
            model,
            batch,
            buckets=int(future_sketch_buckets),
            min_token_weight=float(future_sketch_min_token_weight),
            topk=int(future_sketch_topk),
            windows=tuple(future_sketch_windows),
        )
    future_sketch_weight = float(future_sketch_aux_weight)
    if future_sketch_weight > 0.0 and float(aux_grad_budget) > 0.0:
        max_contribution = float(aux_grad_budget) * float(seq_loss.detach().clamp_min(1e-6).cpu().item())
        raw_contribution = future_sketch_weight * float(future_sketch_loss.detach().clamp_min(1e-6).cpu().item())
        if raw_contribution > max_contribution:
            future_sketch_weight *= max_contribution / raw_contribution
    state_sketch_loss = next(model.parameters()).new_tensor(0.0)
    if float(state_sketch_aux_weight) > 0.0:
        state_sketch_loss = _state_sketch_aux_loss(
            model,
            batch,
            buckets=int(state_sketch_buckets),
            topk=int(state_sketch_topk),
            min_token_weight=float(state_sketch_min_token_weight),
        )
    state_sketch_weight = float(state_sketch_aux_weight)
    if state_sketch_weight > 0.0 and float(aux_grad_budget) > 0.0:
        max_contribution = float(aux_grad_budget) * float(seq_loss.detach().clamp_min(1e-6).cpu().item())
        raw_contribution = state_sketch_weight * float(state_sketch_loss.detach().clamp_min(1e-6).cpu().item())
        if raw_contribution > max_contribution:
            state_sketch_weight *= max_contribution / raw_contribution
    target_sketch_loss = next(model.parameters()).new_tensor(0.0)
    if float(target_sketch_aux_weight) > 0.0:
        target_sketch_loss = _target_sketch_aux_loss(
            model,
            batch,
            buckets=int(target_sketch_buckets),
            topk=int(target_sketch_topk),
            min_token_weight=float(target_sketch_min_token_weight),
        )
    target_sketch_weight = float(target_sketch_aux_weight)
    if target_sketch_weight > 0.0 and float(aux_grad_budget) > 0.0:
        reference_loss = seq_loss
        if float(decoder_loss_weight) <= 0.0:
            reference_loss = (
                float(policy_head_loss_weight) * policy_loss
                + float(intent_head_loss_weight) * intent_head_loss
                + float(intent_contrastive_weight) * intent_loss
                + float(retrieval_contrastive_weight) * retrieval_loss
            )
        max_contribution = float(aux_grad_budget) * float(reference_loss.detach().clamp_min(1e-6).cpu().item())
        raw_contribution = target_sketch_weight * float(target_sketch_loss.detach().clamp_min(1e-6).cpu().item())
        if raw_contribution > max_contribution:
            target_sketch_weight *= max_contribution / raw_contribution
    decoder_future_sketch_loss = next(model.parameters()).new_tensor(0.0)
    if float(decoder_future_sketch_aux_weight) > 0.0:
        decoder_future_sketch_loss = _decoder_future_sketch_aux_loss(
            model,
            batch,
            buckets=int(decoder_future_sketch_buckets),
            min_token_weight=float(decoder_future_sketch_min_token_weight),
            topk=int(decoder_future_sketch_topk),
            windows=tuple(decoder_future_sketch_windows),
        )
    decoder_future_sketch_weight = float(decoder_future_sketch_aux_weight)
    if decoder_future_sketch_weight > 0.0 and float(aux_grad_budget) > 0.0:
        max_contribution = float(aux_grad_budget) * float(seq_loss.detach().clamp_min(1e-6).cpu().item())
        raw_contribution = decoder_future_sketch_weight * float(
            decoder_future_sketch_loss.detach().clamp_min(1e-6).cpu().item()
        )
        if raw_contribution > max_contribution:
            decoder_future_sketch_weight *= max_contribution / raw_contribution
    decoder_drift_alignment_loss = next(model.parameters()).new_tensor(0.0)
    if float(decoder_drift_alignment_weight) > 0.0:
        decoder_drift_alignment_loss = _decoder_drift_alignment_loss(
            model,
            batch,
            min_token_weight=float(decoder_drift_alignment_min_token_weight),
            contrastive_temperature=float(decoder_drift_alignment_temperature),
        )
    decoder_drift_weight = float(decoder_drift_alignment_weight)
    if decoder_drift_weight > 0.0 and float(aux_grad_budget) > 0.0:
        max_contribution = float(aux_grad_budget) * float(seq_loss.detach().clamp_min(1e-6).cpu().item())
        raw_contribution = decoder_drift_weight * float(
            decoder_drift_alignment_loss.detach().clamp_min(1e-6).cpu().item()
        )
        if raw_contribution > max_contribution:
            decoder_drift_weight *= max_contribution / raw_contribution
    return (
        float(decoder_loss_weight) * seq_loss
        + float(decoder_multi_token_ce_weight) * multi_token_ce_loss
        + float(decoder_source_copy_anchor_weight) * source_copy_anchor_loss
        + float(decoder_eos_loss_weight) * eos_loss
        + float(decoder_json_structure_loss_weight) * json_structure_loss
        + float(decoder_control_token_suppression_weight) * control_suppression_loss
        + float(negative_decoder_loss_weight) * negative_decoder_loss
        + float(negative_divergent_decoder_loss_weight) * negative_divergent_decoder_loss
        + float(negative_first_token_margin_weight) * negative_first_token_margin_loss
        + float(teacher_distill_weight) * distill_loss
        + float(encoder_rep_distill_weight) * encoder_rep_distill_loss
        + float(retrieval_contrastive_weight) * retrieval_loss
        + float(retrieval_ternary_aware_weight) * ternary_loss
        + float(retrieval_ternary_teacher_distill_weight) * ternary_teacher_loss
        + float(retrieval_ternary_reconstruction_weight) * ternary_reconstruction_loss
        + float(retrieval_hard_negative_weight) * hard_negative_loss
        + float(policy_head_loss_weight) * policy_loss
        + float(intent_head_loss_weight) * intent_head_loss
        + float(intent_contrastive_weight) * intent_loss
        + float(future_bow_aux_weight) * future_bow_loss
        + future_sketch_weight * future_sketch_loss
        + state_sketch_weight * state_sketch_loss
        + target_sketch_weight * target_sketch_loss
        + decoder_future_sketch_weight * decoder_future_sketch_loss
        + decoder_drift_weight * decoder_drift_alignment_loss
    )


def _evaluate(
    model: torch.nn.Module,
    dataset: EncDecJsonlDataset,
    *,
    batch_size: int,
    device: torch.device,
    max_batches: int,
    decoder_loss_weight: float = 1.0,
    decoder_eos_loss_weight: float = 0.0,
    decoder_json_structure_loss_weight: float = 0.0,
    decoder_token_weight_alpha: float = 0.0,
    decoder_multi_token_ce_weight: float = 0.0,
    decoder_multi_token_horizons: tuple[int, ...] = (),
    decoder_multi_token_min_token_weight: float = 0.0,
    decoder_seq_eos_token_weight: float = 1.0,
    decoder_stop_token_loss_weight: float = 1.0,
    decoder_eos_token_id: int = 2,
    decoder_stop_token_ids: tuple[int, ...] = (),
    decoder_json_structure_token_ids: tuple[int, ...] = (),
    decoder_control_token_suppression_weight: float = 0.0,
    decoder_control_token_ids: tuple[int, ...] = (),
    negative_decoder_loss_weight: float = 0.0,
    negative_divergent_decoder_loss_weight: float = 0.0,
    negative_first_token_margin_weight: float = 0.0,
    negative_first_token_margin: float = 1.0,
    retrieval_contrastive_weight: float = 0.0,
    retrieval_temperature: float = 0.05,
    retrieval_ternary_aware_weight: float = 0.0,
    retrieval_ternary_threshold_ratio: float = 0.20,
    retrieval_ternary_group_size: int = 16,
    retrieval_ternary_residual_dims: int = 64,
    retrieval_ternary_teacher_distill_weight: float = 0.0,
    retrieval_ternary_teacher_temperature: float = 0.05,
    retrieval_ternary_reconstruction_weight: float = 0.0,
    retrieval_hard_negative_weight: float = 0.0,
    retrieval_hard_negative_ternary: bool = True,
    policy_head_loss_weight: float = 0.0,
    intent_head_loss_weight: float = 0.0,
    intent_contrastive_weight: float = 0.0,
    intent_contrastive_temperature: float = 0.10,
    encoder_rep_distill_weight: float = 0.0,
    future_bow_aux_weight: float = 0.0,
    future_bow_buckets: int = 512,
    future_sketch_aux_weight: float = 0.0,
    future_sketch_buckets: int = 256,
    future_sketch_min_token_weight: float = 1.2,
    future_sketch_topk: int = 8,
    future_sketch_windows: tuple[int, ...] = (),
    state_sketch_aux_weight: float = 0.0,
    state_sketch_buckets: int = 256,
    state_sketch_topk: int = 12,
    state_sketch_min_token_weight: float = 1.2,
    target_sketch_aux_weight: float = 0.0,
    target_sketch_buckets: int = 256,
    target_sketch_topk: int = 12,
    target_sketch_min_token_weight: float = 1.2,
    decoder_future_sketch_aux_weight: float = 0.0,
    decoder_future_sketch_buckets: int = 256,
    decoder_future_sketch_min_token_weight: float = 1.2,
    decoder_future_sketch_topk: int = 8,
    decoder_future_sketch_windows: tuple[int, ...] = (),
    decoder_drift_alignment_weight: float = 0.0,
    decoder_drift_alignment_min_token_weight: float = 1.2,
    decoder_drift_alignment_temperature: float = 0.0,
    decoder_source_copy_anchor_weight: float = 0.0,
    decoder_source_copy_min_token_weight: float = 1.2,
    aux_grad_budget: float = 0.0,
    teacher_model: torch.nn.Module | None = None,
) -> dict[str, Any]:
    if len(dataset) <= 0:
        return {"eval_examples": 0, "eval_batches": 0, "eval_loss": None}
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    was_training = model.training
    model.eval()
    losses: list[float] = []
    seen_examples = 0
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            if max_batches > 0 and index > max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = _weighted_loss(
                model,
                batch,
                decoder_loss_weight=float(decoder_loss_weight),
                decoder_eos_loss_weight=float(decoder_eos_loss_weight),
                decoder_json_structure_loss_weight=float(decoder_json_structure_loss_weight),
                decoder_token_weight_alpha=float(decoder_token_weight_alpha),
                decoder_multi_token_ce_weight=float(decoder_multi_token_ce_weight),
                decoder_multi_token_horizons=tuple(decoder_multi_token_horizons),
                decoder_multi_token_min_token_weight=float(decoder_multi_token_min_token_weight),
                decoder_seq_eos_token_weight=float(decoder_seq_eos_token_weight),
                decoder_stop_token_loss_weight=float(decoder_stop_token_loss_weight),
                decoder_eos_token_id=int(decoder_eos_token_id),
                decoder_stop_token_ids=tuple(decoder_stop_token_ids),
                decoder_json_structure_token_ids=tuple(decoder_json_structure_token_ids),
                decoder_control_token_suppression_weight=float(decoder_control_token_suppression_weight),
                decoder_control_token_ids=tuple(decoder_control_token_ids),
                negative_decoder_loss_weight=float(negative_decoder_loss_weight),
                negative_divergent_decoder_loss_weight=float(negative_divergent_decoder_loss_weight),
                negative_first_token_margin_weight=float(negative_first_token_margin_weight),
                negative_first_token_margin=float(negative_first_token_margin),
                retrieval_contrastive_weight=float(retrieval_contrastive_weight),
                retrieval_temperature=float(retrieval_temperature),
                retrieval_ternary_aware_weight=float(retrieval_ternary_aware_weight),
                retrieval_ternary_threshold_ratio=float(retrieval_ternary_threshold_ratio),
                retrieval_ternary_group_size=int(retrieval_ternary_group_size),
                retrieval_ternary_residual_dims=int(retrieval_ternary_residual_dims),
                retrieval_ternary_teacher_distill_weight=float(retrieval_ternary_teacher_distill_weight),
                retrieval_ternary_teacher_temperature=float(retrieval_ternary_teacher_temperature),
                retrieval_ternary_reconstruction_weight=float(retrieval_ternary_reconstruction_weight),
                retrieval_hard_negative_weight=float(retrieval_hard_negative_weight),
                retrieval_hard_negative_ternary=bool(retrieval_hard_negative_ternary),
                policy_head_loss_weight=float(policy_head_loss_weight),
                intent_head_loss_weight=float(intent_head_loss_weight),
                intent_contrastive_weight=float(intent_contrastive_weight),
                intent_contrastive_temperature=float(intent_contrastive_temperature),
                encoder_rep_distill_weight=float(encoder_rep_distill_weight),
                future_bow_aux_weight=float(future_bow_aux_weight),
                future_bow_buckets=int(future_bow_buckets),
                future_sketch_aux_weight=float(future_sketch_aux_weight),
                future_sketch_buckets=int(future_sketch_buckets),
                future_sketch_min_token_weight=float(future_sketch_min_token_weight),
                future_sketch_topk=int(future_sketch_topk),
                future_sketch_windows=tuple(future_sketch_windows),
                state_sketch_aux_weight=float(state_sketch_aux_weight),
                state_sketch_buckets=int(state_sketch_buckets),
                state_sketch_topk=int(state_sketch_topk),
                state_sketch_min_token_weight=float(state_sketch_min_token_weight),
                target_sketch_aux_weight=float(target_sketch_aux_weight),
                target_sketch_buckets=int(target_sketch_buckets),
                target_sketch_topk=int(target_sketch_topk),
                target_sketch_min_token_weight=float(target_sketch_min_token_weight),
                decoder_future_sketch_aux_weight=float(decoder_future_sketch_aux_weight),
                decoder_future_sketch_buckets=int(decoder_future_sketch_buckets),
                decoder_future_sketch_min_token_weight=float(decoder_future_sketch_min_token_weight),
                decoder_future_sketch_topk=int(decoder_future_sketch_topk),
                decoder_future_sketch_windows=tuple(decoder_future_sketch_windows),
                decoder_drift_alignment_weight=float(decoder_drift_alignment_weight),
                decoder_drift_alignment_min_token_weight=float(decoder_drift_alignment_min_token_weight),
                decoder_drift_alignment_temperature=float(decoder_drift_alignment_temperature),
                decoder_source_copy_anchor_weight=float(decoder_source_copy_anchor_weight),
                decoder_source_copy_min_token_weight=float(decoder_source_copy_min_token_weight),
                aux_grad_budget=float(aux_grad_budget),
                teacher_model=teacher_model,
            )
            losses.append(float(loss.detach().cpu().item()))
            seen_examples += int(batch["labels"].shape[0])
    if was_training:
        model.train()
    return {
        "eval_examples": seen_examples,
        "eval_batches": len(losses),
        "eval_loss": sum(losses) / len(losses) if losses else None,
    }


def _preset_config(name: str, vocab_size: int):
    from specs.config import ModelConfig

    normalized = str(name).strip().lower()
    if normalized == "tiny":
        return ModelConfig(
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            vocab_size=int(vocab_size),
            attn_impl="eager",
            activation="silu",
            norm="layer",
            max_position_embeddings=512,
        )
    if normalized in {"100m", "agentkernel-lite-100m", "lite-100m"}:
        return ModelConfig(
            d_model=640,
            n_heads=10,
            n_layers=6,
            d_ff=2048,
            vocab_size=int(vocab_size),
            attn_impl="eager",
            activation="silu",
            norm="layer",
            max_position_embeddings=4096,
        )
    raise ValueError(f"unknown preset: {name}")


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(int(param.numel()) for param in model.parameters())


def _trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(int(param.numel()) for param in model.parameters() if param.requires_grad)


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def _clone_trainable_grads(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [
        param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param, memory_format=torch.preserve_format)
        for param in parameters
    ]


def _grad_dot(left: list[torch.Tensor], right: list[torch.Tensor]) -> torch.Tensor:
    value = left[0].new_zeros(())
    for left_tensor, right_tensor in zip(left, right):
        value = value + (left_tensor.float() * right_tensor.float()).sum()
    return value


def _grad_norm_sq(grad: list[torch.Tensor]) -> torch.Tensor:
    value = grad[0].new_zeros(())
    for tensor in grad:
        value = value + tensor.float().square().sum()
    return value


def _apply_pcgrad_snapshots(
    parameters: list[torch.nn.Parameter],
    snapshots: list[list[torch.Tensor]],
) -> dict[str, float | int]:
    if not snapshots:
        return {"pcgrad_snapshots": 0, "pcgrad_conflicts": 0, "pcgrad_min_cosine": 0.0}
    projected = [[tensor.clone() for tensor in snapshot] for snapshot in snapshots]
    reference_norms = [_grad_norm_sq(snapshot).clamp_min(1e-12) for snapshot in snapshots]
    conflicts = 0
    min_cosine = 1.0
    for idx, grad in enumerate(projected):
        for other_idx, other in enumerate(snapshots):
            if idx == other_idx:
                continue
            dot = _grad_dot(grad, other)
            cosine = float(
                (dot / (_grad_norm_sq(grad).sqrt() * reference_norms[other_idx].sqrt()).clamp_min(1e-12))
                .detach()
                .item()
            )
            min_cosine = min(min_cosine, cosine)
            if dot.item() < 0:
                conflicts += 1
                scale = dot / reference_norms[other_idx]
                for tensor_idx in range(len(grad)):
                    grad[tensor_idx].sub_(
                        scale.to(device=grad[tensor_idx].device, dtype=grad[tensor_idx].dtype) * other[tensor_idx]
                    )
    for parameter_idx, parameter in enumerate(parameters):
        merged = projected[0][parameter_idx]
        for grad in projected[1:]:
            merged = merged + grad[parameter_idx]
        parameter.grad = merged.to(device=parameter.device, dtype=parameter.dtype)
    return {"pcgrad_snapshots": len(snapshots), "pcgrad_conflicts": conflicts, "pcgrad_min_cosine": min_cosine}


def _trainable_anchor_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    anchors: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            anchors[name] = param.detach().float().cpu().clone()
    return anchors


def _anchor_weight_loss(model: torch.nn.Module, anchors: dict[str, torch.Tensor]) -> torch.Tensor:
    if not anchors:
        return next(model.parameters()).new_tensor(0.0)
    loss: torch.Tensor | None = None
    terms = 0
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in anchors:
            continue
        target = anchors[name].to(device=param.device, dtype=torch.float32)
        term = F.mse_loss(param.float(), target)
        loss = term if loss is None else loss + term
        terms += 1
    if loss is None:
        return next(model.parameters()).new_tensor(0.0)
    return loss / float(max(1, terms))


def _freeze_encoder_parameters(model: torch.nn.Module) -> int:
    frozen = 0
    for name, param in model.named_parameters():
        if (
            name.startswith("enc_embed.")
            or name.startswith("enc_pos_embed.")
            or name.startswith("encoder.")
            or name.startswith("enc_norm.")
        ):
            param.requires_grad_(False)
            frozen += int(param.numel())
    return frozen


def _freeze_token_embedding_parameters(model: torch.nn.Module) -> int:
    frozen = 0
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if name.startswith("enc_embed.") or name.startswith("dec_embed."):
            identity = id(param)
            if identity not in seen:
                frozen += int(param.numel())
                seen.add(identity)
            param.requires_grad_(False)
    return frozen


def _freeze_lm_head_parameters(model: torch.nn.Module) -> int:
    frozen = 0
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if name.startswith("lm_head."):
            identity = id(param)
            if identity not in seen:
                frozen += int(param.numel())
                seen.add(identity)
            param.requires_grad_(False)
    return frozen


def _freeze_decoder_parameters(model: torch.nn.Module) -> int:
    frozen = 0
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if (
            name.startswith("dec_embed.")
            or name.startswith("dec_pos_embed.")
            or name.startswith("decoder.")
            or name.startswith("dec_norm.")
            or name.startswith("lm_head.")
        ):
            identity = id(param)
            if identity not in seen:
                frozen += int(param.numel())
                seen.add(identity)
            param.requires_grad_(False)
    return frozen


def _freeze_encoder_layers_through(model: torch.nn.Module, last_layer_index: int) -> int:
    frozen = 0
    prefixes = tuple(f"encoder.{index}." for index in range(max(-1, int(last_layer_index)) + 1))
    if not prefixes:
        return 0
    for name, param in model.named_parameters():
        if name.startswith(prefixes):
            if param.requires_grad:
                frozen += int(param.numel())
            param.requires_grad_(False)
    return frozen


def _freeze_trainable_bitnet_linear_weights(model: torch.nn.Module) -> int:
    from compress.quantization import TrainableBitNetLinear

    frozen = 0
    for module in model.modules():
        if isinstance(module, TrainableBitNetLinear):
            if module.weight.requires_grad:
                module.weight.requires_grad_(False)
                frozen += int(module.weight.numel())
    return frozen


def _freeze_dense_linear_weights(model: torch.nn.Module) -> int:
    frozen = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.weight.requires_grad:
            module.weight.requires_grad_(False)
            frozen += int(module.weight.numel())
    return frozen


def _apply_trainable_name_filter(
    model: torch.nn.Module,
    include_patterns: str,
    exclude_patterns: str = "",
) -> dict[str, Any]:
    includes = [re.compile(item.strip()) for item in str(include_patterns or "").split(",") if item.strip()]
    excludes = [re.compile(item.strip()) for item in str(exclude_patterns or "").split(",") if item.strip()]
    if not includes:
        return {"enabled": False, "trainable_parameters": int(_trainable_parameter_count(model))}

    trainable_names: list[str] = []
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        keep = any(pattern.search(name) for pattern in includes)
        if keep and excludes:
            keep = not any(pattern.search(name) for pattern in excludes)
        param.requires_grad_(bool(keep))
        count = int(param.numel())
        if keep:
            trainable += count
            trainable_names.append(name)
        else:
            frozen += count
    return {
        "enabled": True,
        "frozen_parameters": frozen,
        "include": [pattern.pattern for pattern in includes],
        "exclude": [pattern.pattern for pattern in excludes],
        "trainable_parameter_names": trainable_names[:64],
        "trainable_parameter_name_count": len(trainable_names),
        "trainable_parameters": trainable,
    }


def _materialize_lazy_modules(model: torch.nn.Module) -> None:
    for module in model.modules():
        ensure_self_attn = getattr(module, "_ensure_self_attn", None)
        if callable(ensure_self_attn):
            ensure_self_attn()


def _convert_trainable_bitnet_to_dense(model: torch.nn.Module) -> int:
    from compress.quantization import TrainableBitNetLinear

    converted = 0
    for name, child in list(model.named_children()):
        if isinstance(child, TrainableBitNetLinear):
            layer = torch.nn.Linear(child.in_features, child.out_features, bias=child.bias is not None).to(
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            with torch.no_grad():
                layer.weight.copy_(child.ternary_weight().to(device=layer.weight.device, dtype=layer.weight.dtype))
                if layer.bias is not None and child.bias is not None:
                    layer.bias.copy_(child.bias.detach().to(device=layer.bias.device, dtype=layer.bias.dtype))
            setattr(model, name, layer)
            converted += 1
            continue
        converted += _convert_trainable_bitnet_to_dense(child)
    return converted


def _convert_trainable_bitnet_to_quantized(model: torch.nn.Module) -> int:
    from compress.quantization import TrainableBitNetLinear

    converted = 0
    for name, child in list(model.named_children()):
        if isinstance(child, TrainableBitNetLinear):
            layer = child.to_quantized()
            setattr(model, name, layer)
            converted += 1
            continue
        converted += _convert_trainable_bitnet_to_quantized(child)
    return converted


def _checkpoint_path(output_dir: Path, step: int) -> Path:
    return output_dir / "checkpoints" / f"step_{int(step):08d}.pt"


def _tokenizer_token_id(tokenizer: Any | None, token: str) -> int | None:
    if tokenizer is None:
        return None
    if hasattr(tokenizer, "token_to_id"):
        value = tokenizer.token_to_id(token)
    elif hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "token_to_id"):
        value = tokenizer.tokenizer.token_to_id(token)
    else:
        try:
            value = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            value = None
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return value


def _single_token_ids(tokenizer: Any | None, texts: tuple[str, ...]) -> tuple[int, ...]:
    token_ids: list[int] = []
    if tokenizer is None:
        return ()
    for text in texts:
        try:
            encoded = tokenizer(
                text,
                max_length=8,
                padding="max_length",
                truncation=True,
                add_special_tokens=False,
            )
            ids = [int(token_id) for token_id, mask in zip(encoded.get("input_ids", []), encoded.get("attention_mask", [])) if int(mask) > 0]
        except Exception:
            ids = []
        if len(ids) == 1 and ids[0] >= 0:
            token_ids.append(ids[0])
    return tuple(sorted(set(token_ids)))


def _initialize_expanded_agentkernel_rows(
    weight: torch.Tensor,
    *,
    old_vocab_size: int,
    tokenizer: Any | None,
) -> None:
    """Seed newly added structural tokens from existing structural rows."""
    if tokenizer is None:
        return
    row_sources: dict[str, str] = {
        "<AK_SOURCE_SLOTS>": "<AK_CONTEXT>",
        "<AK_STRUCTURED>": "<AK_JSON>",
        "<AK_ACTION_RESPOND>": "<AK_RESPOND>",
        "<AK_ACTION_ASK_USER>": "<AK_ASK_USER>",
        "<AK_ACTION_EXTENSION_REQUEST>": "<AK_EXTENSION>",
        "<AK_ACTION_SAVE_MEMORY>": "<AK_SAVE_MEMORY>",
        "<AK_CONTENT>": "<AK_ANSWER>",
        "</AK_CONTENT>": "<AK_ANSWER>",
        "<AK_TASK_TYPE>": "<AK_DECISION>",
        "<AK_INTENT>": "<AK_DECISION>",
        "<AK_FIELD>": "<AK_SLOT>",
        "<AK_FIELD_NAME>": "<AK_SLOT_NAME>",
        "<AK_FIELD_VALUE>": "<AK_SLOT_VALUE>",
        "<AK_FIELDS>": "<AK_SLOT>",
        "<AK_FRESHNESS>": "<AK_SOURCE_TYPE>",
        "<AK_END>": "</s>",
    }
    for index in range(1, 25):
        row_sources[f"<AK_COPY_USER_SOURCE_{index}>"] = "<AK_SLOT_VALUE>"
    for token, source in row_sources.items():
        target_id = _tokenizer_token_id(tokenizer, token)
        source_id = _tokenizer_token_id(tokenizer, source)
        if target_id is None or source_id is None:
            continue
        if target_id < old_vocab_size or target_id >= int(weight.shape[0]):
            continue
        if source_id >= old_vocab_size or source_id >= int(weight.shape[0]):
            continue
        weight[target_id, :].copy_(weight[source_id, :])


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoint_dir = output_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("step_*.pt"))
    return candidates[-1] if candidates else None


def _save_training_checkpoint(
    *,
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    losses: list[float],
    eval_history: list[dict[str, Any]],
    include_optimizer: bool,
) -> Path:
    path = _checkpoint_path(output_dir, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": int(step),
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "losses": [float(loss) for loss in losses],
        "eval_history": eval_history,
        "include_optimizer": bool(include_optimizer),
    }
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    latest = path.parent / "latest.json"
    latest.write_text(json.dumps({"step": int(step), "path": str(path)}, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    vocab_mismatch: str,
    tokenizer: Any | None = None,
) -> tuple[int, list[float], list[dict[str, Any]]]:
    if path.is_dir():
        from tensor.io_safetensors import safetensor_load

        payload: dict[str, Any] = {
            "step": 0,
            "model_state_dict": safetensor_load(str(path / "model.safetensors")),
            "losses": [],
            "eval_history": [],
        }
    elif path.name.endswith(".safetensors"):
        from tensor.io_safetensors import safetensor_load

        payload = {
            "step": 0,
            "model_state_dict": safetensor_load(str(path)),
            "losses": [],
            "eval_history": [],
        }
    else:
        payload = torch.load(path, map_location=device)
    state = payload["model_state_dict"]
    allowed_new_keys = {"enc_pos_embed.weight"}
    allowed_new_prefixes = (
        "agent_controller.",
        "agent_controller_norm.",
        "agent_controller_gate",
    )
    if str(vocab_mismatch).strip().lower() == "expand":
        current = model.state_dict()
        patched: dict[str, torch.Tensor] = {}
        for key, tensor in state.items():
            if key not in current:
                continue
            target = current[key]
            if tuple(tensor.shape) == tuple(target.shape):
                patched[key] = tensor
                continue
            if key in {"enc_embed.weight", "dec_embed.weight", "lm_head.weight"} and len(tensor.shape) == 2:
                if tensor.shape[1] != target.shape[1] or tensor.shape[0] > target.shape[0]:
                    raise RuntimeError(
                        f"cannot expand checkpoint tensor {key}: checkpoint={tuple(tensor.shape)} target={tuple(target.shape)}"
                    )
                expanded = target.detach().clone()
                expanded[: tensor.shape[0], :].copy_(tensor.to(dtype=expanded.dtype, device=expanded.device))
                _initialize_expanded_agentkernel_rows(expanded, old_vocab_size=int(tensor.shape[0]), tokenizer=tokenizer)
                patched[key] = expanded
                continue
            raise RuntimeError(
                f"checkpoint tensor shape mismatch for {key}: checkpoint={tuple(tensor.shape)} target={tuple(target.shape)}"
            )
        missing, unexpected = model.load_state_dict(patched, strict=False)
        unexpected = [
            key
            for key in unexpected
            if not key.startswith("future_bow_aux_head.")
            and not key.startswith("future_sketch_aux_head.")
            and not key.startswith("state_sketch_aux_head.")
            and not key.startswith("target_sketch_aux_head.")
            and not key.startswith("decoder_future_sketch_aux_head.")
        ]
        missing = [
            key
            for key in missing
            if key not in {"enc_embed.weight", "dec_embed.weight", "lm_head.weight", *allowed_new_keys}
            and not key.startswith("retrieval_query_head.")
            and not key.startswith("retrieval_doc_head.")
            and not key.startswith("agent_policy_heads.")
            and not key.startswith("agent_intent_head.")
            and not key.startswith(allowed_new_prefixes)
        ]
        if unexpected or missing:
            raise RuntimeError(f"partial checkpoint load mismatch: missing={missing} unexpected={unexpected}")
        optimizer = None
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [
            key
            for key in missing
            if key not in allowed_new_keys
            and not key.startswith("retrieval_query_head.")
            and not key.startswith("retrieval_doc_head.")
            and not key.startswith("agent_policy_heads.")
            and not key.startswith("agent_intent_head.")
            and not key.startswith(allowed_new_prefixes)
        ]
        unexpected = [
            key
            for key in unexpected
            if not key.startswith("future_bow_aux_head.")
            and not key.startswith("future_sketch_aux_head.")
            and not key.startswith("state_sketch_aux_head.")
            and not key.startswith("target_sketch_aux_head.")
            and not key.startswith("decoder_future_sketch_aux_head.")
        ]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint load mismatch: missing={missing} unexpected={unexpected}")
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    losses = [float(loss) for loss in payload.get("losses", [])]
    eval_history = payload.get("eval_history", [])
    if not isinstance(eval_history, list):
        eval_history = []
    return int(payload.get("step", 0) or 0), losses, eval_history


def _save_manifest(
    *,
    output_dir: Path,
    model_dir: Path,
    tokenizer_dir: Path,
    dataset_manifest: dict[str, Any],
    config,
    parameter_count: int,
    tokenizer_kind: str,
    tokenizer_name: str,
    training_summary: dict[str, Any],
    browser_bitnet_manifest_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path = output_dir / "agentkernel_lite_encdec_manifest.json"
    replaces_surfaces = ["chat_decision_generation", "context_grounded_reply_generation"]
    if getattr(config, "retrieval_head_dim", None):
        replaces_surfaces.extend(
            [
                "neural_retrieval_query_embedding",
                "neural_retrieval_doc_embedding",
                "retrieval_namespace_policy",
            ]
        )
    if bool(getattr(config, "agent_policy_heads", False)):
        replaces_surfaces.extend(
            [
                "controller_confidence_estimation",
                "controller_ood_estimation",
                "controller_verification_need_estimation",
                "controller_action_validity_estimation",
            ]
        )
    if int(getattr(config, "agent_controller_dim", 0) or 0) > 0:
        replaces_surfaces.extend(
            [
                "controller_latent_state_sidecar",
                "controller_state_sketch_training",
            ]
        )
    dataset_objective = str(dataset_manifest.get("objective", "")).strip()
    source_counts = dict(dataset_manifest.get("source_counts", {}) or {})
    task_type_counts = dict(dataset_manifest.get("task_type_counts", {}) or {})
    if "pocketpal_user_slots" in source_counts:
        replaces_surfaces.extend(
            [
                "user_slot_conditioned_generation",
                "profile_slot_update_policy",
                "local_privacy_boundary_policy",
                "local_memory_write_policy",
                "installed_extension_routing_policy",
                "missing_slot_question_policy",
                "generic_user_context_grounded_reply",
            ]
        )
    if any(str(key).startswith("slot_") for key in task_type_counts):
        replaces_surfaces.extend(["parameterized_user_configuration"])
    if "controller_trace" in dataset_objective:
        replaces_surfaces.extend(
            [
                "agentkernel_action_space_selection",
                "artifact_repair_action_policy",
                "source_inspection_policy",
            ]
        )
    if int(getattr(config, "agent_intent_labels", 0) or 0) > 1:
        replaces_surfaces.append("agent_intent_policy")
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_bundle",
        "model_family": "agentkernel_lite_encdec_v1",
        "chat_contract": {
            "primary_action": "respond",
            "code_execution": False,
            "structured_decision": True,
            "extensions_may_be_suggested": True,
        },
        "manifest_path": str(manifest_path),
        "model_dir": str(model_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "tokenizer_kind": tokenizer_kind,
        "tokenizer_name": tokenizer_name,
        "dataset_manifest_path": str(dataset_manifest.get("manifest_path", "")),
        "model_config": asdict(config),
        "parameter_count": int(parameter_count),
        "training_summary": training_summary,
        "runtime_targets": {
            "browser": "model_stack_browser_bitnet_encoder_decoder",
            "kernel": "agentkernel_lite_rust_wasm_loop",
        },
        "replaces_surfaces": sorted(dict.fromkeys(replaces_surfaces)),
    }
    if browser_bitnet_manifest_path is not None:
        manifest["browser_bitnet_manifest_path"] = str(browser_bitnet_manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _attach_tokenizer_to_browser_bitnet(
    *,
    browser_bitnet_manifest_path: Path,
    tokenizer_dir: Path,
    tokenizer_kind: str,
    tokenizer,
) -> None:
    if not browser_bitnet_manifest_path.exists():
        return
    browser_dir = browser_bitnet_manifest_path.parent
    target_dir = browser_dir / "tokenizer"
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in tokenizer_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, target_dir / path.name)
    payload = json.loads(browser_bitnet_manifest_path.read_text(encoding="utf-8"))
    payload["tokenizer"] = {
        "kind": str(tokenizer_kind),
        "path": "tokenizer/tokenizer.json" if (target_dir / "tokenizer.json").exists() else "tokenizer/tokenizer_config.json",
        "config_path": "tokenizer/tokenizer_config.json",
        "pad_token_id": int(getattr(tokenizer, "pad_token_id", 0) or 0),
        "bos_token_id": int(getattr(tokenizer, "bos_token_id", 1) or 1),
        "eos_token_id": int(getattr(tokenizer, "eos_token_id", 2) or 2),
        "unk_token_id": int(getattr(tokenizer, "unk_token_id", 3) or 3),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer)),
    }
    browser_bitnet_manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def train(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    _install_model_stack_path(repo_root)
    _seed_everything(int(args.seed))
    bitnet_training_env: dict[str, str] = {}
    if bool(getattr(args, "bitnet_qat", 0)):
        training_forward = str(getattr(args, "bitnet_training_forward", "") or "").strip()
        if training_forward:
            os.environ["MODEL_STACK_TRAINABLE_BITNET_TRAINING_FORWARD"] = training_forward
            bitnet_training_env["MODEL_STACK_TRAINABLE_BITNET_TRAINING_FORWARD"] = training_forward
        if bool(getattr(args, "bitnet_strict_ternary_forward", 0)):
            if not training_forward:
                os.environ["MODEL_STACK_TRAINABLE_BITNET_TRAINING_FORWARD"] = "packed_int4_ste"
                bitnet_training_env["MODEL_STACK_TRAINABLE_BITNET_TRAINING_FORWARD"] = "packed_int4_ste"
            os.environ["MODEL_STACK_TRAINABLE_BITNET_STRICT_TERNARY_FORWARD"] = "1"
            bitnet_training_env["MODEL_STACK_TRAINABLE_BITNET_STRICT_TERNARY_FORWARD"] = "1"
        activation_quant = str(getattr(args, "bitnet_activation_quant", "") or "").strip()
        if activation_quant:
            os.environ["MODEL_STACK_TRAINABLE_BITNET_ACT_QUANT"] = activation_quant
            bitnet_training_env["MODEL_STACK_TRAINABLE_BITNET_ACT_QUANT"] = activation_quant
        activation_bits = int(getattr(args, "bitnet_activation_quant_bits", 0) or 0)
        if activation_bits > 0:
            os.environ["MODEL_STACK_TRAINABLE_BITNET_ACT_QUANT_BITS"] = str(activation_bits)
            bitnet_training_env["MODEL_STACK_TRAINABLE_BITNET_ACT_QUANT_BITS"] = str(activation_bits)

    from runtime.checkpoint import save_pretrained
    from runtime.seq2seq import EncoderDecoderLM
    from export.exporter import export_model
    from specs.export import ExportConfig
    from compress.apply import apply_compression

    dataset_manifest = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).resolve()
    model_dir = output_dir / "model"
    tokenizer_dir = output_dir / "tokenizer"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_vocab_mismatch = str(args.checkpoint_vocab_mismatch).strip().lower()
    checkpoint_continuation = bool(
        str(getattr(args, "init_from_checkpoint", "") or "").strip()
        or str(getattr(args, "resume_from", "") or "").strip()
        or bool(getattr(args, "resume_latest", 0))
    )
    if (
        checkpoint_continuation
        and checkpoint_vocab_mismatch == "expand"
        and (bool(getattr(args, "freeze_token_embeddings", 0)) or bool(getattr(args, "freeze_lm_head", 0)))
    ):
        raise SystemExit(
            "Refusing checkpoint vocab expansion while token embeddings or lm_head are frozen. "
            "New token rows need training; use the checkpoint tokenizer via --tokenizer-source-dir, "
            "or unfreeze embeddings/lm_head for an explicit vocab-expansion run."
        )

    tokenizer = _load_tokenizer(args, dataset_manifest=dataset_manifest)
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    config = _preset_config(str(args.preset), vocab_size=vocab_size)
    config.pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    config.attn_dropout = float(getattr(args, "attn_dropout", 0.0) or 0.0)
    config.mlp_dropout = float(getattr(args, "mlp_dropout", 0.0) or 0.0)
    config.resid_dropout = float(getattr(args, "resid_dropout", 0.0) or 0.0)
    config.encoder_position_embeddings = bool(args.encoder_position_embeddings)
    retrieval_head_dim = int(getattr(args, "retrieval_head_dim", 0) or 0)
    config.retrieval_head_dim = retrieval_head_dim if retrieval_head_dim > 0 else None
    config.agent_policy_heads = bool(getattr(args, "agent_policy_heads", 0))
    intent_label_count = int(getattr(args, "agent_intent_labels", 0) or 0)
    if intent_label_count <= 0:
        intent_labels = dataset_manifest.get("intent_labels", {}) or {}
        if isinstance(intent_labels, dict) and intent_labels:
            intent_label_count = max(int(value) for value in intent_labels.values()) + 1
    config.agent_intent_labels = max(0, int(intent_label_count))
    config.agent_controller_dim = max(0, int(getattr(args, "agent_controller_dim", 0) or 0))
    config.agent_controller_apply_policy = bool(getattr(args, "agent_controller_apply_policy", 1))
    config.agent_controller_apply_retrieval = bool(getattr(args, "agent_controller_apply_retrieval", 0))
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=vocab_size)
    _materialize_lazy_modules(model)
    parameter_count = _parameter_count(model)
    frozen_encoder_parameters = 0
    frozen_token_embedding_parameters = 0
    frozen_lm_head_parameters = 0
    frozen_decoder_parameters = 0
    frozen_encoder_layer_parameters = 0
    frozen_dense_linear_weight_parameters = 0
    tokenizer_kind = "byte" if bool(args.byte_tokenizer) else str(args.tokenizer_kind)
    tokenizer_name = str(args.tokenizer_name) if tokenizer_kind == "hf" else tokenizer_kind

    dry_summary = {
        "dry_run": bool(args.dry_run),
        "preset": str(args.preset),
        "device": str(args.device),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "parameter_count": int(parameter_count),
        "tokenizer_kind": tokenizer_kind,
        "tokenizer_vocab_size": int(vocab_size),
        "dataset_objective": str(dataset_manifest.get("objective", "")),
        "primary_action": "respond" if str(dataset_manifest.get("objective", "")).lower() in {"chat", "text"} else "mixed",
        "code_execution": False,
        "extension_counts": dict(dataset_manifest.get("extension_counts", {}) or {}),
        "checkpoint_vocab_mismatch": checkpoint_vocab_mismatch,
        "tokenizer_source_dir": str(getattr(args, "tokenizer_source_dir", "") or ""),
        "checkpoint_every": int(args.checkpoint_every),
        "eval_every": int(args.eval_every),
        "bitnet_qat": bool(args.bitnet_qat),
        "bitnet_training_env": dict(bitnet_training_env),
        "init_from_checkpoint": str(args.init_from_checkpoint),
        "train_dataset_path": str(dataset_manifest.get("train_dataset_path", "")),
        "eval_dataset_path": str(dataset_manifest.get("eval_dataset_path", "")),
        "retrieval_contrastive_weight": float(args.retrieval_contrastive_weight),
        "retrieval_temperature": float(args.retrieval_temperature),
        "retrieval_head_dim": int(retrieval_head_dim),
        "retrieval_ternary_aware_weight": float(args.retrieval_ternary_aware_weight),
        "retrieval_ternary_threshold_ratio": float(args.retrieval_ternary_threshold_ratio),
        "retrieval_ternary_group_size": int(args.retrieval_ternary_group_size),
        "retrieval_ternary_residual_dims": int(args.retrieval_ternary_residual_dims),
        "retrieval_ternary_teacher_distill_weight": float(args.retrieval_ternary_teacher_distill_weight),
        "retrieval_ternary_teacher_temperature": float(args.retrieval_ternary_teacher_temperature),
        "retrieval_ternary_reconstruction_weight": float(args.retrieval_ternary_reconstruction_weight),
        "retrieval_hard_negative_weight": float(args.retrieval_hard_negative_weight),
        "retrieval_hard_negative_ternary": bool(args.retrieval_hard_negative_ternary),
        "agent_policy_heads": bool(args.agent_policy_heads),
        "agent_controller_dim": int(config.agent_controller_dim),
        "agent_controller_apply_policy": bool(config.agent_controller_apply_policy),
        "agent_controller_apply_retrieval": bool(config.agent_controller_apply_retrieval),
        "policy_head_loss_weight": float(args.policy_head_loss_weight),
        "agent_intent_labels": int(config.agent_intent_labels),
        "intent_head_loss_weight": float(args.intent_head_loss_weight),
        "intent_contrastive_weight": float(args.intent_contrastive_weight),
        "intent_contrastive_temperature": float(args.intent_contrastive_temperature),
        "encoder_rep_distill_weight": float(args.encoder_rep_distill_weight),
        "future_bow_aux_weight": float(args.future_bow_aux_weight),
        "future_bow_buckets": int(args.future_bow_buckets),
        "future_bow_weight_schedule": str(args.future_bow_weight_schedule),
        "future_sketch_aux_weight": float(args.future_sketch_aux_weight),
        "future_sketch_buckets": int(args.future_sketch_buckets),
        "future_sketch_min_token_weight": float(args.future_sketch_min_token_weight),
        "future_sketch_topk": int(args.future_sketch_topk),
        "future_sketch_windows": list(_parse_positive_ints(str(args.future_sketch_windows))),
        "state_sketch_aux_weight": float(args.state_sketch_aux_weight),
        "state_sketch_buckets": int(args.state_sketch_buckets),
        "state_sketch_topk": int(args.state_sketch_topk),
        "state_sketch_min_token_weight": float(args.state_sketch_min_token_weight),
        "target_sketch_aux_weight": float(args.target_sketch_aux_weight),
        "target_sketch_buckets": int(args.target_sketch_buckets),
        "target_sketch_topk": int(args.target_sketch_topk),
        "target_sketch_min_token_weight": float(args.target_sketch_min_token_weight),
        "decoder_future_sketch_aux_weight": float(args.decoder_future_sketch_aux_weight),
        "decoder_future_sketch_buckets": int(args.decoder_future_sketch_buckets),
        "decoder_future_sketch_min_token_weight": float(args.decoder_future_sketch_min_token_weight),
        "decoder_future_sketch_topk": int(args.decoder_future_sketch_topk),
        "decoder_future_sketch_windows": list(_parse_positive_ints(str(args.decoder_future_sketch_windows))),
        "decoder_drift_alignment_weight": float(args.decoder_drift_alignment_weight),
        "decoder_drift_alignment_min_token_weight": float(args.decoder_drift_alignment_min_token_weight),
        "decoder_drift_alignment_temperature": float(args.decoder_drift_alignment_temperature),
        "decoder_source_copy_anchor_weight": float(args.decoder_source_copy_anchor_weight),
        "decoder_source_copy_min_token_weight": float(args.decoder_source_copy_min_token_weight),
        "aux_grad_budget": float(args.aux_grad_budget),
        "pcgrad_decoder_aux": bool(args.pcgrad_decoder_aux),
        "decoder_loss_weight": float(args.decoder_loss_weight),
        "decoder_eos_loss_weight": float(args.decoder_eos_loss_weight),
        "decoder_json_structure_loss_weight": float(args.decoder_json_structure_loss_weight),
        "decoder_control_token_suppression_weight": float(args.decoder_control_token_suppression_weight),
        "decoder_token_weight_alpha": float(args.decoder_token_weight_alpha),
        "decoder_multi_token_ce_weight": float(args.decoder_multi_token_ce_weight),
        "decoder_multi_token_horizons": list(_parse_positive_ints(str(args.decoder_multi_token_horizons))),
        "decoder_multi_token_min_token_weight": float(args.decoder_multi_token_min_token_weight),
        "decoder_stop_token_loss_weight": float(args.decoder_stop_token_loss_weight),
        "decoder_seq_eos_token_weight": float(args.decoder_seq_eos_token_weight),
        "anchor_weight_loss_weight": float(args.anchor_weight_loss_weight),
        "negative_decoder_loss_weight": float(args.negative_decoder_loss_weight),
        "negative_divergent_decoder_loss_weight": float(args.negative_divergent_decoder_loss_weight),
        "negative_first_token_margin_weight": float(args.negative_first_token_margin_weight),
        "negative_first_token_margin": float(args.negative_first_token_margin),
        "teacher_distill_weight": float(args.teacher_distill_weight),
        "teacher_distill_temperature": float(args.teacher_distill_temperature),
        "attn_dropout": float(args.attn_dropout),
        "mlp_dropout": float(args.mlp_dropout),
        "resid_dropout": float(args.resid_dropout),
        "freeze_encoder": bool(args.freeze_encoder),
        "freeze_token_embeddings": bool(args.freeze_token_embeddings),
        "freeze_decoder": bool(args.freeze_decoder),
        "freeze_encoder_layers_through": int(args.freeze_encoder_layers_through),
        "freeze_bitnet_linear_weights": bool(args.freeze_bitnet_linear_weights),
        "max_retrieval_query_tokens": int(args.max_retrieval_query_tokens),
        "max_retrieval_doc_tokens": int(args.max_retrieval_doc_tokens),
        "max_retrieval_negatives": int(args.max_retrieval_negatives),
    }
    if bool(args.dry_run):
        tokenizer.save_pretrained(str(tokenizer_dir))
        return _save_manifest(
            output_dir=output_dir,
            model_dir=model_dir,
            tokenizer_dir=tokenizer_dir,
            dataset_manifest=dataset_manifest,
            config=config,
            parameter_count=parameter_count,
            tokenizer_kind=tokenizer_kind,
            tokenizer_name=tokenizer_name,
            training_summary=dry_summary,
        )

    train_stream_specs = [str(item) for item in (dataset_manifest.get("train_stream_specs") or []) if str(item).strip()]
    if train_stream_specs:
        train_dataset = EncDecHfStreamIterableDataset(
            train_stream_specs,
            tokenizer,
            max_encoder_tokens=int(args.max_encoder_tokens),
            max_decoder_tokens=int(args.max_decoder_tokens),
            max_retrieval_query_tokens=int(args.max_retrieval_query_tokens),
            max_retrieval_doc_tokens=int(args.max_retrieval_doc_tokens),
            max_retrieval_negatives=int(args.max_retrieval_negatives),
            weight=float(dataset_manifest.get("stream_weight", 3.0) or 3.0),
            shuffle=bool(args.train_shuffle),
            shuffle_buffer_size=int(args.parquet_shuffle_buffer_size),
            seed=int(args.seed),
        )
        print(
            json.dumps(
                {
                    "event": "hf_stream_train_dataset",
                    "specs": train_stream_specs,
                    "estimated_rows": len(train_dataset),
                    "stream_weight": float(dataset_manifest.get("stream_weight", 3.0) or 3.0),
                },
                sort_keys=True,
            )
        )
    else:
        train_dataset = _load_encdec_dataset(
            Path(str(dataset_manifest["train_dataset_path"])),
            tokenizer,
            max_encoder_tokens=int(args.max_encoder_tokens),
            max_decoder_tokens=int(args.max_decoder_tokens),
            max_retrieval_query_tokens=int(args.max_retrieval_query_tokens),
            max_retrieval_doc_tokens=int(args.max_retrieval_doc_tokens),
            max_retrieval_negatives=int(args.max_retrieval_negatives),
            shuffle=bool(args.train_shuffle),
            shuffle_buffer_size=int(args.parquet_shuffle_buffer_size),
            require_retrieval_pair=bool(args.parquet_require_retrieval_pair),
            action_include=tuple(item.strip() for item in str(args.parquet_action_include).split(",") if item.strip()),
            task_type_include=tuple(item.strip() for item in str(args.parquet_task_type_include).split(",") if item.strip()),
            seed=int(args.seed),
        )
    if len(train_dataset) <= 0:
        raise SystemExit("training dataset is empty")
    eval_dataset = None
    eval_stream_specs = [str(item) for item in (dataset_manifest.get("eval_stream_specs") or []) if str(item).strip()]
    if eval_stream_specs:
        eval_dataset = EncDecHfStreamIterableDataset(
            eval_stream_specs,
            tokenizer,
            max_encoder_tokens=int(args.max_encoder_tokens),
            max_decoder_tokens=int(args.max_decoder_tokens),
            max_retrieval_query_tokens=int(args.max_retrieval_query_tokens),
            max_retrieval_doc_tokens=int(args.max_retrieval_doc_tokens),
            max_retrieval_negatives=int(args.max_retrieval_negatives),
            weight=float(dataset_manifest.get("stream_weight", 3.0) or 3.0),
            shuffle=False,
            shuffle_buffer_size=0,
            seed=int(args.seed) + 1009,
        )
        print(
            json.dumps(
                {
                    "event": "hf_stream_eval_dataset",
                    "specs": eval_stream_specs,
                    "estimated_rows": len(eval_dataset),
                },
                sort_keys=True,
            )
        )
    eval_path_value = str(dataset_manifest.get("eval_dataset_path", "") or "")
    eval_path = Path(eval_path_value) if eval_path_value else None
    if eval_dataset is None and eval_path is not None and eval_path.exists():
        eval_dataset = _load_encdec_dataset(
            eval_path,
            tokenizer,
            max_encoder_tokens=int(args.max_encoder_tokens),
            max_decoder_tokens=int(args.max_decoder_tokens),
            max_retrieval_query_tokens=int(args.max_retrieval_query_tokens),
            max_retrieval_doc_tokens=int(args.max_retrieval_doc_tokens),
            max_retrieval_negatives=int(args.max_retrieval_negatives),
            shuffle=False,
            shuffle_buffer_size=0,
            require_retrieval_pair=bool(args.parquet_require_retrieval_pair),
            action_include=tuple(item.strip() for item in str(args.parquet_action_include).split(",") if item.strip()),
            task_type_include=tuple(item.strip() for item in str(args.parquet_task_type_include).split(",") if item.strip()),
            seed=int(args.seed),
        )
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    train_is_iterable = isinstance(train_dataset, IterableDataset)
    loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=False if train_is_iterable else bool(args.train_shuffle),
        num_workers=0,
        collate_fn=_collate,
        generator=None if train_is_iterable else generator,
    )
    device = torch.device(str(args.device))
    model.to(device)
    model.train()
    init_checkpoint_path: Path | None = None
    if str(args.init_from_checkpoint).strip():
        init_checkpoint_path = Path(str(args.init_from_checkpoint)).expanduser().resolve()
        if not init_checkpoint_path.exists():
            raise FileNotFoundError(f"init checkpoint not found: {init_checkpoint_path}")
        _load_training_checkpoint(
            init_checkpoint_path,
            model=model,
            optimizer=None,
            device=device,
            vocab_mismatch=str(args.checkpoint_vocab_mismatch),
            tokenizer=tokenizer,
        )
        print(json.dumps({"event": "initialized", "checkpoint": str(init_checkpoint_path)}, sort_keys=True))
    if bool(args.freeze_encoder):
        frozen_encoder_parameters = _freeze_encoder_parameters(model)
        print(
            json.dumps(
                {
                    "event": "encoder_frozen",
                    "frozen_parameters": int(frozen_encoder_parameters),
                    "trainable_parameters": int(_trainable_parameter_count(model)),
                },
                sort_keys=True,
            )
        )
    if bool(args.freeze_token_embeddings):
        frozen_token_embedding_parameters = _freeze_token_embedding_parameters(model)
        print(
            json.dumps(
                {
                    "event": "token_embeddings_frozen",
                    "frozen_parameters": int(frozen_token_embedding_parameters),
                    "trainable_parameters": int(_trainable_parameter_count(model)),
                },
                sort_keys=True,
            )
        )
    if bool(getattr(args, "freeze_lm_head", 0)):
        frozen_lm_head_parameters = _freeze_lm_head_parameters(model)
        print(
            json.dumps(
                {
                    "event": "lm_head_frozen",
                    "frozen_parameters": int(frozen_lm_head_parameters),
                    "trainable_parameters": int(_trainable_parameter_count(model)),
                },
                sort_keys=True,
            )
        )
    if bool(args.freeze_decoder):
        frozen_decoder_parameters = _freeze_decoder_parameters(model)
        print(
            json.dumps(
                {
                    "event": "decoder_frozen",
                    "frozen_parameters": int(frozen_decoder_parameters),
                    "trainable_parameters": int(_trainable_parameter_count(model)),
                },
                sort_keys=True,
            )
        )
    if bool(getattr(args, "freeze_dense_linear_weights", 0)):
        newly_frozen_dense_linear_weight_parameters = _freeze_dense_linear_weights(model)
        frozen_dense_linear_weight_parameters += int(newly_frozen_dense_linear_weight_parameters)
        print(
            json.dumps(
                {
                    "event": "dense_linear_weights_frozen",
                    "frozen_parameters": int(newly_frozen_dense_linear_weight_parameters),
                    "total_frozen_parameters": int(frozen_dense_linear_weight_parameters),
                    "trainable_parameters": int(_trainable_parameter_count(model)),
                },
                sort_keys=True,
            )
        )
    if int(args.freeze_encoder_layers_through) >= 0:
        frozen_encoder_layer_parameters = _freeze_encoder_layers_through(model, int(args.freeze_encoder_layers_through))
        print(
            json.dumps(
                {
                    "event": "encoder_layers_frozen",
                    "frozen_parameters": int(frozen_encoder_layer_parameters),
                    "through_layer": int(args.freeze_encoder_layers_through),
                    "trainable_parameters": int(_trainable_parameter_count(model)),
                },
                sort_keys=True,
            )
        )
    teacher_model = None
    if (
        float(getattr(args, "teacher_distill_weight", 0.0) or 0.0) > 0.0
        or float(getattr(args, "retrieval_ternary_teacher_distill_weight", 0.0) or 0.0) > 0.0
        or float(getattr(args, "encoder_rep_distill_weight", 0.0) or 0.0) > 0.0
    ):
        # Capture the dense initialized model before BitNet QAT replaces linear
        # layers. For init-from-checkpoint runs this is the trained dense teacher;
        # for resume-only runs callers should prefer init-from-checkpoint for a
        # stable teacher and resume only for non-distillation continuation.
        teacher_model = copy.deepcopy(model).to(device).eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        print(
            json.dumps(
                {
                    "event": "teacher_distill_enabled",
                    "weight": float(args.teacher_distill_weight),
                    "temperature": float(args.teacher_distill_temperature),
                    "encoder_rep_distill_weight": float(args.encoder_rep_distill_weight),
                    "retrieval_ternary_teacher_distill_weight": float(args.retrieval_ternary_teacher_distill_weight),
                    "retrieval_ternary_teacher_temperature": float(args.retrieval_ternary_teacher_temperature),
                },
                sort_keys=True,
            )
        )
    qat_summary: dict[str, Any] | None = None
    trainable_parameter_count_before_export = int(_trainable_parameter_count(model))
    if bool(args.bitnet_qat):
        bitnet_qat_include = tuple(
            item.strip() for item in str(getattr(args, "bitnet_qat_include", "") or "").split(",") if item.strip()
        )
        bitnet_qat_exclude = tuple(
            item.strip() for item in str(getattr(args, "bitnet_qat_exclude", "") or "").split(",") if item.strip()
        )
        qat_summary = apply_compression(
            model,
            quant={
                "scheme": "bitnet_qat",
                "include": bitnet_qat_include or None,
                "exclude": bitnet_qat_exclude or None,
            },
        )
        trainable_parameter_count_before_export = int(_trainable_parameter_count(model))
        print(
            json.dumps(
                {
                    "event": "bitnet_qat_enabled",
                    "modules": int((qat_summary.get("quant") or {}).get("num", 0)),
                    "training_env": dict(bitnet_training_env),
                    "trainable_parameters": int(trainable_parameter_count_before_export),
                },
                sort_keys=True,
            )
        )
    if bool(getattr(args, "freeze_lm_head", 0)):
        frozen_lm_head_parameters = _freeze_lm_head_parameters(model)
        trainable_parameter_count_before_export = int(_trainable_parameter_count(model))
        print(
            json.dumps(
                {
                    "event": "lm_head_frozen_after_qat",
                    "frozen_parameters": int(frozen_lm_head_parameters),
                    "trainable_parameters": int(trainable_parameter_count_before_export),
                },
                sort_keys=True,
            )
        )
    if bool(args.freeze_encoder):
        frozen_encoder_parameters = _freeze_encoder_parameters(model)
        trainable_parameter_count_before_export = int(_trainable_parameter_count(model))
        print(
            json.dumps(
                {
                    "event": "encoder_frozen_after_qat",
                    "frozen_parameters": int(frozen_encoder_parameters),
                    "trainable_parameters": int(trainable_parameter_count_before_export),
                },
                sort_keys=True,
            )
        )
    if bool(args.freeze_token_embeddings):
        frozen_token_embedding_parameters = _freeze_token_embedding_parameters(model)
        trainable_parameter_count_before_export = int(_trainable_parameter_count(model))
        print(
            json.dumps(
                {
                    "event": "token_embeddings_frozen_after_qat",
                    "frozen_parameters": int(frozen_token_embedding_parameters),
                    "trainable_parameters": int(trainable_parameter_count_before_export),
                },
                sort_keys=True,
            )
        )
    if bool(args.freeze_decoder):
        frozen_decoder_parameters = _freeze_decoder_parameters(model)
        trainable_parameter_count_before_export = int(_trainable_parameter_count(model))
        print(
            json.dumps(
                {
                    "event": "decoder_frozen_after_qat",
                    "frozen_parameters": int(frozen_decoder_parameters),
                    "trainable_parameters": int(trainable_parameter_count_before_export),
                },
                sort_keys=True,
            )
        )
    if bool(getattr(args, "freeze_dense_linear_weights", 0)):
        newly_frozen_dense_linear_weight_parameters = _freeze_dense_linear_weights(model)
        frozen_dense_linear_weight_parameters += int(newly_frozen_dense_linear_weight_parameters)
        trainable_parameter_count_before_export = int(_trainable_parameter_count(model))
        print(
            json.dumps(
                {
                    "event": "dense_linear_weights_frozen_after_qat",
                    "frozen_parameters": int(newly_frozen_dense_linear_weight_parameters),
                    "total_frozen_parameters": int(frozen_dense_linear_weight_parameters),
                    "trainable_parameters": int(trainable_parameter_count_before_export),
                },
                sort_keys=True,
            )
        )
    if int(args.freeze_encoder_layers_through) >= 0:
        frozen_encoder_layer_parameters = _freeze_encoder_layers_through(model, int(args.freeze_encoder_layers_through))
        trainable_parameter_count_before_export = int(_trainable_parameter_count(model))
        print(
            json.dumps(
                {
                    "event": "encoder_layers_frozen_after_qat",
                    "frozen_parameters": int(frozen_encoder_layer_parameters),
                    "through_layer": int(args.freeze_encoder_layers_through),
                    "trainable_parameters": int(trainable_parameter_count_before_export),
                },
                sort_keys=True,
            )
        )
    frozen_bitnet_linear_weight_parameters = 0
    if bool(getattr(args, "freeze_bitnet_linear_weights", 0)):
        frozen_bitnet_linear_weight_parameters = _freeze_trainable_bitnet_linear_weights(model)
        trainable_parameter_count_before_export = int(_trainable_parameter_count(model))
        print(
            json.dumps(
                {
                    "event": "bitnet_linear_weights_frozen",
                    "frozen_parameters": int(frozen_bitnet_linear_weight_parameters),
                    "trainable_parameters": int(trainable_parameter_count_before_export),
                },
                sort_keys=True,
            )
        )
    trainable_name_filter_summary = _apply_trainable_name_filter(
        model,
        include_patterns=str(getattr(args, "trainable_name_include", "") or ""),
        exclude_patterns=str(getattr(args, "trainable_name_exclude", "") or ""),
    )
    if bool(trainable_name_filter_summary.get("enabled")):
        trainable_parameter_count_before_export = int(_trainable_parameter_count(model))
        print(
            json.dumps(
                {
                    "event": "trainable_name_filter_applied",
                    **trainable_name_filter_summary,
                    "trainable_parameters": int(trainable_parameter_count_before_export),
                },
                sort_keys=True,
            )
        )
    anchor_parameters: dict[str, torch.Tensor] = {}
    if float(getattr(args, "anchor_weight_loss_weight", 0.0) or 0.0) > 0.0:
        anchor_parameters = _trainable_anchor_parameters(model)
        print(
            json.dumps(
                {
                    "event": "anchor_weight_loss_enabled",
                    "weight": float(args.anchor_weight_loss_weight),
                    "anchored_parameter_tensors": len(anchor_parameters),
                    "anchored_parameters": int(sum(tensor.numel() for tensor in anchor_parameters.values())),
                },
                sort_keys=True,
            )
        )
    if float(getattr(args, "future_bow_aux_weight", 0.0) or 0.0) > 0.0:
        model.future_bow_aux_head = torch.nn.Linear(int(config.d_model), int(args.future_bow_buckets)).to(device)
        print(
            json.dumps(
                {
                    "event": "future_bow_aux_enabled",
                    "buckets": int(args.future_bow_buckets),
                    "weight": float(args.future_bow_aux_weight),
                    "parameters": int(sum(param.numel() for param in model.future_bow_aux_head.parameters())),
                },
                sort_keys=True,
            )
        )
    if float(getattr(args, "future_sketch_aux_weight", 0.0) or 0.0) > 0.0:
        future_sketch_windows = _parse_positive_ints(str(args.future_sketch_windows))
        future_sketch_head_dim = int(args.future_sketch_buckets) * max(1, len(future_sketch_windows))
        model.future_sketch_aux_head = torch.nn.Linear(int(config.d_model), int(future_sketch_head_dim)).to(device)
        print(
            json.dumps(
                {
                    "event": "future_sketch_aux_enabled",
                    "buckets": int(args.future_sketch_buckets),
                    "windows": list(future_sketch_windows),
                    "weight": float(args.future_sketch_aux_weight),
                    "topk": int(args.future_sketch_topk),
                    "min_token_weight": float(args.future_sketch_min_token_weight),
                    "aux_grad_budget": float(args.aux_grad_budget),
                    "parameters": int(sum(param.numel() for param in model.future_sketch_aux_head.parameters())),
                },
                sort_keys=True,
            )
        )
    if float(getattr(args, "state_sketch_aux_weight", 0.0) or 0.0) > 0.0:
        model.state_sketch_aux_head = torch.nn.Linear(int(config.d_model), int(args.state_sketch_buckets)).to(device)
        print(
            json.dumps(
                {
                    "event": "state_sketch_aux_enabled",
                    "buckets": int(args.state_sketch_buckets),
                    "weight": float(args.state_sketch_aux_weight),
                    "topk": int(args.state_sketch_topk),
                    "min_token_weight": float(args.state_sketch_min_token_weight),
                    "aux_grad_budget": float(args.aux_grad_budget),
                    "parameters": int(sum(param.numel() for param in model.state_sketch_aux_head.parameters())),
                },
                sort_keys=True,
            )
        )
    if float(getattr(args, "target_sketch_aux_weight", 0.0) or 0.0) > 0.0:
        model.target_sketch_aux_head = torch.nn.Linear(int(config.d_model), int(args.target_sketch_buckets)).to(device)
        print(
            json.dumps(
                {
                    "event": "target_sketch_aux_enabled",
                    "buckets": int(args.target_sketch_buckets),
                    "weight": float(args.target_sketch_aux_weight),
                    "topk": int(args.target_sketch_topk),
                    "min_token_weight": float(args.target_sketch_min_token_weight),
                    "aux_grad_budget": float(args.aux_grad_budget),
                    "parameters": int(sum(param.numel() for param in model.target_sketch_aux_head.parameters())),
                },
                sort_keys=True,
            )
        )
    if float(getattr(args, "decoder_future_sketch_aux_weight", 0.0) or 0.0) > 0.0:
        decoder_future_sketch_windows = _parse_positive_ints(str(args.decoder_future_sketch_windows))
        if not decoder_future_sketch_windows:
            decoder_future_sketch_windows = (8, 32)
        decoder_future_sketch_head_dim = int(args.decoder_future_sketch_buckets) * len(decoder_future_sketch_windows)
        model.decoder_future_sketch_aux_head = torch.nn.Linear(
            int(config.d_model),
            int(decoder_future_sketch_head_dim),
        ).to(device)
        print(
            json.dumps(
                {
                    "event": "decoder_future_sketch_aux_enabled",
                    "buckets": int(args.decoder_future_sketch_buckets),
                    "windows": list(decoder_future_sketch_windows),
                    "weight": float(args.decoder_future_sketch_aux_weight),
                    "topk": int(args.decoder_future_sketch_topk),
                    "min_token_weight": float(args.decoder_future_sketch_min_token_weight),
                    "aux_grad_budget": float(args.aux_grad_budget),
                    "parameters": int(sum(param.numel() for param in model.decoder_future_sketch_aux_head.parameters())),
                },
                sort_keys=True,
            )
        )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    resume_path: Path | None = None
    if str(args.resume_from).strip():
        resume_path = Path(str(args.resume_from)).expanduser().resolve()
    elif bool(args.resume_latest):
        resume_path = _latest_checkpoint(output_dir)
    losses: list[float] = []
    eval_history: list[dict[str, Any]] = []
    start_step = 0
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        start_step, losses, eval_history = _load_training_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
            vocab_mismatch=str(args.checkpoint_vocab_mismatch),
            tokenizer=tokenizer,
        )
        print(json.dumps({"event": "resumed", "checkpoint": str(resume_path), "step": start_step}, sort_keys=True))
    iterator = iter(loader)
    last_step = start_step
    for step in range(start_step + 1, int(args.max_steps) + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        loss_kwargs = {
            "decoder_loss_weight": float(args.decoder_loss_weight),
            "decoder_eos_loss_weight": float(args.decoder_eos_loss_weight),
            "decoder_json_structure_loss_weight": float(args.decoder_json_structure_loss_weight),
            "decoder_token_weight_alpha": float(args.decoder_token_weight_alpha),
            "decoder_multi_token_ce_weight": _scheduled_aux_weight(
                float(args.decoder_multi_token_ce_weight),
                step=step,
                total_steps=int(args.max_steps),
                schedule=str(args.future_bow_weight_schedule),
            ),
            "decoder_multi_token_horizons": _parse_positive_ints(str(args.decoder_multi_token_horizons)),
            "decoder_multi_token_min_token_weight": float(args.decoder_multi_token_min_token_weight),
            "decoder_seq_eos_token_weight": float(args.decoder_seq_eos_token_weight),
            "decoder_stop_token_loss_weight": float(args.decoder_stop_token_loss_weight),
            "decoder_eos_token_id": int(getattr(tokenizer, "eos_token_id", 2) or 2),
            "decoder_stop_token_ids": _single_token_ids(tokenizer, ("</AK_CONTENT>", "<AK_END>")),
            "decoder_json_structure_token_ids": _single_token_ids(tokenizer, ("{", "}", "[", "]", "\"", ":", ",")),
            "decoder_control_token_suppression_weight": float(args.decoder_control_token_suppression_weight),
            "decoder_control_token_ids": _single_token_ids(
                tokenizer,
                (
                    "<AK_STRUCTURED>",
                    "<AK_ACTION_RESPOND>",
                    "<AK_ACTION_ASK_USER>",
                    "<AK_ACTION_EXTENSION_REQUEST>",
                    "<AK_ACTION_SAVE_MEMORY>",
                    "<AK_CONTENT>",
                    "</AK_CONTENT>",
                    "<AK_TASK_TYPE>",
                    "<AK_INTENT>",
                    "<AK_FIELD>",
                    "<AK_FIELD_NAME>",
                    "<AK_FIELD_VALUE>",
                    "<AK_FIELDS>",
                    "<AK_FRESHNESS>",
                    "<AK_END>",
                    *[f"<AK_COPY_USER_SOURCE_{index}>" for index in range(1, 25)],
                ),
            ),
            "negative_decoder_loss_weight": float(args.negative_decoder_loss_weight),
            "negative_divergent_decoder_loss_weight": float(args.negative_divergent_decoder_loss_weight),
            "negative_first_token_margin_weight": float(args.negative_first_token_margin_weight),
            "negative_first_token_margin": float(args.negative_first_token_margin),
            "retrieval_contrastive_weight": float(args.retrieval_contrastive_weight),
            "retrieval_temperature": float(args.retrieval_temperature),
            "retrieval_ternary_aware_weight": float(args.retrieval_ternary_aware_weight),
            "retrieval_ternary_threshold_ratio": float(args.retrieval_ternary_threshold_ratio),
            "retrieval_ternary_group_size": int(args.retrieval_ternary_group_size),
            "retrieval_ternary_residual_dims": int(args.retrieval_ternary_residual_dims),
            "retrieval_ternary_teacher_distill_weight": float(args.retrieval_ternary_teacher_distill_weight),
            "retrieval_ternary_teacher_temperature": float(args.retrieval_ternary_teacher_temperature),
            "retrieval_ternary_reconstruction_weight": float(args.retrieval_ternary_reconstruction_weight),
            "retrieval_hard_negative_weight": float(args.retrieval_hard_negative_weight),
            "retrieval_hard_negative_ternary": bool(args.retrieval_hard_negative_ternary),
            "teacher_model": teacher_model,
            "teacher_distill_weight": float(args.teacher_distill_weight),
            "teacher_distill_temperature": float(args.teacher_distill_temperature),
            "policy_head_loss_weight": float(args.policy_head_loss_weight),
            "intent_head_loss_weight": float(args.intent_head_loss_weight),
            "intent_contrastive_weight": float(args.intent_contrastive_weight),
            "intent_contrastive_temperature": float(args.intent_contrastive_temperature),
            "encoder_rep_distill_weight": float(args.encoder_rep_distill_weight),
            "future_bow_aux_weight": _scheduled_aux_weight(
                float(args.future_bow_aux_weight),
                step=step,
                total_steps=int(args.max_steps),
                schedule=str(args.future_bow_weight_schedule),
            ),
            "future_bow_buckets": int(args.future_bow_buckets),
            "future_sketch_aux_weight": _scheduled_aux_weight(
                float(args.future_sketch_aux_weight),
                step=step,
                total_steps=int(args.max_steps),
                schedule=str(args.future_bow_weight_schedule),
            ),
            "future_sketch_buckets": int(args.future_sketch_buckets),
            "future_sketch_min_token_weight": float(args.future_sketch_min_token_weight),
            "future_sketch_topk": int(args.future_sketch_topk),
            "future_sketch_windows": _parse_positive_ints(str(args.future_sketch_windows)),
            "state_sketch_aux_weight": _scheduled_aux_weight(
                float(args.state_sketch_aux_weight),
                step=step,
                total_steps=int(args.max_steps),
                schedule=str(args.future_bow_weight_schedule),
            ),
            "state_sketch_buckets": int(args.state_sketch_buckets),
            "state_sketch_topk": int(args.state_sketch_topk),
            "state_sketch_min_token_weight": float(args.state_sketch_min_token_weight),
            "target_sketch_aux_weight": _scheduled_aux_weight(
                float(args.target_sketch_aux_weight),
                step=step,
                total_steps=int(args.max_steps),
                schedule=str(args.future_bow_weight_schedule),
            ),
            "target_sketch_buckets": int(args.target_sketch_buckets),
            "target_sketch_topk": int(args.target_sketch_topk),
            "target_sketch_min_token_weight": float(args.target_sketch_min_token_weight),
            "decoder_future_sketch_aux_weight": _scheduled_aux_weight(
                float(args.decoder_future_sketch_aux_weight),
                step=step,
                total_steps=int(args.max_steps),
                schedule=str(args.future_bow_weight_schedule),
            ),
            "decoder_future_sketch_buckets": int(args.decoder_future_sketch_buckets),
            "decoder_future_sketch_min_token_weight": float(args.decoder_future_sketch_min_token_weight),
            "decoder_future_sketch_topk": int(args.decoder_future_sketch_topk),
            "decoder_future_sketch_windows": _parse_positive_ints(str(args.decoder_future_sketch_windows)),
            "decoder_drift_alignment_weight": _scheduled_aux_weight(
                float(args.decoder_drift_alignment_weight),
                step=step,
                total_steps=int(args.max_steps),
                schedule=str(args.future_bow_weight_schedule),
            ),
            "decoder_drift_alignment_min_token_weight": float(args.decoder_drift_alignment_min_token_weight),
            "decoder_drift_alignment_temperature": float(args.decoder_drift_alignment_temperature),
            "decoder_source_copy_anchor_weight": float(args.decoder_source_copy_anchor_weight),
            "decoder_source_copy_min_token_weight": float(args.decoder_source_copy_min_token_weight),
            "aux_grad_budget": float(args.aux_grad_budget),
        }
        pcgrad_stats: dict[str, float | int] = {"pcgrad_snapshots": 0, "pcgrad_conflicts": 0, "pcgrad_min_cosine": 0.0}
        use_pcgrad_decoder_aux = (
            bool(getattr(args, "pcgrad_decoder_aux", 0))
            and float(loss_kwargs["decoder_future_sketch_aux_weight"]) > 0.0
        )
        if use_pcgrad_decoder_aux:
            trainable_parameters = _trainable_parameters(model)
            primary_kwargs = dict(loss_kwargs)
            primary_kwargs["decoder_future_sketch_aux_weight"] = 0.0
            aux_kwargs = dict(loss_kwargs)
            for key in (
                "decoder_loss_weight",
                "decoder_eos_loss_weight",
                "decoder_json_structure_loss_weight",
                "decoder_control_token_suppression_weight",
                "negative_decoder_loss_weight",
                "negative_divergent_decoder_loss_weight",
                "negative_first_token_margin_weight",
                "retrieval_contrastive_weight",
                "retrieval_ternary_aware_weight",
                "retrieval_ternary_teacher_distill_weight",
                "retrieval_ternary_reconstruction_weight",
                "retrieval_hard_negative_weight",
                "teacher_distill_weight",
                "encoder_rep_distill_weight",
                "policy_head_loss_weight",
                "intent_head_loss_weight",
                "intent_contrastive_weight",
                "future_bow_aux_weight",
                "future_sketch_aux_weight",
                "state_sketch_aux_weight",
                "target_sketch_aux_weight",
                "decoder_source_copy_anchor_weight",
            ):
                aux_kwargs[key] = 0.0
            aux_kwargs["aux_grad_budget"] = 0.0
            optimizer.zero_grad(set_to_none=True)
            primary_loss = _weighted_loss(model, batch, **primary_kwargs)
            if anchor_parameters and float(args.anchor_weight_loss_weight) > 0.0:
                primary_loss = primary_loss + float(args.anchor_weight_loss_weight) * _anchor_weight_loss(
                    model,
                    anchor_parameters,
                )
            primary_loss.backward()
            primary_grads = _clone_trainable_grads(trainable_parameters)
            optimizer.zero_grad(set_to_none=True)
            aux_loss = _weighted_loss(model, batch, **aux_kwargs)
            aux_loss.backward()
            aux_grads = _clone_trainable_grads(trainable_parameters)
            pcgrad_stats = _apply_pcgrad_snapshots(trainable_parameters, [primary_grads, aux_grads])
            loss = primary_loss.detach() + aux_loss.detach()
        else:
            loss = _weighted_loss(model, batch, **loss_kwargs)
            if anchor_parameters and float(args.anchor_weight_loss_weight) > 0.0:
                loss = loss + float(args.anchor_weight_loss_weight) * _anchor_weight_loss(model, anchor_parameters)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad_norm))
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)
        last_step = step
        if step % max(1, int(args.log_every)) == 0:
            print(json.dumps({"step": step, "loss": loss_value, **pcgrad_stats}, sort_keys=True))
        if eval_dataset is not None and int(args.eval_every) > 0 and step % int(args.eval_every) == 0:
            eval_result = _evaluate(
                model,
                eval_dataset,
                batch_size=int(args.eval_batch_size) if int(args.eval_batch_size) > 0 else int(args.batch_size),
                device=device,
                max_batches=int(args.max_eval_batches),
                decoder_loss_weight=float(args.decoder_loss_weight),
                decoder_eos_loss_weight=float(args.decoder_eos_loss_weight),
                decoder_json_structure_loss_weight=float(args.decoder_json_structure_loss_weight),
                decoder_token_weight_alpha=float(args.decoder_token_weight_alpha),
                decoder_multi_token_ce_weight=float(args.decoder_multi_token_ce_weight),
                decoder_multi_token_horizons=_parse_positive_ints(str(args.decoder_multi_token_horizons)),
                decoder_multi_token_min_token_weight=float(args.decoder_multi_token_min_token_weight),
                decoder_seq_eos_token_weight=float(args.decoder_seq_eos_token_weight),
                decoder_stop_token_loss_weight=float(args.decoder_stop_token_loss_weight),
                decoder_eos_token_id=int(getattr(tokenizer, "eos_token_id", 2) or 2),
                decoder_stop_token_ids=_single_token_ids(tokenizer, ("</AK_CONTENT>", "<AK_END>")),
                decoder_json_structure_token_ids=_single_token_ids(tokenizer, ("{", "}", "[", "]", "\"", ":", ",")),
                decoder_control_token_suppression_weight=float(args.decoder_control_token_suppression_weight),
                decoder_control_token_ids=_single_token_ids(
                    tokenizer,
                    (
                        "<AK_STRUCTURED>",
                        "<AK_ACTION_RESPOND>",
                        "<AK_ACTION_ASK_USER>",
                        "<AK_ACTION_EXTENSION_REQUEST>",
                        "<AK_ACTION_SAVE_MEMORY>",
                        "<AK_CONTENT>",
                        "</AK_CONTENT>",
                        "<AK_TASK_TYPE>",
                        "<AK_INTENT>",
                        "<AK_FIELD>",
                        "<AK_FIELD_NAME>",
                        "<AK_FIELD_VALUE>",
                        "<AK_FIELDS>",
                        "<AK_FRESHNESS>",
                        "<AK_END>",
                        *[f"<AK_COPY_USER_SOURCE_{index}>" for index in range(1, 25)],
                    ),
                ),
                negative_decoder_loss_weight=float(args.negative_decoder_loss_weight),
                negative_divergent_decoder_loss_weight=float(args.negative_divergent_decoder_loss_weight),
                negative_first_token_margin_weight=float(args.negative_first_token_margin_weight),
                negative_first_token_margin=float(args.negative_first_token_margin),
                retrieval_contrastive_weight=float(args.retrieval_contrastive_weight),
                retrieval_temperature=float(args.retrieval_temperature),
                retrieval_ternary_aware_weight=float(args.retrieval_ternary_aware_weight),
                retrieval_ternary_threshold_ratio=float(args.retrieval_ternary_threshold_ratio),
                retrieval_ternary_group_size=int(args.retrieval_ternary_group_size),
                retrieval_ternary_residual_dims=int(args.retrieval_ternary_residual_dims),
                retrieval_ternary_teacher_distill_weight=float(args.retrieval_ternary_teacher_distill_weight),
                retrieval_ternary_teacher_temperature=float(args.retrieval_ternary_teacher_temperature),
                retrieval_ternary_reconstruction_weight=float(args.retrieval_ternary_reconstruction_weight),
                retrieval_hard_negative_weight=float(args.retrieval_hard_negative_weight),
                retrieval_hard_negative_ternary=bool(args.retrieval_hard_negative_ternary),
                policy_head_loss_weight=float(args.policy_head_loss_weight),
                intent_head_loss_weight=float(args.intent_head_loss_weight),
                intent_contrastive_weight=float(args.intent_contrastive_weight),
                intent_contrastive_temperature=float(args.intent_contrastive_temperature),
                encoder_rep_distill_weight=float(args.encoder_rep_distill_weight),
                future_bow_aux_weight=float(args.future_bow_aux_weight),
                future_bow_buckets=int(args.future_bow_buckets),
                future_sketch_aux_weight=float(args.future_sketch_aux_weight),
                future_sketch_buckets=int(args.future_sketch_buckets),
                future_sketch_min_token_weight=float(args.future_sketch_min_token_weight),
                future_sketch_topk=int(args.future_sketch_topk),
                future_sketch_windows=_parse_positive_ints(str(args.future_sketch_windows)),
                state_sketch_aux_weight=float(args.state_sketch_aux_weight),
                state_sketch_buckets=int(args.state_sketch_buckets),
                state_sketch_topk=int(args.state_sketch_topk),
                state_sketch_min_token_weight=float(args.state_sketch_min_token_weight),
                target_sketch_aux_weight=float(args.target_sketch_aux_weight),
                target_sketch_buckets=int(args.target_sketch_buckets),
                target_sketch_topk=int(args.target_sketch_topk),
                target_sketch_min_token_weight=float(args.target_sketch_min_token_weight),
                decoder_future_sketch_aux_weight=float(args.decoder_future_sketch_aux_weight),
                decoder_future_sketch_buckets=int(args.decoder_future_sketch_buckets),
                decoder_future_sketch_min_token_weight=float(args.decoder_future_sketch_min_token_weight),
                decoder_future_sketch_topk=int(args.decoder_future_sketch_topk),
                decoder_future_sketch_windows=_parse_positive_ints(str(args.decoder_future_sketch_windows)),
                decoder_drift_alignment_weight=float(args.decoder_drift_alignment_weight),
                decoder_drift_alignment_min_token_weight=float(args.decoder_drift_alignment_min_token_weight),
                decoder_drift_alignment_temperature=float(args.decoder_drift_alignment_temperature),
                decoder_source_copy_anchor_weight=float(args.decoder_source_copy_anchor_weight),
                decoder_source_copy_min_token_weight=float(args.decoder_source_copy_min_token_weight),
                aux_grad_budget=float(args.aux_grad_budget),
                teacher_model=teacher_model,
            )
            eval_result = {"step": step, **eval_result}
            eval_history.append(eval_result)
            print(json.dumps({"event": "eval", **eval_result}, sort_keys=True))
        if int(args.checkpoint_every) > 0 and step % int(args.checkpoint_every) == 0:
            checkpoint_path = _save_training_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                step=step,
                losses=losses,
                eval_history=eval_history,
                include_optimizer=bool(args.checkpoint_include_optimizer),
            )
            print(json.dumps({"event": "checkpoint", "path": str(checkpoint_path), "step": step}, sort_keys=True))

    if bool(args.save_final_checkpoint) and last_step > start_step:
        checkpoint_path = _save_training_checkpoint(
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            step=last_step,
            losses=losses,
            eval_history=eval_history,
            include_optimizer=bool(args.checkpoint_include_optimizer),
        )
        print(json.dumps({"event": "checkpoint", "final": True, "path": str(checkpoint_path), "step": last_step}, sort_keys=True))

    if hasattr(model, "future_bow_aux_head"):
        delattr(model, "future_bow_aux_head")
    if hasattr(model, "future_sketch_aux_head"):
        delattr(model, "future_sketch_aux_head")
    if hasattr(model, "state_sketch_aux_head"):
        delattr(model, "state_sketch_aux_head")
    if hasattr(model, "target_sketch_aux_head"):
        delattr(model, "target_sketch_aux_head")
    if hasattr(model, "decoder_future_sketch_aux_head"):
        delattr(model, "decoder_future_sketch_aux_head")
    tokenizer.save_pretrained(str(tokenizer_dir))
    model = model.eval()
    qat_dense_converted = 0
    qat_quantized_converted = 0
    if bool(args.bitnet_qat):
        dense_model = copy.deepcopy(model)
        qat_dense_converted = _convert_trainable_bitnet_to_dense(dense_model)
        save_pretrained(dense_model.eval().cpu(), config, str(model_dir))
        del dense_model
    else:
        save_pretrained(model.cpu(), config, str(model_dir))
    browser_bitnet_manifest_path = None
    if bool(args.export_browser_bitnet):
        if bool(args.bitnet_qat):
            qat_quantized_converted = _convert_trainable_bitnet_to_quantized(model)
            export_quantize = None
        else:
            export_quantize = "bitnet"
            model = model.cpu()
        browser_bitnet_manifest_path = export_model(
            model.cpu(),
            ExportConfig(
                target="browser-bitnet",
                outdir=str(output_dir / "browser_bitnet"),
                quantize=export_quantize,
                quant_spin=False,
                quant_weight_opt="none",
                quant_activation_quant=None,
                max_seq_len=max(int(args.max_encoder_tokens), int(args.max_decoder_tokens)),
            ),
            model_cfg=config,
        )
        _attach_tokenizer_to_browser_bitnet(
            browser_bitnet_manifest_path=Path(browser_bitnet_manifest_path),
            tokenizer_dir=tokenizer_dir,
            tokenizer_kind=tokenizer_kind,
            tokenizer=tokenizer,
        )
    training_summary = {
        **dry_summary,
        "dry_run": False,
        "completed_steps": int(last_step),
        "start_step": int(start_step),
        "resumed_from": "" if resume_path is None else str(resume_path),
        "last_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "eval_history": eval_history,
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "browser_bitnet_exported": browser_bitnet_manifest_path is not None,
        "bitnet_qat_summary": qat_summary,
        "bitnet_qat_dense_modules_saved": int(qat_dense_converted),
        "bitnet_qat_quantized_modules_exported": int(qat_quantized_converted),
        "initialized_from": "" if init_checkpoint_path is None else str(init_checkpoint_path),
        "freeze_encoder": bool(args.freeze_encoder),
        "freeze_token_embeddings": bool(args.freeze_token_embeddings),
        "freeze_lm_head": bool(getattr(args, "freeze_lm_head", 0)),
        "freeze_decoder": bool(args.freeze_decoder),
        "freeze_dense_linear_weights": bool(getattr(args, "freeze_dense_linear_weights", 0)),
        "frozen_encoder_parameters": int(frozen_encoder_parameters),
        "frozen_token_embedding_parameters": int(frozen_token_embedding_parameters),
        "frozen_lm_head_parameters": int(frozen_lm_head_parameters),
        "frozen_decoder_parameters": int(frozen_decoder_parameters),
        "frozen_dense_linear_weight_parameters": int(frozen_dense_linear_weight_parameters),
        "freeze_encoder_layers_through": int(args.freeze_encoder_layers_through),
        "frozen_encoder_layer_parameters": int(frozen_encoder_layer_parameters),
        "freeze_bitnet_linear_weights": bool(args.freeze_bitnet_linear_weights),
        "frozen_bitnet_linear_weight_parameters": int(frozen_bitnet_linear_weight_parameters),
        "trainable_name_filter": trainable_name_filter_summary,
        "anchor_weight_loss_weight": float(args.anchor_weight_loss_weight),
        "anchored_parameter_count": int(sum(tensor.numel() for tensor in anchor_parameters.values())),
        "trainable_parameter_count_before_export": int(trainable_parameter_count_before_export),
        "trainable_parameter_count": int(_trainable_parameter_count(model)),
    }
    return _save_manifest(
        output_dir=output_dir,
        model_dir=model_dir,
        tokenizer_dir=tokenizer_dir,
        dataset_manifest=dataset_manifest,
        config=config,
        parameter_count=parameter_count,
        tokenizer_kind=tokenizer_kind,
        tokenizer_name=tokenizer_name,
        training_summary=training_summary,
        browser_bitnet_manifest_path=browser_bitnet_manifest_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", default="artifacts/agentkernel_lite_encdec/run")
    parser.add_argument("--preset", default="agentkernel-lite-100m", choices=("agentkernel-lite-100m", "100m", "tiny"))
    parser.add_argument("--tokenizer-kind", default="byte", choices=("agentkernel-bpe", "hf", "byte"))
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--tokenizer-source-dir", default="")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=32768)
    parser.add_argument("--tokenizer-max-texts", type=int, default=200000)
    parser.add_argument("--agentkernel-special-tokens", type=int, choices=(0, 1), default=1)
    parser.add_argument("--byte-tokenizer", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-encoder-tokens", type=int, default=1024)
    parser.add_argument("--max-decoder-tokens", type=int, default=512)
    parser.add_argument("--max-retrieval-query-tokens", type=int, default=96)
    parser.add_argument("--max-retrieval-doc-tokens", type=int, default=256)
    parser.add_argument("--max-retrieval-negatives", type=int, default=0)
    parser.add_argument("--retrieval-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--retrieval-temperature", type=float, default=0.05)
    parser.add_argument("--retrieval-head-dim", type=int, default=0)
    parser.add_argument("--retrieval-ternary-aware-weight", type=float, default=0.0)
    parser.add_argument("--retrieval-ternary-threshold-ratio", type=float, default=0.20)
    parser.add_argument("--retrieval-ternary-group-size", type=int, default=16)
    parser.add_argument("--retrieval-ternary-residual-dims", type=int, default=64)
    parser.add_argument("--retrieval-ternary-teacher-distill-weight", type=float, default=0.0)
    parser.add_argument("--retrieval-ternary-teacher-temperature", type=float, default=0.05)
    parser.add_argument("--retrieval-ternary-reconstruction-weight", type=float, default=0.0)
    parser.add_argument("--retrieval-hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--retrieval-hard-negative-ternary", type=int, choices=(0, 1), default=1)
    parser.add_argument("--agent-policy-heads", type=int, choices=(0, 1), default=0)
    parser.add_argument("--agent-controller-dim", type=int, default=0)
    parser.add_argument("--agent-controller-apply-policy", type=int, choices=(0, 1), default=1)
    parser.add_argument("--agent-controller-apply-retrieval", type=int, choices=(0, 1), default=0)
    parser.add_argument("--policy-head-loss-weight", type=float, default=0.0)
    parser.add_argument("--agent-intent-labels", type=int, default=0)
    parser.add_argument("--intent-head-loss-weight", type=float, default=0.0)
    parser.add_argument("--intent-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--intent-contrastive-temperature", type=float, default=0.10)
    parser.add_argument("--encoder-rep-distill-weight", type=float, default=0.0)
    parser.add_argument("--future-bow-aux-weight", type=float, default=0.0)
    parser.add_argument("--future-bow-buckets", type=int, default=512)
    parser.add_argument("--future-bow-weight-schedule", choices=("constant", "warmup_cosine_decay"), default="constant")
    parser.add_argument("--future-sketch-aux-weight", type=float, default=0.0)
    parser.add_argument("--future-sketch-buckets", type=int, default=256)
    parser.add_argument("--future-sketch-min-token-weight", type=float, default=1.2)
    parser.add_argument("--future-sketch-topk", type=int, default=8)
    parser.add_argument("--future-sketch-windows", default="")
    parser.add_argument("--state-sketch-aux-weight", type=float, default=0.0)
    parser.add_argument("--state-sketch-buckets", type=int, default=256)
    parser.add_argument("--state-sketch-topk", type=int, default=12)
    parser.add_argument("--state-sketch-min-token-weight", type=float, default=1.2)
    parser.add_argument("--target-sketch-aux-weight", type=float, default=0.0)
    parser.add_argument("--target-sketch-buckets", type=int, default=256)
    parser.add_argument("--target-sketch-topk", type=int, default=12)
    parser.add_argument("--target-sketch-min-token-weight", type=float, default=1.2)
    parser.add_argument("--decoder-future-sketch-aux-weight", type=float, default=0.0)
    parser.add_argument("--decoder-future-sketch-buckets", type=int, default=256)
    parser.add_argument("--decoder-future-sketch-min-token-weight", type=float, default=1.2)
    parser.add_argument("--decoder-future-sketch-topk", type=int, default=8)
    parser.add_argument("--decoder-future-sketch-windows", default="8,32")
    parser.add_argument("--decoder-drift-alignment-weight", type=float, default=0.0)
    parser.add_argument("--decoder-drift-alignment-min-token-weight", type=float, default=1.2)
    parser.add_argument("--decoder-drift-alignment-temperature", type=float, default=0.0)
    parser.add_argument("--decoder-source-copy-anchor-weight", type=float, default=0.0)
    parser.add_argument("--decoder-source-copy-min-token-weight", type=float, default=1.2)
    parser.add_argument("--aux-grad-budget", type=float, default=0.0)
    parser.add_argument("--pcgrad-decoder-aux", type=int, choices=(0, 1), default=0)
    parser.add_argument("--decoder-loss-weight", type=float, default=1.0)
    parser.add_argument("--decoder-eos-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoder-json-structure-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoder-control-token-suppression-weight", type=float, default=0.0)
    parser.add_argument("--decoder-token-weight-alpha", type=float, default=0.0)
    parser.add_argument("--decoder-multi-token-ce-weight", type=float, default=0.0)
    parser.add_argument("--decoder-multi-token-horizons", default="2,4")
    parser.add_argument("--decoder-multi-token-min-token-weight", type=float, default=0.0)
    parser.add_argument("--decoder-stop-token-loss-weight", type=float, default=1.0)
    parser.add_argument("--decoder-seq-eos-token-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight-loss-weight", type=float, default=0.0)
    parser.add_argument("--negative-decoder-loss-weight", type=float, default=0.0)
    parser.add_argument("--negative-divergent-decoder-loss-weight", type=float, default=0.0)
    parser.add_argument("--negative-first-token-margin-weight", type=float, default=0.0)
    parser.add_argument("--negative-first-token-margin", type=float, default=1.0)
    parser.add_argument("--teacher-distill-weight", type=float, default=0.0)
    parser.add_argument("--teacher-distill-temperature", type=float, default=1.0)
    parser.add_argument("--parquet-shuffle-buffer-size", type=int, default=8192)
    parser.add_argument("--train-shuffle", type=int, choices=(0, 1), default=1)
    parser.add_argument("--parquet-require-retrieval-pair", type=int, choices=(0, 1), default=0)
    parser.add_argument("--parquet-action-include", default="")
    parser.add_argument("--parquet-task-type-include", default="")
    parser.add_argument("--freeze-encoder", type=int, choices=(0, 1), default=0)
    parser.add_argument("--freeze-token-embeddings", type=int, choices=(0, 1), default=0)
    parser.add_argument("--freeze-lm-head", type=int, choices=(0, 1), default=0)
    parser.add_argument("--freeze-decoder", type=int, choices=(0, 1), default=0)
    parser.add_argument("--freeze-encoder-layers-through", type=int, default=-1)
    parser.add_argument("--encoder-position-embeddings", type=int, choices=(0, 1), default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--mlp-dropout", type=float, default=0.0)
    parser.add_argument("--resid-dropout", type=float, default=0.0)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--checkpoint-include-optimizer", type=int, choices=(0, 1), default=1)
    parser.add_argument("--checkpoint-vocab-mismatch", choices=("strict", "expand"), default="strict")
    parser.add_argument("--init-from-checkpoint", default="")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--resume-latest", type=int, choices=(0, 1), default=0)
    parser.add_argument("--save-final-checkpoint", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", type=int, choices=(0, 1), default=1)
    parser.add_argument("--export-browser-bitnet", type=int, choices=(0, 1), default=1)
    parser.add_argument("--bitnet-qat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--bitnet-qat-include", default="")
    parser.add_argument("--bitnet-qat-exclude", default="")
    parser.add_argument("--bitnet-training-forward", default="")
    parser.add_argument("--bitnet-strict-ternary-forward", type=int, choices=(0, 1), default=0)
    parser.add_argument("--bitnet-activation-quant", default="")
    parser.add_argument("--bitnet-activation-quant-bits", type=int, default=0)
    parser.add_argument("--freeze-bitnet-linear-weights", type=int, choices=(0, 1), default=0)
    parser.add_argument("--freeze-dense-linear-weights", type=int, choices=(0, 1), default=0)
    parser.add_argument("--trainable-name-include", default="")
    parser.add_argument("--trainable-name-exclude", default="")
    args = parser.parse_args()
    manifest = train(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
