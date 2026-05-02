#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
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
        "<AK_RET_CODE>",
        "<AK_OOD>",
        "<AK_ARTIFACT_REPAIR>",
        "<AK_SOURCE_INSPECT>",
        "<AK_PATCH_BUILD>",
        "<AK_SAFE_STOP>",
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
    target = tokenizer(
        str(row["decoder_text"]),
        max_length=int(max_decoder_tokens),
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
    )
    labels = list(target["input_ids"])
    if labels and int(labels[0]) == int(decoder_start_token_id):
        labels = labels[1:]
    labels = labels[: int(max_decoder_tokens)]
    while len(labels) < int(max_decoder_tokens):
        labels.append(int(pad_token_id))
    decoder_input_ids = [int(decoder_start_token_id), *labels[:-1]]
    label_ids = [token if token != int(pad_token_id) else -100 for token in labels]
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

    return {
        "enc_input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
        "enc_attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        "dec_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
        "labels": torch.tensor(label_ids, dtype=torch.long),
        "loss_weight": torch.tensor(float(row.get("weight", 1.0) or 1.0), dtype=torch.float32),
        "retrieval_query_ids": torch.tensor(retrieval_query["input_ids"], dtype=torch.long),
        "retrieval_query_attention_mask": torch.tensor(retrieval_query["attention_mask"], dtype=torch.long),
        "retrieval_doc_ids": torch.tensor(retrieval_doc["input_ids"], dtype=torch.long),
        "retrieval_doc_attention_mask": torch.tensor(retrieval_doc["attention_mask"], dtype=torch.long),
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
    }


class EncDecParquetIterableDataset(IterableDataset):
    columns = [
        "encoder_text",
        "decoder_text",
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


def _weighted_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    decoder_loss_weight: float = 1.0,
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
) -> torch.Tensor:
    logits = None
    needs_decoder_logits = float(decoder_loss_weight) > 0.0 or (
        teacher_model is not None and float(teacher_distill_weight) > 0.0
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
        valid_tokens = (batch["labels"] != -100).to(dtype=token_losses.dtype)
        per_example = (token_losses * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp_min(1.0)
        weights = batch["loss_weight"].to(dtype=per_example.dtype).clamp_min(0.0)
        seq_loss = (per_example * weights).sum() / weights.sum().clamp_min(1e-6)
    else:
        seq_loss = next(model.parameters()).new_tensor(0.0)
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
    return (
        float(decoder_loss_weight) * seq_loss
        + float(teacher_distill_weight) * distill_loss
        + float(retrieval_contrastive_weight) * retrieval_loss
        + float(retrieval_ternary_aware_weight) * ternary_loss
        + float(retrieval_ternary_teacher_distill_weight) * ternary_teacher_loss
        + float(retrieval_ternary_reconstruction_weight) * ternary_reconstruction_loss
        + float(retrieval_hard_negative_weight) * hard_negative_loss
        + float(policy_head_loss_weight) * policy_loss
    )


def _evaluate(
    model: torch.nn.Module,
    dataset: EncDecJsonlDataset,
    *,
    batch_size: int,
    device: torch.device,
    max_batches: int,
    decoder_loss_weight: float = 1.0,
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


def _freeze_trainable_bitnet_linear_weights(model: torch.nn.Module) -> int:
    from compress.quantization import TrainableBitNetLinear

    frozen = 0
    for module in model.modules():
        if isinstance(module, TrainableBitNetLinear):
            if module.weight.requires_grad:
                module.weight.requires_grad_(False)
                frozen += int(module.weight.numel())
    return frozen


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
) -> tuple[int, list[float], list[dict[str, Any]]]:
    payload = torch.load(path, map_location=device)
    state = payload["model_state_dict"]
    allowed_new_keys = {"enc_pos_embed.weight"}
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
                patched[key] = expanded
                continue
            raise RuntimeError(
                f"checkpoint tensor shape mismatch for {key}: checkpoint={tuple(tensor.shape)} target={tuple(target.shape)}"
            )
        missing, unexpected = model.load_state_dict(patched, strict=False)
        unexpected = list(unexpected)
        missing = [
            key
            for key in missing
            if key not in {"enc_embed.weight", "dec_embed.weight", "lm_head.weight", *allowed_new_keys}
            and not key.startswith("retrieval_query_head.")
            and not key.startswith("retrieval_doc_head.")
            and not key.startswith("agent_policy_heads.")
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
        ]
        unexpected = list(unexpected)
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
    dataset_objective = str(dataset_manifest.get("objective", "")).strip()
    if "controller_trace" in dataset_objective:
        replaces_surfaces.extend(
            [
                "agentkernel_action_space_selection",
                "artifact_repair_action_policy",
                "source_inspection_policy",
            ]
        )
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
    model = EncoderDecoderLM(config, tie_embeddings=True, vocab_size=vocab_size)
    _materialize_lazy_modules(model)
    parameter_count = _parameter_count(model)
    frozen_encoder_parameters = 0
    frozen_token_embedding_parameters = 0
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
        "policy_head_loss_weight": float(args.policy_head_loss_weight),
        "decoder_loss_weight": float(args.decoder_loss_weight),
        "teacher_distill_weight": float(args.teacher_distill_weight),
        "teacher_distill_temperature": float(args.teacher_distill_temperature),
        "attn_dropout": float(args.attn_dropout),
        "mlp_dropout": float(args.mlp_dropout),
        "resid_dropout": float(args.resid_dropout),
        "freeze_encoder": bool(args.freeze_encoder),
        "freeze_token_embeddings": bool(args.freeze_token_embeddings),
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

    train_dataset = _load_encdec_dataset(
        Path(str(dataset_manifest["train_dataset_path"])),
        tokenizer,
        max_encoder_tokens=int(args.max_encoder_tokens),
        max_decoder_tokens=int(args.max_decoder_tokens),
        max_retrieval_query_tokens=int(args.max_retrieval_query_tokens),
        max_retrieval_doc_tokens=int(args.max_retrieval_doc_tokens),
        max_retrieval_negatives=int(args.max_retrieval_negatives),
        shuffle=True,
        shuffle_buffer_size=int(args.parquet_shuffle_buffer_size),
        require_retrieval_pair=bool(args.parquet_require_retrieval_pair),
        action_include=tuple(item.strip() for item in str(args.parquet_action_include).split(",") if item.strip()),
        task_type_include=tuple(item.strip() for item in str(args.parquet_task_type_include).split(",") if item.strip()),
        seed=int(args.seed),
    )
    if len(train_dataset) <= 0:
        raise SystemExit("training dataset is empty")
    eval_dataset = None
    eval_path_value = str(dataset_manifest.get("eval_dataset_path", "") or "")
    eval_path = Path(eval_path_value) if eval_path_value else None
    if eval_path is not None and eval_path.exists():
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
        shuffle=False if train_is_iterable else True,
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
    teacher_model = None
    if (
        float(getattr(args, "teacher_distill_weight", 0.0) or 0.0) > 0.0
        or float(getattr(args, "retrieval_ternary_teacher_distill_weight", 0.0) or 0.0) > 0.0
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
        loss = _weighted_loss(
            model,
            batch,
            decoder_loss_weight=float(args.decoder_loss_weight),
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
            teacher_model=teacher_model,
            teacher_distill_weight=float(args.teacher_distill_weight),
            teacher_distill_temperature=float(args.teacher_distill_temperature),
            policy_head_loss_weight=float(args.policy_head_loss_weight),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad_norm))
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)
        last_step = step
        if step % max(1, int(args.log_every)) == 0:
            print(json.dumps({"step": step, "loss": loss_value}, sort_keys=True))
        if eval_dataset is not None and int(args.eval_every) > 0 and step % int(args.eval_every) == 0:
            eval_result = _evaluate(
                model,
                eval_dataset,
                batch_size=int(args.eval_batch_size) if int(args.eval_batch_size) > 0 else int(args.batch_size),
                device=device,
                max_batches=int(args.max_eval_batches),
                decoder_loss_weight=float(args.decoder_loss_weight),
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
        "frozen_encoder_parameters": int(frozen_encoder_parameters),
        "frozen_token_embedding_parameters": int(frozen_token_embedding_parameters),
        "freeze_bitnet_linear_weights": bool(args.freeze_bitnet_linear_weights),
        "frozen_bitnet_linear_weight_parameters": int(frozen_bitnet_linear_weight_parameters),
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
    parser.add_argument("--policy-head-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoder-loss-weight", type=float, default=1.0)
    parser.add_argument("--teacher-distill-weight", type=float, default=0.0)
    parser.add_argument("--teacher-distill-temperature", type=float, default=1.0)
    parser.add_argument("--parquet-shuffle-buffer-size", type=int, default=8192)
    parser.add_argument("--parquet-require-retrieval-pair", type=int, choices=(0, 1), default=0)
    parser.add_argument("--parquet-action-include", default="")
    parser.add_argument("--parquet-task-type-include", default="")
    parser.add_argument("--freeze-encoder", type=int, choices=(0, 1), default=0)
    parser.add_argument("--freeze-token-embeddings", type=int, choices=(0, 1), default=0)
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
    args = parser.parse_args()
    manifest = train(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
