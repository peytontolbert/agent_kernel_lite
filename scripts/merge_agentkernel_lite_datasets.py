#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MERGE_COLUMNS = [
    "example_id",
    "split",
    "source_type",
    "source_id",
    "task_type",
    "encoder_text",
    "decoder_text",
    "negative_decoder_text",
    "negative_loss_weight",
    "action",
    "weight",
    "intent_label",
    "intent_label_id",
    "retrieval_query_text",
    "retrieval_doc_text",
    "retrieval_loss_weight",
    "query_confidence_target",
    "retrieval_coverage_target",
    "ood_query_target",
    "ood_evidence_target",
    "answer_confidence_target",
    "needs_verification_target",
    "paper_action_validity_target",
]

POLICY_TARGET_COLUMNS = (
    "query_confidence_target",
    "retrieval_coverage_target",
    "ood_query_target",
    "ood_evidence_target",
    "answer_confidence_target",
    "needs_verification_target",
    "paper_action_validity_target",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_jsonl(inputs: list[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as out_handle:
        for path in inputs:
            with path.open("r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    if not line.strip():
                        continue
                    out_handle.write(line)
                    count += 1
    return count


def _iter_dataset_rows(path: Path):
    if path.is_dir() and any(path.glob("*.parquet")):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("merging parquet datasets requires pyarrow") from exc
        for shard in sorted(path.glob("*.parquet")):
            parquet_file = pq.ParquetFile(shard)
            available = set(parquet_file.schema_arrow.names)
            read_columns = [column for column in MERGE_COLUMNS if column in available]
            for batch in parquet_file.iter_batches(batch_size=2048, columns=read_columns):
                yield from batch.to_pylist()
        return
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("merging parquet datasets requires pyarrow") from exc
        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        read_columns = [column for column in MERGE_COLUMNS if column in available]
        for batch in parquet_file.iter_batches(batch_size=2048, columns=read_columns):
            yield from batch.to_pylist()
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


class DatasetWriter:
    def __init__(self, output_dir: Path, *, output_format: str, parquet_shard_size: int) -> None:
        self.output_dir = output_dir
        self.output_format = str(output_format)
        self.parquet_shard_size = max(1, int(parquet_shard_size))
        self.counts = {"train": 0, "eval": 0}
        self.shard_counts = {"train": 0, "eval": 0}
        self.buffers: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
        self.handles: dict[str, Any] = {}
        if self.output_format == "jsonl":
            self.train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
            self.eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
            self.handles["train"] = self.train_path.open("w", encoding="utf-8")
            self.handles["eval"] = self.eval_path.open("w", encoding="utf-8")
        elif self.output_format == "parquet":
            self.train_path = output_dir / "train"
            self.eval_path = output_dir / "eval"
            self.train_path.mkdir(parents=True, exist_ok=True)
            self.eval_path.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"unknown output format: {self.output_format}")

    def _normalize(self, row: dict[str, Any], *, split: str) -> dict[str, Any]:
        normalized = {key: row.get(key) for key in MERGE_COLUMNS}
        normalized["split"] = str(normalized.get("split") or split)
        normalized["encoder_text"] = str(normalized.get("encoder_text") or "")
        normalized["decoder_text"] = str(normalized.get("decoder_text") or "")
        normalized["action"] = str(normalized.get("action") or row.get("target_action") or "")
        normalized["weight"] = float(normalized.get("weight") or 1.0)
        normalized["retrieval_loss_weight"] = float(normalized.get("retrieval_loss_weight") or 0.0)
        for key in POLICY_TARGET_COLUMNS:
            value = normalized.get(key)
            if value is None or value == "":
                normalized[key] = None
            else:
                try:
                    normalized[key] = float(value)
                except (TypeError, ValueError):
                    normalized[key] = None
        for key in (
            "example_id",
            "source_type",
            "source_id",
            "task_type",
            "retrieval_query_text",
            "retrieval_doc_text",
        ):
            normalized[key] = "" if normalized.get(key) is None else str(normalized.get(key))
        return normalized

    def _write_parquet_shard(self, split: str) -> None:
        rows = self.buffers[split]
        if not rows:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("parquet output requires pyarrow") from exc
        shard_index = self.shard_counts[split]
        path = (self.train_path if split == "train" else self.eval_path) / f"part-{shard_index:05d}.parquet"
        schema = pa.schema(
            [
                pa.field("example_id", pa.string()),
                pa.field("split", pa.string()),
                pa.field("source_type", pa.string()),
                pa.field("source_id", pa.string()),
                pa.field("task_type", pa.string()),
                pa.field("encoder_text", pa.string()),
                pa.field("decoder_text", pa.string()),
                pa.field("action", pa.string()),
                pa.field("weight", pa.float32()),
                pa.field("retrieval_query_text", pa.string()),
                pa.field("retrieval_doc_text", pa.string()),
                pa.field("retrieval_loss_weight", pa.float32()),
                pa.field("query_confidence_target", pa.float32()),
                pa.field("retrieval_coverage_target", pa.float32()),
                pa.field("ood_query_target", pa.float32()),
                pa.field("ood_evidence_target", pa.float32()),
                pa.field("answer_confidence_target", pa.float32()),
                pa.field("needs_verification_target", pa.float32()),
                pa.field("paper_action_validity_target", pa.float32()),
            ]
        )
        table = pa.Table.from_pylist([self._normalize(row, split=split) for row in rows], schema=schema)
        pq.write_table(table, path, compression="zstd")
        self.buffers[split] = []
        self.shard_counts[split] += 1

    def write(self, split: str, row: dict[str, Any]) -> None:
        if split not in {"train", "eval"}:
            raise ValueError(f"unknown split: {split}")
        self.counts[split] += 1
        if self.output_format == "jsonl":
            self.handles[split].write(json.dumps(self._normalize(row, split=split), ensure_ascii=False, sort_keys=True) + "\n")
            return
        self.buffers[split].append(row)
        if len(self.buffers[split]) >= self.parquet_shard_size:
            self._write_parquet_shard(split)

    def close(self) -> None:
        if self.output_format == "jsonl":
            for handle in self.handles.values():
                handle.close()
            return
        self._write_parquet_shard("train")
        self._write_parquet_shard("eval")


def _sum_dicts(manifests: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for manifest in manifests:
        values = manifest.get(key, {})
        if not isinstance(values, dict):
            continue
        for item_key, value in values.items():
            try:
                amount = int(value)
            except (TypeError, ValueError):
                continue
            out[str(item_key)] = out.get(str(item_key), 0) + amount
    return dict(sorted(out.items()))


def _sum_ints(manifests: list[dict[str, Any]], key: str) -> int:
    total = 0
    for manifest in manifests:
        try:
            total += int(manifest.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
    return total


def merge_datasets(args: argparse.Namespace) -> dict[str, Any]:
    manifest_paths = [Path(item).expanduser().resolve() for item in args.dataset_manifest]
    manifests = [_load_json(path) for path in manifest_paths]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = DatasetWriter(
        output_dir,
        output_format=str(args.output_format),
        parquet_shard_size=int(args.parquet_shard_size),
    )
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    try:
        for manifest in manifests:
            train_source = Path(str(manifest["train_dataset_path"]))
            eval_source = Path(str(manifest["eval_dataset_path"]))
            for row in _iter_dataset_rows(train_source):
                writer.write("train", row)
            for row in _iter_dataset_rows(eval_source):
                writer.write("eval", row)
    finally:
        writer.close()
    first = manifests[0] if manifests else {}
    manifest = {
        **first,
        "artifact_kind": "agentkernel_lite_encdec_distill_dataset",
        "objective": str(args.objective or first.get("objective", "chat")),
        "dataset_format": writer.output_format,
        "manifest_path": str(manifest_path),
        "train_dataset_path": str(writer.train_path),
        "eval_dataset_path": str(writer.eval_path),
        "merged_source_manifest_paths": [str(path) for path in manifest_paths],
        "total_examples": writer.counts["train"] + writer.counts["eval"],
        "train_examples": writer.counts["train"],
        "eval_examples": writer.counts["eval"],
        "train_shards": int(writer.shard_counts["train"]),
        "eval_shards": int(writer.shard_counts["eval"]),
        "source_counts": _sum_dicts(manifests, "source_counts"),
        "task_type_counts": _sum_dicts(manifests, "task_type_counts"),
        "target_action_counts": _sum_dicts(manifests, "target_action_counts"),
        "source_action_counts": _sum_dicts(manifests, "source_action_counts"),
        "extension_counts": _sum_dicts(manifests, "extension_counts"),
    }
    retrieval_pair_count = _sum_ints(manifests, "retrieval_pair_count")
    if retrieval_pair_count:
        manifest["retrieval_pair_count"] = retrieval_pair_count
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objective", default="chat")
    parser.add_argument("--output-format", choices=("jsonl", "parquet"), default="jsonl")
    parser.add_argument("--parquet-shard-size", type=int, default=50000)
    args = parser.parse_args()
    print(json.dumps(merge_datasets(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
