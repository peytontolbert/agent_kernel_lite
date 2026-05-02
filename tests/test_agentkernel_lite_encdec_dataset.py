from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paper(title: str, abstract: str, paper_id: str) -> dict[str, object]:
    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "text": abstract,
        "categories": "cs.AI",
        "year": 2026,
        "source_file": "unit",
        "row_index": 0,
    }


def test_paper_query_uses_surface_tokens_without_filtering_common_words() -> None:
    builder = _load_script("build_agentkernel_lite_encdec_dataset")

    assert builder._term_tokens("The paper with model data", limit=10) == [
        "the",
        "paper",
        "with",
        "model",
        "data",
    ]


def test_research_multitask_examples_use_agentkernel_tokens() -> None:
    builder = _load_script("build_agentkernel_lite_encdec_dataset")
    rows = [
        _paper(
            "Neural Retrieval for Scientific Assistants",
            "This paper studies neural retrieval for scientific assistant systems, including candidate ranking and grounded answer synthesis from long research documents.",
            "paper-a",
        ),
        _paper(
            "Database Index Maintenance",
            "This paper studies database index maintenance policies for write-heavy transactional systems and evaluates compact scheduling rules.",
            "paper-b",
        ),
        _paper(
            "Compiler Optimization Traces",
            "This paper studies compiler optimization traces and their relationship to program transformations in practical toolchains.",
            "paper-c",
        ),
    ]

    examples = builder._research_multitask_examples_from_rows(
        rows,
        max_examples=3,
        negative_count=2,
        objective="chat",
        seed=11,
    )

    task_types = {row["task_type"] for row in examples}
    assert {
        "query_rewrite",
        "rerank_candidates",
        "evidence_sufficiency",
        "answer_synthesis",
    } <= task_types
    joined_inputs = "\n".join(str(row["encoder_text"]) for row in examples)
    assert "<AK_GATHER_CONTEXT>" in joined_inputs
    assert "<AK_RERANK>" in joined_inputs
    assert "<AK_EVIDENCE>" in joined_inputs
    assert "<AK_ANSWER>" in joined_inputs


def test_build_dataset_accepts_paper_text_path(tmp_path: Path) -> None:
    builder = _load_script("build_agentkernel_lite_encdec_dataset")
    paper_path = tmp_path / "papers.jsonl"
    rows = [
        _paper(
            "Grounded Research Chat",
            "This paper studies grounded research chat using retrieval, candidate reranking, evidence checks, and answer synthesis over scientific text. It evaluates how assistant responses improve when the context compiler supplies source-specific evidence before generation.",
            "paper-1",
        ),
        _paper(
            "Distributed Storage Recovery",
            "This paper studies distributed storage recovery algorithms for replicated systems after partial node failures and delayed writes. It compares recovery policies using synthetic traces, repair latency measurements, and consistency checks across storage nodes.",
            "paper-2",
        ),
    ]
    paper_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    manifest = builder.build_dataset(
        repo_root=REPO_ROOT,
        output_dir=tmp_path / "dataset",
        trajectory_root=tmp_path / "missing_trajectories",
        docs_root=tmp_path / "missing_docs",
        sft_root=tmp_path / "missing_sft",
        paper_chunk_root=Path(""),
        paper_text_dataset="",
        paper_text_split="train",
        paper_text_path=paper_path,
        max_trajectory_files=0,
        max_doc_files=0,
        max_sft_files=0,
        max_sft_rows_per_file=0,
        max_research_examples=0,
        max_research_files=0,
        max_paper_text_examples=2,
        paper_text_negative_count=1,
        paper_text_streaming=False,
        eval_fraction=0.25,
        max_examples=0,
        objective="chat",
        code_trace_mode="explain",
        seed=3,
    )

    assert manifest["source_counts"]["research_retrieval_multitask"] >= 8
    assert manifest["target_action_counts"]["gather_context"] > 0
    assert "<AK_GATHER_CONTEXT>" in manifest["agentkernel_special_tokens"]
    assert Path(manifest["train_dataset_path"]).exists()
    assert Path(manifest["eval_dataset_path"]).exists()


def test_agentkernel_bpe_can_register_custom_special_tokens(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    pytest.importorskip("torch.nn")
    trainer = _load_script("train_agentkernel_lite_encdec")
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    row = {
        "encoder_text": "<AK_CHAT> <AK_GATHER_CONTEXT> <AK_USER> Find papers about neural retrieval.",
        "decoder_text": json.dumps({"action": "gather_context", "content": "neural retrieval"}),
    }
    train_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    eval_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    tokenizer = trainer._train_agentkernel_bpe(
        {
            "train_dataset_path": str(train_path),
            "eval_dataset_path": str(eval_path),
        },
        vocab_size=128,
        max_texts=100,
        use_agentkernel_special_tokens=True,
    )

    special_id = tokenizer.tokenizer.token_to_id("<AK_GATHER_CONTEXT>")
    assert special_id is not None
    encoded = tokenizer(
        "<AK_GATHER_CONTEXT> neural retrieval",
        max_length=16,
        padding="max_length",
        truncation=True,
    )
    assert special_id in encoded["input_ids"]
