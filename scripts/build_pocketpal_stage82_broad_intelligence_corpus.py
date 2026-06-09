#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

INTENT_LABELS = {
    "plan": 0,
    "action_items": 1,
    "rewrite": 2,
    "translation": 3,
    "web_search": 4,
    "casual": 5,
    "source_echo": 6,
    "saved_data": 7,
    "ask_user": 8,
    "summary": 9,
    "title": 10,
    "checklist": 11,
    "risks": 12,
    "json": 13,
    "ranking": 14,
    "extraction": 15,
    "subject": 16,
    "brainstorm": 17,
}


def _stable_id(*parts: object) -> str:
    return hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _compact(value: object, *, limit: int = 1200) -> str:
    text = " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split())
    return text[:limit].rstrip()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row
    except OSError:
        return


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _ak_decision(task_type: str, content: str, *, action: str = "respond") -> str:
    return (
        f"<AK_STRUCTURED> <AK_ACTION_{action.upper()}> <AK_TASK_TYPE> {task_type} "
        f"<AK_CONTENT> {_compact(content, limit=1000)} </AK_CONTENT> <AK_END>"
    )


def _json_decision(task_type: str, content: str, *, action: str = "respond") -> str:
    return json.dumps(
        {"action": action, "content": content, "proposal_metadata": {"task_type": task_type}},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _prompt(*, agent_name: str, instruction: str, intent: str, task_type: str, user: str, context: str = "") -> str:
    return (
        "<AK_CHAT> <AK_RESPOND> PocketPal broad intelligence curriculum example.\n"
        "<AK_AGENT_ACTIVE>\n"
        f"Agent name: {agent_name}\n"
        f"Agent instruction: {instruction}\n"
        "Retrieval policy: auto\n"
        "Tool policy: ask_before_extensions\n"
        "Action policy: respond_or_ask\n"
        "The active agent instruction is the primary task contract for this turn.\n"
        "</AK_AGENT_ACTIVE>\n"
        f"<AK_TASK_HINT> intent={intent} task={task_type} source_text_required=true\n"
        "<AK_CONTEXT> Saved user data: none\n"
        f"{('<AK_CONTEXT> ' + context + chr(10)) if context else ''}"
        f"<AK_USER> {user}\n"
        "Return AK structured tokens for the active agent decision."
    )


def _base_row(
    *,
    source_id: str,
    source_type: str,
    task_type: str,
    intent: str,
    encoder_text: str,
    content: str,
    weight: float,
    retrieval_query_text: str = "",
    retrieval_doc_text: str = "",
    state_text: str = "",
    split: str = "train",
) -> dict[str, Any] | None:
    content = _compact(content, limit=1000)
    if not content or intent not in INTENT_LABELS:
        return None
    decoder_text = _ak_decision(task_type, content)
    return {
        "action": "respond",
        "decoder_loss_weight": 1.0,
        "decoder_text": decoder_text,
        "encoder_text": encoder_text,
        "example_id": _stable_id("stage82", source_id, encoder_text, decoder_text),
        "expected_content": content,
        "intent_label": intent,
        "intent_label_id": INTENT_LABELS[intent],
        "json_decoder_text": _json_decision(task_type, content),
        "negative_decoder_text": None,
        "negative_loss_weight": None,
        "retrieval_doc_text": retrieval_doc_text,
        "retrieval_loss_weight": 1.0 if retrieval_query_text and retrieval_doc_text else 0.0,
        "retrieval_query_text": retrieval_query_text,
        "source_id": source_id,
        "source_type": source_type,
        "split": split,
        "state_text": state_text or content,
        "task_type": task_type,
        "weight": float(weight),
    }


def _skill_rows(path: Path, *, max_rows: int, repeat: int, seed: int) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Stage 82 skill ingestion requires pyarrow") from exc
    if not path.exists():
        return []
    table = pq.read_table(
        path,
        columns=[
            "id",
            "repo",
            "source_path",
            "primitive_type",
            "skill_kind",
            "qualname",
            "name",
            "summary",
            "use_when",
            "patch_relevance",
            "risks",
            "verification_hints",
            "score",
        ],
    )
    rows = table.to_pylist()
    rows = [row for row in rows if _compact(row.get("summary"), limit=80)]
    rows.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
    rng = random.Random(seed)
    head = rows[: max_rows * 3]
    rng.shuffle(head)
    rows = head[:max_rows]
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        repo = _compact(row.get("repo"), limit=80)
        source_path = _compact(row.get("source_path"), limit=180)
        qualname = _compact(row.get("qualname"), limit=220)
        summary = _compact(row.get("summary"), limit=500)
        use_when = _compact(row.get("use_when"), limit=420)
        risks = _compact(row.get("risks"), limit=320)
        verify = _compact(row.get("verification_hints"), limit=360)
        patch_relevance = _compact(row.get("patch_relevance"), limit=360)
        doc = (
            f"Repo: {repo}. Path: {source_path}. Symbol/card: {qualname}. "
            f"Summary: {summary}. Use when: {use_when}. Patch relevance: {patch_relevance}. "
            f"Risks: {risks}. Verification: {verify}."
        )
        examples = [
            (
                "summary",
                "active_agent_summary",
                "Skill Summary Agent",
                "Summarize the retrieved code skill card into one practical capability note.",
                f"Summarize the useful capability in {repo}:{source_path}.",
                f"{summary} Use it when {use_when}.",
                2.1,
            ),
            (
                "risks",
                "active_agent_risks",
                "Verification Agent",
                "Identify risk and verification hooks before using a retrieved skill.",
                f"What can go wrong when using {qualname}, and how should it be verified?",
                f"Risks: {risks or 'unknown behavior if used outside its intended context'}. Verification: {verify or 'run the relevant tests and inspect outputs for regressions'}.",
                2.0,
            ),
            (
                "plan",
                "active_agent_plan",
                "Local Planning Agent",
                "Plan how to apply a retrieved repo skill without executing tools directly.",
                f"Plan a safe local-agent step using {qualname}.",
                f"First retrieve the relevant card for {source_path}. Then apply it only when the request matches: {use_when}. Check patch relevance: {patch_relevance}. Finish by verifying: {verify}.",
                2.0,
            ),
        ]
        for rep in range(repeat):
            intent, task_type, agent, instruction, user, content, weight = examples[(index + rep) % len(examples)]
            made = _base_row(
                source_id=f"skill:{row.get('id')}:{rep}",
                source_type="stage82_openclaw_hermes_skill",
                task_type=task_type,
                intent=intent,
                encoder_text=_prompt(
                    agent_name=agent,
                    instruction=instruction,
                    intent=intent,
                    task_type=task_type,
                    user=user,
                    context=f"Retrieved skill card: {doc}",
                ),
                content=content,
                weight=weight,
                retrieval_query_text=user,
                retrieval_doc_text=doc,
                state_text=f"repo={repo} path={source_path} skill={qualname}",
            )
            if made:
                out.append(made)
    return out


def _concept_rows(path: Path, *, max_rows: int, repeat: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(_iter_jsonl(path)):
        if index >= max_rows:
            break
        repo = _compact(row.get("repo_id"), limit=100)
        name = _compact(row.get("name"), limit=140)
        kind = _compact(row.get("kind"), limit=80)
        uri = _compact(row.get("uri"), limit=220)
        summary = _compact(row.get("summary") or row.get("doc") or row.get("code"), limit=850)
        code = _compact(row.get("code"), limit=900)
        if not repo or not name or not summary:
            continue
        doc = f"Repo: {repo}. Concept: {name}. Kind: {kind}. URI: {uri}. Summary/code: {summary}."
        examples = [
            (
                "summary",
                "active_agent_summary",
                f"{name} is a {kind or 'repository concept'} in {repo}. {summary}",
                f"Summarize {name} from {repo} for a local private assistant.",
            ),
            (
                "extraction",
                "active_agent_extraction",
                f"repo={repo}; concept={name}; kind={kind or 'unknown'}; uri={uri}",
                f"Extract the repo, concept, kind, and URI from the retrieved concept card.",
            ),
        ]
        if code:
            examples.append(
                (
                    "plan",
                    "active_agent_plan",
                    f"Use the {name} context by first confirming the relevant file or symbol, then apply a minimal change, then verify behavior against the surrounding repository contract.",
                    f"Plan how an agent should use the retrieved concept {name} without overreaching.",
                )
            )
        for rep in range(repeat):
            intent, task_type, content, user = examples[(index + rep) % len(examples)]
            made = _base_row(
                source_id=f"concept:{repo}:{name}:{rep}",
                source_type="stage82_repo_concept",
                task_type=task_type,
                intent=intent,
                encoder_text=_prompt(
                    agent_name="Repository Reasoning Agent",
                    instruction="Use retrieved code/repo concept context to answer compactly and groundedly.",
                    intent=intent,
                    task_type=task_type,
                    user=user,
                    context=f"Retrieved concept: {doc}",
                ),
                content=content,
                weight=1.9,
                retrieval_query_text=user,
                retrieval_doc_text=doc,
                state_text=f"repo={repo} concept={name} kind={kind}",
            )
            if made:
                out.append(made)
    return out


def _commit_rows(path: Path, *, max_rows: int, repeat: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(_iter_jsonl(path)):
        if index >= max_rows:
            break
        repo = _compact(row.get("entity_id"), limit=100)
        text = _compact(row.get("text"), limit=360)
        graph = row.get("graph_context") if isinstance(row.get("graph_context"), list) else []
        files = ", ".join(_compact(item, limit=120) for item in graph[:4])
        if not repo or not text:
            continue
        doc = f"Repo: {repo}. Commit/change: {text}. Related files: {files}."
        content = f"Change intent: {text}. Inspect related files: {files or 'none listed'}. Verify that the behavior named by the commit still works and no adjacent path regressed."
        for rep in range(repeat):
            intent = "action_items" if rep % 2 == 0 else "checklist"
            task_type = "active_agent_action_items" if intent == "action_items" else "active_agent_checklist"
            user = f"Turn this repository change into a compact agent checklist: {text}"
            made = _base_row(
                source_id=f"commit:{row.get('id')}:{rep}",
                source_type="stage82_commit_episode",
                task_type=task_type,
                intent=intent,
                encoder_text=_prompt(
                    agent_name="Change Review Agent",
                    instruction="Convert retrieved repository change context into concrete verification steps.",
                    intent=intent,
                    task_type=task_type,
                    user=user,
                    context=f"Retrieved change episode: {doc}",
                ),
                content=content,
                weight=1.7,
                retrieval_query_text=user,
                retrieval_doc_text=doc,
                state_text=f"repo={repo} files={files}",
            )
            if made:
                out.append(made)
    return out


def _paper_rows(path: Path, *, max_rows: int, repeat: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(_iter_jsonl(path)):
        if index >= max_rows:
            break
        title = _compact(row.get("paper_title"), limit=180)
        abstract = _compact(row.get("paper_abstract") or row.get("paper_text"), limit=900)
        paper_id = _compact(row.get("paper_id"), limit=80)
        if not title or not abstract:
            continue
        doc = f"Paper: {title}. ID: {paper_id}. Abstract/context: {abstract}"
        examples = [
            (
                "summary",
                "active_agent_summary",
                f"{title}: {abstract}",
                f"Summarize the training or modeling idea from this paper context.",
            ),
            (
                "ranking",
                "active_agent_ranking",
                f"Most relevant idea: use the method only when its assumptions match the target task; then compare it against simpler baselines and verify held-out behavior.",
                f"Rank the strongest transferable lesson from this paper for a tiny local model.",
            ),
        ]
        for rep in range(repeat):
            intent, task_type, content, user = examples[(index + rep) % len(examples)]
            made = _base_row(
                source_id=f"paper:{paper_id}:{rep}",
                source_type="stage82_paper_repo_alignment",
                task_type=task_type,
                intent=intent,
                encoder_text=_prompt(
                    agent_name="Research Transfer Agent",
                    instruction="Extract a grounded research lesson from retrieved paper context.",
                    intent=intent,
                    task_type=task_type,
                    user=user,
                    context=f"Retrieved paper context: {doc}",
                ),
                content=content,
                weight=1.8,
                retrieval_query_text=user,
                retrieval_doc_text=doc,
                state_text=f"paper_id={paper_id} title={title}",
            )
            if made:
                out.append(made)
    return out


def _load_rows_from_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = _read_json(manifest_path)
    rows = _read_jsonl(Path(manifest["train_dataset_path"]))
    rows.extend(_read_jsonl(Path(manifest["eval_dataset_path"])))
    return rows


def _split_row(row: dict[str, Any], eval_fraction: float) -> str:
    if str(row.get("split") or "") == "eval":
        return "eval"
    key = str(row.get("example_id") or _stable_id(row.get("encoder_text", ""), row.get("decoder_text", "")))
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket < float(eval_fraction) else "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", default=str(REPO_ROOT / "tmp/pocketpal_stage78_scaled_agentic_corpus/agentkernel_lite_encdec_dataset_manifest.json"))
    parser.add_argument("--intent-anchor-manifest", default=str(REPO_ROOT / "tmp/pocketpal_stage80_intent_boundary_overfit_probe/agentkernel_lite_encdec_dataset_manifest.json"))
    parser.add_argument("--repo-library-root", default="/data/repository_library")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "tmp/pocketpal_stage82_broad_intelligence_corpus"))
    parser.add_argument("--max-skill-rows", type=int, default=3200)
    parser.add_argument("--max-concept-rows", type=int, default=1400)
    parser.add_argument("--max-commit-rows", type=int, default=3500)
    parser.add_argument("--max-paper-rows", type=int, default=700)
    parser.add_argument("--skill-repeat", type=int, default=2)
    parser.add_argument("--concept-repeat", type=int, default=2)
    parser.add_argument("--commit-repeat", type=int, default=1)
    parser.add_argument("--paper-repeat", type=int, default=2)
    parser.add_argument("--anchor-repeat-limit", type=int, default=7800)
    parser.add_argument("--eval-fraction", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=8201)
    args = parser.parse_args()

    random.seed(int(args.seed))
    repo_root = Path(args.repo_library_root).resolve()
    base_manifest_path = Path(args.base_manifest).resolve()
    base_manifest = _read_json(base_manifest_path)
    train_rows = _read_jsonl(Path(base_manifest["train_dataset_path"]))
    eval_rows = _read_jsonl(Path(base_manifest["eval_dataset_path"]))

    anchor_rows: list[dict[str, Any]] = []
    anchor_manifest_path = Path(args.intent_anchor_manifest).resolve()
    if anchor_manifest_path.exists() and int(args.anchor_repeat_limit) > 0:
        anchor_rows = _load_rows_from_manifest(anchor_manifest_path)[: int(args.anchor_repeat_limit)]
        for row in anchor_rows:
            row["source_type"] = str(row.get("source_type") or "stage82_intent_anchor_replay")
            row["weight"] = float(row.get("weight", 1.0) or 1.0) * 1.2

    additions: list[dict[str, Any]] = []
    additions.extend(
        _skill_rows(
            repo_root / "exports/repo_skills_openclaw_hermes_viewer/parquet/skills_all.parquet",
            max_rows=int(args.max_skill_rows),
            repeat=int(args.skill_repeat),
            seed=int(args.seed),
        )
    )
    additions.extend(
        _concept_rows(
            repo_root / "models/exports/repo_concepts.jsonl",
            max_rows=int(args.max_concept_rows),
            repeat=int(args.concept_repeat),
        )
    )
    additions.extend(
        _commit_rows(
            repo_root / "models/exports/commit_episodes.jsonl",
            max_rows=int(args.max_commit_rows),
            repeat=int(args.commit_repeat),
        )
    )
    additions.extend(
        _paper_rows(
            repo_root / "models/exports/paper_repo_span_align.jsonl",
            max_rows=int(args.max_paper_rows),
            repeat=int(args.paper_repeat),
        )
    )

    dedup: dict[str, dict[str, Any]] = {}
    for row in [*train_rows, *eval_rows, *anchor_rows, *additions]:
        copied = deepcopy(row)
        key = str(copied.get("example_id") or _stable_id(copied.get("encoder_text", ""), copied.get("decoder_text", "")))
        copied["example_id"] = key
        dedup[key] = copied

    combined_train: list[dict[str, Any]] = []
    combined_eval: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    task_type_counts: dict[str, int] = {}
    intent_counts: dict[str, int] = {}
    for row in sorted(dedup.values(), key=lambda item: str(item.get("example_id", ""))):
        source = str(row.get("source_type") or "unknown")
        task = str(row.get("task_type") or "unknown")
        intent = str(row.get("intent_label") or row.get("intent") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        task_type_counts[task] = task_type_counts.get(task, 0) + 1
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        split = _split_row(row, float(args.eval_fraction))
        row["split"] = split
        if split == "eval":
            combined_eval.append(row)
        else:
            combined_train.append(row)

    output_dir = Path(args.output_dir).resolve()
    train_path = output_dir / "agentkernel_lite_encdec_train.jsonl"
    eval_path = output_dir / "agentkernel_lite_encdec_eval.jsonl"
    manifest_path = output_dir / "agentkernel_lite_encdec_dataset_manifest.json"
    _write_jsonl(train_path, combined_train)
    _write_jsonl(eval_path, combined_eval)
    manifest = {
        "artifact_kind": "agentkernel_lite_encdec_stage82_broad_intelligence_corpus",
        "base_examples": len(train_rows) + len(eval_rows),
        "base_manifest": str(base_manifest_path),
        "dataset_format": "jsonl",
        "eval_dataset_path": str(eval_path),
        "eval_examples": len(combined_eval),
        "intent_anchor_examples": len(anchor_rows),
        "intent_anchor_manifest": str(anchor_manifest_path),
        "intent_counts": dict(sorted(intent_counts.items())),
        "intent_labels": INTENT_LABELS,
        "local_additions_before_dedup": len(additions),
        "manifest_path": str(manifest_path),
        "objective": "pocketpal_stage82_broad_intelligence_corpus",
        "source_counts": dict(sorted(source_counts.items())),
        "task_type_counts": dict(sorted(task_type_counts.items())),
        "total_examples": len(combined_train) + len(combined_eval),
        "train_dataset_path": str(train_path),
        "train_examples": len(combined_train),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "train_examples": len(combined_train),
                "eval_examples": len(combined_eval),
                "local_additions_before_dedup": len(additions),
                "intent_anchor_examples": len(anchor_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
