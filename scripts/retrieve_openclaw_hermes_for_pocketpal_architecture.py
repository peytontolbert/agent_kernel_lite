#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_SKILL_DATASET = "/data/repo_skills_miner/artifacts/hf_openclaw_hermes_skills/data/train.parquet"

ARCHITECTURE_QUERIES = [
    {
        "area": "hybrid_controller",
        "query": "agent harness controller separates routing policy intent confidence verification retrieval from content generation decoder",
        "pocketpal_signal": "PocketPal has encoder policy heads and an intent head, but direct generation is still treated as a broad controller surface.",
    },
    {
        "area": "active_agent_contract",
        "query": "active agent instruction contract primary task boundary action policy tool policy retrieval policy prepared turn executor",
        "pocketpal_signal": "PocketPal prompts encode active agent instruction, retrieval policy, tool policy, and action policy directly into the encoder text.",
    },
    {
        "area": "skill_retrieval",
        "query": "harness skill retrieval use_when patch relevance verification hints tool registry skills selected before action",
        "pocketpal_signal": "v182 added OpenClaw/Hermes harness skill retrieval examples, but retrieval currently trains gather_context more than downstream action selection.",
    },
    {
        "area": "memory_state",
        "query": "long lived agent memory persistent state local memory slots session state self improvement learn from experience",
        "pocketpal_signal": "PocketPal has browser-local memory, slots, agents, data sources, and session export/restore.",
    },
    {
        "area": "tool_approval_security",
        "query": "tool approval permission boundary extension request web search requires user approval do not execute before approval",
        "pocketpal_signal": "PocketPal extension_request decisions carry extension_id capability query max_sources and requires_user_approval.",
    },
    {
        "area": "verification_eval",
        "query": "agent harness verification gates regression tests benchmark adapter promotion criteria trace scoring failure replay",
        "pocketpal_signal": "PocketPal has many narrow gates and eval JSONs, but promotion state is fragmented across tmp files.",
    },
    {
        "area": "fallback_repair",
        "query": "agent harness fallback repair malformed output structured decision repair retry prompt quality issue fallback deterministic rule",
        "pocketpal_signal": "PocketPal repairs malformed JSON and falls back when active-agent decisions fail quality checks.",
    },
    {
        "area": "browser_runtime",
        "query": "browser wasm runtime webgpu bitnet local model cached model worker parity export manifest runtime fallback",
        "pocketpal_signal": "PocketPal exports browser BitNet WebGPU/WASM bundles and loads them through a worker.",
    },
    {
        "area": "self_improvement",
        "query": "self improving agent creates skills from experience updates skills versioning autonomous learning trace store",
        "pocketpal_signal": "PocketPal has failure replay datasets, but no explicit closed-loop skill credit assignment for its controller decisions.",
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_stringify(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{key} {_stringify(item)}" for key, item in value.items())
    return str(value)


def _skill_text(row: dict[str, Any]) -> str:
    fields = [
        "source_repo",
        "source_path",
        "primitive_type",
        "primitive_subtype",
        "skill_kind",
        "qualname",
        "annotation_summary",
        "annotation_primitive_labels",
        "annotation_use_when",
        "annotation_patch_relevance",
        "annotation_risks",
        "annotation_verification_hints",
        "source_excerpt",
    ]
    return "\n".join(_stringify(row.get(field)) for field in fields)


def _short(text: Any, limit: int = 420) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


def _load_skill_rows(dataset_path: Path, max_rows: int) -> list[dict[str, Any]]:
    table = pq.read_table(dataset_path)
    rows = table.to_pylist()
    if max_rows > 0:
        rows = rows[:max_rows]
    return rows


def _retrieve(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    corpus = [_skill_text(row) for row in rows]
    queries = [item["query"] for item in ARCHITECTURE_QUERIES]
    vectorizer = TfidfVectorizer(max_features=120_000, ngram_range=(1, 2), min_df=2, stop_words="english")
    matrix = vectorizer.fit_transform(corpus + queries)
    skill_matrix = matrix[: len(corpus)]
    query_matrix = matrix[len(corpus) :]
    scores = cosine_similarity(query_matrix, skill_matrix)
    out: list[dict[str, Any]] = []
    for query_index, query in enumerate(ARCHITECTURE_QUERIES):
        ranked = scores[query_index].argsort()[::-1][:top_k]
        for rank, row_index in enumerate(ranked, start=1):
            row = rows[int(row_index)]
            out.append(
                {
                    "area": query["area"],
                    "rank": rank,
                    "score": float(scores[query_index, int(row_index)]),
                    "query": query["query"],
                    "pocketpal_signal": query["pocketpal_signal"],
                    "skill_id": row.get("id", ""),
                    "source_repo": row.get("source_repo", ""),
                    "source_path": row.get("source_path", ""),
                    "primitive_type": row.get("primitive_type", ""),
                    "primitive_subtype": row.get("primitive_subtype", ""),
                    "skill_kind": row.get("skill_kind", ""),
                    "qualname": row.get("qualname", ""),
                    "annotation_summary": row.get("annotation_summary", ""),
                    "annotation_use_when": _stringify(row.get("annotation_use_when")),
                    "annotation_patch_relevance": _stringify(row.get("annotation_patch_relevance")),
                    "annotation_risks": _stringify(row.get("annotation_risks")),
                    "annotation_verification_hints": _stringify(row.get("annotation_verification_hints")),
                    "source_excerpt": _short(row.get("source_excerpt"), 900),
                }
            )
    return out


def _latest_training_manifest(repo_root: Path) -> dict[str, Any]:
    manifests = sorted(
        (repo_root / "artifacts").glob("pocketpal_controller_100m_v*/agentkernel_lite_encdec_manifest.json"),
        key=lambda path: path.stat().st_mtime,
    )
    return _load_json(manifests[-1]) if manifests else {}


def _manifest_digest(manifest: dict[str, Any]) -> dict[str, Any]:
    config = manifest.get("model_config", {}) or manifest.get("model", {})
    config = config if isinstance(config, dict) else {}
    summary = manifest.get("training_summary", {})
    summary = summary if isinstance(summary, dict) else {}
    lite = manifest.get("agentkernel_lite", {})
    lite = lite if isinstance(lite, dict) else {}
    return {
        "manifest_path": manifest.get("manifest_path", ""),
        "model_dir": manifest.get("model_dir", "") or lite.get("source_model_dir", ""),
        "parameter_count": manifest.get("parameter_count") or lite.get("parameter_count"),
        "model_family": manifest.get("model_family", "") or lite.get("model_family", ""),
        "source_bundle_manifest_path": lite.get("source_bundle_manifest_path", ""),
        "retrieval_head_dim": config.get("retrieval_head_dim"),
        "agent_policy_heads": config.get("agent_policy_heads"),
        "agent_intent_labels": config.get("agent_intent_labels"),
        "browser_bitnet_exported": summary.get("browser_bitnet_exported"),
        "dataset_objective": summary.get("dataset_objective"),
        "completed_steps": summary.get("completed_steps"),
        "replaces_surfaces": manifest.get("replaces_surfaces", []),
    }


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    latest_digest: dict[str, Any],
    web_digest: dict[str, Any],
    dataset_path: Path,
) -> None:
    by_area: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_area.setdefault(str(row["area"]), []).append(row)

    lines = [
        "# PocketPal Architecture Retrieval Review",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Skill dataset: `{dataset_path}`",
        "",
        "## Architecture Snapshot",
        "",
        f"- Latest trained bundle: `{latest_digest.get('manifest_path', '')}`",
        f"- Latest trained heads: retrieval_head_dim=`{latest_digest.get('retrieval_head_dim')}`, agent_policy_heads=`{latest_digest.get('agent_policy_heads')}`, agent_intent_labels=`{latest_digest.get('agent_intent_labels')}`.",
        f"- Latest objective: `{latest_digest.get('dataset_objective')}`, steps=`{latest_digest.get('completed_steps')}`.",
        f"- Packaged web bundle: `{web_digest.get('manifest_path', '')}`",
        f"- Packaged web heads: retrieval_head_dim=`{web_digest.get('retrieval_head_dim')}`, agent_policy_heads=`{web_digest.get('agent_policy_heads')}`, agent_intent_labels=`{web_digest.get('agent_intent_labels')}`.",
        "",
        "## Retrieval-Backed Findings",
        "",
        "The OpenClaw/Hermes skills point toward treating PocketPal as a small local controller with retrieval, policy, memory, and verification surfaces around it. They do not support making the tiny decoder the full autonomous agent.",
        "",
        "1. Keep the tiny model in the action/control plane. Use encoder heads for intent, retrieval need, confidence, OOD, action validity, and verification need. Let the decoder produce constrained decisions or short text only when the route is low entropy.",
        "2. Make active-agent policy explicit outside the decoder. OpenClaw-style prepared turn boundaries and tool/plugin contracts match PocketPal's active-agent preamble, but the runtime should enforce action policy after model output.",
        "3. Retrieval should condition behavior, not just train an `<AK_GATHER_CONTEXT>` token. The harness skill records include use-when, risks, patch relevance, and verification hints; PocketPal should pass retrieved operators into the decision pipeline before direct generation.",
        "4. Memory and skill learning should be credit-assigned. Hermes-style self-improvement maps to saving successful PocketPal traces as new skill examples with outcome fields, not just adding more synthetic curricula.",
        "5. Promotion needs a single verifier artifact. OpenClaw/Hermes patterns emphasize guardrails, tests, and verification; PocketPal currently has many eval JSONs but no single promotion ledger attached to a bundle.",
        "6. Check browser parity before trusting architecture claims. If the app loads a bundle without policy/retrieval heads, the UI cannot use the latest controller architecture even if training manifests contain those heads.",
        "",
    ]

    for area in [item["area"] for item in ARCHITECTURE_QUERIES]:
        matches = by_area.get(area, [])[:5]
        signal = next(item["pocketpal_signal"] for item in ARCHITECTURE_QUERIES if item["area"] == area)
        lines.extend([f"## {area}", "", f"PocketPal signal: {signal}", ""])
        for match in matches:
            lines.append(
                f"- `{match['source_repo']}:{match['source_path']}` score={match['score']:.3f} "
                f"summary={_short(match['annotation_summary'], 220)}"
            )
            use_when = _short(match.get("annotation_use_when"), 260)
            if use_when:
                lines.append(f"  Use when: {use_when}")
            verify = _short(match.get("annotation_verification_hints"), 260)
            if verify:
                lines.append(f"  Verify: {verify}")
        lines.append("")

    lines.extend(
        [
            "## Concrete Next Changes",
            "",
            "- Add a PocketPal promotion ledger that combines agent gates, direct prompt eval, retrieval top-1, intent confusion, browser parity, malformed count, and zero-pass tasks into one Parquet row per bundle.",
            "- Add a runtime action-policy verifier that normalizes every model decision, rejects disallowed actions, expands safe slot placeholders, and routes invalid/high-entropy tasks to deterministic fallback or teacher-backed generation.",
            "- Convert OpenClaw/Hermes skill retrieval examples into controller examples with action targets: retrieve_skill -> choose_operator -> choose_verifier -> generate constrained decision.",
            "- Store successful and failed PocketPal turns as trace skills with retrieved skills, model decision, fallback reason, final output, and pass/fail credit assignment.",
            "- Update the deployed web model bundle when a promoted bundle has policy/retrieval heads, and run browser parity before switching the app default.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--skill-dataset", default=DEFAULT_SKILL_DATASET)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-skill-rows", type=int, default=0)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    dataset_path = Path(args.skill_dataset).expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else repo_root / "artifacts" / "pocketpal_architecture_retrieval" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    skill_rows = _load_skill_rows(dataset_path, int(args.max_skill_rows))
    retrieved = _retrieve(skill_rows, int(args.top_k))
    table = pa.Table.from_pylist(retrieved)
    pq.write_table(table, output_dir / "openclaw_hermes_architecture_matches.parquet", compression="zstd")

    latest_digest = _manifest_digest(_latest_training_manifest(repo_root))
    web_manifest_path = repo_root / "web" / "models" / "pocketpal_controller_100m_bitnet_dev" / "manifest.json"
    web_manifest = _load_json(web_manifest_path)
    web_manifest["manifest_path"] = str(web_manifest_path)
    web_digest = _manifest_digest(web_manifest)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "skill_dataset": str(dataset_path),
        "skill_rows": len(skill_rows),
        "match_rows": len(retrieved),
        "latest_training": latest_digest,
        "packaged_web": web_digest,
        "output_dir": str(output_dir),
    }
    (output_dir / "pocketpal_architecture_retrieval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "pocketpal_architecture_retrieval_review.md"
    _write_report(report_path, retrieved, latest_digest, web_digest, dataset_path)
    docs_path = repo_root / "docs" / "pocketpal_architecture_retrieval_review.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
