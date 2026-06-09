#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
from typing import Any


def parse_label(label: str) -> dict[str, str]:
    parts = [part.strip() for part in label.split(" | ")]
    prompt = parts[0] if parts else label
    source = ""
    seed = ""
    for part in parts[1:]:
        if part.startswith("seed="):
            seed = part.split("=", 1)[1].strip()
        elif not source:
            source = part
    return {"prompt": prompt, "source": source, "seed": seed}


def connected_components(labels: list[str], matrix: list[list[float]], threshold: float) -> list[list[int]]:
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(len(labels)):
        if start in seen:
            continue
        queue: deque[int] = deque([start])
        seen.add(start)
        component: list[int] = []
        while queue:
            idx = queue.popleft()
            component.append(idx)
            for other, cosine in enumerate(matrix[idx]):
                if other not in seen and other != idx and float(cosine) >= threshold:
                    seen.add(other)
                    queue.append(other)
        components.append(component)
    return components


def unique_prompts(labels: list[str]) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for label in labels:
        prompt = parse_label(label)["prompt"]
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
    return prompts


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a sequence-level curriculum from FLUX student gradient cosines.")
    parser.add_argument("--input", required=True, help="JSON from analyze_flux_student_concept_gradients.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--anchor-label", default="", help="Exact label to use as the curriculum anchor. Defaults to lowest-loss label.")
    parser.add_argument("--safe-cosine", type=float, default=0.75)
    parser.add_argument("--bridge-cosine", type=float, default=0.25)
    parser.add_argument("--cluster-cosine", type=float, default=0.75)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    gradient = data.get("gradient_cosines", {})
    labels = list(gradient.get("prompts") or [])
    matrix = gradient.get("cosine_matrix") or []
    losses: dict[str, float] = {str(key): float(value) for key, value in (gradient.get("losses") or {}).items()}
    if not labels or not matrix:
        raise ValueError("input does not contain gradient_cosines.prompts and cosine_matrix")

    if args.anchor_label:
        if args.anchor_label not in labels:
            raise ValueError(f"anchor label not found: {args.anchor_label}")
        anchor_index = labels.index(args.anchor_label)
    else:
        anchor_index = min(range(len(labels)), key=lambda idx: losses.get(labels[idx], float("inf")))
    anchor = labels[anchor_index]

    rows: list[dict[str, Any]] = []
    safe: list[str] = []
    bridge: list[str] = []
    risky: list[str] = []
    for idx, label in enumerate(labels):
        cosine = float(matrix[anchor_index][idx])
        loss = losses.get(label, 0.0)
        risk = float(loss * max(1.0 - cosine, 0.0))
        if idx == anchor_index or cosine >= args.safe_cosine:
            tier = "ring0_safe"
            safe.append(label)
        elif cosine >= args.bridge_cosine:
            tier = "ring1_bridge"
            bridge.append(label)
        else:
            tier = "ring2_risky"
            risky.append(label)
        rows.append(
            {
                "label": label,
                **parse_label(label),
                "anchor_cosine": cosine,
                "loss": loss,
                "risk": risk,
                "tier": tier,
            }
        )
    rows.sort(key=lambda row: ({"ring0_safe": 0, "ring1_bridge": 1, "ring2_risky": 2}[row["tier"]], row["risk"]))

    components = connected_components(labels, matrix, float(args.cluster_cosine))
    clusters = []
    for cluster_index, component in enumerate(sorted(components, key=lambda value: (-len(value), value[0]))):
        cluster_labels = [labels[idx] for idx in component]
        clusters.append(
            {
                "cluster": cluster_index,
                "size": len(cluster_labels),
                "labels": cluster_labels,
                "prompts": unique_prompts(cluster_labels),
                "mean_loss": sum(losses.get(label, 0.0) for label in cluster_labels) / max(len(cluster_labels), 1),
                "mean_anchor_cosine": sum(float(matrix[anchor_index][idx]) for idx in component) / max(len(component), 1),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tiers = {
        "ring0_safe": safe,
        "ring1_bridge": bridge,
        "ring2_risky": risky,
        "ring0_plus_ring1": safe + bridge,
        "all_ordered": [str(row["label"]) for row in rows],
    }
    for tier, tier_labels in tiers.items():
        write_lines(output_dir / f"{tier}_sequence_keys.txt", tier_labels)
        write_lines(output_dir / f"{tier}_prompts.txt", unique_prompts(tier_labels))

    for cluster in clusters:
        stem = f"cluster_{int(cluster['cluster']):02d}"
        write_lines(output_dir / f"{stem}_sequence_keys.txt", list(cluster["labels"]))
        write_lines(output_dir / f"{stem}_prompts.txt", list(cluster["prompts"]))

    summary = {
        "input": str(args.input),
        "anchor": anchor,
        "safe_cosine": float(args.safe_cosine),
        "bridge_cosine": float(args.bridge_cosine),
        "cluster_cosine": float(args.cluster_cosine),
        "tier_counts": {tier: len(values) for tier, values in tiers.items() if tier != "all_ordered"},
        "rows": rows,
        "clusters": clusters,
        "recommendation": [
            "Train ring0_safe first with the anchor replay weight high.",
            "Add ring1_bridge only after ring0 eval preserves the anchor.",
            "Keep ring2_risky out of the run until a bridge cluster becomes coherent.",
            "Use --sequence-include-key-file rather than prompt-only filters for exact source/seed control.",
        ],
    }
    (output_dir / "curriculum_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
