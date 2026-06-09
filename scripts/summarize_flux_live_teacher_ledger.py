#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean


METRICS = (
    "loss",
    "flow_loss",
    "endpoint_loss",
    "multistep_terminal_loss",
    "multistep_terminal_spatial_loss",
    "on_policy_teacher_loss",
    "decoded_image_loss",
    "decoded_lowfreq_loss",
    "decoded_edge_loss",
    "prompt_contrast_loss",
    "prompt_sensitivity_ratio",
)


def finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def summarize_rows(rows: list[dict[str, object]], *, tail: int) -> dict[str, object]:
    if not rows:
        return {"rows": 0}
    tail_rows = rows[-tail:] if tail > 0 else rows
    out: dict[str, object] = {
        "rows": len(rows),
        "first_step": rows[0].get("step"),
        "last_step": rows[-1].get("step"),
        "tail_rows": len(tail_rows),
    }
    for metric in METRICS:
        values = [finite_number(row.get(metric)) for row in tail_rows]
        values = [value for value in values if value is not None]
        if values:
            out[f"{metric}_tail_mean"] = mean(values)
            out[f"{metric}_tail_last"] = values[-1]
    horizons: dict[str, list[float]] = defaultdict(list)
    sources: dict[str, int] = defaultdict(int)
    prompts: set[str] = set()
    nonzero_prompt_contrast = 0
    decoded_rows = 0
    for row in tail_rows:
        horizon = row.get("multistep_consistency_horizon")
        terminal = finite_number(row.get("multistep_terminal_loss"))
        if horizon is not None and terminal is not None:
            horizons[str(horizon)].append(terminal)
        source = str(row.get("source_name") or "")
        if source:
            sources[source] += 1
        prompt = str(row.get("prompt") or "")
        if prompt:
            prompts.add(prompt)
        if (finite_number(row.get("prompt_contrast_loss")) or 0.0) > 0:
            nonzero_prompt_contrast += 1
        if (finite_number(row.get("decoded_image_loss")) or 0.0) > 0:
            decoded_rows += 1
    out["unique_prompts_tail"] = len(prompts)
    out["source_counts_tail"] = dict(sorted(sources.items()))
    out["decoded_active_fraction_tail"] = decoded_rows / max(len(tail_rows), 1)
    out["prompt_contrast_active_fraction_tail"] = nonzero_prompt_contrast / max(len(tail_rows), 1)
    out["terminal_loss_by_horizon_tail"] = {
        horizon: mean(values) for horizon, values in sorted(horizons.items()) if values
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--tail", type=int, default=100)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    recent_errors: deque[str] = deque(maxlen=5)
    with args.ledger.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                recent_errors.append(f"line {line_no}: {exc}")
                continue
            if isinstance(row, dict) and "step" in row:
                rows.append(row)

    summary = summarize_rows(rows, tail=args.tail)
    if recent_errors:
        summary["parse_errors"] = list(recent_errors)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
