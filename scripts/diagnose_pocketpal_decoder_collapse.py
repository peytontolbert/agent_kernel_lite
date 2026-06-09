#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")


def _tokens(text: str) -> list[str]:
    return [item.lower() for item in TOKEN_RE.findall(str(text or ""))]


def _content_ngrams(text: str, n: int = 4) -> Counter[str]:
    toks = _tokens(text)
    return Counter(" ".join(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1)))


def _overlap(left: str, right: str) -> float:
    a = set(_tokens(left))
    b = set(_tokens(right))
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def diagnose(path: Path) -> dict[str, Any]:
    payload = _load(path)
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"expected gate results list in {path}")
    failed = [row for row in results if not bool(row.get("passed"))]
    all_content = "\n".join(str(row.get("content") or "") for row in results)
    failed_content = "\n".join(str(row.get("content") or "") for row in failed)
    failure_reasons: Counter[str] = Counter()
    gate_failures: dict[str, list[str]] = {}
    by_gate: dict[str, dict[str, Any]] = {}
    for row in results:
        gate_id = str(row.get("id") or "")
        failures = [str(item) for item in row.get("failures", []) if str(item)]
        for item in failures:
            failure_reasons[item] += 1
        gate_failures[gate_id] = failures
        content = str(row.get("content") or "")
        output = str(row.get("output") or "")
        by_gate[gate_id] = {
            "passed": bool(row.get("passed")),
            "content": content,
            "failures": failures,
            "content_token_count": len(_tokens(content)),
            "output_token_count": len(_tokens(output)),
            "json_token_leak": bool(re.search(r"\b(action|proposal_metadata|task_type|active_agent)\b", content)),
        }
    ngrams = _content_ngrams(failed_content, 3)
    all_ngrams = _content_ngrams(all_content, 3)
    memorized = [
        {"ngram": key, "count": value}
        for key, value in ngrams.most_common(25)
        if value >= 2 or all_ngrams[key] >= 2
    ]
    cross_gate_content: defaultdict[str, list[str]] = defaultdict(list)
    for row in results:
        content = " ".join(_tokens(str(row.get("content") or ""))[:8])
        if content:
            cross_gate_content[content].append(str(row.get("id") or ""))
    repeated_answers = [
        {"prefix": prefix, "gates": gates}
        for prefix, gates in sorted(cross_gate_content.items(), key=lambda item: (-len(item[1]), item[0]))
        if len(gates) >= 2
    ]
    return {
        "input_path": str(path),
        "bundle_dir": payload.get("bundle_dir"),
        "passed": payload.get("passed"),
        "required_passed": payload.get("required_passed"),
        "required_total": payload.get("required_total"),
        "failure_count": len(failed),
        "failure_reasons": dict(failure_reasons.most_common()),
        "memorized_failed_ngrams": memorized,
        "repeated_answer_prefixes": repeated_answers[:20],
        "by_gate": by_gate,
    }


def compare(paths: list[Path]) -> dict[str, Any]:
    reports = [diagnose(path) for path in paths]
    return {
        "reports": reports,
        "summary": [
            {
                "path": report["input_path"],
                "bundle_dir": report.get("bundle_dir"),
                "required_passed": report.get("required_passed"),
                "required_total": report.get("required_total"),
                "failure_count": report.get("failure_count"),
                "top_failure_reasons": dict(list(report.get("failure_reasons", {}).items())[:8]),
                "top_repeated_prefixes": report.get("repeated_answer_prefixes", [])[:5],
            }
            for report in reports
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate_json", nargs="+")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    result = compare([Path(item).expanduser().resolve() for item in args.gate_json])
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
