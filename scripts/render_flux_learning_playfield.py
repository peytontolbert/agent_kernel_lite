#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path


DEFAULT_KEYS = [
    "loss",
    "endpoint_loss",
    "lowfreq_endpoint_loss",
    "x0_loss",
    "lowfreq_x0_loss",
    "decoded_loss",
    "decoded_multiscale_loss",
    "decoded_highfreq_loss",
]


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name, path
    name, path = value.split("=", 1)
    return name.strip(), Path(path)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def copy_asset(path: Path, asset_dir: Path) -> str:
    asset_dir.mkdir(parents=True, exist_ok=True)
    target = asset_dir / path.name
    if target.exists():
        stem = path.stem
        suffix = path.suffix
        counter = 1
        while target.exists():
            target = asset_dir / f"{stem}_{counter}{suffix}"
            counter += 1
    shutil.copy2(path, target)
    return f"assets/{target.name}"


def plotly_series_div(runs: list[tuple[str, list[dict[str, object]]]], keys: list[str]) -> str:
    traces = []
    for run_name, rows in runs:
        for key in keys:
            xs = [int(row["step"]) for row in rows if key in row and "step" in row]
            ys = [float(row[key]) for row in rows if key in row and "step" in row]
            if xs:
                traces.append({"x": xs, "y": ys, "mode": "lines+markers", "name": f"{run_name}:{key}"})
    return f"""
<div id="loss_curves" class="chart"></div>
<script>
Plotly.newPlot('loss_curves', {json.dumps(traces)}, {{
  title: 'Training Scalars',
  xaxis: {{title: 'step'}},
  yaxis: {{title: 'value'}},
  template: 'plotly_white'
}});
</script>
"""


def pcgrad_div(runs: list[tuple[str, list[dict[str, object]]]]) -> str:
    traces = []
    for run_name, rows in runs:
        for key in ("pcgrad_conflicts", "pcgrad_min_cosine"):
            xs = [int(row["step"]) for row in rows if key in row and "step" in row]
            ys = [float(row[key]) for row in rows if key in row and "step" in row]
            if xs:
                traces.append({"x": xs, "y": ys, "mode": "lines+markers", "name": f"{run_name}:{key}"})
    if not traces:
        return ""
    return f"""
<div id="pcgrad_curves" class="chart"></div>
<script>
Plotly.newPlot('pcgrad_curves', {json.dumps(traces)}, {{
  title: 'PCGrad Conflict Surface',
  xaxis: {{title: 'step'}},
  yaxis: {{title: 'conflicts / cosine'}},
  template: 'plotly_white'
}});
</script>
"""


def gradient_heatmap_div(path: Path) -> str:
    if not path.exists():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    cosines = data.get("gradient_cosines") or {}
    prompts = cosines.get("prompts") or []
    matrix = cosines.get("cosine_matrix") or []
    losses = cosines.get("losses") or {}
    negative_pairs = cosines.get("negative_pairs") or []
    rows = data.get("rows") or []
    prompt_block_figs = []
    for loss_name in ("silhouette_lowfreq", "texture_highfreq", "endpoint_full", "flow_field"):
        z = []
        y = []
        block_count = 0
        for row in rows:
            blocks = row.get("top_blocks", {}).get(loss_name, [])
            if not blocks:
                continue
            block_count = max(block_count, max(int(item["block"]) for item in blocks) + 1)
        block_count = max(block_count, 8)
        for row in rows:
            values = [0.0] * block_count
            for item in row.get("top_blocks", {}).get(loss_name, []):
                block = int(item["block"])
                if block < block_count:
                    values[block] = float(item["grad_norm"])
            z.append(values)
            y.append(str(row.get("prompt", "")))
        if z:
            div_id = f"block_heatmap_{loss_name}"
            prompt_block_figs.append(
                f"""
<div id="{div_id}" class="chart"></div>
<script>
Plotly.newPlot('{div_id}', [{{
  z: {json.dumps(z)},
  x: {json.dumps([f"block {idx}" for idx in range(block_count)])},
  y: {json.dumps(y)},
  type: 'heatmap',
  colorscale: 'Viridis'
}}], {{
  title: 'Block Gradient Pressure: {loss_name}',
  template: 'plotly_white'
}});
</script>
"""
            )
    loss_rows = "".join(
        f"<tr><td>{html.escape(str(prompt))}</td><td>{float(value):.4f}</td></tr>"
        for prompt, value in sorted(losses.items())
    )
    pair_rows = "".join(
        f"<tr><td>{html.escape(str(row.get('a')))}</td><td>{html.escape(str(row.get('b')))}</td><td>{float(row.get('cosine')):.4f}</td></tr>"
        for row in negative_pairs
    )
    return f"""
<div id="gradient_heatmap" class="chart"></div>
<script>
Plotly.newPlot('gradient_heatmap', [{{
  z: {json.dumps(matrix)},
  x: {json.dumps(prompts)},
  y: {json.dumps(prompts)},
  type: 'heatmap',
  zmin: -1,
  zmax: 1,
  colorscale: 'RdBu'
}}], {{
  title: 'Adapter Gradient Cosine Matrix',
  template: 'plotly_white'
}});
</script>
<h2>Endpoint Losses In Cosine Probe</h2>
<table><tr><th>prompt</th><th>endpoint loss</th></tr>{loss_rows}</table>
<h2>Negative Gradient Pairs</h2>
<table><tr><th>A</th><th>B</th><th>cosine</th></tr>{pair_rows}</table>
{''.join(prompt_block_figs)}
"""


def gallery(title: str, paths: list[Path], asset_dir: Path) -> str:
    cards = []
    for path in paths:
        if not path.exists():
            continue
        rel = copy_asset(path, asset_dir)
        cards.append(
            f"<figure><img src='{html.escape(rel)}'><figcaption>{html.escape(path.parent.name + '/' + path.name)}</figcaption></figure>"
        )
    if not cards:
        return ""
    return f"<h2>{html.escape(title)}</h2><div class='gallery'>{''.join(cards)}</div>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an image-model learning playfield dashboard.")
    parser.add_argument("--run", action="append", default=[], help="name=/path/to/ledger.jsonl")
    parser.add_argument("--gradient-json", default="")
    parser.add_argument("--sample", action="append", default=[], help="path to contact_sheet.png or sample png")
    parser.add_argument("--reference", action="append", default=[], help="path to teacher/reference image")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    asset_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [(name, read_jsonl(path)) for name, path in (parse_run(value) for value in args.run)]
    keys = [item.strip() for item in args.keys.split(",") if item.strip()]
    html_parts = [
        "<html><head><meta charset='utf-8'><title>Flux Learning Playfield</title>",
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#222}.chart{width:100%;height:520px;margin-bottom:28px}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}.gallery img{width:100%;height:auto;border:1px solid #ccc}figure{margin:0}figcaption{font-size:12px;color:#555;margin-top:4px}table{border-collapse:collapse;margin-bottom:24px}td,th{border:1px solid #ddd;padding:6px 10px;text-align:left}</style>",
        "</head><body><h1>Flux Learning Playfield</h1>",
        plotly_series_div(runs, keys),
        pcgrad_div(runs),
        gradient_heatmap_div(Path(args.gradient_json)) if args.gradient_json else "",
        gallery("Teacher References", [Path(value) for value in args.reference], asset_dir),
        gallery("Student Samples", [Path(value) for value in args.sample], asset_dir),
        "</body></html>",
    ]
    output_path = output_dir / "index.html"
    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(json.dumps({"dashboard": str(output_path)}), flush=True)


if __name__ == "__main__":
    main()
