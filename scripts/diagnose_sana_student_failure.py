#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F

try:
    from transformers import CLIPModel, CLIPProcessor
except Exception:  # pragma: no cover
    CLIPModel = None
    CLIPProcessor = None


def as_feature_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    for key in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        candidate = getattr(value, key, None)
        if isinstance(candidate, torch.Tensor):
            if key == "last_hidden_state":
                return candidate[:, 0]
            return candidate
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"could not extract feature tensor from {type(value).__name__}")


def to_tensor_rgb(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


def ensure_same_size(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.shape[-2:] == b.shape[-2:]:
        return a, b
    h = min(a.shape[-2], b.shape[-2])
    w = min(a.shape[-1], b.shape[-1])
    a2 = F.interpolate(a.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
    b2 = F.interpolate(b.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
    return a2, b2


def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[0], x[1], x[2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def sobel_edges(gray: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
    g = gray.unsqueeze(0).unsqueeze(0)
    ex = F.conv2d(g, kx.view(1, 1, 3, 3), padding=1)
    ey = F.conv2d(g, ky.view(1, 1, 3, 3), padding=1)
    mag = torch.sqrt(ex.square() + ey.square() + 1e-8)
    return mag.squeeze(0).squeeze(0)


def ssim_simple(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = ensure_same_size(a, b)
    c1 = 0.01**2
    c2 = 0.03**2
    ga = rgb_to_gray(a)
    gb = rgb_to_gray(b)

    def blur(x: torch.Tensor) -> torch.Tensor:
        k = torch.ones((1, 1, 7, 7), dtype=torch.float32) / 49.0
        return F.conv2d(x.unsqueeze(0).unsqueeze(0), k, padding=3).squeeze(0).squeeze(0)

    mu_a = blur(ga)
    mu_b = blur(gb)
    var_a = blur(ga * ga) - mu_a * mu_a
    var_b = blur(gb * gb) - mu_b * mu_b
    cov = blur(ga * gb) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2) + 1e-8)
    return float(ssim_map.mean().item())


def hist_js_divergence(a: torch.Tensor, b: torch.Tensor, bins: int = 64) -> float:
    a, b = ensure_same_size(a, b)
    pa = []
    pb = []
    for c in range(3):
        ha = torch.histc(a[c], bins=bins, min=0.0, max=1.0)
        hb = torch.histc(b[c], bins=bins, min=0.0, max=1.0)
        ha = ha / (ha.sum() + 1e-8)
        hb = hb / (hb.sum() + 1e-8)
        pa.append(ha)
        pb.append(hb)
    p = torch.cat(pa, dim=0)
    q = torch.cat(pb, dim=0)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * torch.log((p + 1e-8) / (m + 1e-8)))
    kl_qm = torch.sum(q * torch.log((q + 1e-8) / (m + 1e-8)))
    return float(0.5 * (kl_pm + kl_qm))


def edge_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = ensure_same_size(a, b)
    ea = sobel_edges(rgb_to_gray(a))
    eb = sobel_edges(rgb_to_gray(b))
    return float(torch.mean(torch.abs(ea - eb)).item())


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = ensure_same_size(a, b)
    mse = float(torch.mean((a - b) ** 2).item())
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def l1(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = ensure_same_size(a, b)
    return float(torch.mean(torch.abs(a - b)).item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())


def read_metrics_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    by_idx: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return by_idx
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        row = json.loads(text)
        idx = int(row.get("index", -1))
        if idx >= 0:
            by_idx[idx] = row
    return by_idx


def parse_prompt_file(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    if not path.exists():
        return out
    index = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("{"):
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                row = {}
            prompt = str(row.get("prompt") or row.get("text") or row.get("caption") or "").strip()
        else:
            prompt = text
        if prompt:
            out[index] = prompt
            index += 1
    return out


def collect_triplets(compare_dir: Path) -> list[dict[str, Any]]:
    student_paths = sorted(compare_dir.glob("student_*.png"))
    if not student_paths:
        raise FileNotFoundError(f"No student_*.png files found in {compare_dir}")
    triplets: list[dict[str, Any]] = []
    for student_path in student_paths:
        stem = student_path.stem
        suffix = stem.split("_")[-1]
        try:
            index = int(suffix)
        except ValueError:
            continue
        teacher_path = compare_dir / f"teacher_{index:03d}.png"
        ref_path = compare_dir / f"ref_{index:03d}.png"
        if not teacher_path.exists():
            continue
        triplets.append(
            {
                "index": index,
                "student_path": student_path,
                "teacher_path": teacher_path,
                "ref_path": ref_path if ref_path.exists() else None,
            }
        )
    if not triplets:
        raise ValueError("No valid (student, teacher) image pairs found.")
    return sorted(triplets, key=lambda r: int(r["index"]))


def build_contact_sheet(rows: list[dict[str, Any]], out_path: Path, title: str, cell: int = 256) -> None:
    if not rows:
        return
    n = len(rows)
    cols = 3
    label_h = 84
    top_h = 36
    canvas = Image.new("RGB", (cols * cell, top_h + n * (cell + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    headers = ("teacher", "student", "abs diff (x4)")
    for i, row in enumerate(rows):
        y = top_h + i * (cell + label_h)
        t = Image.open(row["teacher_path"]).convert("RGB").resize((cell, cell), Image.Resampling.BILINEAR)
        s = Image.open(row["student_path"]).convert("RGB").resize((cell, cell), Image.Resampling.BILINEAR)
        ta = np.asarray(t).astype(np.float32) / 255.0
        sa = np.asarray(s).astype(np.float32) / 255.0
        d = np.clip(np.abs(sa - ta) * 4.0, 0.0, 1.0)
        d_img = Image.fromarray((d * 255.0).astype(np.uint8))
        for c, img in enumerate((t, s, d_img)):
            x = c * cell
            canvas.paste(img, (x, y))
            draw.text((x + 6, y + cell + 4), headers[c], fill=(0, 0, 0), font=font)
        draw.text(
            (6, y + cell + 20),
            (
                f"idx={row['index']} fail={row['failure_score']:.3f} "
                f"clip_gap={row.get('clip_prompt_gap', float('nan')):.3f} "
                f"ssim={row['ssim_teacher_student']:.3f} edge_l1={row['edge_l1_teacher_student']:.3f}"
            ),
            fill=(0, 0, 0),
            font=font,
        )
    canvas.save(out_path)


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    compare_dir = Path(args.compare_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    triplets = collect_triplets(compare_dir)
    prior_metrics = read_metrics_jsonl(compare_dir / "metrics.jsonl")
    prompt_map: dict[int, str] = {}
    if args.prompt_file:
        prompt_map = parse_prompt_file(Path(args.prompt_file))
    else:
        for idx, row in prior_metrics.items():
            prompt = row.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                prompt_map[idx] = prompt.strip()

    device = torch.device(args.device if (torch.cuda.is_available() or not args.device.startswith("cuda")) else "cpu")
    clip_enabled = bool(args.enable_clip)
    if clip_enabled and (CLIPModel is None or CLIPProcessor is None):
        raise RuntimeError("transformers CLIP components unavailable. Install transformers or pass --no-enable-clip.")

    processor = None
    model = None
    if clip_enabled:
        dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
        processor = CLIPProcessor.from_pretrained(args.clip_model)
        model = CLIPModel.from_pretrained(args.clip_model, torch_dtype=dtype).to(device).eval()

    rows: list[dict[str, Any]] = []
    for item in triplets:
        idx = int(item["index"])
        t = to_tensor_rgb(Path(item["teacher_path"]))
        s = to_tensor_rgb(Path(item["student_path"]))
        t, s = ensure_same_size(t, s)

        record: dict[str, Any] = {
            "index": idx,
            "teacher_path": str(item["teacher_path"]),
            "student_path": str(item["student_path"]),
            "ref_path": str(item["ref_path"]) if item["ref_path"] is not None else None,
            "prompt": prompt_map.get(idx, ""),
            "l1_teacher_student": l1(t, s),
            "ssim_teacher_student": ssim_simple(t, s),
            "psnr_teacher_student": psnr(t, s),
            "edge_l1_teacher_student": edge_l1(t, s),
            "hist_js_teacher_student": hist_js_divergence(t, s),
        }

        if item["ref_path"] is not None:
            r = to_tensor_rgb(Path(item["ref_path"]))
            r, s2 = ensure_same_size(r, s)
            r, t2 = ensure_same_size(r, t)
            record.update(
                {
                    "l1_teacher_ref": l1(t2, r),
                    "l1_student_ref": l1(s2, r),
                    "ssim_teacher_ref": ssim_simple(t2, r),
                    "ssim_student_ref": ssim_simple(s2, r),
                    "edge_l1_teacher_ref": edge_l1(t2, r),
                    "edge_l1_student_ref": edge_l1(s2, r),
                }
            )
            record["student_minus_teacher_ref_l1"] = record["l1_student_ref"] - record["l1_teacher_ref"]
            record["student_minus_teacher_ref_ssim"] = record["ssim_student_ref"] - record["ssim_teacher_ref"]

        if clip_enabled and processor is not None and model is not None:
            pil_teacher = Image.open(item["teacher_path"]).convert("RGB")
            pil_student = Image.open(item["student_path"]).convert("RGB")
            prompts = [record["prompt"], record["prompt"]]
            inputs = processor(text=prompts, images=[pil_teacher, pil_student], return_tensors="pt", padding=True, truncation=True)
            px = inputs["pixel_values"].to(device=device, dtype=model.dtype)
            ids = inputs["input_ids"].to(device)
            mask = inputs["attention_mask"].to(device)
            img_feat = as_feature_tensor(model.get_image_features(pixel_values=px))
            txt_feat = as_feature_tensor(model.get_text_features(input_ids=ids, attention_mask=mask))
            img_feat = F.normalize(img_feat.float(), dim=-1)
            txt_feat = F.normalize(txt_feat.float(), dim=-1)
            record["clip_prompt_teacher"] = cosine(img_feat[0], txt_feat[0])
            record["clip_prompt_student"] = cosine(img_feat[1], txt_feat[1])
            record["clip_prompt_gap"] = record["clip_prompt_teacher"] - record["clip_prompt_student"]
            record["clip_teacher_student_image_cosine"] = cosine(img_feat[0], img_feat[1])

        # Higher means "more likely failing" relative to teacher behavior.
        failure_score = (
            2.0 * (1.0 - float(record["ssim_teacher_student"]))
            + 1.0 * float(record["edge_l1_teacher_student"])
            + 0.5 * float(record["hist_js_teacher_student"])
        )
        if "clip_prompt_gap" in record:
            failure_score += max(0.0, float(record["clip_prompt_gap"])) * 1.5
        if "student_minus_teacher_ref_l1" in record:
            failure_score += max(0.0, float(record["student_minus_teacher_ref_l1"])) * 2.0
        record["failure_score"] = float(failure_score)

        rows.append(record)

    rows_sorted = sorted(rows, key=lambda r: float(r["failure_score"]), reverse=True)
    worst = rows_sorted[: max(1, min(args.top_k, len(rows_sorted)))]
    best = list(reversed(rows_sorted[-max(1, min(args.top_k, len(rows_sorted))) :]))

    def mean(key: str) -> float | None:
        vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    summary = {
        "compare_dir": str(compare_dir),
        "count": len(rows),
        "clip_enabled": clip_enabled,
        "mean_failure_score": mean("failure_score"),
        "mean_ssim_teacher_student": mean("ssim_teacher_student"),
        "mean_edge_l1_teacher_student": mean("edge_l1_teacher_student"),
        "mean_hist_js_teacher_student": mean("hist_js_teacher_student"),
        "mean_clip_prompt_gap": mean("clip_prompt_gap"),
        "mean_clip_teacher_student_image_cosine": mean("clip_teacher_student_image_cosine"),
        "mean_student_minus_teacher_ref_l1": mean("student_minus_teacher_ref_l1"),
        "mean_student_minus_teacher_ref_ssim": mean("student_minus_teacher_ref_ssim"),
    }

    report = {
        "summary": summary,
        "worst_examples": worst,
        "best_examples": best,
        "rows": rows_sorted,
    }
    (output_dir / "failure_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / "failure_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows_sorted:
            handle.write(json.dumps(row) + "\n")

    build_contact_sheet(
        worst,
        output_dir / "worst_cases_contact_sheet.png",
        title="Worst student failures vs teacher (ranked by failure_score)",
        cell=args.cell,
    )
    build_contact_sheet(
        best,
        output_dir / "best_cases_contact_sheet.png",
        title="Best student matches vs teacher (lowest failure_score)",
        cell=args.cell,
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(str(output_dir / "failure_report.json"), flush=True)
    print(str(output_dir / "worst_cases_contact_sheet.png"), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep-diagnose SANA student failure against teacher outputs.")
    parser.add_argument("--compare-dir", required=True, help="Directory with teacher_XXX.png, student_XXX.png, optional ref_XXX.png")
    parser.add_argument("--output-dir", required=True, help="Where diagnostics artifacts are written")
    parser.add_argument("--prompt-file", default="", help="Optional prompt file matching sample order")
    parser.add_argument("--top-k", type=int, default=16, help="How many best/worst samples to visualize")
    parser.add_argument("--cell", type=int, default=256, help="Cell size in output contact sheets")
    parser.add_argument("--enable-clip", action="store_true", help="Enable CLIP-based prompt and image embedding metrics")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
