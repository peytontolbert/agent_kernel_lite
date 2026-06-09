#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path


def human_bytes(value: float) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} GiB"


def estimate(path: Path) -> dict[str, object]:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is not a PyTorch ZIP checkpoint")

    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        storage_infos = [info for info in infos if "/data/" in info.filename]
        storage_bytes = sum(info.file_size for info in storage_infos)
        size_histogram = Counter(info.file_size for info in storage_infos)

    fp32_params = storage_bytes / 4.0
    estimates = {
        "fp32": storage_bytes,
        "fp16": fp32_params * 2.0,
        "int8": fp32_params,
        "int4": fp32_params / 2.0,
        "bitnet_2bit_weights_only": fp32_params / 4.0,
    }
    return {
        "path": str(path),
        "checkpoint_bytes": path.stat().st_size,
        "zip_entries": len(infos),
        "storage_entries": len(storage_infos),
        "storage_bytes": storage_bytes,
        "estimated_fp32_params_if_all_fp32": int(fp32_params),
        "estimates_bytes": {name: int(value) for name, value in estimates.items()},
        "estimates_human": {name: human_bytes(value) for name, value in estimates.items()},
        "largest_storage_entries": [
            {"count": count, "bytes_each": size, "total": count * size}
            for size, count in size_histogram.most_common(12)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate mobile-friendly sizes for a PyTorch ZIP checkpoint without importing torch."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="/data/resumebot/checkpoints/final_finetuned_model.pt",
        help="Path to a PyTorch ZIP checkpoint.",
    )
    args = parser.parse_args()
    result = estimate(Path(args.checkpoint))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
