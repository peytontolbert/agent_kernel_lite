#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed a FLUX final latent refiner inside a student checkpoint.")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--final-latent-refiner", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    student_path = Path(args.student_checkpoint)
    refiner_path = Path(args.final_latent_refiner)
    output_path = Path(args.output)
    checkpoint = torch.load(student_path, map_location="cpu")
    refiner = torch.load(refiner_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"student checkpoint is not a dict: {student_path}")
    if not isinstance(refiner, dict) or "model" not in refiner or "config" not in refiner:
        raise ValueError(f"final latent refiner checkpoint is missing model/config: {refiner_path}")
    checkpoint["embedded_final_latent_refiner"] = {
        "model": refiner["model"],
        "config": refiner["config"],
        "source": str(refiner_path),
    }
    checkpoint.setdefault("metadata", {})
    if isinstance(checkpoint["metadata"], dict):
        checkpoint["metadata"]["embedded_final_latent_refiner"] = str(refiner_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print({"output": str(output_path), "refiner": str(refiner_path)})


if __name__ == "__main__":
    main()
