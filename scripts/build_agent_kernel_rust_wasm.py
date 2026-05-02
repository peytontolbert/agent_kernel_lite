#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def build_and_sync(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    crate_dir = (repo_root / args.crate_dir).resolve()
    export_dir = Path(args.export_dir).resolve()
    if not crate_dir.exists():
        raise SystemExit(f"Rust/WASM crate does not exist: {crate_dir}")

    if not bool(args.no_build):
        subprocess.run(
            ["wasm-pack", "build", "--target", "web", "--release"],
            cwd=crate_dir,
            check=True,
        )

    source_dir = export_dir / "src"
    pkg_dir = export_dir / "pkg"
    source_dir.mkdir(parents=True, exist_ok=True)
    pkg_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(crate_dir / "Cargo.toml", export_dir / "Cargo.toml")
    lock_path = crate_dir / "Cargo.lock"
    if lock_path.exists():
        shutil.copy2(lock_path, export_dir / "Cargo.lock")
    _copy_tree_contents(crate_dir / "src", source_dir)
    _copy_tree_contents(crate_dir / "pkg", pkg_dir)

    print(f"synced {crate_dir} -> {export_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--crate-dir", default="agent_kernel_rust_wasm")
    parser.add_argument(
        "--export-dir",
        default="/data/repository_library/exports/agent_kernel/wasm/agent_kernel_lite_core",
    )
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    build_and_sync(args)


if __name__ == "__main__":
    main()
