from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO_ROOT / "scripts" / "build_pocketpal_agent_quality_dataset.py"
    spec = importlib.util.spec_from_file_location("build_pocketpal_agent_quality_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_agent_quality_dataset_preserves_user_agent_instructions(tmp_path: Path) -> None:
    builder = _load_script()

    manifest = builder.build_dataset(tmp_path / "agent_quality", eval_fraction=0.25)

    rows = _read_jsonl(Path(manifest["train_dataset_path"])) + _read_jsonl(Path(manifest["eval_dataset_path"]))
    assert manifest["total_examples"] == len(rows)
    assert {"respond", "ask_user", "extension_request", "gather_context"} <= set(manifest["target_action_counts"])
    joined_inputs = "\n".join(str(row["encoder_text"]) for row in rows)
    assert "<AK_AGENT_ACTIVE>" in joined_inputs
    assert "Agent instruction:" in joined_inputs
    assert "The active agent instruction is the primary task contract" in joined_inputs
    assert "Saved user data:" in joined_inputs
    assert "Stale selected paper context:" in joined_inputs
    assert "Use stale paper context only when the current user request asks" in joined_inputs


def test_agent_quality_dataset_emits_structured_actions(tmp_path: Path) -> None:
    builder = _load_script()

    manifest = builder.build_dataset(tmp_path / "agent_quality", eval_fraction=0.25)

    rows = _read_jsonl(Path(manifest["train_dataset_path"])) + _read_jsonl(Path(manifest["eval_dataset_path"]))
    by_action = {row["action"]: json.loads(str(row["decoder_text"])) for row in rows}
    assert by_action["extension_request"]["proposal_metadata"]["requires_user_approval"] is True
    assert by_action["gather_context"]["action"] == "gather_context"
    assert by_action["ask_user"]["action"] == "ask_user"
