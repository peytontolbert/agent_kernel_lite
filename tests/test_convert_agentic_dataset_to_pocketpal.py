from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO_ROOT / "scripts" / "convert_agentic_dataset_to_pocketpal.py"
    spec = importlib.util.spec_from_file_location("convert_agentic_dataset_to_pocketpal", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_converts_messages_to_pocketpal_respond_dataset(tmp_path: Path) -> None:
    converter = _load_script()
    input_path = tmp_path / "agentic.jsonl"
    row = {
        "id": "respond-1",
        "messages": [
            {"role": "user", "content": "Help me plan the next app task."},
            {"role": "assistant", "content": "Start by choosing the smallest useful behavior to test."},
        ],
    }
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    manifest = converter.convert_dataset(
        argparse.Namespace(
            input_path=str(input_path),
            hf_dataset="",
            hf_split="train",
            hf_streaming="0",
            source_name="unit_agentic",
            output_dir=str(tmp_path / "out"),
            max_rows=10,
            eval_fraction=0.0,
        )
    )

    assert manifest["total_examples"] == 1
    assert manifest["target_action_counts"] == {"respond": 1}
    train_path = Path(manifest["train_dataset_path"])
    converted = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])
    assert converted["task_type"] == "converted_agentic_instruction"
    assert "<AK_PROFILE>" in converted["encoder_text"]
    assert json.loads(converted["decoder_text"])["action"] == "respond"


def test_converts_tool_call_to_extension_request(tmp_path: Path) -> None:
    converter = _load_script()
    input_path = tmp_path / "tools.jsonl"
    row = {
        "id": "tool-1",
        "messages": [
            {"role": "user", "content": "Add dentist appointment tomorrow."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calendar.create_event",
                            "arguments": {"title": "Dentist", "date": "tomorrow"},
                        }
                    }
                ],
            },
        ],
    }
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    manifest = converter.convert_dataset(
        argparse.Namespace(
            input_path=str(input_path),
            hf_dataset="",
            hf_split="train",
            hf_streaming="0",
            source_name="unit_tool",
            output_dir=str(tmp_path / "out"),
            max_rows=10,
            eval_fraction=0.0,
        )
    )

    assert manifest["total_examples"] == 1
    assert manifest["target_action_counts"] == {"extension_request": 1}
    converted = json.loads(Path(manifest["train_dataset_path"]).read_text(encoding="utf-8").splitlines()[0])
    target = json.loads(converted["decoder_text"])
    assert target["action"] == "extension_request"
    metadata = target["proposal_metadata"]
    assert metadata["extension_id"] == "calendar"
    assert metadata["capability"] == "calendar.create_event"
    assert metadata["requires_user_approval"] is True
    assert metadata["tool_args_present"] is True
