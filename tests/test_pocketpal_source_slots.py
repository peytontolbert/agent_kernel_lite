from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pocketpal_source_slots import compile_source_slots
from scripts.pocketpal_source_slots import expand_source_pointers
from scripts.pocketpal_source_slots import pointerize_exact_text
from scripts.pocketpal_source_slots import source_slots_encoder_block


def test_source_slots_pointerize_and_expand_user_text() -> None:
    slots = compile_source_slots(
        user_text="hey john i need the report by friday because the client is asking",
        max_slots=8,
    )

    pointerized = pointerize_exact_text(
        "Please send john a polished note: hey john i need the report by friday because the client is asking",
        slots,
    )

    assert "<AK_COPY_USER_SOURCE_1>" in pointerized
    assert expand_source_pointers(pointerized, slots).endswith(
        "hey john i need the report by friday because the client is asking"
    )


def test_source_slots_encoder_block_lists_copy_tokens() -> None:
    slots = compile_source_slots(user_text="meeting moved to 11 and priya needs the updated slides")
    block = source_slots_encoder_block(slots)

    assert "<AK_SOURCE_SLOTS>" in block
    assert "<AK_COPY_USER_SOURCE_1>" in block
    assert "priya needs the updated slides" in block
