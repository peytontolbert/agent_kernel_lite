#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_eval_module():
    path = ROOT / "scripts" / "evaluate_pocketpal_agent_gates.py"
    spec = importlib.util.spec_from_file_location("evaluate_pocketpal_agent_gates", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load evaluator helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(action: str, content: str, task_type: str) -> str:
    return json.dumps(
        {"action": action, "content": content, "proposal_metadata": {"task_type": task_type}},
        ensure_ascii=False,
        sort_keys=True,
    )


def _row(
    *,
    split: str,
    index: int,
    prompt: str,
    content: str,
    task_type: str,
    intent: str,
    intent_labels: dict[str, int],
    weight: float,
    negative: str = "",
    action: str = "respond",
) -> dict[str, Any]:
    return {
        "example_id": f"v171_{split}_{index:07d}",
        "split": split,
        "source_type": "pocketpal_v171_source_transform_curriculum",
        "source_id": f"v171_{task_type}_{index:07d}",
        "task_type": task_type,
        "encoder_text": prompt,
        "decoder_text": _payload(action, content, task_type),
        "negative_decoder_text": negative,
        "negative_loss_weight": 1.0 if negative else 0.0,
        "action": action,
        "weight": float(weight),
        "intent_label": intent,
        "intent_label_id": int(intent_labels.get(intent, -1)),
    }


SOURCE_CASES = [
    {
        "text": "Hi how are you?",
        "rewrite": "Hello, I hope you are well.",
        "summary": "- Greeting: Hi, how are you?",
        "extract": "Greeting; asks how the recipient is doing",
        "classify": "greeting",
        "translate_es": "Hola, ¿cómo estás?",
        "plan": "- Reply politely.\n- Ask how they are doing.",
        "search": "No web search needed; this is a greeting.",
        "memory": "Remember that the user greeted someone and asked how they are doing.",
    },
    {
        "text": "hey john send the report by friday because the client is asking",
        "rewrite": "Hello John, could you please send the report by Friday? The client is asking. Thank you.",
        "summary": "- John should send the report by Friday because the client is asking.",
        "extract": "John; report; Friday; client is asking",
        "classify": "professional_message",
        "translate_es": "Hola John, envía el informe antes del viernes porque el cliente lo está pidiendo.",
        "plan": "- Ask John for the report.\n- Keep the Friday deadline.\n- Mention the client request.",
        "search": "No web search needed; this is a writing task.",
        "memory": "Remember that John needs to send the report by Friday because the client is asking.",
    },
    {
        "text": "vendor invoice INV-2048 is blocked until finance approves $1,200",
        "rewrite": "The vendor invoice INV-2048 is blocked until Finance approves $1,200.",
        "summary": "- Vendor invoice INV-2048 is blocked until Finance approves $1,200.",
        "extract": "INV-2048; Finance approval; $1,200; blocked vendor invoice",
        "classify": "finance_status",
        "translate_es": "La factura del proveedor INV-2048 está bloqueada hasta que Finanzas apruebe $1,200.",
        "plan": "- Confirm Finance owns the approval.\n- Track invoice INV-2048.\n- Follow up on the $1,200 approval.",
        "search": "No web search needed; this is internal invoice information.",
        "memory": "Remember that vendor invoice INV-2048 is blocked until Finance approves $1,200.",
    },
    {
        "text": "move the launch review to Tuesday because QA needs more time",
        "rewrite": "Please move the launch review to Tuesday because QA needs more time.",
        "summary": "- Move the launch review to Tuesday because QA needs more time.",
        "extract": "launch review; Tuesday; QA needs more time",
        "classify": "scheduling",
        "translate_es": "Mueve la revisión de lanzamiento al martes porque QA necesita más tiempo.",
        "plan": "- Reschedule the launch review for Tuesday.\n- Note that QA needs more time.\n- Notify attendees.",
        "search": "No web search needed; this is a scheduling request.",
        "memory": "Remember that the launch review should move to Tuesday because QA needs more time.",
    },
    {
        "text": "Nora has the south entrance badge code GATE-17 for Thursday at 2 PM",
        "rewrite": "Nora has the south entrance badge code GATE-17 for Thursday at 2 PM.",
        "summary": "- Nora has south entrance badge code GATE-17 for Thursday at 2 PM.",
        "extract": "Nora; south entrance; GATE-17; Thursday at 2 PM",
        "classify": "access_code",
        "translate_es": "Nora tiene el código de acceso de la entrada sur GATE-17 para el jueves a las 2 PM.",
        "plan": "- Keep Nora's badge code available.\n- Use GATE-17 at the south entrance.\n- Apply it Thursday at 2 PM.",
        "search": "No web search needed; this is saved access information.",
        "memory": "Remember that Nora has south entrance badge code GATE-17 for Thursday at 2 PM.",
    },
    {
        "text": "research current TestFlight external tester limits and summarize sources",
        "rewrite": "Please research the current TestFlight external tester limits and summarize the sources.",
        "summary": "- Research current TestFlight external tester limits and summarize sources.",
        "extract": "TestFlight; external tester limits; summarize sources",
        "classify": "web_search",
        "translate_es": "Investiga los límites actuales de probadores externos de TestFlight y resume las fuentes.",
        "plan": "- Search current Apple TestFlight documentation.\n- Compare source dates.\n- Summarize the external tester limits.",
        "search": "Search the web for current TestFlight external tester limits and summarize up to 5 reliable sources.",
        "memory": "Remember that the user wants current TestFlight external tester limits summarized from sources.",
    },
]


TRANSFORMS = [
    (
        "rewrite",
        "Professional Rewrite Agent",
        "Rewrite the user's provided text professionally. Preserve facts and intent. Do not use placeholders unless the matching text slot exists.",
        "rewrite",
        7.0,
    ),
    (
        "summary",
        "Bullet Summary Agent",
        "Convert the user's provided text into a concise bullet summary. Do not answer conversationally.",
        "summary",
        7.0,
    ),
    (
        "extract",
        "Extraction Agent",
        "Extract the important names, dates, codes, amounts, topics, or requested facts from the user's provided text.",
        "extract",
        5.0,
    ),
    (
        "classify",
        "Intent Classifier",
        "Classify the user's request into one short label. Return only the label.",
        "classify",
        4.0,
    ),
    (
        "translate_es",
        "Spanish Translator",
        "Translate the user's provided text into Spanish. Preserve meaning and do not answer as a chatbot.",
        "translate",
        4.0,
    ),
    (
        "plan",
        "Planning Agent",
        "Turn the user's provided text into a short actionable plan. Preserve concrete names, dates, values, and constraints.",
        "plan",
        4.0,
    ),
    (
        "search",
        "Search Request Agent",
        "Decide whether the user's request needs web search. If it does, write a precise search request. If it does not, say no web search is needed and why.",
        "search",
        4.0,
    ),
    (
        "memory",
        "Memory Agent",
        "Write the useful memory that should be saved from the user's provided text. Preserve exact details.",
        "memory",
        3.0,
    ),
]


def build_rows(train_repeats: int, eval_repeats: int, intent_labels: dict[str, int], seed: int) -> list[dict[str, Any]]:
    ev = _load_eval_module()
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    idx = 0
    negative_named = _payload("respond", "Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you.", "v171_negative_named_slots_unavailable")
    negative_source_echo = _payload("respond", "- Summary: [[SOURCE_TEXT]]", "v171_negative_raw_source_echo")

    for split, repeats in (("train", train_repeats), ("eval", eval_repeats)):
        for _ in range(int(repeats)):
            cases = list(SOURCE_CASES)
            rng.shuffle(cases)
            for case in cases:
                transforms = list(TRANSFORMS)
                rng.shuffle(transforms)
                for key, name, instruction, intent, weight in transforms:
                    prompt = ev._agent_prompt(
                        name=name,
                        instruction=instruction,
                        user_text=case["text"],
                        text_slots={"SOURCE_TEXT": case["text"]},
                    )
                    neg = negative_named if key == "rewrite" else negative_source_echo if key == "summary" else ""
                    rows.append(
                        _row(
                            split=split,
                            index=idx,
                            prompt=prompt,
                            content=case[key],
                            task_type=f"v171_source_{key}",
                            intent=intent,
                            intent_labels=intent_labels,
                            weight=weight,
                            negative=neg,
                        )
                    )
                    idx += 1

            for name, item, deadline, reason in [
                ("John", "report", "Friday", "the client is asking"),
                ("Priya", "slide deck", "Monday", "the team is waiting"),
                ("Lena", "budget draft", "June 3", "Finance is waiting"),
            ]:
                prompt = ev._agent_prompt(
                    name="Professional Rewrite Agent",
                    instruction="Rewrite the user's provided text professionally. Preserve facts and intent. Use existing placeholders only for matching available text slots.",
                    user_text=f"yo {name.lower()} send the {item} by {deadline.lower()} because {reason}",
                    text_slots={"NAME": name, "ITEM": item, "DEADLINE": deadline, "REASON": reason},
                )
                rows.append(
                    _row(
                        split=split,
                        index=idx,
                        prompt=prompt,
                        content="Hi [[NAME]], could you please send the [[ITEM]] by [[DEADLINE]]? [[REASON]]. Thank you.",
                        task_type="v171_named_slots_available",
                        intent="rewrite",
                        intent_labels=intent_labels,
                        weight=4.0,
                    )
                )
                idx += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/pocketpal_v171_source_transform_curriculum")
    parser.add_argument("--intent-labels-json", default="tmp/pocketpal_v168h_broad_slot_contract_mix/agentkernel_lite_encdec_dataset_manifest.json")
    parser.add_argument("--train-repeats", type=int, default=500)
    parser.add_argument("--eval-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1710)
    args = parser.parse_args()

    intent_labels: dict[str, int] = {}
    source_manifest_path = Path(args.intent_labels_json)
    if source_manifest_path.exists():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        intent_labels = {str(k): int(v) for k, v in source_manifest.get("intent_labels", {}).items()}

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.train_repeats, args.eval_repeats, intent_labels, args.seed)
    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "eval"]
    train_path = out / "pocketpal_v171_source_transform_train.jsonl"
    eval_path = out / "pocketpal_v171_source_transform_eval.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with eval_path.open("w", encoding="utf-8") as handle:
        for row in eval_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "manifest_path": str((out / "agentkernel_lite_encdec_dataset_manifest.json").resolve()),
        "objective": "pocketpal_v171_source_transform_curriculum",
        "train_dataset_path": str(train_path.resolve()),
        "eval_dataset_path": str(eval_path.resolve()),
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "intent_labels": intent_labels,
        "source_counts": {"pocketpal_v171_source_transform_curriculum": len(rows)},
    }
    (out / "agentkernel_lite_encdec_dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
