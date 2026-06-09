# Agentic Datasets for PocketPal

PocketPal should not chase lower perplexity on generic web text. The target is
lower loss on user-centered control behavior:

- understand a user's request
- inspect profile slots and local context
- choose `respond`, `gather_context`, `save_memory`, `ask_user`, or
  `extension_request`
- follow privacy and approval policy
- summarize extension results without pretending an action already happened

## Best Public Dataset Families

### 1. Function Calling

Use for extension routing, argument formation, and choosing no tool when no tool
is needed.

Good candidates:

- Berkeley Function Calling Leaderboard dataset
- Salesforce `xlam-function-calling-60k`
- Glaive function-calling variants

Convert these into PocketPal examples by mapping each function/API to an
installed extension capability:

```json
{
  "action": "extension_request",
  "content": "Requesting approval to use calendar:create_event.",
  "proposal_metadata": {
    "extension_id": "calendar",
    "capability": "calendar.create_event",
    "requires_user_approval": true
  }
}
```

Do not train the model to execute the API silently or to reconstruct nested
tool arguments. PocketPal should request approval when an extension touches
user data or external state; the runtime can pass the original user request to
the approved extension.

### 2. Multi-Turn Tool/User Benchmarks

Use for policy following, asking clarification questions, and maintaining state
across user turns.

Good candidates:

- tau-bench retail and airline
- AppWorld
- AndroidWorld or mobile-app agent traces if converted carefully

These are better as curriculum sources than as direct answer data. Convert
their traces into decisions:

- `ask_user` when required info is missing
- `extension_request` when an API/tool action is needed
- `respond` after tool results
- `save_memory` only when the user explicitly wants durable memory

### 3. General Agentic Instruction Data

Use sparingly for broad helpfulness and instruction following.

Good candidates:

- Microsoft `orca-agentinstruct-1M-v1`
- other permissively licensed agentic/instruction SFT sets after filtering

This should not dominate the mix. It improves language quality, but it does not
teach PocketPal's local-first control contract unless converted or wrapped with
PocketPal slots.

### 4. Personal Utility Synthetic Data

Keep generating this in-repo. It is the highest-signal data for PocketPal
because it directly matches the target runtime:

- profile slots
- local notes
- reminders
- privacy rules
- extension manifests
- approval policy
- local context snippets

The current generator lives in
`scripts/build_agentkernel_lite_encdec_dataset.py` under
`--max-user-slot-examples`.

## Recommended Mixture

For the next serious PocketPal controller run:

| Source | Share |
| --- | ---: |
| PocketPal synthetic slots/memory/extensions | 40% |
| Function-calling datasets converted to extension requests | 25% |
| Multi-turn tool/user traces converted to decisions | 20% |
| Existing research/paper curriculum | 10% |
| General agentic instruction SFT | 5% |

For the iPhone-target model, prefer fewer high-signal examples over huge noisy
mixtures. A 50k to 200k curated controller dataset is more useful than millions
of generic chat rows.

## Conversion Rules

Every external dataset row should become this contract:

```text
<AK_PROFILE> user slots
<AK_CONTEXT> local context or tool result
<AK_EXTENSION> installed extension manifest, if relevant
<AK_USER> request
```

Target:

```json
{
  "action": "respond | gather_context | save_memory | ask_user | extension_request",
  "content": "...",
  "proposal_metadata": {}
}
```

Reject or downweight rows that:

- execute a tool without approval
- reveal private user data to a third party
- rely on cloud-only assumptions
- train the model to claim it performed an action when it only proposed one
- require long chain-of-thought style targets

## Converter

Use `scripts/convert_agentic_dataset_to_pocketpal.py` for first-pass conversion
of local JSON/JSONL exports or Hugging Face datasets.

Local file or directory:

```bash
/home/peyton/miniconda3/envs/ai/bin/python scripts/convert_agentic_dataset_to_pocketpal.py \
  --input-path data/external_agentic/function_calls.jsonl \
  --source-name function_calls_local \
  --output-dir artifacts/pocketpal_external_function_calls \
  --max-rows 50000
```

Hugging Face dataset:

```bash
/home/peyton/miniconda3/envs/ai/bin/python scripts/convert_agentic_dataset_to_pocketpal.py \
  --hf-dataset Salesforce/xlam-function-calling-60k \
  --hf-split train \
  --source-name xlam_function_calling_60k \
  --output-dir artifacts/pocketpal_external_xlam \
  --max-rows 50000
```

The converter handles common schemas:

- `messages` / `conversations` / `turns`
- `instruction` + `input` + `output`
- `prompt` / `query` / `question` with `answer` / `response`
- assistant `tool_calls`, `function_call`, or row-level `function_calls`

Rows with tool calls become `extension_request` examples. Rows without tool
calls become lower-weight `respond` examples. Review samples before a real run;
the converter is intentionally conservative but external datasets vary.

## Practical Order

1. Train on PocketPal synthetic controller data only.
2. Add function-calling data mapped to installed extensions.
3. Add multi-turn user/tool traces.
4. Add research data back as one context source.
5. Add a small amount of general agentic instruction data for answer quality.

This keeps the model pointed at being a personal on-device agent instead of
drifting back into a generic assistant.
