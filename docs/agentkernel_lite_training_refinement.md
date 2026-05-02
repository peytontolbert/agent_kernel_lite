# AgentKernel Lite Training Refinement

## Current Diagnosis

The current Qwen -> dense teacher -> BitNet student pipeline is structurally complete, but the training objective is not aligned well enough with the browser assistant we want.

The strongest signal is that both the dense teacher and the BitNet student fail the same generation probes. That means the main issue is upstream of BitNet compression: the dense intermediate has learned the protocol and rough action shape, but it has not learned a reliable assistant behavior distribution.

## What Is Going Wrong

### 1. Eval loss is rewarding target imitation, not assistant quality

The final dense teacher reached a good eval loss on the held-out split, and the BitNet student followed it. But the held-out split is generated from the same templates and teacher relabeling scheme as training. It is not a realistic behavioral eval.

Observed behavior:

- `Action:` / `Content:` format is usually correct.
- `respond` vs `gather_context` is often correct.
- Grounded content is weak, repetitive, or title-hallucinates.
- Selected-paper followups sometimes collapse into generic paper language.

So token loss is currently measuring whether the model matches the synthetic target distribution, not whether the model is a good chat assistant with research context.

### 2. The dataset is too template-shaped

The curriculum still contains many targets shaped like:

- `Based on the retrieved evidence...`
- `This paper focuses on...`
- `I am using the selected paper [P1]...`
- `The selected paper...`

Those are useful bootstrap patterns, but at this scale they dominate the style. A 100M model learns the shell more strongly than the semantic behavior.

Direct chat is especially underrepresented. In the 30k split, only a tiny seed set is true direct chat, while most direct answers are topic templates derived from paper terms. That teaches "explain a jumbled paper keyword phrase" more than "be a helpful assistant."

### 3. Teacher relabeling is constrained by weak old targets

The Qwen relabeler asks Qwen to rewrite an existing weak target. This preserves too much of the old target's structure. If the old target is generic, Qwen often produces a better generic target, not a genuinely natural answer.

For the next dataset, Qwen should be asked to produce the ideal target from the browser prompt and evidence only, without seeing the old weak target except as optional metadata for the expected action.

### 4. Planning and answering are mixed in one decoder objective

The model is being asked to learn:

- whether to gather context,
- how to rewrite the query,
- how to rerank candidates,
- how to answer directly,
- how to answer from selected context,
- how to answer from retrieved evidence.

That is fine architecturally, but the target format currently compresses these into one `Action/Content` string. The model learns a single generic response channel instead of a crisp loop state machine.

The decoder needs a small action language with explicit tokens, for example:

```text
<AK_DECISION> <AK_GATHER_CONTEXT> <AK_QUERY> ...
<AK_DECISION> <AK_RESPOND> <AK_ANSWER> ...
<AK_DECISION> <AK_USE_CONTEXT> <AK_TARGET_CONTEXT> P1 <AK_ANSWER> ...
```

Then the UI can parse the action tokens deterministically while the model still generates the actual answer text.

### 5. Retrieval ranking is not trained enough

The training runs used `retrieval_contrastive_weight=0.0`, so the encoder was not actually trained as the retrieval/ranking model. The encoder-decoder model is learning to consume context, but not learning a strong embedding space for retrieving abstracts or selecting papers.

If we want the single model to replace a separate embedding model, we need retrieval-pair training:

- query -> relevant abstract/title
- query -> selected paper full-text span
- selected-paper followup -> active selected context
- hard negatives from the same broad area
- in-batch contrastive loss

The existing trainer already has a retrieval contrastive path, but the curriculum needs high-quality `retrieval_query_text`, `retrieval_doc_text`, and nonzero `retrieval_loss_weight`.

### 6. Browser prompt distribution and training prompt distribution do not match tightly enough

The browser sends large prompts containing:

- context packet JSON,
- reading notes,
- answer scaffolds,
- retrieved evidence blocks,
- recent conversation,
- mode instructions.

The default generation probes and much of training use shorter synthetic prompts. The model is then evaluated and deployed under a different prompt distribution.

Training should be generated from the exact Rust/WASM context compiler and browser prompt builder. The train row should contain the actual prompt the model will see in production.

### 7. Browser fallbacks hide model failure and contaminate design feedback

The browser still has deterministic fallback answers and heuristic planner overrides. They are useful for product resilience, but they make it difficult to tell whether the model has learned the behavior.

For training/eval, we need a strict no-fallback harness:

- compile prompt using the same browser/Rust path,
- run local model,
- parse decision,
- score behavior,
- no fallback answer,
- no heuristic rewrite after decode.

## Refined Training Plan

### Stage A: Build a loop-native dataset

Generate examples from real browser loop states, not only synthetic paper rows.

Required task families:

- normal chat with no retrieval,
- ambiguous chat where retrieval is not needed,
- paper search request -> `gather_context`,
- paper search with candidates -> rerank/select,
- answer from retrieved evidence,
- selected-paper followup -> use active context, no new retrieval,
- weak/off-topic evidence -> ask for narrower query or answer cautiously,
- think mode synthesis,
- deep research evidence-by-evidence synthesis.

Each row should include:

- exact compiled browser/Rust prompt,
- explicit expected action token,
- expected answer text,
- retrieval pair fields when relevant,
- source/evidence ids available in prompt,
- mode label.

### Stage B: Regenerate teacher targets without old-target anchoring

Use Qwen as a teacher, but prompt it as:

```text
Given this exact AgentKernel Lite runtime prompt, produce the ideal structured decision.
Use only evidence ids in the prompt.
If action is gather_context, output only the query/retrieval decision.
If action is respond, answer the user naturally.
Do not invent titles or source ids.
```

Do not show Qwen the old weak target. Only show the expected action when the row is intentionally supervised as `respond` or `gather_context`.

Reject teacher outputs that:

- cite unavailable evidence ids,
- invent titles not present in prompt,
- mention "training example",
- use generic fallback language when evidence is present,
- fail selected-context behavior.

### Stage C: Train dense with multi-objective loss

Train dense first with:

- decoder CE loss,
- retrieval contrastive loss for encoder ranking,
- action-token loss weighted higher than prose tokens,
- selected-context examples weighted higher,
- direct chat weighted higher than current split.

Suggested starting weights:

- direct chat: `3.0`
- selected-paper followup: `4.0`
- answer from retrieved evidence: `3.5`
- gather_context action/query: `2.5`
- rerank/select: `3.0`
- off-topic evidence refusal/caution: `2.5`
- retrieval contrastive weight: start `0.05`, tune to `0.1`

### Stage D: Distill BitNet only after dense passes behavior

Do not BitNet-distill just because dense eval loss is low. Gate dense with behavior probes first.

Promotion requirements before BitNet:

- action parse pass rate >= 95%,
- selected-paper followup pass rate >= 90%,
- direct chat pass rate >= 90%,
- grounded answer pass rate >= 85%,
- title hallucination rate near zero on evidence prompts,
- no token-soup/repetition on no-fallback decode.

Then distill:

- initialize BitNet from dense,
- use dense teacher KL,
- include CE on Qwen targets,
- use the same behavior harness every checkpoint,
- promote only checkpoints that pass generation probes after quantized conversion.

## Immediate Next Changes

1. Add a new dataset builder that calls the Rust/WASM prompt compiler or mirrors it exactly, so training prompts match browser prompts.
2. Update Qwen relabeling to generate from prompt-only, not old weak targets.
3. Convert decoder targets to explicit action tokens instead of prose-only `Action/Content` lines.
4. Populate retrieval contrastive fields and turn on encoder retrieval loss.
5. Add a no-fallback browser parity/eval harness as the promotion gate.
6. Re-train dense until behavior probes pass, then redo dense -> BitNet.

## Bottom Line

Yes, we are training the wrong objective for the quality we want. The current pipeline proves that the model can learn the shell of AgentKernel Lite, but it does not yet train the model as a robust assistant. The refined path is to train on exact runtime prompts, improve teacher targets, explicitly supervise loop actions, train the encoder for retrieval, and gate BitNet distillation on behavioral generation quality instead of eval loss alone.
