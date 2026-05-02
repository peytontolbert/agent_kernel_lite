# AgentKernel Lite Browser Decoding Strategy

Production target: iPhone/Safari WASM first, with WebGPU as an optional desktop path.

## Current Browser Runtime

- The model-stack browser runtime uses cached autoregressive decoding through `createGenerationSession(...).next(tokenId)`.
- This avoids re-encoding the prompt and reuses decoder key/value cache.
- It is not speculative decoding yet: the target model still verifies one token at a time.

## Implemented Fast Path

- Retrieval appears immediately after index ranking.
- Evidence selection is now a separate `Select Evidence` step.
- If retrieval has a clear top candidate, selector decode is skipped.
- If selector decode is needed, it uses the fixed control prefix:

```text
Action: gather_context
Content: selected_candidate_id=
```

- Selector generation is capped to a short token budget and stops as soon as a valid `selected_candidate_id` is decoded.

## Runtime Upgrade Added

The browser model-stack runtime now exposes:

```js
session.nextMany(tokenIds)
session.cloneState()
session.restoreState(state)
```

`nextMany` decodes a known span against the cached encoder memory and cached decoder state. The worker uses this in two places:

- Batched control-prefix consumption for short selector prompts.
- Strict all-or-nothing n-gram speculation during deterministic decoding.
- Conservative probabilistic n-gram speculation during sampled decoding.

If the draft span is fully accepted, the span cache is kept. If any draft token fails verification, the worker restores the previous cache and falls back to the normal one-token path.

For non-zero-temperature chat, draft tokens are accepted only when the target model keeps the token inside the active top-p distribution, ranks it near the top, and passes a probability draw. This is intentionally conservative; it improves repeated-phrase spans without letting an n-gram draft override the target distribution.

The worker also tracks acceptance per generation. Chat speculation starts only after enough generated text exists to make n-gram proposals meaningful, skips one-token chat drafts, and backs off temporarily when accepted draft-token rate is poor. This prevents failed speculation from becoming a consistent slowdown on Safari/WASM.

## Speculative Decode Requirements

Full speculative decoding in the browser needs a higher-level target-runtime API that can verify a draft span and return accepted lengths/probabilities:

```js
session.verifyDraft(draftTokenIds, {
  method: "strict",
  bonusToken: true
})
```

or:

```js
session.nextMany(draftTokenIds)
```

The current browser session now has the lower-level span primitive. It does not yet implement probabilistic rejection-sampling acceptance.

## Practical Next Step

Next runtime work:

1. Add `session.verifyDraft(...)` on top of `nextMany`.
2. Return accepted token count, accepted ids, and next logits.
3. Replace the current conservative probabilistic approximation with full rejection-sampling residual acceptance.
4. Add a tiny learned draft head/model if n-gram acceptance is too low.

The first draft source is n-gram/suffix proposals because it needs no second model and is safe for Safari. A tiny learned draft head can come later if n-gram acceptance is too low.
