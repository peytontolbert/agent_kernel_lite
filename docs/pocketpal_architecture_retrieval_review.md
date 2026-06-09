# PocketPal Architecture Retrieval Review

Generated: 2026-05-18T13:49:37.232668+00:00
Skill dataset: `/data/repo_skills_miner/artifacts/hf_openclaw_hermes_skills/data/train.parquet`

## Architecture Snapshot

- Latest trained bundle: `/data/agent_kernel_lite/artifacts/pocketpal_controller_100m_v201a_failure_replay_from_v200a/agentkernel_lite_encdec_manifest.json`
- Latest trained heads: retrieval_head_dim=`128`, agent_policy_heads=`True`, agent_intent_labels=`18`.
- Latest objective: `pocketpal_v201_failure_replay_from_v200a_eval`, steps=`100`.
- Packaged web bundle: `/data/agent_kernel_lite/web/models/pocketpal_controller_100m_bitnet_dev/manifest.json`
- Packaged web heads: retrieval_head_dim=`None`, agent_policy_heads=`False`, agent_intent_labels=`None`.

## Retrieval-Backed Findings

The OpenClaw/Hermes skills point toward treating PocketPal as a small local controller with retrieval, policy, memory, and verification surfaces around it. They do not support making the tiny decoder the full autonomous agent.

1. Keep the tiny model in the action/control plane. Use encoder heads for intent, retrieval need, confidence, OOD, action validity, and verification need. Let the decoder produce constrained decisions or short text only when the route is low entropy.
2. Make active-agent policy explicit outside the decoder. OpenClaw-style prepared turn boundaries and tool/plugin contracts match PocketPal's active-agent preamble, but the runtime should enforce action policy after model output.
3. Retrieval should condition behavior, not just train an `<AK_GATHER_CONTEXT>` token. The harness skill records include use-when, risks, patch relevance, and verification hints; PocketPal should pass retrieved operators into the decision pipeline before direct generation.
4. Memory and skill learning should be credit-assigned. Hermes-style self-improvement maps to saving successful PocketPal traces as new skill examples with outcome fields, not just adding more synthetic curricula.
5. Promotion needs a single verifier artifact. OpenClaw/Hermes patterns emphasize guardrails, tests, and verification; PocketPal currently has many eval JSONs but no single promotion ledger attached to a bundle.
6. Check browser parity before trusting architecture claims. If the app loads a bundle without policy/retrieval heads, the UI cannot use the latest controller architecture even if training manifests contain those heads.

## hybrid_controller

PocketPal signal: PocketPal has encoder policy heads and an intent head, but direct generation is still treated as a broad controller surface.

- `openclaw/openclaw:docs/plugins/sdk-agent-harness.md` score=0.304 summary=Documentation section describing the Agent Harness plugin.
  Use when: When working with the SDK and need to understand how to use the Agent Harness plugin.
  Verify: Check if the sidebar title matches 'Agent Harness' and that the content provides accurate information about the plugin.
- `openclaw/openclaw:src/plugin-sdk/agent-harness.ts` score=0.297 summary=A TypeScript configuration file for the plugin SDK's agent harness.
  Use when: When setting up or modifying the agent harness in the plugin SDK.
  Verify: Check that the configuration matches the expected behavior of the agent harness.
- `openclaw/openclaw:docs/plugins/sdk-agent-harness.md` score=0.257 summary=A documentation section explaining the selection policy for the SDK agent harness.
  Use when: When working with the SDK agent harness and need to understand its selection policy.
  Verify: Check if the selection policy aligns with the intended behavior of the SDK agent harness.
- `openclaw/openclaw:docs/plugins/sdk-agent-harness.md` score=0.234 summary=Provides guidance on terminal outcome classification in SDK agent harness pairing.
  Use when: When setting up or troubleshooting SDK agent harness pairing to understand terminal outcomes.
  Verify: Check if examples and explanations align with actual implementation.
- `openclaw/openclaw:docs/plugins/sdk-agent-harness.md` score=0.214 summary=Registers a harness in the SDK agent.
  Use when: When setting up or configuring the SDK agent to include a new harness.
  Verify: Check if the harness registration process is correctly documented and follows the expected steps.

## active_agent_contract

PocketPal signal: PocketPal prompts encode active agent instruction, retrieval policy, tool policy, and action policy directly into the encoder text.

- `openclaw/openclaw:src/agents/tool-policy.conformance.ts` score=0.386 summary=Configuration file for tool policy conformance in TypeScript
  Use when: Setting up tool policy conformance rules
  Verify: Run tests that cover tool policy conformance scenarios
- `openclaw/openclaw:src/agents/tool-policy-match.ts` score=0.301 summary=This skill card represents a symbol index in TypeScript within the 'tool-policy-match.ts' file of the 'openclaw/openclaw' repository.
  Use when: When working with TypeScript code that involves symbol indexing.
  Verify: Check if the symbol index is used correctly in the context of the code.
- `openclaw/openclaw:src/agents/tool-policy.plugin-only-allowlist.test.ts` score=0.294 summary=Tests symbol index functionality in tool-policy plugin.
  Use when: developing or testing tool-policy plugin
  Verify: run 'npm test' in project root
- `openclaw/openclaw:src/agents/tool-policy.ts` score=0.293 summary=Retrieve symbol index from tool policy
  Use when: Need to access symbol information in tool policy
  Verify: Check if symbol index is correctly retrieved and used
- `NousResearch/hermes-agent:tests/agent/test_plugin_llm.py` score=0.265 summary=Tests the default policy of the TrustGate to ensure it blocks model overrides.
  Use when: developing or testing the TrustGate's policy implementation
  Verify: run the test locally and check that it fails without the correct policy implementation

## skill_retrieval

PocketPal signal: v182 added OpenClaw/Hermes harness skill retrieval examples, but retrieval currently trains gather_context more than downstream action selection.

- `NousResearch/hermes-agent:tests/gateway/test_agent_cache.py` score=0.202 summary=Tests cache busting configuration extraction including live tool registry generation.
  Use when: Developing or testing cache busting functionality in Hermes Agent.
  Verify: Run tests locally and check if cache busting works as expected.
- `openclaw/openclaw:src/commands/doctor-cron-legacy-delivery.ts` score=0.194 summary=A function in TypeScript that does not perform any operations and has no side effects.
  Use when: when you need a placeholder or a stub function
  Verify: check if there are any calls to this function
- `NousResearch/hermes-agent:tools/registry.py` score=0.182 summary=Deregisters a tool from the registry.
  Use when: When you need to remove a tool from the registry.
  Verify: Check if the tool is no longer listed in the registry after calling this method.
- `NousResearch/hermes-agent:tests/conftest.py` score=0.180 summary=Resets tool registry caches in tests.
  Use when: testing development
  Verify: run tests after applying patch
- `NousResearch/hermes-agent:tools/registry.py` score=0.171 summary=Retrieves the schema for a tool by its name.
  Use when: When you need to fetch the schema of a specific tool from the registry.
  Verify: Check if the returned schema matches the expected structure for the given tool name.

## memory_state

PocketPal signal: PocketPal has browser-local memory, slots, agents, data sources, and session export/restore.

- `openclaw/openclaw:extensions/open-prose/skills/prose/primitives/session.md` score=0.257 summary=Understand and manage persistent agent memory in session context.
  Use when: when dealing with long-running sessions that require maintaining state across multiple interactions.
  Verify: test with various session scenarios to ensure memory retention. check for any error messages related to memory operations.
- `NousResearch/hermes-agent:tests/run_agent/test_background_review.py` score=0.248 summary=Tests that the background review summary is attributed to the self-improvement loop.
  Use when: development testing
  Verify: check output contains 'self-improvement loop' run in development environment
- `openclaw/openclaw:extensions/open-prose/skills/prose/primitives/session.md` score=0.244 summary=Manages persistent state within session contexts.
  Use when: when working with applications that require maintaining state across multiple requests or sessions.
  Verify: verify by checking if the session state is correctly saved and restored between requests. test with different scenarios to ensure state consistency.
- `openclaw/openclaw:extensions/open-prose/skills/prose/primitives/session.md` score=0.216 summary=Manages persistent state by reading memory.
  Use when: when interacting with systems that require maintaining state across sessions.
  Verify: check if the system maintains consistent state across different sessions. verify that no unauthorized access occurs during memory read operations.
- `NousResearch/hermes-agent:website/docs/user-guide/features/memory.md` score=0.197 summary=Provides information on how Persistent Memory works.
  Use when: When users need to understand how Persistent Memory functions.
  Verify: Check if examples and diagrams accurately represent how Persistent Memory operates.

## tool_approval_security

PocketPal signal: PocketPal extension_request decisions carry extension_id capability query max_sources and requires_user_approval.

- `openclaw/openclaw:extensions/discord/src/approval-runtime.ts` score=0.326 summary=A TypeScript file containing runtime approval logic for Discord.
  Use when: When implementing or reviewing approval processes in a Discord bot.
  Verify: Check that the changes do not break existing approval workflows. Test the updated approval logic in a development environment.
- `openclaw/openclaw:src/plugin-sdk/approval-runtime.ts` score=0.307 summary=Retrieve symbol index from approval runtime
  Use when: Need to access symbol information in approval runtime
  Verify: Check if the returned symbol index matches expected values
- `openclaw/openclaw:docs/tools/exec-approvals.md` score=0.304 summary=Describes the approval flow process in the exec-tools section of the documentation.
  Use when: When updating or reviewing the approval flow process in the exec-tools section.
  Verify: Check if the steps outlined in the document match the current workflow. Verify that all required permissions are correctly listed.
- `NousResearch/hermes-agent:tools/approval.py` score=0.301 summary=A function in the Hermes Agent's approval module that fires an approval hook.
  Use when: When needing to trigger an approval process within the Hermes Agent.
  Verify: Check if the function is called correctly in other parts of the codebase.
- `NousResearch/hermes-agent:tools/approval.py` score=0.299 summary=Function to submit pending approvals in the Hermes Agent repository.
  Use when: When submitting approvals in the Hermes Agent system.
  Verify: Check function signature and implementation for correctness.

## verification_eval

PocketPal signal: PocketPal has many narrow gates and eval JSONs, but promotion state is fragmented across tmp files.

- `openclaw/openclaw:docs/plugins/sdk-agent-harness.md` score=0.295 summary=Documentation section describing the Agent Harness plugin.
  Use when: When working with the SDK and need to understand how to use the Agent Harness plugin.
  Verify: Check if the sidebar title matches 'Agent Harness' and that the content provides accurate information about the plugin.
- `openclaw/openclaw:src/plugin-sdk/agent-harness.ts` score=0.292 summary=A TypeScript configuration file for the plugin SDK's agent harness.
  Use when: When setting up or modifying the agent harness in the plugin SDK.
  Verify: Check that the configuration matches the expected behavior of the agent harness.
- `openclaw/openclaw:docs/plugins/sdk-agent-harness.md` score=0.222 summary=Provides guidance on terminal outcome classification in SDK agent harness pairing.
  Use when: When setting up or troubleshooting SDK agent harness pairing to understand terminal outcomes.
  Verify: Check if examples and explanations align with actual implementation.
- `openclaw/openclaw:docs/plugins/sdk-agent-harness.md` score=0.219 summary=A documentation section explaining the selection policy for the SDK agent harness.
  Use when: When working with the SDK agent harness and need to understand its selection policy.
  Verify: Check if the selection policy aligns with the intended behavior of the SDK agent harness.
- `openclaw/openclaw:docs/plugins/sdk-agent-harness.md` score=0.210 summary=Registers a harness in the SDK agent.
  Use when: When setting up or configuring the SDK agent to include a new harness.
  Verify: Check if the harness registration process is correctly documented and follows the expected steps.

## fallback_repair

PocketPal signal: PocketPal repairs malformed JSON and falls back when active-agent decisions fail quality checks.

- `NousResearch/hermes-agent:tests/run_agent/test_repair_tool_call_name.py` score=0.296 summary=Tests mixed separators and suffix in repair tool call name.
  Use when: testing repair functionality
  Verify: check if test passes with expected output
- `openclaw/openclaw:docs/plugins/sdk-agent-harness.md` score=0.277 summary=Documentation section describing the Agent Harness plugin.
  Use when: When working with the SDK and need to understand how to use the Agent Harness plugin.
  Verify: Check if the sidebar title matches 'Agent Harness' and that the content provides accurate information about the plugin.
- `NousResearch/hermes-agent:tests/run_agent/test_repair_tool_call_name.py` score=0.275 summary=TestClassLikeEmissions is a class in the tests/run_agent/test_repair_tool_call_name.py module of the NousResearch/hermes-agent repository.
  Use when: When working with classes in Python codebases.
  Verify: Check if the class has expected methods and attributes.
- `openclaw/openclaw:src/plugin-sdk/agent-harness.ts` score=0.274 summary=A TypeScript configuration file for the plugin SDK's agent harness.
  Use when: When setting up or modifying the agent harness in the plugin SDK.
  Verify: Check that the configuration matches the expected behavior of the agent harness.
- `NousResearch/hermes-agent:tests/run_agent/test_repair_tool_call_name.py` score=0.272 summary=Tests existing behavior still works
  Use when: bug_fix feature_addition
  Verify: run tests locally

## browser_runtime

PocketPal signal: PocketPal exports browser BitNet WebGPU/WASM bundles and loads them through a worker.

- `openclaw/openclaw:docs/gateway/local-model-services.md` score=0.289 summary=Provides instructions for starting local model servers on demand before OpenClaw model requests.
  Use when: When setting up or troubleshooting local model services in the OpenClaw project.
  Verify: Check if the local model servers start successfully before making model requests.
- `NousResearch/hermes-agent:website/docs/user-guide/features/fallback-providers.md` score=0.289 summary=Provides examples of using a local model as a fallback for cloud models.
  Use when: When setting up fallback providers in a system that uses both cloud and local models.
  Verify: Check if the local model responds correctly to test inputs. Ensure network access to the cloud model for comparison.
- `openclaw/openclaw:docs/gateway/local-models.md` score=0.243 summary=Provides instructions on setting up a hybrid configuration using LM Studio and a large local model for handling responses via the API.
  Use when: When setting up a hybrid model configuration for response generation using LM Studio and a local model.
  Verify: Test the hybrid configuration by sending sample requests through the API. Check if the local model falls back correctly when the hosted model fails.
- `openclaw/openclaw:src/secrets/runtime-manifest.runtime.ts` score=0.240 summary=A TypeScript configuration file containing runtime manifest settings.
  Use when: when setting up or configuring the runtime environment of the project.
  Verify: check if the changes in the runtime manifest align with the expected behavior. run tests that depend on the runtime configuration.
- `openclaw/openclaw:docs/gateway/local-model-services.md` score=0.238 summary=Provides configuration details for local model services in the gateway.
  Use when: Developing or deploying local model services in the gateway.
  Verify: Check if the configuration matches expected values. Test connectivity and response time.

## self_improvement

PocketPal signal: PocketPal has failure replay datasets, but no explicit closed-loop skill credit assignment for its controller decisions.

- `NousResearch/hermes-agent:tests/tools/test_skills_hub_clawhub.py` score=0.416 summary=Tests the handling of nested skill payloads in ClawHubSource.
  Use when: developing and testing the ClawHubSource class
  Verify: run the test locally using pytest
- `NousResearch/hermes-agent:tests/tools/test_skills_hub_clawhub.py` score=0.406 summary=Tests searching for repairs in a poisoned cache using an exact slug lookup.
  Use when: Developing or testing cache-related features in the ClawHubSource module.
  Verify: Check that the test passes when expected and fails when cache behavior is altered.
- `NousResearch/hermes-agent:tests/tools/test_skills_hub_clawhub.py` score=0.382 summary=Tests that search falls back to exact slug when search results are irrelevant.
  Use when: Developing or testing search functionality in ClawHubSource.
  Verify: Run tests locally and check if search results match expected behavior.
- `NousResearch/hermes-agent:tests/tools/test_skills_hub_clawhub.py` score=0.229 summary=Tests the ClawHubSource class in Hermes Agent's test suite.
  Use when: developing or testing the ClawHubSource class verifying network interactions in the Hermes Agent
  Verify: run tests locally and check for network activity verify that expected network requests are made
- `NousResearch/hermes-agent:tests/tools/test_skills_hub_clawhub.py` score=0.216 summary=Tests if searching with a space-separated query matches a hyphenated slug.
  Use when: Developing or testing search functionality in ClawHubSource.
  Verify: Check that the test passes with various query strings. Ensure the search function correctly converts spaces to hyphens.

## Concrete Next Changes

- Add a PocketPal promotion ledger that combines agent gates, direct prompt eval, retrieval top-1, intent confusion, browser parity, malformed count, and zero-pass tasks into one Parquet row per bundle.
- Add a runtime action-policy verifier that normalizes every model decision, rejects disallowed actions, expands safe slot placeholders, and routes invalid/high-entropy tasks to deterministic fallback or teacher-backed generation.
- Convert OpenClaw/Hermes skill retrieval examples into controller examples with action targets: retrieve_skill -> choose_operator -> choose_verifier -> generate constrained decision.
- Store successful and failed PocketPal turns as trace skills with retrieved skills, model decision, fallback reason, final output, and pass/fail credit assignment.
- Update the deployed web model bundle when a promoted bundle has policy/retrieval heads, and run browser parity before switching the app default.

## Deeper Gap Matrix

### 1. Trained Architecture Is Ahead Of The Shipped Browser Bundle

Latest trained `v201a` advertises:

- `retrieval_head_dim=128`
- `agent_policy_heads=True`
- `agent_intent_labels=18`
- replaced surfaces including `agent_intent_policy`, `controller_confidence_estimation`, `controller_ood_estimation`, `controller_verification_need_estimation`, and neural retrieval embeddings

The packaged browser model at `web/models/pocketpal_controller_100m_bitnet_dev/manifest.json` is sourced from:

`/data/agent_kernel_lite/artifacts/pocketpal_controller_100m_v10_from_v9/agentkernel_lite_encdec_manifest.json`

and its manifest reports:

- `retrieval_head_dim=None`
- `agent_policy_heads=False`
- `agent_intent_labels=None`
- graph support `retrieval_embeddings=False`

This is the largest practical gap. Even if training improves the controller heads, the active browser bundle cannot use them until the promoted bundle is exported and wired into the app.

### 2. Browser Worker Requests Intent Logits That The Packaged Runtime Does Not Expose

The worker calls `modelStackRuntime.agentIntentLogits(...)` and throws if that method is absent. The packaged runtime exposes retrieval embedding methods, but no `agentIntentLogits` implementation is present in `web/models/pocketpal_controller_100m_bitnet_dev/runtime/encdec_runtime.js`.

Relevant code:

- [llm-worker.js](/data/agent_kernel_lite/web/js/llm-worker.js:913) checks for `agentIntentLogits`.
- [encdec_runtime.js](/data/agent_kernel_lite/web/models/pocketpal_controller_100m_bitnet_dev/runtime/encdec_runtime.js:812) exposes retrieval embedding methods, but no intent/policy head runtime.

This means the UI path probably logs `agent intent unavailable` and falls back to regex/runtime heuristics. The model may look worse or less intelligent in the browser because the trained policy heads are not actually in the live loop.

### 3. The Runtime Is Already Compensating For Decoder Weakness

PocketPal's web runtime has a strong rule-based safety layer:

- malformed decision repair
- action allowlist
- extension metadata validation
- placeholder validation
- source-preservation checks
- retry with constrained decoder prefix
- deterministic fallback answer

Relevant code:

- [agent-kernel-app.js](/data/agent_kernel_lite/web/js/agent-kernel-app.js:3740) repairs malformed decision JSON.
- [agent-kernel-app.js](/data/agent_kernel_lite/web/js/agent-kernel-app.js:3822) decides whether an active-agent result needs fallback.
- [agent-kernel-app.js](/data/agent_kernel_lite/web/js/agent-kernel-app.js:5401) retries malformed active-agent decoding with a constrained prefix.
- [agent-kernel-app.js](/data/agent_kernel_lite/web/js/agent-kernel-app.js:5422) finalizes through fallback if the decoder output fails.

This is consistent with the OpenClaw/Hermes retrieval results: keep the small model in the action/control plane and enforce policy outside the decoder.

### 4. Training Supports Policy Heads, But Promotion Does Not Enforce Their Runtime Use

The training script supports:

- retrieval contrastive loss
- hard-negative retrieval
- policy-head regression targets
- intent-head classification
- intent contrastive loss

Relevant code:

- [train_agentkernel_lite_encdec.py](/data/agent_kernel_lite/scripts/train_agentkernel_lite_encdec.py:1099) computes policy-head losses for confidence, retrieval coverage, OOD, answer confidence, verification need, and action validity.
- [train_agentkernel_lite_encdec.py](/data/agent_kernel_lite/scripts/train_agentkernel_lite_encdec.py:1129) computes intent-head loss.
- [train_agentkernel_lite_encdec.py](/data/agent_kernel_lite/scripts/train_agentkernel_lite_encdec.py:1377) combines decoder, retrieval, policy, and intent losses.
- [seq2seq.py](/data/agentkernel/other_repos/model-stack/runtime/seq2seq.py:110) defines retrieval heads.
- [seq2seq.py](/data/agentkernel/other_repos/model-stack/runtime/seq2seq.py:117) defines policy heads.
- [seq2seq.py](/data/agentkernel/other_repos/model-stack/runtime/seq2seq.py:220) defines intent logits.

The missing piece is a promotion rule that proves those heads are exported, loaded, and used in the browser.

### 5. OpenClaw/Hermes Implication

The retrieval matches repeatedly point to:

- OpenClaw agent harness/plugin contracts
- tool-policy conformance
- approval runtimes
- terminal outcome classification
- Hermes tool registry and approval hooks
- Hermes self-improvement/background review traces
- persistent memory and session state

Mapped to PocketPal, that says the architecture should be:

1. `compile_turn`: deterministic active-agent contract, slots, memory, tool policy, retrieval policy
2. `classify`: tiny encoder intent/policy heads
3. `retrieve`: local skills/memory/docs when the policy head requests it
4. `decide`: constrained JSON decision, not broad natural language
5. `verify`: runtime policy checks plus task-specific content checks
6. `fallback`: deterministic answer or teacher-backed route if verification fails
7. `learn`: store trace as a skill row with credit assignment

The current implementation has pieces of this, but they are split across training, `/tmp` eval files, old web export, and frontend heuristics.

## Priority Fix Order

1. Export a promoted model bundle that includes retrieval, policy, and intent heads.
2. Add browser runtime support for `agentIntentLogits` and policy-head logits, or stop calling unavailable methods.
3. Make browser parity a hard promotion gate.
4. Add a single Parquet promotion ledger per bundle.
5. Convert OpenClaw/Hermes retrieval examples into controller traces with explicit action/policy/verifier labels.
6. Treat direct decoder text generation as a fallback-limited path, not the main intelligence layer.
