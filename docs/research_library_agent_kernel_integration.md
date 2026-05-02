# Research Library Integration For Agent Kernel

Status: opt-in research adapter implemented; retrieval clients still bounded/read-only  
Date: 2026-04-27

For the deeper continuous-learning and TOLBERT synchronization design, see
`docs/research_library_continuous_learning_architecture.md`.
For the full 1M-paper parquet source-of-truth and incremental code-span sync
plan, see `docs/research_library_full_corpus_tolbert_sync.md`.

## Goal

Connect the local research, repository, paper, algorithm, and model assets into `agent_kernel` as first-class retrieval and training resources. The objective is not just browsing. The objective is to let the kernel use the library before, during, and after coding work:

- before a task: route the request to relevant repositories, papers, algorithms, and prior skills;
- during a task: retrieve bounded, provenance-rich context as the planner and editor need it;
- after a task: write useful outcomes back to memory and use solved work to improve rankers, adapters, and curricula.

`/data/research_library` is not currently mounted as a single directory. The effective research-library roots on this machine are:

- `/data/repository_library`
- `/data/TOLBERT_BRAIN`
- `/data/arxiv`
- `/data/algorithms`
- `/data/algorithms_library`
- `/data/checkpoints`
- `/data/huggingface` and `/data/huggingface_cache`

## Available Coverage

| Asset | Local Path | Coverage | Immediate Use |
| --- | --- | ---: | --- |
| Repository graph library | `/data/repository_library/exports` | 184 export dirs; manifest reports 178 repos; 24G exports | repo selection, symbol/API grounding, source snippets, repo QA |
| Repo graph HF dataset | `PeytonT/repo_graph` | README reports 15,499,276 rows across configs | portable graph training and retrieval snapshots |
| Paper text HF dataset | `PeytonT/1m_papers_text` | README reports 1,000,000 full-text papers | paper retrieval, concept grounding, training corpora |
| Full 1M paper text parquet | `/arxiv/huggingface/paper_text_1m_dedup_v1` | 1,000,000 paper rows; 21G compressed parquet | source-of-truth paper text for retrieval and training |
| P1 full-paper chunk dataset | `/data/tmp/p1_full_paper_lm_hf_all_chunks` | 17,117,443 chunk examples; 965,038 kept papers; 35G parquet | full-text chunk training and chunk-level retrieval |
| Paper graph / paper universe | `/data/repository_library/exports/_paper_universe` | 1,000,000 paper nodes; metadata/full-text embeddings; 20,000,062 KNN edges | immediate 1M-paper search, neighborhood expansion, topic routing |
| TOLBERT joint code/paper brain | `/data/TOLBERT_BRAIN/data/joint_v2` | 101,118 code spans; 3,939,404 paper spans; 336,480 nodes; derived projection, not full source corpus | cross-domain hierarchical retrieval and routing |
| arXiv metadata | `/data/arxiv/arxiv-metadata-oai-snapshot.json` | 3,016,131 metadata rows; 23G local arXiv tree | metadata filters, paper provenance, PDF fetching |
| Algorithm catalog | `/data/algorithms` | 972 algorithms; 970 problem schemas | algorithm choice, constraints, complexity checks |
| Algorithm implementations | `/data/algorithms_library/Python` | 1,373 Python files | reference implementations and competitive programming patterns |
| Local model checkpoints | `/data/checkpoints` and `/data/repository_library/models` | paper embedding, keyword, world-model, verifier/coder adapters | retrieval embedding, keyword extraction, offline training |

## Existing Agent Kernel Hooks

The kernel already has several relevant surfaces:

- `agent_kernel/config.py` includes `use_tolbert_context`, `use_graph_memory`, `use_retrieval_proposals`, and TOLBERT runtime paths.
- `agent_kernel/extensions/tolbert.py` starts and queries the local TOLBERT service.
- `scripts/tolbert_service.py` inserts `/data/TOLBERT_BRAIN` into `sys.path` and loads `tolbert_brain.brain.TolbertBrain`.
- `agent_kernel/extensions/tolbert_assets.py` builds kernel-specific TOLBERT nodes, spans, and runtime bundles.
- `scripts/build_agentkernel_tolbert_assets.py` materializes those assets under `var/tolbert/agentkernel`.

This means the right path is incremental: add read-only research adapters first, then route their output into the existing context/retrieval/TOLBERT surfaces.

## Current Kernel Entrypoint

The first integration slice is now live and intentionally separate from the
benchmark browser:

- `config/research_library_sources.json` declares the local source registry.
- `agent_kernel/research_library/sources.py` builds source-level status records
  for the full paper corpus, chunk corpus, paper graph, repository exports,
  TOLBERT projection, algorithms, algorithm implementations, and checkpoints.
- `scripts/build_research_library_status.py` writes the runtime snapshot to
  `var/research_library/status.json`.
- `agent_kernel/research_library/models.py` exposes trained-model query helpers
  over that snapshot, so runtime code can select existing adapters/checkpoints
  by source, asset type, and group without retraining.
- `agent_kernel/research_library/context.py` wraps the kernel context provider
  and adds bounded research-library chunks into `ContextPacket.control` and the
  retrieval payload.
- `agent_kernel/extensions/runtime_modeling_adapter.py` now composes the
  research-library provider with the existing TOLBERT provider only when
  `AGENT_KERNEL_USE_RESEARCH_LIBRARY_CONTEXT=1`.

Adapter contract:

- The research library is off by default. `KernelConfig.use_research_library_context`
  defaults to false unless `AGENT_KERNEL_USE_RESEARCH_LIBRARY_CONTEXT=1`.
- The adapter is detachable. With the flag off, `build_context_provider()`
  returns the normal TOLBERT provider or no provider, and Agent Kernel continues
  without research-library context.
- Standalone research context also requires
  `AGENT_KERNEL_RESEARCH_LIBRARY_STANDALONE_CONTEXT=1`; otherwise the adapter
  only wraps an existing base context provider.
- The adapter must not modify prompt templates, policy logic, or core payload
  compaction. It emits bounded context chunks, retrieval evidence, and optional
  retrieval-guidance commands through the context packet only.

Current live status from `var/research_library/status.json`:

- 11/11 registered sources available, 0 partial, 0 missing.
- 1,000,000 full-paper rows in `/arxiv/huggingface/paper_text_1m_dedup_v1`.
- 17,117,443 full-paper chunk examples in `/data/tmp/p1_full_paper_lm_hf_all_chunks`.
- 1,000,000 paper graph rows, 20,000,062 paper KNN edges, and 3,000,000 paper-topic edges.
- 178 repository exports and 74,472 mined repository skills.
- 1,942 algorithm catalog rows and 1,373 Python implementation files.
- 35 trained model assets registered: 11 repository/paper LoRA adapters, 17
  digital-world-model assets, and 7 TOLBERT checkpoint/cache assets.
- Model asset types currently include 14 PEFT adapters, 10 HF full models,
  3 lightweight models, 1 FAISS index, 6 TOLBERT checkpoints, and 1 retrieval
  cache.

Runtime behavior:

- with the adapter flag off, no research-library provider is attached;
- with the adapter flag on and TOLBERT enabled, research-library context
  augments TOLBERT retrieval instead of replacing it;
- with `AGENT_KERNEL_RESEARCH_LIBRARY_STANDALONE_CONTEXT=1`, the research
  adapter can provide a context packet even when TOLBERT context is disabled;
- context chunks include source inventory, selected trained models, paper
  backbone paths/counts, relevant repository exports, and algorithm matches;
- paper-content hits can now be compiled into `research:applied_guidance`
  chunks. These are concise constraints derived from retrieved paper text, not
  title/id matching;
- for exact-field probes, the adapter may emit compact key/value evidence and a
  matching `recommended_command_spans` entry in the context packet. This remains
  adapter data: disconnecting the adapter removes the behavior entirely;
- model routing diversifies by model family so the kernel sees TOLBERT,
  paper/repo adapters, coder adapters, and memory assets rather than many
  checkpoints from one family.

Current content-probe status:

- Earlier leftist-window probes were useful diagnostics but are not clean
  evidence: they were too narrow, and one run depended on a direct core repair
  path that has been removed.
- The cleaner `farad_table3` probe uses body/table content from paper
  `1003.1582` and a hidden verifier script so the task contract does not leak
  exact values.
- Adapter-only run
  `var/research_library/hard_farad_table3_probe_report_r10_adapter_only_exact_chunks.json`
  retrieved the correct table values with research on, while the no-research
  variant produced a generic summary. It still failed because the live model
  wrote `/workspace/...` heredoc/Python commands instead of a relative
  `answer.txt` field file.
- A temporary core payload-compaction change did make the same probe pass, but
  that change was reverted. Current evidence says the adapter is improving
  content grounding, while execution-control reliability remains a kernel/model
  issue outside the standalone adapter contract.

This is the intended role split: TOLBERT should increasingly handle semantic
selection and reranking over the paper/repository corpus, while the research
adapter compiles selected spans into kernel-usable constraints. The current
live TOLBERT research checkpoint/cache is still the older bundle under
`/data/TOLBERT_BRAIN/checkpoints/tolbert_brain`; the 3.9M-span joint-v2
projection exists, but `/data/TOLBERT_BRAIN/checkpoints/tolbert_brain_joint_v2`
is not present yet, so joint-v2 is not the active full-corpus TOLBERT runtime.

## Adapter Architecture

Add a small research-library package under `agent_kernel/research_library/` with read-only clients and a context provider.

Recommended modules:

- `sources.py`: validates local paths, reports counts and freshness, and loads `config/research_library_sources.json`.
- `repository_client.py`: reads `/data/repository_library/exports/_manifest.json`, per-repo graph exports, QA indexes, source snippets, and optional MCP tools.
- `paper_client.py`: queries paper universe metadata, topic/category/year nodes, KNN edges, and embedding-backed candidates.
- `algorithm_client.py`: loads `/data/algorithms/algorithms.jsonl`, `/data/algorithms/problems.jsonl`, and implementation snippets from `/data/algorithms_library/Python`.
- `models.py`: reads the generated source snapshot and exposes existing trained
  model assets for runtime selection before any retraining path is considered.
- `context.py`: composes source inventory, trained-model routing, repo export
  hints, and algorithm matches into bounded context chunks.
- `tolbert_bridge.py`: converts repository, paper, and algorithm candidates into TOLBERT-compatible spans or source-span paths.
- `context_provider.py`: future deeper retrieval clients for source-grounded
  snippets; the current `context.py` module already provides the first runtime
  packet integration.

The context provider should return structured records, not plain text blobs:

```json
{
  "source_type": "paper|repository|algorithm|model",
  "source_id": "stable id",
  "title": "human readable title",
  "rank": 0,
  "score": 0.0,
  "summary": "short synthesized summary",
  "evidence": [
    {
      "path": "/data/...",
      "locator": "line/span/node/commit/page",
      "snippet": "bounded excerpt"
    }
  ],
  "provenance": {
    "dataset": "local path or HF id",
    "adapter": "adapter name",
    "retrieved_at": "ISO timestamp"
  }
}
```

## Integration Phases

### Phase 0: Source Registry And Health Checks

Implemented. `config/research_library_sources.json` is a structured list of
source descriptors, not a flat root map. `scripts/build_research_library_status.py`
produces `var/research_library/status.json` as a cheap read-only snapshot with
path existence, file counts, dataset sizes, manifest versions, row metrics, and
missing required files.

### Phase 1: Read-Only Retrieval Clients

Implement direct clients with no training and no mutation:

- repository lookup by task text, repo language, repo name, symbol name, artifact path, and QA index;
- paper lookup by title/abstract/fulltext embeddings, categories, year, topic, and KNN neighborhood;
- algorithm lookup by problem description, constraints, tags, and complexity.

Do not load the full paper corpus or all TOLBERT spans into process memory. Use metadata-first search, memory-mapped arrays where available, parquet column projection, and top-k limits.

### Phase 2: Kernel Context Provider

Implemented as a wrapper around the existing context-provider factory. Config
fields:

- `AGENT_KERNEL_USE_RESEARCH_LIBRARY_CONTEXT`
- `AGENT_KERNEL_RESEARCH_LIBRARY_STANDALONE_CONTEXT`
- `AGENT_KERNEL_RESEARCH_LIBRARY_CONFIG_PATH`
- `AGENT_KERNEL_RESEARCH_LIBRARY_STATUS_PATH`
- `AGENT_KERNEL_RESEARCH_LIBRARY_CONTEXT_MAX_CHUNKS`
- `AGENT_KERNEL_RESEARCH_LIBRARY_CONTEXT_MAX_MODELS`
- `AGENT_KERNEL_RESEARCH_LIBRARY_CONTEXT_MAX_REPOSITORIES`
- `AGENT_KERNEL_RESEARCH_LIBRARY_CONTEXT_MAX_ALGORITHMS`

Context budgets should be strict. The planner should get compact candidates. The editor should only get source-grounded snippets when it asks for a specific implementation detail. The verifier should get constraints, paper claims, algorithm complexity, and repository-specific test conventions.

### Phase 3: TOLBERT Bridge

Use the existing TOLBERT integration instead of building a second brain. There are two viable modes:

- mount mode: add `/data/TOLBERT_BRAIN/data/joint_v2` spans, nodes, label maps, and cache paths as external runtime sources;
- merge mode: build a kernel-specific bundle that keeps local Agent Kernel spans and links them to external code/paper/algorithm spans.

Mount mode is the first target because it is reversible and lower risk. Merge mode is better later for training and retained memory.

The bridge should use TOLBERT for:

- hierarchical route prediction from task text to L1-L5 concepts;
- cross-domain retrieval from repository code spans to papers and back;
- candidate expansion when a benchmark task mentions a concept but not a repository-specific API;
- curriculum generation by finding nearby problems, papers, and implementations.

### Phase 4: Training And Ranker Updates

Training is not the first action. The source registry now exposes existing
trained assets, so runtime adapters should first use the local model catalog:

- repository/paper PEFT adapters under `/data/repository_library/models/checkpoints`;
- planner, verifier, actuator, coder, memory, and world-model assets under
  `/data/checkpoints/digital_world_model`;
- TOLBERT epochs and retrieval cache under `/data/TOLBERT_BRAIN/checkpoints/tolbert_brain`.

Use `trained_model_asset_catalog()` or `iter_trained_model_assets()` from
`agent_kernel.research_library` to choose these assets. Schedule retraining only
when an asset is missing, incompatible with a newer corpus generation, or loses
against the current baseline.

After read-only retrieval and existing-model routing are stable, use solved
kernel traces as labels:

- positive examples: context items referenced in successful patches, plans, tests, or verifier decisions;
- hard negatives: retrieved items ignored in failed attempts;
- outcome labels: patch accepted, tests pass, benchmark resolved, verifier caught issue.

Candidate training targets:

- repo/paper/algorithm reranker;
- TOLBERT route calibration for benchmark tasks;
- paper keyword extractor refresh using `PeytonT/1m-papers-abstract-keywords`;
- repository QA adapter refresh for high-value repos;
- curriculum generator for SWE-Bench, RE-Bench, MLE-Bench, and Codeforces-style tasks.

### Phase 5: Autonomous Usage

Once adapters are stable, make the kernel use them automatically:

- SWE-Bench: identify relevant upstream repo patterns, failing-test conventions, historical API behavior, and neighboring issue fixes.
- SWE-ReBench: use repo graphs and source snippets for robust patch localization without overfitting to fixed task ids.
- RE-Bench: retrieve systems, ML, and optimization papers plus reference code from the repository graph.
- MLE-Bench: retrieve model, feature engineering, metric, and competition-pattern papers while keeping Kaggle data access credential-gated.
- Codeforces/CodeContests: retrieve algorithm candidates, constraints, complexity checks, and implementation templates.

## Repository Library Adapter Details

The repository library already exposes the strongest immediate adapter. Its docs describe an MCP server:

```bash
PYTHONPATH=/data/repository_library /home/peyton/miniconda3/envs/ai/bin/python -m scripts.repo_library_mcp
```

Useful tools include:

- `list_repos`
- `get_repo`
- `get_repo_profile`
- `list_repo_skills`
- `build_repo_skill`
- `repository_qa`
- `find_relevant_repos`
- `get_repo_graph`
- `get_source_snippet`
- `search_arxiv_papers`
- `download_arxiv_pdfs`
- `search_algorithms`
- `get_algorithm_problem`
- `get_algorithm_implementations`

Recommended kernel flow:

1. `find_relevant_repos` for task routing.
2. `get_repo_profile` for repo-level conventions and available indices.
3. `list_repo_skills` and `build_repo_skill` when a repo-specific skill exists or can be built cheaply.
4. `repository_qa` for targeted questions.
5. `get_repo_graph` and `get_source_snippet` for concrete code grounding.

For the first implementation, prefer direct file reads for status and metadata, then call the MCP tools only for query operations where the tool already implements ranking or snippet extraction.

## Paper Adapter Details

The paper universe should be used for retrieval, not raw context stuffing. High-value features:

- paper metadata and topic/category/year nodes;
- title/abstract/fulltext embeddings;
- KNN edges for neighborhood expansion;
- paper-to-topic and paper-to-category edges;
- public HF dataset ids for portable rebuilds.

Paper candidates should include title, authors/year if available, abstract summary, category/topic, score, and a short source locator. Full paper text should only be pulled when the task requires it.

## Algorithm Adapter Details

The algorithm library is small enough to load fully. The adapter should map task text to:

- algorithm id and aliases;
- problem id;
- constraints and invalid-use warnings;
- expected time and space complexity;
- implementation files from `/data/algorithms_library/Python`;
- testable examples when available.

This is especially useful for Codeforces/CodeContests and for verifier checks where an implementation violates known constraints.

## Contamination And Benchmark Safety

Benchmark progress needs clean evidence. The adapters should log every retrieved source into the run artifact:

- source type;
- source id;
- dataset/path;
- retrieval query;
- top-k rank;
- whether the item was injected into the model context.

For benchmark tasks, the adapter must not retrieve target patches, evaluator answers, hidden tests, or generated predictions as context. Public repository code, public papers, public docs, public algorithms, and task-visible problem statements are acceptable. Benchmark-specific caches should be separated from general research-library caches.

## Recommended First PR

Keep the first PR deliberately narrow:

1. Add `config/research_library_sources.json`.
2. Add `agent_kernel/research_library/sources.py`.
3. Add `scripts/build_research_library_status.py`.
4. Add tests with tiny temporary fixtures.
5. Document the source inventory and health-check command.

Expected output:

```bash
python scripts/build_research_library_status.py \
  --config config/research_library_sources.json \
  --output var/research_library/status.json
```

The status JSON should be visible in the benchmark browser later, but it should not block benchmark execution.

## Recommended Second PR

Implement query clients and a CLI:

```bash
python scripts/query_research_library.py \
  --query "optimize sparse matrix multiplication in scipy" \
  --repos 5 \
  --papers 5 \
  --algorithms 5
```

The output should be structured JSON plus a human-readable summary.

## Recommended Third PR

Wire `context_provider.py` into the planner/editor/verifier path behind `AGENT_KERNEL_RESEARCH_LIBRARY_ENABLED=1`. Start with metadata candidates only. Add snippet injection only when a subcomponent requests details for a selected candidate.

## Recommended Fourth PR

Mount TOLBERT external spans and cache paths from `/data/TOLBERT_BRAIN/data/joint_v2`. Measure:

- retrieval latency;
- context token growth;
- accepted patch rate;
- failed-test localization accuracy;
- benchmark contamination log completeness.

Only after those metrics are stable should we merge external spans into a retained kernel-specific TOLBERT bundle.

## Definition Of Done

The integration is useful when a benchmark or coding run can show:

- a source health snapshot from the research library;
- top-k repository, paper, and algorithm candidates for the task;
- bounded snippets with provenance;
- retrieval logs attached to the run artifact;
- no benchmark contamination;
- measurable improvement in patch localization, planning accuracy, or verifier catches.
