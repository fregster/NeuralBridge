# God-Class Refactor Plan

## Background

Following the adoption of new class-design standards (see [docs/CODING_STANDARDS.md](../CODING_STANDARDS.md)),
a codebase audit identified ten source files exceeding the 500-line soft limit and eight concrete classes
that lack a `Protocol` or `ABC` interface. This document describes the staged plan to bring every file into
compliance, ordered from lowest to highest effort so value is delivered continuously.

### Standards being enforced

| Rule | Limit |
|------|-------|
| Class file size (soft limit) | 500 lines |
| Public methods per class | 15 |
| Instance variables per class | 10 |
| Single-sentence class responsibility | Required |
| Framework-First (Protocol/ABC before concrete class) | Required for all new classes |

### Current violation summary

| File | Lines | Protocol? | Priority |
|------|-------|-----------|----------|
| `config_flow.py` | 2,630 | ❌ | 🔴 Critical |
| `router_engine.py` | 1,129 | ❌ | 🟠 High |
| `agent_benchmark.py` | 958 | ❌ | 🟠 High |
| `pipeline_executor.py` | ~~687~~ **584** ✅ | ✅ | ✅ Stage 2c done |
| `guard_rail.py` | ~~649~~ **452** ✅ | ✅ | ✅ Stage 2b done |
| `confirmation_flows.py` | ~~613~~ **253** ✅ | ✅ | ✅ Stage 2a done |
| `benchmark_probes.py` | ~~559~~ **< 200** ✅ | ✅ | ✅ Stage 1a done |
| `conversation.py` | ~~557~~ **435** ✅ | ✅ | ✅ Stage 2d done |
| `web_search_client.py` | ~~535~~ **< 350** ✅ | ✅ | ✅ Stage 1b done |
| `llm_agent_proxy.py` | ~~526~~ **515** ✅ | ✅ | ✅ Stage 1c done |
| `agent_backend.py` | 497 | ✅ | ⚠️ Approaching |
| `response_cache.py` | 414 | ✅ | ✅ Stage 1d done |

---

## Stage 1 — Quick Wins ✅ *Complete*

**Estimated effort**: 2–3 days
**Risk**: Low — no architectural changes, no new public interfaces
**Goal**: Eliminate all "just over" violations (526–559 lines) and add missing Protocols to already-simple
classes. Deliver immediate compliance on the easiest cases while building team habits around the
Framework-First pattern.

### 1a. `benchmark_probes.py` (559 → < 200) ✅

`BenchmarkProbe` is a single large dataclass populated with many probe definitions. The file is oversized
because all probe payloads are inlined rather than grouped.

**Action**: Extract probe definitions into a `probes/` sub-package:
- `probes/__init__.py` — re-exports `BenchmarkProbe`
- `probes/ollama_probes.py` — Ollama-specific probe instances
- `probes/integration_probes.py` — HA integration probe instances
- Keep `benchmark_probes.py` as a thin façade that imports from the sub-package, or delete and rename

No interface change required; the existing `BenchmarkProbe` dataclass is already the correct abstraction.

### 1b. `web_search_client.py` (535 → < 350) ✅

`WebSearchProvider(Protocol)` already exists ✅. The file is over-limit because three provider
implementations (`BraveSearchProvider`, `BraveAnswersProvider`, `BraveAnswersCombinedProvider`) and the
`WebSearchClient` orchestrator all live in one file.

**Action**: Split into:
- `web_search_client.py` — keep `WebSearchClient` + `WebSearchProvider` Protocol only
- `providers/brave_search.py` — `BraveSearchProvider`
- `providers/brave_answers.py` — `BraveAnswersProvider`
- `providers/brave_combined.py` — `BraveAnswersCombinedProvider`

No interface changes. `WebSearchClient` already depends on the Protocol, so the providers can move without
changing call sites.

### 1c. `llm_agent_proxy.py` (526 → 515) ✅

`LLMAgentProxy` mixes two concerns: prompt construction (system prompt, entity context injection, user text
enrichment) and transport dispatch (`_process_ollama`, `_process_integrated`).

**Action**:
- Introduce `LLMAgentProxyProtocol(Protocol)` with a single `process(user_input) -> ConversationResult`
  method
- Extract `PromptEnricher` (the four `_build_*` / `_render_*` helpers) into `prompt_enricher.py`
- `LLMAgentProxy` slims to construction + delegation to `PromptEnricher` + two transport methods

### 1d. Add missing Protocol to `response_cache.py` (414) ✅

`response_cache.py` has no interface. Before it crosses the threshold, introduce:
- `ResponseCacheProtocol(Protocol)` — `get`, `put`, `invalidate`, `clear`
- Concrete `ResponseCache` implements it unchanged

---

## Stage 2 — Focused Decompositions ✅ *Complete*

**Estimated effort**: 4–6 days
**Risk**: Low–moderate — files are split along clear seams, mostly self-contained
**Goal**: Bring the three 600–690 line moderate violations into compliance and introduce missing Protocols
for the mid-tier classes. Also tighten `conversation.py` which is currently a thin pass-through class whose
methods technically belong to delegated components.

### 2a. `confirmation_flows.py` (613 → 253) ✅

`ConfirmationFlows` handles three distinct concerns: guard rail action dispatch, high-stakes confirmation
resolution, and preference confirmation/suggestion. Each concern has a clear natural boundary.

**Action**: Introduce `ConfirmationHandlerProtocol(Protocol)` with `handle(user_input) -> ConversationResult`.
Extract into:
- `confirmation_flows.py` — `ConfirmationFlows` as a thin coordinator (< 150 lines)
- `guard_rail_action_handler.py` — `_apply_guard_rail_action`, `_check_guardrails` logic
- `high_stakes_handler.py` — `_check_high_stakes`, `_resolve_high_stakes_confirmation`
- `preference_confirmation_handler.py` — `_resolve_preference_confirmation`, `_analyse_and_suggest`,
  `_fire_preferences_updated`

### 2b. `guard_rail.py` (649 → 452) ✅

The file contains three distinct types: `GuardRailChecker` (the main logic), `GuardRailCache`, and
`HighStakesCache`. The two cache classes are clearly separable.

**Action**:
- Introduce `GuardRailCheckerProtocol(Protocol)` with `check_input` / `check_output`
- Move `GuardRailCache` and `HighStakesCache` to `guard_rail_cache.py`
- Split `GuardRailChecker` check backends into:
  - `_check_with_profanity` + `_check_with_detoxify` + `_evaluate_detoxify_scores` → `toxicity_checker.py`
  - `_check_with_rules` → inline in `GuardRailChecker` (small)
  - `_check_with_ai` → `ai_safety_checker.py`
- `guard_rail.py` retains `GuardRailChecker` as an orchestrator only

### 2c. `pipeline_executor.py` (687 → 584) ✅

`PipelineExecutor` mixes orchestration (the pipeline loop) with retry mechanics and individual agent
dispatch. These have different rates of change.

**Action**:
- Introduce `PipelineExecutorProtocol(Protocol)` with a single `execute(user_input) -> ConversationResult`
- Extract retry/tracking logic into `agent_retry_engine.py` — `AgentRetryEngine` handling
  `_try_agent_with_retries`, `_try_agent_with_tracking`, `_record_agent_failure`
- Extract compound fragment processing into `fragment_processor.py` — `FragmentProcessor`
- `PipelineExecutor` becomes the orchestrator: owns `_run_pipeline` and delegates to sub-components

### 2d. `conversation.py` (557 → 435) ✅

`NeuralBridgeAgent` currently acts as a god-hub that re-declares ~40 methods which are thin delegations to
`PipelineExecutor`, `RouterEngine`, `ConfirmationFlows`, etc. It is a HA framework requirement that this
class exists, but it should contain *only* HA integration boilerplate and thin delegation.

**Action**:
- Audit every method: any method with a body > 3 lines that is also present in a delegated component is
  a delegation that was never completed
- Remove duplicated method bodies; each should be a single `return await self._executor.method()` call or
  equivalent
- Target: `NeuralBridgeAgent` owns lifecycle (`__init__`, `async_will_remove_from_hass`,
  `async_process`, `_create_result`, `_create_error_result`) plus HA property stubs only (< 200 lines)
- Introduce `ConversationAgentProtocol(Protocol)` mirroring the HA `ConversationEntity` surface for
  testability

---

## Stage 3 — Architectural Splits ✅ COMPLETE

**Completed**: All new files created, 1295 tests pass, `make check` green at 100% coverage.

**Estimated effort**: 6–10 days
**Risk**: Moderate — requires new `Protocol` contracts, affects multiple call sites, tests need updating
**Goal**: Decompose the two large (950–1,130 line) subsystems using the Strategy and Backend patterns.
Each subsystem gets a proper Protocol, and each concrete transport/strategy gets its own file.

### 3a. `agent_benchmark.py` (958 → target < 300 per file)

The file contains data models, a profile store, a benchmarker orchestrator, a scheduler, probe runners,
and telemetry update logic — five separate responsibilities.

**Proposed split**:

| New file | Contents | Target lines |
|----------|----------|--------------|
| `benchmark_models.py` | `ScoreBreakdown`, `BenchmarkStatus`, `ProbeResult`, `BenchmarkProfile` (incl. `to_dict`/`from_dict`) | ~250 |
| `benchmark_profile_store.py` | `BenchmarkProfileProtocol(Protocol)` + profile CRUD methods extracted from `AgentBenchmarker` | ~150 |
| `benchmark_scheduler.py` | `async_schedule_benchmark`, `cancel_pending`, `cancel_all_pending` | ~100 |
| `benchmark_probe_runner.py` | `_run_ollama_probes`, `_run_single_ollama_probe`, `_run_existing_probes`, `_get_ollama_client` | ~200 |
| `benchmark_telemetry.py` | `update_realworld_telemetry`, `_apply_probe_results`, `_check_version` | ~150 |
| `agent_benchmark.py` | `AgentBenchmarkerProtocol(Protocol)` + `AgentBenchmarker` as thin orchestrator | ~150 |

### 3b. `router_engine.py` (1,129 → target < 300 per file)

`RouterEngine` contains: a rich `RouterDecision` value object, backend dispatch (Ollama router, existing
HA agent), decision parsing, fallback logic, stats dispatch, and logging. The backends are interchangeable
by design — the Strategy pattern is the natural fit.

**Proposed split**:

| New file | Contents | Target lines |
|----------|----------|--------------|
| `router_decision.py` | `RouterDecision` dataclass + `__eq__`, `__hash__`, `__repr__`, parsing helpers | ~150 |
| `router_backend.py` | `RouterBackendProtocol(Protocol)` — `call(config, prompt) -> str \| None` | ~30 |
| `router_backends/ollama_router_backend.py` | `OllamaRouterBackend` | ~60 |
| `router_backends/integrated_router_backend.py` | `IntegratedRouterBackend` (wraps `_call_existing_agent_for_routing`) | ~80 |
| `router_engine.py` | `RouterEngine` — orchestration, fallback, stats dispatch, logging only | ~250 |

The `RouterDecision` line 74–717 gap (which is currently non-class utility code / constants) should also
be reviewed during this stage and any utility functions extracted to a `router_utils.py`.

---

## Stage 4 — God Class Surgery

**Estimated effort**: 10–15 days
**Risk**: High — the largest single refactor in the codebase; touches every config UI flow and all tests
**Goal**: Decompose `NeuralBridgeOptionsFlowHandler` (2,630 lines, ~45 methods) into focused, single-
responsibility step handler classes. The HA config flow architecture constrains how steps are registered,
so the decomposition uses a *Mixin* pattern rather than full separate classes to remain compatible with
`config_entries.OptionsFlow`.

### Current step inventory by concern

| Concern | Steps |
|---------|-------|
| Bootstrap / navigation | `init`, `done`, `back_to_main`, `user` |
| Agent management (add/edit/delete) | `add_agent`, `manage_agents`, `confirm_delete_agent`, `configure_ollama`, `configure_integrated`, `configure_local`, `configure_routing_agent`, `edit_agent_ollama`, `edit_agent_integrated`, `edit_agent_local` |
| Web search configuration | `configure_web_search`, `edit_agent_web_search` |
| Guard rails | `configure_guard_rails`, `guard_rails_settings`, `configure_guard_rail_rules`, `configure_high_stakes` |
| Advanced / settings | `advanced_settings`, `default_prompt`, `language_settings` |
| Benchmarks | `run_benchmarks_now` |

### Proposed decomposition

**Approach**: Extract each concern into a mixin module. `NeuralBridgeOptionsFlowHandler` inherits from all
mixins and provides only navigation bootstrap and the shared state (`self.options`, `self._lang`). Because
HA resolves `async_step_*` methods via `getattr`, the mixin approach works transparently.

| New file | Mixin class | Steps | Target lines |
|----------|-------------|-------|--------------|
| `config_flow_agent_steps.py` | `AgentStepsMixin` | `add_agent`, `manage_agents`, `confirm_delete_agent`, `configure_ollama`, `configure_integrated`, `configure_local`, `configure_routing_agent`, `edit_agent_*` | ~650 |
| `config_flow_web_search_steps.py` | `WebSearchStepsMixin` | `configure_web_search`, `edit_agent_web_search` + validation helpers | ~350 |
| `config_flow_guard_rail_steps.py` | `GuardRailStepsMixin` | `configure_guard_rails`, `guard_rails_settings`, `configure_guard_rail_rules`, `configure_high_stakes` + parse helpers | ~350 |
| `config_flow_advanced_steps.py` | `AdvancedStepsMixin` | `advanced_settings`, `default_prompt`, `language_settings` | ~250 |
| `config_flow_benchmark_steps.py` | `BenchmarkStepsMixin` | `run_benchmarks_now`, `_build_benchmark_status_table` | ~150 |
| `config_flow.py` | `NeuralBridgeConfigFlow` + `NeuralBridgeOptionsFlowHandler` (inherits all mixins) | `init`, `done`, `back_to_main`, shared helpers (`_lang`, `_s`, `_migrate_legacy_router_agents`) | ~200 |

**Introduce `OptionsFlowProtocol(Protocol)`** for the shared surface (`options`, `hass`, `_lang`) so each
mixin can be type-checked independently without importing the concrete handler class.

### Migration strategy

Because this is the highest-risk stage, it should be executed in sub-steps, each committed and passing all
tests before proceeding:

1. Extract `BenchmarkStepsMixin` first (smallest, self-contained)
2. Extract `WebSearchStepsMixin`
3. Extract `GuardRailStepsMixin`
4. Extract `AdvancedStepsMixin`
5. Extract `AgentStepsMixin` (largest, most complex)
6. Final cleanup: `config_flow.py` becomes the thin host

---

## Cross-Cutting Work (all stages)

These tasks span multiple stages and should be tracked separately:

- [ ] Add `Protocol` stubs for `GuardRailChecker`, `ConfirmationFlows`, `PipelineExecutor`,
  `RouterEngine`, `AgentBenchmarker`, `NeuralBridgeAgent`, `LLMAgentProxy`, `ResponseCache`
  (stub files can be created at the start of each stage, concrete classes updated at end)
- [ ] Ensure every new stub Protocol has a corresponding in-memory fake in `tests/fixtures/`
- [ ] Maintain 100% test coverage throughout — run `make check` after every sub-step
- [ ] Update `docs/STRUCTURE.md` to reflect the new file layout after each stage completes
- [ ] Update `CLAUDE.md` and `.github/.copilot-instructions.md` if any agent-type or routing
  concepts are renamed during the refactor

---

## Completion Checklist

- [x] Stage 1 complete — all "just over" files under 500 lines
- [x] Stage 2 complete — all moderate violations resolved, `conversation.py` is delegation-only
- [x] Stage 3 complete — `router_engine.py` and `agent_benchmark.py` decomposed with Protocols
- [ ] Stage 4 complete — `config_flow.py` decomposed into mixin modules
- [ ] Zero files exceed 500 lines without documented justification
- [ ] Every concrete class has a preceding `Protocol` or `ABC`
- [ ] `make check` passes at 100% coverage after every stage
