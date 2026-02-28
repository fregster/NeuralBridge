# NeuralBridge Directory Structure

## Root Directory

```
NeuralBridge/
├── .github/
│   ├── .copilot-instructions.md     Project development guidelines
│   └── workflows/
│       ├── test.yml                 Testing pipeline
│       └── validate.yml             HACS/Hassfest validation
├── custom_components/
│   └── neuralbridge/                Integration package
│       ├── __init__.py              Integration setup and teardown
│       ├── manifest.json            HACS and Home Assistant metadata
│       ├── const.py                 All constants, defaults, and prompt templates
│       │
│       ├── ── Core ──
│       ├── conversation.py          NeuralBridgeAgent — HA lifecycle + thin delegation
│       ├── config_flow.py           UI configuration flows
│       ├── sensor.py                Home Assistant statistics sensors
│       │
│       ├── ── Pipeline ──
│       ├── pipeline_executor.py     PipelineExecutor — run-pipeline orchestration
│       ├── agent_retry_engine.py    AgentRetryEngine — circuit-breaker + retry + failure tracking
│       ├── fragment_processor.py    FragmentProcessor — compound multi-fragment splitting
│       ├── router_engine.py         RouterEngine — thin routing orchestrator (delegation only)
│       ├── router_decision.py       RouterDecision — value object + parsing helpers
│       ├── router_backend.py        RouterBackendProtocol — structural Protocol for backends
│       ├── router_utils.py          Router utility / helper functions
│       ├── router_backends/         Concrete router backend implementations
│       │   ├── __init__.py
│       │   ├── ollama_router_backend.py     OllamaRouterBackend — direct-HTTP Ollama backend
│       │   └── integrated_router_backend.py IntegratedRouterBackend — HA conversation-service backend
│       ├── llm_agent_proxy.py       LLMAgentProxy — INTEGRATED and OLLAMA transport dispatch
│       ├── prompt_builder.py        PromptBuilder — system/user prompt construction
│       ├── agent_backend.py         AgentBackend — generic agent dispatch helpers
│       │
│       ├── ── Confirmation & Guard Rails ──
│       ├── confirmation_flows.py    ConfirmationFlows — high-stakes + preference coordinator
│       ├── high_stakes_handler.py   HighStakesHandler — dangerous-command confirmation
│       ├── preference_confirmation_handler.py  PreferenceConfirmationHandler — preference suggestions
│       ├── guard_rail_action_handler.py        GuardRailActionHandler — block/warn/log dispatch
│       ├── guard_rail.py            GuardRailChecker — four-stage safety pipeline orchestrator
│       ├── guard_rail_types.py      GuardRailResult dataclass and shared types
│       ├── toxicity_checker.py      ToxicityChecker — stages 2 & 3 (profanity + detoxify ML)
│       ├── ai_safety_checker.py     AISafetyChecker — stage 4 (Ollama AI classification)
│       ├── guard_rail_cache.py      Re-export of GuardRailCache + HighStakesCache
│       ├── pending_cache.py         PendingCache — in-flight request deduplication
│       │
│       ├── ── Memory & State ──
│       ├── circuit_breaker.py       CircuitBreaker — per-agent failure tracking and bypass
│       ├── response_cache.py        ResponseCache — TTL-based response cache
│       ├── session_memory.py        SessionMemory — Ollama per-conversation history
│       ├── preference_memory.py     PreferenceMemory — adaptive preference store
│       ├── preference_analyser.py   PreferenceAnalyser — extract preferences from conversation
│       ├── entity_context.py        EntityContextCache — HA entity/sensor context cache
│       ├── persistent_store.py      PersistentStore — HA storage-backed config persistence
│       ├── statistics.py            AgentStatistics — per-agent request/latency counters
│       │
│       ├── ── Benchmarking ──
│       ├── agent_benchmark.py       AgentBenchmarker — thin orchestrator + Protocol
│       ├── benchmark_models.py      ScoreBreakdown, BenchmarkStatus, ProbeResult, BenchmarkProfile
│       ├── benchmark_profile_store.py  BenchmarkProfileProtocol — CRUD Protocol
│       ├── benchmark_probe_runner.py   BenchmarkProbeRunner — Ollama + existing probe execution
│       ├── benchmark_telemetry.py      Telemetry helpers: _check_version, update_realworld_telemetry
│       ├── benchmark_scheduler.py      BenchmarkScheduler — warm-up scheduling + pending tasks
│       ├── benchmark_probes.py      BenchmarkProbe — probe definitions (imports from probes/)
│       ├── probes/                  Probe definition subpackage
│       │   ├── __init__.py
│       │   ├── ollama_probes.py     Ollama-specific probe instances
│       │   └── integration_probes.py  HA integration probe instances
│       │
│       ├── ── Web Search ──
│       ├── web_search_client.py     WebSearchClient — search orchestrator + Protocol
│       ├── providers/               Search provider implementations
│       │   ├── __init__.py
│       │   ├── brave_search.py      BraveSearchProvider
│       │   ├── brave_answers.py     BraveAnswersProvider
│       │   └── brave_combined.py    BraveAnswersCombinedProvider
│       │
│       ├── ── Internationalisation ──
│       ├── languages_loader.py      YAML language file loader
│       ├── prompts_loader.py        YAML prompt template loader
│       ├── languages/               Translated string YAML files
│       ├── prompts/                 Prompt template YAML files
│       ├── strings.json             UI string keys
│       └── translations/
│           └── en.json              English UI translations
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  Shared fixtures
│   ├── unit/                        Unit tests (one file per module)
│   │   ├── test_agent_backend.py
│   │   ├── test_agent_benchmark.py
│   │   ├── test_benchmark_probes.py
│   │   ├── test_circuit_breaker.py
│   │   ├── test_config_flow.py
│   │   ├── test_conversation.py
│   │   ├── test_entity_context.py
│   │   ├── test_guard_rail.py
│   │   ├── test_guard_rail_cache.py
│   │   ├── test_hacs_manifest.py
│   │   ├── test_init.py
│   │   ├── test_languages_loader.py
│   │   ├── test_llm_agent_proxy.py
│   │   ├── test_ollama_client.py
│   │   ├── test_pending_cache.py
│   │   ├── test_persistent_store.py
│   │   ├── test_preference_analyser.py
│   │   ├── test_preference_memory.py
│   │   ├── test_prompts_loader.py
│   │   ├── test_response_cache.py
│   │   ├── test_sensor.py
│   │   ├── test_session_memory.py
│   │   ├── test_statistics.py
│   │   ├── test_toxicity_checker.py
│   │   ├── test_translations.py
│   │   └── test_web_search_client.py
│   ├── integration/
│   │   ├── test_full_flow.py
│   │   └── test_guard_rail_integration.py
│   └── fixtures/
│       └── conversation_samples.py
├── docs/
│   ├── CODING_STANDARDS.md          Code quality quick reference
│   ├── CONFIGURATION_EXAMPLES.md    Annotated real-world setups
│   ├── DEVELOPMENT_GUIDE.md         Full development standards
│   ├── STRUCTURE.md                 This file
│   ├── architecture/
│   │   └── README.md                System design and component map
│   ├── user-guide/
│   │   ├── quick-start.md           5-minute setup guide
│   │   ├── installation.md          Full installation walkthrough
│   │   ├── agent-configuration.md   All agent options and settings
│   │   └── usage.md                 Day-to-day usage guide
│   └── api/
│       └── conversation_entity.md   Developer API reference
├── config/
│   └── examples/
│       ├── basic_configuration.yaml
│       ├── advanced_configuration.yaml
│       └── alexa_integration.yaml
├── CLAUDE.md                        Claude Code instructions (root — required)
├── LICENSE
├── README.md                        Project overview (root — required by HACS/GitHub)
├── hacs.json                        HACS metadata
├── pytest.ini                       Pytest configuration
├── pyproject.toml                   Black / Ruff / Mypy configuration
└── requirements_test.txt            Testing dependencies
```

## Key Directories

### `/custom_components/neuralbridge/`

The main integration code. This is what gets installed by users via HACS or manually.

**Core files:**
- `conversation.py` — `NeuralBridgeAgent`: HA lifecycle, `async_process`, and thin delegation to sub-components
- `config_flow.py` — the options flow UI for adding and managing agents
- `const.py` — all configuration keys, defaults, and the built-in router classification prompt

**Pipeline:**
- `pipeline_executor.py` — `PipelineExecutor`: run-pipeline loop, compound fragment handling, cache checks
- `agent_retry_engine.py` — `AgentRetryEngine`: circuit-breaker integration, retry counting, failure recording
- `fragment_processor.py` — `FragmentProcessor`: splits compound queries into parallel fragment calls
- `router_engine.py` — `RouterEngine`: thin orchestrator — delegates to backend subpackage; fallback, stats dispatch
- `router_decision.py` — `RouterDecision` value object + `_parse_router_response` / `_apply_router_decision` helpers
- `router_backend.py` — `RouterBackendProtocol`: structural `Protocol` contract for all router backends
- `router_utils.py` — shared router utility functions (similarity helpers, etc.)
- `router_backends/` — `OllamaRouterBackend` (direct HTTP) and `IntegratedRouterBackend` (HA conversation service)
- `llm_agent_proxy.py` — `LLMAgentProxy`: INTEGRATED and direct-OLLAMA transport dispatch with prompt enrichment
- `prompt_builder.py` — `PromptBuilder`: constructs system prompts, injects entity/area/sensor context
- `agent_backend.py` — generic agent dispatch and backend utilities

**Confirmation & Guard Rails:**
- `confirmation_flows.py` — `ConfirmationFlows`: thin coordinator for guard rail + high-stakes + preference flows
- `high_stakes_handler.py` — `HighStakesHandler`: dangerous-command confirmation resolution
- `preference_confirmation_handler.py` — `PreferenceConfirmationHandler`: preference suggestion and confirmation
- `guard_rail_action_handler.py` — `GuardRailActionHandler`: block/warn/log action dispatch
- `guard_rail.py` — `GuardRailChecker`: four-stage safety pipeline orchestrator
- `guard_rail_types.py` — `GuardRailResult` dataclass and shared guard rail types
- `toxicity_checker.py` — `ToxicityChecker`: stage 2 (better-profanity) + stage 3 (detoxify ML model)
- `ai_safety_checker.py` — `AISafetyChecker`: stage 4 Ollama AI classification
- `guard_rail_cache.py` — re-export shim for `GuardRailCache` + `HighStakesCache`
- `pending_cache.py` — `PendingCache`: in-flight request deduplication

**Memory & State:**
- `circuit_breaker.py` — `CircuitBreaker`: skips failing agents during cooldown
- `response_cache.py` — `ResponseCache`: TTL-based cache keyed by agent ID + query hash
- `session_memory.py` — `SessionMemory`: per-session Ollama message history with TTL and eviction
- `preference_memory.py` — `PreferenceMemory`: adaptive preference store
- `preference_analyser.py` — `PreferenceAnalyser`: extracts preferences from free-text conversation
- `entity_context.py` — `EntityContextCache`: HA entity/sensor context caching
- `persistent_store.py` — `PersistentStore`: HA storage-backed configuration persistence
- `statistics.py` — `AgentStatistics`: tracks requests, success/failure, latency per agent
- `sensor.py` — exposes `AgentStatistics` as HA sensor entities

**Benchmarking:**
- `agent_benchmark.py` — `AgentBenchmarkerProtocol` + `AgentBenchmarker`: thin orchestrator, delegation wrappers, re-exports
- `benchmark_models.py` — `ScoreBreakdown`, `BenchmarkStatus`, `ProbeResult`, `BenchmarkProfile` + `_evaluate_probe`, `_compute_scores`
- `benchmark_profile_store.py` — `BenchmarkProfileProtocol`: CRUD structural interface for profile management
- `benchmark_probe_runner.py` — `BenchmarkProbeRunner`: Ollama and existing-HA probe execution + `_apply_probe_results`
- `benchmark_telemetry.py` — `_check_version`, `update_realworld_telemetry` free functions
- `benchmark_scheduler.py` — `BenchmarkScheduler`: warm-up delay, task lifecycle, `cancel_pending` / `cancel_all_pending`
- `benchmark_probes.py` — `BenchmarkProbe`: probe definitions (thin façade over `probes/`)
- `probes/` — probe definition subpackage (ollama + integration probes)

**Web Search:**
- `web_search_client.py` — `WebSearchClient`: search orchestrator + `WebSearchProvider` Protocol
- `providers/` — provider implementations: `BraveSearchProvider`, `BraveAnswersProvider`, `BraveAnswersCombinedProvider`

**Internationalisation:**
- `languages_loader.py` — YAML language file loader
- `prompts_loader.py` — YAML prompt template loader
- `ollama_client.py` — async HTTP client for Ollama `generate()` and `chat()` endpoints

100% coverage is required. Tests follow the Arrange / Act / Assert pattern with async
test functions using `pytest-homeassistant-custom-component` fixtures.

### `/docs/`

All user and developer documentation. No platform-specific generated files belong here.

### `/config/examples/`

Sample YAML snippets demonstrating real-world NeuralBridge configurations for reference.
These are not deployed or loaded automatically.
