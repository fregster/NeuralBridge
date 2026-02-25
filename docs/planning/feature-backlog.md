# NeuralBridge Feature Backlog

> Features identified from smart home voice interaction analysis (Alexa parity and beyond).
> Features 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13 are implemented (see git history).
> Feature 14 is partially implemented (14b: `confidence` field in `RouterDecision`;
> 14c: agent-type manifest passed to router prompt).
> Features below are planned but not yet implemented.

---

## Already Implemented (reference summary)

| Feature | Summary | Key files |
|---|---|---|
| 1 | Guard rail content filtering (input + output) | `guard_rail.py`, `conversation.py` |
| 2 | Compound command splitting (`CONF_SPLIT_COMPOUND_COMMANDS`) | `conversation.py`, `const.py`, `config_flow.py` |
| 3 | Web search agent with Brave Search API | `web_search_client.py`, `conversation.py` |
| 4 | High-stakes action confirmation (`CONF_HIGH_STAKES_ENABLED`) | `conversation.py`, `guard_rail.py`, `const.py` |
| 5 | Session memory + context window management | `session_memory.py`, `ollama_client.py` |
| 6 | Response verbosity / Brief Mode (`CONF_RESPONSE_VERBOSITY`) | `conversation.py`, `const.py`, `config_flow.py` |
| 8 | Timer / reminder `intent_hint` routing signal | `conversation.py`, `router_classification.txt`, `statistics.py` |
| 9 | Semantic (normalised) cache keying | `response_cache.py`, `conversation.py` |
| 10 | Language passthrough to router + Ollama prompts | `conversation.py` |
| 11 | Broadcast / announcement via `tts.speak` | `conversation.py`, `config_flow.py` |
| 12 | Shopping / to-do list routing via `intent_hint` | `entity_context.py`, `router_classification.txt` |
| 13 | Brave Answers provider + combined provider | `web_search_client.py`, `config_flow.py` |
| 14b | `confidence` field in `RouterDecision`; skip flag-promotion when `"low"` | `conversation.py`, `const.py` |
| 14c | Agent-type manifest appended to router prompt | `conversation.py` |

---

## Feature 7 — Multi-User / Speaker Profile Routing

**Problem:** All users get the same routing and guard rail settings regardless of who is speaking.

**Plan:**
1. HA passes `user_id` on `ConversationInput.context.user_id` when voice profiles are configured.
2. Add `CONF_USER_PROFILES` in options — a list of profile entries, each with:
   - `user_id` — selected from HA users list in config flow.
   - `preferred_agent_id` — optional soft routing hint (try first, not exclusive).
   - `guard_rails_always_on` — bool, force guard rails on for this user.
   - `max_complexity` — int cap; processing agents with a complexity rating above this value are excluded.
3. In `_compute_result`, resolve `user_input.context.user_id` against the profile list before routing.
4. Apply matched profile constraints before router and agent selection.
5. **Privacy:** `user_id` is used for lookup only and never logged (PII).
6. Config flow: "User Profiles" options step — list of HA users with per-user settings form.

**Files affected:** `conversation.py`, `const.py`, `config_flow.py`, `strings.json`, language YAMLs.

---

## Feature 14 — Agent Benchmark Profiling

**Problem:** When a user adds a new Ollama agent (e.g. `qwen3:0.6b` vs `llama3.1:8b`) or an existing integration (cloud LLM), NeuralBridge has no intrinsic understanding of that agent's performance characteristics or capabilities. Priority is set manually and blindly. There is no way for the user to make informed tuning decisions ("which of my agents is fastest?", "which can follow instructions reliably?") and no data to power future advanced routing modes (e.g. "Route to the fastest capable agent for this complexity score").

**Design decisions:**
- **No static taxonomy** — no hardcoded model name → tier mappings (maintenance burden). All data is discovered at runtime.
- **Three complementary layers:** (1) metadata discovery from the agent's own API, (2) active benchmark probe suite run asynchronously in the background, (3) passive telemetry continuously enriched from real traffic.
- **User control first** — results are exposed as sensor attributes for informed tuning decisions. Future routing enhancements are empowered by the stored profile but not wired in this feature.
- **Ollama cold-start aware** — a configurable warm-up delay before first probe run ensures the model is loaded. After the first successful run the benchmark does NOT re-run automatically — only on explicit toggle or manual service call.
- **Failed/pending agents always retry** — `re_benchmark_on_save` remains `true` until a successful profile exists, so every config save retries until the agent is reachable.

---

### Layer 1 — Metadata Discovery (free, one-shot at registration)

**Ollama agents** — call `/api/show` immediately after agent is saved (no inference, near-instant).  Extracts:
- `details.family` → `model_family` (e.g. `"qwen3"`, `"llama3"`)
- `details.parameter_size` → `parameter_count_billions` (e.g. `7.0`, `0.6`)
- `details.quantization_level` → `quantization` (e.g. `"Q4_K_M"`, `"F16"`)
- `modelinfo["general.context_length"]` → `context_window` (e.g. `32768`)
- `details.format` → model format (`"gguf"` etc.)
- `size` (bytes) from the `/api/show` response root

**Existing integration / cloud agents** — no universal introspection API. Store `model_name` from the agent config; all other metadata fields remain `null` until probes populate the performance metrics.

---

### Layer 2 — Active Benchmark Probe Suite (async background, ~10-30 s)

Eight probes in chat format. Each probe is time-boxed individually (15 s default). The suite runs as `asyncio.create_task()` — never blocks the event loop or real conversations.

| # | Probe name | Prompt (user turn) | Pass criterion | Dimension |
|---|---|---|---|---|
| 1 | `instruction_exact` | `"Reply with only the number 42. Nothing else."` | Response stripped == `"42"` | `instruction_following` |
| 2 | `instruction_format` | `"List three colours. Use a numbered list."` | Contains `"1."`, `"2."`, `"3."` | `instruction_following` |
| 3 | `math_basic` | `"What is 17 multiplied by 23? Reply with only the number."` | Contains `"391"` | `reasoning` |
| 4 | `reasoning_time` | `"A train travels at 60 mph for 90 miles. How many minutes does the journey take? Reply with only the number."` | Contains `"90"` | `reasoning` |
| 5 | `smart_home_intent` | `"You are a smart home assistant. The user says: 'turn off the kitchen lights'. What HA domain handles this? Reply with one word."` | Contains `"light"` | `smart_home_intent` |
| 6 | `factual_recall` | `"What is the chemical symbol for gold? Reply with only the symbol."` | Response stripped (case-insensitive) == `"au"` | `factual` |
| 7 | `multi_turn_memory` | System: no system prompt. Turn 1: `"My secret code is BANANA"`. Turn 2: `"What is my secret code? Reply with only the code."` | Contains `"banana"` (case-insensitive) | `memory` |
| 8 | `conciseness` | `"In exactly one word, describe the colour of the sky on a clear daytime day."` | Word count of stripped response == 1 AND contains `"blue"` | `instruction_following` |

**Scoring:**
- Each probe: `1` = pass, `0` = fail
- `score_instruction_following`: probes 1 + 2 + 8 (0-3)
- `score_reasoning`: probes 3 + 4 (0-2)
- `score_smart_home_intent`: probe 5 (0-1)
- `score_factual`: probe 6 (0-1)
- `score_memory`: probe 7 (0-1)
- `capability_score`: sum of all probes (0-8)

**Performance metrics extracted per probe (Ollama only):**
- `eval_count` / `eval_duration` → tokens/sec for that probe
- `prompt_eval_count` / `prompt_eval_duration` → prompt ingestion speed
- Median and P95 latency across all probes → `median_latency_ms`, `p95_latency_ms`
- Median tokens/sec across probes → `probe_tokens_per_sec`
- Median prompt ingestion speed → `probe_prompt_tps`

**Warm-up lifecycle:**
```
Agent saved (new) → metadata discovery runs immediately
                  → BenchmarkProfile(status=SCHEDULED, re_benchmark_on_save=True) stored
                  → asyncio.create_task(warm_up_then_probe(delay=CONF_BENCHMARK_WARM_UP_DELAY))
                     ... delay elapses ...
                  → probes run → status=COMPLETE, re_benchmark_on_save=False
                               OR status=FAILED, re_benchmark_on_save=True (stays true)

Agent saved (edit, re_benchmark_on_save=True) → probes run immediately (no warm-up delay, already loaded)
Agent saved (edit, re_benchmark_on_save=False) → no probe run
Service call neuralbridge.run_benchmark(agent_id) → probes run immediately, ignores toggle
```

---

### Layer 3 — Passive Telemetry (continuous enrichment from real traffic)

Modify `OllamaClient.generate()` and `OllamaClient.chat()` to return an `OllamaResponse` dataclass instead of a bare `str | None`:

```python
@dataclass
class OllamaResponse:
    content: str
    eval_count: int | None = None          # output tokens
    eval_duration_ns: int | None = None    # time to generate output tokens
    prompt_eval_count: int | None = None   # input tokens processed
    prompt_eval_duration_ns: int | None = None

    @property
    def tokens_per_second(self) -> float | None: ...  # eval_count / (eval_duration_ns / 1e9)
    @property
    def prompt_tokens_per_second(self) -> float | None: ...
```

After every successful Ollama call in `_process_with_ollama`, update `BenchmarkProfile.realworld_tokens_per_sec` with a rolling weighted average (new = 0.2 × sample + 0.8 × existing, initialised from first sample). Store `realworld_sample_count` so the sensor can show confidence ("based on N real requests").

---

### New module: `agent_benchmark.py`

**`BenchmarkStatus`** — `StrEnum`: `PENDING`, `SCHEDULED`, `RUNNING`, `COMPLETE`, `FAILED`, `SKIPPED`

**`BenchmarkProbe`** — dataclass:  `name`, `dimension`, `messages: list[dict]`, `expected_contains: list[str] | None`, `expected_exact: str | None`, `word_count_check: bool`, `timeout_seconds`

**`ProbeResult`** — dataclass: `probe_name`, `dimension`, `passed`, `latency_ms`, `eval_count`, `eval_duration_ns`, `prompt_eval_count`, `prompt_eval_duration_ns`  *(response text NOT stored — privacy; probes contain no PII but establish the principle)*

**`BenchmarkProfile`** — dataclass (fully serialisable to dict for HA storage):
```
agent_id, agent_name, agent_type, status, re_benchmark_on_save,
benchmark_timestamp, benchmark_duration_ms, error,
model_name, model_family, parameter_count_billions, quantization,
context_window, model_size_bytes,
median_latency_ms, p95_latency_ms,
probe_tokens_per_sec, probe_prompt_tps,
realworld_tokens_per_sec, realworld_sample_count,
score_instruction_following, score_reasoning, score_smart_home_intent,
score_factual, score_memory, capability_score,
probe_results: dict[str, bool]  # probe_name → pass/fail (no response text)
```

**`AgentBenchmarker`** — manages all profiles:
- `async_load()` / `async_save()` — HA storage (`neuralbridge.benchmark`, version 1)
- `ensure_profile(agent_config)` — idempotent create-or-return
- `remove_profile(agent_id)` — called when agent is deleted from config
- `async_run_metadata(agent_id, agent_config)` — Ollama `/api/show` call only
- `async_schedule_benchmark(agent_id, agent_config, delay_seconds)` — create_task with warm-up
- `async_run_benchmark(agent_id, agent_config)` — full probe suite (sets status=RUNNING, fires events, handles exceptions)
- `update_realworld_telemetry(agent_id, response: OllamaResponse)` — rolling average update
- `cancel_pending(agent_id)` — cancel warm-up task if agent is deleted before it fires

---

### HA Events fired

| Event | Payload |
|---|---|
| `neuralbridge_benchmark_started` | `{agent_id, agent_name, agent_type}` |
| `neuralbridge_benchmark_complete` | `{agent_id, agent_name, capability_score, median_latency_ms, probe_tokens_per_sec, status}` |
| `neuralbridge_benchmark_failed` | `{agent_id, agent_name, error}` |

---

### HA Service

`neuralbridge.run_benchmark` — service with `agent_id: str` field. Triggers immediate probe run regardless of `re_benchmark_on_save` state. Registered in `__init__.py`.

---

### Sensor changes (`sensor.py`)

Add `benchmark` sub-dict to `NeuralBridgeAgentSensor.extra_state_attributes`:

```python
"benchmark": {
    "status": "complete",
    "capability_score": 6,          # /8
    "score_instruction_following": 3,
    "score_reasoning": 2,
    "score_smart_home_intent": 1,
    "score_factual": 0,
    "score_memory": 0,
    "median_latency_ms": 1240.0,
    "p95_latency_ms": 1890.0,
    "probe_tokens_per_sec": 42.3,
    "probe_prompt_tps": 310.5,
    "realworld_tokens_per_sec": 38.1,
    "realworld_sample_count": 17,
    "model_family": "qwen3",
    "parameter_count_billions": 0.6,
    "quantization": "Q4_K_M",
    "context_window": 32768,
    "last_benchmarked": 1740000000.0,
    "re_benchmark_on_save": false,
}
```

`NeuralBridgeAgentSensor` receives a reference to `AgentBenchmarker` and reads from it on each `async_update_ha_state()` call (same dispatcher signal pattern as `AgentStatistics`).

---

### Config flow changes (`config_flow.py`)

**Per-agent options form** (both add and edit flows) — add:
- `CONF_AGENT_RE_BENCHMARK` (`re_benchmark_on_save`) — `BooleanSelector`, default `True` for new agents, auto-set to `False` after first successful run and exposed as editable toggle for re-runs.
- Help text: *"Re-run benchmark when this agent is saved. Enabled automatically until a benchmark result exists."*

**Global options form** — add:
- `CONF_BENCHMARK_WARM_UP_DELAY` — `NumberSelector(min=10, max=300, step=5, unit_of_measurement="s")`, default `60`. Help: *"Seconds to wait after HA starts before benchmarking newly added agents (allows Ollama to load the model)."*

---

### New constants (`const.py`)

```python
CONF_AGENT_RE_BENCHMARK: Final = "re_benchmark_on_save"
CONF_BENCHMARK_WARM_UP_DELAY: Final = "benchmark_warm_up_delay"
DEFAULT_BENCHMARK_WARM_UP_DELAY: Final = 60
BENCHMARK_STORAGE_KEY: Final = f"{DOMAIN}.benchmark"
BENCHMARK_STORAGE_VERSION: Final = 1
EVENT_BENCHMARK_STARTED: Final = f"{DOMAIN}_benchmark_started"
EVENT_BENCHMARK_COMPLETE: Final = f"{DOMAIN}_benchmark_complete"
EVENT_BENCHMARK_FAILED: Final = f"{DOMAIN}_benchmark_failed"
SERVICE_RUN_BENCHMARK: Final = "run_benchmark"
DATA_BENCHMARKER: Final = "benchmarker"
```

---

### Files affected

| File | Change |
|---|---|
| `agent_benchmark.py` | **New** — entire module (probes, profile, benchmarker) |
| `ollama_client.py` | `generate()` + `chat()` return `OllamaResponse` instead of `str \| None`; add `async_show_model()` for `/api/show` |
| `conversation.py` | Import + wire `AgentBenchmarker`; capture `OllamaResponse.eval_*` for passive telemetry; trigger metadata + schedule on agent add/edit |
| `sensor.py` | Accept `AgentBenchmarker` reference; add `benchmark` sub-dict to agent sensor attributes |
| `statistics.py` | No changes — passive telemetry lives in `BenchmarkProfile`, not `AgentStats` |
| `__init__.py` | Instantiate + load `AgentBenchmarker`; register `neuralbridge.run_benchmark` service; store in `hass.data` under `DATA_BENCHMARKER` |
| `config_flow.py` | `re_benchmark_on_save` toggle + `benchmark_warm_up_delay` global option |
| `const.py` | New constants above |
| `strings.json` | Labels + descriptions for new config fields and events |
| Language YAMLs (5) | Translated strings for new fields |

---

### Test plan (`tests/unit/test_agent_benchmark.py` + `tests/integration/test_benchmark_integration.py`)

**Unit — `BenchmarkProfile` dataclass:**
- `test_profile_defaults` — status=PENDING, re_benchmark_on_save=True, all scores 0
- `test_profile_serialization_round_trip` — to_dict() → from_dict() preserves all fields
- `test_profile_serialization_partial` — missing optional fields deserialize to None gracefully

**Unit — probe evaluation logic:**
- `test_probe_instruction_exact_pass` / `_fail` / `_whitespace_trimmed`
- `test_probe_instruction_format_pass` / `_fail_missing_numbers`
- `test_probe_math_pass` / `_fail_wrong_answer` / `_fail_with_surrounding_text`
- `test_probe_reasoning_time_pass` / `_fail`
- `test_probe_smart_home_intent_pass` / `_fail_unrelated_word`
- `test_probe_factual_pass_lowercase` / `_pass_uppercase` / `_fail`
- `test_probe_multi_turn_memory_pass` / `_fail_no_recall`
- `test_probe_conciseness_pass` / `_fail_multi_word` / `_fail_wrong_colour`

**Unit — scoring:**
- `test_score_instruction_following_all_pass` — 3/3
- `test_score_instruction_following_partial` — 1/3
- `test_score_reasoning_all_pass` / `_partial` / `_none`
- `test_capability_score_is_sum_of_all_probes`
- `test_capability_score_max_eight`
- `test_capability_score_zero_no_passes`

**Unit — `AgentBenchmarker.async_run_metadata` (Ollama):**
- `test_metadata_ollama_full_response` — all fields extracted correctly
- `test_metadata_ollama_missing_context_window` — falls back to None
- `test_metadata_ollama_missing_quantization` — falls back to None
- `test_metadata_ollama_connection_error` — status stays PENDING, error logged, no crash
- `test_metadata_ollama_non_200` — error handled gracefully
- `test_metadata_existing_integration_skipped` — no HTTP call made, returns immediately

**Unit — `AgentBenchmarker.async_run_benchmark` (probe suite):**
- `test_run_benchmark_all_pass` — status=COMPLETE, re_benchmark_on_save=False
- `test_run_benchmark_all_fail` — status=COMPLETE (not FAILED — probes ran, agent just scored 0), re_benchmark_on_save=False
- `test_run_benchmark_partial_pass` — mixed results, scores computed correctly
- `test_run_benchmark_probe_timeout` — timed-out probe counted as fail, others continue
- `test_run_benchmark_agent_offline` — connection error → status=FAILED, re_benchmark_on_save=True
- `test_run_benchmark_unexpected_exception` — status=FAILED, error captured, re_benchmark_on_save=True
- `test_run_benchmark_extracts_eval_stats` — tokens/sec computed from eval_count/eval_duration
- `test_run_benchmark_missing_eval_stats` — graceful when Ollama omits eval fields
- `test_run_benchmark_median_latency_computed` — median across 8 probe timings
- `test_run_benchmark_p95_latency_computed` — P95 across 8 probe timings
- `test_run_benchmark_duration_recorded` — total elapsed time stored
- `test_run_benchmark_events_fired` — started + complete events fired
- `test_run_benchmark_failed_event_fired` — failed event fired on connection error
- `test_run_benchmark_sets_status_running_during_execution`
- `test_home_assistant_agent_skipped` — LOCAL_HA agent gets status=SKIPPED, no probes
- `test_re_benchmark_toggle_stays_true_after_failure` — FAILED → toggle remains True
- `test_re_benchmark_toggle_cleared_after_success` — COMPLETE → toggle set False

**Unit — warm-up scheduling:**
- `test_schedule_warm_up_runs_after_delay` — task fires after configured delay
- `test_schedule_warm_up_cancelled_on_agent_delete` — `cancel_pending()` prevents probe run
- `test_schedule_warm_up_cancelled_on_shutdown` — all tasks cancelled on unload
- `test_no_warm_up_when_re_benchmark_on_save` — immediate run, no delay

**Unit — passive telemetry:**
- `test_update_realworld_telemetry_first_sample` — initialises from first value
- `test_update_realworld_telemetry_rolling_average` — weighted average applied
- `test_update_realworld_telemetry_missing_eval_stats` — no-op when eval stats absent
- `test_update_realworld_sample_count_increments`

**Unit — storage:**
- `test_async_load_empty_store` — no saved data, starts with empty dict
- `test_async_load_restores_profiles` — saved profiles deserialize correctly
- `test_async_save_persists_all_profiles` — all profiles written
- `test_remove_profile_cleans_up` — deleted agent profile removed from store

**Unit — `OllamaClient` changes:**
- `test_generate_returns_ollama_response_with_eval_stats`
- `test_generate_returns_ollama_response_missing_eval_stats`
- `test_chat_returns_ollama_response_with_eval_stats`
- `test_tokens_per_second_property_calculation`
- `test_prompt_tokens_per_second_property_calculation`
- `test_tokens_per_second_returns_none_when_stats_missing`
- `test_async_show_model_success`
- `test_async_show_model_connection_error`
- `test_async_show_model_non_200`

**Integration — `tests/integration/test_benchmark_integration.py`:**
- `test_benchmark_scheduled_on_new_agent_add` — config entry setup triggers warm-up task
- `test_benchmark_complete_updates_sensor_attributes` — sensor.benchmark sub-dict populated
- `test_benchmark_complete_event_fired_to_hass` — HA event published
- `test_benchmark_failed_event_fired_to_hass` — failed event published
- `test_benchmark_re_run_on_edit_with_toggle` — edit + toggle=True triggers probes
- `test_benchmark_no_re_run_on_edit_without_toggle` — edit + toggle=False, no probes
- `test_service_run_benchmark_triggers_probes` — service call triggers run
- `test_benchmark_sensor_skipped_for_local_ha_agent` — LOCAL_HA sensor shows SKIPPED

---

### Future routing integration (NOT in this feature — empowered by it)

The `BenchmarkProfile.capability_score` and `probe_tokens_per_sec` fields are the foundation for a future **Advanced Routing Mode**:
- Route low-complexity queries (router score ≤ 30) to the fastest agent (`probe_tokens_per_sec` highest) that has `capability_score ≥ 2`
- Route reasoning-heavy queries to the agent with highest `score_reasoning`
- Route smart home intents to LOCAL_HA first; if it fails, fall through to the agent with highest `score_smart_home_intent`
- "Fast mode" = minimise `median_latency_ms`; "Quality mode" = maximise `capability_score`

This feature plants the data; the routing logic follows separately.

**Files affected:** `agent_benchmark.py` (new), `ollama_client.py`, `conversation.py`, `sensor.py`, `__init__.py`, `config_flow.py`, `const.py`, `strings.json`, language YAMLs (5).

**Expected test delta:** ~60 new test cases across unit + integration suites.

---

## Suggested Implementation Order

| Priority | Feature | Rationale |
|---|---|---|
| 1 | **14** — Agent Benchmark Profiling (main layers) | High value for informed tuning; 14b + 14c already done |
| 2 | **7** — Multi-user profiles | Highest complexity; needs HA user profile prerequisites |
