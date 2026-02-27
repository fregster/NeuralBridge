# NeuralBridge Feature Backlog

> Features identified from smart home voice interaction analysis (Alexa parity and beyond).
> All features listed below are implemented except Feature 7.

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
| 14 L1 | Ollama metadata discovery via `/api/show` (model family, params, quantization, context window) | `agent_benchmark.py`, `ollama_client.py` |
| 14 L2 | Active benchmark probe suite (8 probes × 5 dimensions; async background; probe suite versioning) | `agent_benchmark.py`, `benchmark_probes.py`, `sensor.py`, `__init__.py`, `config_flow.py` |
| 14 L3 | Passive telemetry — `OllamaResponse` dataclass; rolling weighted average of real-traffic `eval_*` stats | `ollama_client.py`, `agent_benchmark.py` |
| 14d | Dimension-aware routing strategies; `dimension` + `suggested_agent_order` in `RouterDecision`; per-dimension strategy config on routing agents; capability + strategy blocks injected into router prompt | `router_engine.py`, `conversation.py`, `const.py`, `config_flow.py` |

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

## Future Feature Ideas

### Global Routing Profiles (future)

The `RouterDecision.dimension` field (Feature 14d) and dimension strategies lay the
groundwork for preset routing profiles:

- Named presets (“Speed”, “Quality”, “Privacy / Local-Only”) that set all five dimension
  strategies simultaneously with a single toggle.
- Per-user (Feature 7) dimension strategy overrides — different household members get
  different routing behaviour (e.g. children’s profile → `local_only` on all dimensions).

No implementation planned until Feature 7 multi-user profiles is shipped.

---

## Next Up

| Priority | Feature | Rationale |
|---|---|---|
| 1 | **Feature 7** — Multi-User / Speaker Profile Routing | Only remaining unimplemented feature; unlocks per-user dimension strategies |
