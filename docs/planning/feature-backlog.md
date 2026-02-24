# NeuralBridge Feature Backlog

> Features identified from smart home voice interaction analysis (Alexa parity and beyond).
> Features 1, 3, and 5 are already implemented (see git history).
> Features below are planned but not yet implemented.

---

## Feature 2 — Compound Command Splitting

**Problem:** "Turn off the lights *and* set the thermostat to 22" fails as a single sentence when LOCAL_HA cannot match the composite intent.

**Plan:**
1. Add a `_split_compound_input(text)` module-level helper — detects conjunctions ("and", "then", "also", "after that") and splits on them.
2. Only activate when LOCAL_HA agents are in the pipeline (no point splitting cloud LLM queries).
3. For each fragment, run the full pipeline independently and collect results.
4. Combine responses: join speech text with " · " (or a configurable separator).
5. If any fragment fails, include its failure text in the combined response rather than silently dropping it.
6. Add config toggle `CONF_SPLIT_COMPOUND_COMMANDS` (default `false` — opt-in to avoid breaking existing setups).
7. **Constraint:** max 3 fragments to prevent abuse/resource exhaustion.

**Files affected:** `conversation.py`, `const.py`, `config_flow.py`, `strings.json`, language YAMLs.

---

## Feature 4 — High-Stakes Action Confirmation

**Problem:** Destructive HA actions (unlock door, disarm alarm, open garage) execute immediately with no second check — against the project's security-first principle.

**Plan:**
1. Add `CONF_HIGH_STAKES_DOMAINS` setting — user-configurable list of HA domains requiring confirmation. Defaults: `lock`, `alarm_control_panel`, `cover`, `garage_door`.
2. Add `CONF_HIGH_STAKES_ENABLED` toggle (default `false`).
3. In the LOCAL_HA processing path, after receiving an `action_done` response, inspect the intent response's targeted entity domain against the configured list.
4. If matched: store the pending result in a new `_high_stakes_cache` (same pattern as `_guard_rail_cache`) and return a confirmation prompt ("Are you sure you want to unlock the front door?").
5. On the next turn, detect "yes"/"no" in `_handle_confirmation_check` — already handles guard rail confirms; extend it to also check `_high_stakes_cache`.
6. Fire HA event `neuralbridge_high_stakes_triggered` so automations can layer on a PIN pad or push notification.
7. **Privacy:** log domain + entity friendly name only — never the full intent payload.

**Files affected:** `conversation.py`, `const.py`, `config_flow.py`, `strings.json`, language YAMLs.

---

## Feature 6 — Response Verbosity / Brief Mode

**Problem:** No equivalent to Alexa's Brief Mode ("OK" vs "I've turned on the kitchen lights").

**Plan:**
1. Add `CONF_RESPONSE_VERBOSITY` setting with three levels: `brief`, `normal` (default), `verbose`.
2. Inject the level into the system prompt as a short instruction appended after the base prompt:
   - `brief`: *"Respond with the shortest possible acknowledgement — one to five words."*
   - `verbose`: *"Give detailed, explanatory responses."*
3. For Ollama agents: append the instruction to the system prompt at build time in `_process_with_ollama`.
4. For existing/cloud agents: prepend a brief modifier to `user_input.text` before the service call (e.g. `"[Brief response] Turn off the lights"`).
5. For LOCAL_HA: post-process the response — if `brief` and the HA response is already short, leave it; if long, truncate at the first sentence.
6. Add per-agent override `CONF_AGENT_VERBOSITY` so agents can be configured independently (router = always brief, cloud = always verbose).
7. Config flow: radio-button selector.

**Files affected:** `conversation.py`, `const.py`, `config_flow.py`, `strings.json`, language YAMLs.

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

## Feature 8 — Timer / Reminder Awareness

**Problem:** Timer intents work via LOCAL_HA but there is no dedicated routing signal, so timers risk being routed to cloud agents unnecessarily.

**Plan:**
1. Extend `ROUTER_CLASSIFICATION_PROMPT` with an optional fourth JSON field: `"intent_hint"` — values: `"timer"`, `"reminder"`, `"shopping_list"`, `"announce"`, `"todo"` (and `null` for everything else).
2. Add `intent_hint: str | None` field to `RouterDecision` (default `None`; backward-compatible — parsers that omit it produce `None`).
3. In `_apply_router_decision`: when `intent_hint` is `"timer"` or `"reminder"`, force `local_ha = True` and promote LOCAL_HA agents with `assist_mode = True`.
4. Add timer/reminder examples to the router classification prompt.
5. Record `intent_hint` in `AgentStatistics` so the sensor shows how many timer requests were handled locally.
6. **No new agent type** — purely a routing signal improvement.

**Files affected:** `conversation.py`, `const.py`, `prompts/router_classification.txt`, `statistics.py`.

---

## Feature 9 — Semantic Cache Keying

**Problem:** "What's the weather?" and "How is the weather today?" miss the cache because exact-text keying is used.

**Plan:**
1. Add `CONF_RESPONSE_CACHE_SEMANTIC` toggle (default `false` — opt-in; exact match is safer).
2. Implement `_normalise_cache_key(text)` in `response_cache.py`:
   - Lowercase + strip punctuation.
   - Remove common filler words ("please", "can you", "what's", "what is", "how is", "tell me").
   - Normalise number words ("twenty two" → "22").
   - Light suffix stripping (remove `-ing`, `-ed`, `-s` endings on words > 4 chars).
3. `ResponseCache.get()` and `.store()` accept a `normalise: bool = False` flag.
4. Store both normalised key and original text; on lookup, return a hit only when normalised forms match (avoids false collisions e.g. "lock door" vs "block door").
5. Add `CONF_SEMANTIC_CACHE_TTL` (default 60 s — shorter than exact-match TTL since semantic matching is less precise).
6. **No new pip dependency** — avoid `sentence-transformers` to keep the integration lightweight.

**Files affected:** `response_cache.py`, `conversation.py`, `const.py`, `config_flow.py`, `strings.json`.

---

## Feature 10 — Language Auto-Detection / Passthrough

**Problem:** `user_input.language` from the HA pipeline is available but NeuralBridge does not consume it to influence agent selection or prompt language.

**Plan:**
1. HA already supplies `user_input.language` (e.g. `"en"`, `"de"`, `"fr-FR"`) — no third-party detection library needed.
2. In `_process_with_ollama`: when `user_input.language` differs from the integration's configured `CONF_LANGUAGE`, append *"Respond in {lang_name}."* to the system prompt.
3. In `_classify_with_router`: append `Language: {language}` below the entity context line in the classification prompt to help the model avoid misclassifying non-English home commands.
4. In `_process_with_existing`: already passes `language` through — no change needed.
5. Add `CONF_FORCE_RESPONSE_LANGUAGE` toggle (default `true`) — opt-out for users who prefer the agent responds in its trained language.
6. Area context and entity context injections should preserve any localised HA area/entity names as-is (they are already friendly names from HA state).

**Files affected:** `conversation.py`, `const.py`, `config_flow.py`, `strings.json`, language YAMLs.

---

## Feature 11 — Proactive / Broadcast Announcements

**Problem:** "Announce dinner is ready on all speakers" is a high-volume Alexa pattern with no dedicated NeuralBridge routing path.

**Plan:**
1. Detect announcement intent via `intent_hint == "announce"` (Feature 8) **or** a regex pre-check in `_compute_result`: `r"^(announce|broadcast|tell everyone|say on all(?: the)? speakers?):?\s+"`.
2. Extract the announcement text (the content after the trigger phrase).
3. Add `CONF_ANNOUNCE_MEDIA_PLAYERS` config option — multiselect of `media_player.*` entities (empty = feature disabled).
4. When an announce intent is detected, call `tts.speak` (or `media_player.play_media`) for each configured entity concurrently via `asyncio.gather`.
5. Return a confirmation `ConversationResult` ("Announcing to 3 speakers.") to the Assist UI rather than the TTS text itself.
6. **Privacy:** announcement text is not cached and not logged at info level — only `"Announcement sent to N speakers"` at debug.
7. Feature disabled by default; opt-in via config flow options step.

**Files affected:** `conversation.py`, `const.py`, `config_flow.py`, `strings.json`, language YAMLs, `entity_context.py` (add `media_player` to relevant domains).

---

## Feature 12 — Shopping / To-Do List Integration

**Problem:** "Add milk to the shopping list" routes to cloud/Ollama when LOCAL_HA can handle it directly via the HA `todo` domain.

**Plan:**
1. Add `"todo"` and `"shopping_list"` to the router classification prompt's `local_ha = true` examples.
2. Extend `RELEVANT_DOMAINS` in `entity_context.py` to include `"todo"` and `"shopping_list"` so the entity context summary advertises available lists to the router model.
3. Use `intent_hint == "todo"` (Feature 8) to force LOCAL_HA routing for list management requests.
4. In the LOCAL_HA path: HA's native conversation agent already handles `HassSetTodo`, `HassListTodo` etc. intents — no new service calls needed; correct routing is all that is required.
5. Ensure the assist-mode fall-through check treats `action_done` on todo intents as a success (not a fall-through to cloud).
6. **Low implementation cost** — mostly routing signal changes plus one entity domain addition.

**Files affected:** `entity_context.py`, `prompts/router_classification.txt`, `conversation.py`, `const.py`.

---

## Suggested Implementation Order

| Priority | Feature | Rationale |
|---|---|---|
| 1 | **8** — Timer/reminder + `intent_hint` | Foundational for features 11 and 12 |
| 2 | **12** — To-do list routing | Cheapest win once `intent_hint` exists |
| 3 | **2** — Compound splitting | High user impact; self-contained |
| 4 | **4** — High-stakes confirmation | Security pillar; medium effort |
| 5 | **9** — Semantic cache | Performance win; no new dependencies |
| 6 | **10** — Language passthrough | Low effort; important for i18n users |
| 7 | **6** — Verbosity / brief mode | Quality of life; low effort |
| 8 | **11** — Broadcast announcements | Requires TTS setup; medium effort |
| 9 | **7** — Multi-user profiles | Highest complexity; needs HA user profile prerequisites |
