# Conversation Entity API

_Last updated: February 2026._

---

## NeuralBridgeConversationEntity

The main conversation agent that implements Home Assistant's `ConversationEntity`.
Registered under the entity ID `conversation.neuralbridge`.

### Class Definition

```python
class NeuralBridgeConversationEntity(ConversationEntity):
    """Priority-based AI agent routing conversation entity."""
```

Supports `ConversationEntityFeature.CONTROL` when home control is enabled.

---

## Primary Method

### `async_process(user_input: ConversationInput) -> ConversationResult`

Process a user input through the routing chain.

**Parameters:**
- `user_input` (`ConversationInput`): HA conversation input containing:
  - `text` — raw user text
  - `conversation_id` — session identifier (used for Ollama context memory)
  - `language` — language code
  - `context` — HA context object

**Returns:**
- `ConversationResult` with speech response and conversation metadata

**Routing order:**
1. Confirmation check — handle pending guard-rail or high-stakes `yes`/`no` responses
2. Response cache — return a cached hit immediately (exact or semantic key)
3. Router agents (`is_router=True`, or `priority == 0` as legacy fallback) — classify with `RouterDecision`
4. `_apply_router_decision()` — filter/reorder processing agents based on `RouterDecision` flags
5. Processing agents (priority 1–100) — tried in order, first success returned
6. Output guard rail check per agent (if enabled for that agent)
7. Fallback response if all agents fail

**Example:**
```python
result = await entity.async_process(
    ConversationInput(
        text="turn on the kitchen lights",
        conversation_id="session-abc123",
        language="en",
        context=context,
    )
)
# result.response.speech["plain"]["speech"] contains the text response
```

---

## Key Internal Methods

### `_check_with_routers(user_input, router_agents, processing_agents) -> RouterDecision | None`

Calls each configured router agent in sequence. Router agents are identified by the
`is_router=True` flag; `priority == 0` is accepted as a legacy fallback.
Any agent type is supported for routing (Ollama, Existing HA agent, or LOCAL_HA).

Builds a compact agent-type manifest (e.g. `[Available: home_assistant, ollama]`)
from `processing_agents` and injects it into every router prompt so the model can
tailor its classification to the available agents (see Feature 14c).

Returns `None` to block the request immediately when any router returns a block
signal. Falls back to a default `RouterDecision` (fail-open) if every router
errors — a broken router never silences the assistant.

A `RouterDecision` with `complexity == 0` causes the request to be blocked
immediately without calling any processing agent.

### `_classify_with_router(router_config, user_text) -> RouterDecision | None`

Sends `ROUTER_CLASSIFICATION_PROMPT` (or a custom prompt if configured) to the
router agent back-end. Supports Ollama agents via `OllamaClient` and any HA
conversation agent (existing integration or LOCAL_HA) via the `conversation.process`
service.

Parses the JSON response into a `RouterDecision`. Returns `None` on parse error,
timeout, or network failure — the caller applies the configured fallback behaviour.
When `confidence == "low"`, flag-based promotion (`local_ha` / `web_search`) is
skipped and all eligible agents are returned in priority order.

### `_apply_router_decision(agents, decision) -> list[AgentConfig]`

Filters and reorders the processing agent list based on the router decision:
- `local_ha=True` — only LOCAL_HA agents are returned
- `complexity=0` — empty list is returned (request blocked)
- `complexity=-1` (internal sentinel for `skip_routing` fallback) — all agents
  returned unchanged
- Other complexity scores — agents filtered/sorted by suitability

### `_process_ollama(agent_config, user_input) -> ConversationResult | None`

Calls `OllamaClient.chat()` with session memory context. Applies retry with
exponential back-off and circuit breaker logic. Returns `None` on failure.

### `_process_existing_agent(agent_config, user_input) -> ConversationResult | None`

Calls `hass.services.async_call("conversation", "process")` targeting the configured
entity ID. Returns `None` on failure or timeout.

### `_process_local_ha(agent_config, user_input) -> ConversationResult | None`

Calls the built-in `conversation.home_assistant` agent. When `assist_mode=True`,
only accepts `action_done` responses — non-intent responses return `None` to allow
fall-through to the next agent.

---

## RouterDecision

```python
class RouterDecision:
    local_ha: bool          # True = pin to LOCAL_HA agents
    web_search: bool        # True = route to web-search agent
    complexity: int         # 0 = block; 1–100 = estimated complexity; -1 = skip routing
    intent_hint: str | None # "timer", "reminder", "todo", "shopping_list", "announce", or None
    confidence: str         # "high" (default) or "low" — when "low", flag-promotion is skipped
```

Immutable dataclass (uses `__slots__`). Returned by `_classify_with_router()` and
consumed by `_apply_router_decision()`.

| complexity sentinel | Meaning |
|---|---|
| `0` | Block this request — return an error result immediately |
| `1–100` | Estimated complexity; higher values prefer more capable agents |
| `-1` | `skip_routing` fallback sentinel — all processing agents returned unchanged |

---

## HA Service

### `neuralbridge.clear_conversation`

Clears the Ollama session memory for a given `conversation_id`.

**Service data:**
```yaml
conversation_id: "session-abc123"
```

Useful at the end of an Alexa session or when you want to start a fresh conversation
context without waiting for the 30-minute TTL.

---

## HA Events

### `neuralbridge_guard_rail_triggered`

Fired whenever a guard rail flags content (input or output).

**Event data:**
```python
{
    "agent_id": "abc-123",         # Agent that triggered the flag (or "" for input check)
    "agent_name": "Llama3 Local",
    "action": "block",             # "block" | "warn" | "notify_ask"
    "categories": ["harmful"],     # List of matched categories
}
```

---

## Statistics Sensor

Entity ID: `sensor.neuralbridge_<agent_name>_statistics`

Attributes exposed per agent:

| Attribute | Type | Description |
|---|---|---|
| `request_count` | `int` | Total requests sent to this agent |
| `success_count` | `int` | Successful responses |
| `failure_count` | `int` | Failed responses |
| `timeout_count` | `int` | Timed-out requests |
| `block_count` | `int` | Guard rail / router blocks |
| `average_latency_ms` | `float` | Rolling average response time (successful calls) |
| `success_rate` | `float` | `success_count / request_count` (0.0–1.0) |
| `queries_per_hour` | `float` | Estimated request rate since first request |

---

## Dispatcher Signals

NeuralBridge dispatches `SIGNAL_STATS_UPDATED.format(entry_id=...)` via
`async_dispatcher_send` after every agent call. Subscribe to this signal to react to
real-time statistics updates in Lovelace cards or automations.
