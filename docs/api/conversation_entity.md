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
1. Router agents (priority 0) — classify with `RouterDecision(local_ha, complexity)`
2. `_apply_router_decision()` — filter/reorder processing agents
3. Processing agents (priority 1–100) — tried in order, first success returned
4. Output guard rail check per agent (if enabled for that agent)
5. Fallback response if all agents fail

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

### `_check_with_routers(user_input, router_agents) -> RouterDecision | None`

Calls each priority-0 Ollama router agent in sequence. Returns the first
`RouterDecision` produced. Returns `None` (no routing decision) when no routers are
configured or all routers are unavailable (fail-open behaviour applied by
`_apply_router_decision()`).

A `RouterDecision` with `complexity == 0` causes the request to be blocked
immediately without calling any processing agent.

### `_classify_with_router(router_config, user_text) -> RouterDecision | None`

Sends `ROUTER_CLASSIFICATION_PROMPT.format(user_text=...)` (or a custom prompt if
configured) to the router Ollama agent. Parses the JSON response into a
`RouterDecision(local_ha, complexity)`. Returns `None` on parse error, timeout,
or network failure — the caller applies the configured fallback behaviour.

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
    local_ha: bool    # True = pin to LOCAL_HA agents
    complexity: int   # 0 = block; 1–100 = estimated complexity; -1 = skip routing
```

Immutable dataclass (uses `__slots__`). Returned by `_classify_with_router()` and
consumed by `_apply_router_decision()`.

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
