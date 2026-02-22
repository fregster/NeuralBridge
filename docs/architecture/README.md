# NeuralBridge Architecture

_Last updated: February 2026._

---

## Overview

NeuralBridge is a **priority-based AI agent routing** component for Home Assistant's Assist
pipeline. It replaces a single conversation backend with an ordered list of agents tried in
sequence, with optional router classification and output guard rails.

---

## High-Level Flow

```
User Input (Assist pipeline)
         │
         ▼
┌──────────────────────────────────────────┐
│  Router Agents (Priority 0, optional)    │
│  Small, fast Ollama models.              │
│  Returns RouterDecision(local_ha,        │
│  complexity). complexity=0 blocks.       │
│  Fallback: configurable per router.      │
└────────────┬─────────────────────────────┘
             │ RouterDecision (or None)
             ▼
┌──────────────────────────────────────────┐
│  Processing Agents (Priority 1-100)      │
│  Filtered/ordered by RouterDecision.     │
│  Tried in priority order.                │
│  First success wins.                     │
│  Circuit breaker skips failing agents.   │
│  Response cache avoids redundant calls.  │
└────────────┬─────────────────────────────┘
             │ Response
             ▼
┌──────────────────────────────────────────┐
│  Output Guard Rail (optional)            │
│  Checks response for harmful content.    │
│  Action: block / warn / notify_ask       │
│  Fires neuralbridge_guard_rail_triggered │
└────────────┬─────────────────────────────┘
             │
             ▼
         Response returned to Assist
```

---

## Agent Types

| Type constant | Backend |
|---|---|
| `home_assistant` | `conversation.home_assistant` — local HA intent matching |
| `existing_integration` | Any HA conversation entity (Gemini, ChatGPT, Claude, etc.) |
| `ollama` | Self-hosted Ollama server via HTTP API (`OllamaClient`) |

All three types can be mixed freely in a single configuration.

---

## Router Decision

Router agents return a JSON object which is parsed into a `RouterDecision` dataclass:

```python
@dataclass
class RouterDecision:
    local_ha: bool    # True = pin to LOCAL_HA agents
    complexity: int   # 0 = block; 1–100 = estimated complexity score
```

The `_apply_router_decision()` function in `conversation.py` uses this to filter the
processing agent list before the main routing loop begins.

Built-in classification prompt template (`ROUTER_CLASSIFICATION_PROMPT` in `const.py`):

```
Analyse the user message and respond with ONLY a valid JSON object:
  "local_ha": boolean
  "complexity": integer 1–100 (0 = block)
```

Each router agent can override this with a custom prompt.

---

## Component Map

```
custom_components/neuralbridge/
|
+-- __init__.py            Integration setup, teardown, service registration
|
+-- conversation.py        NeuralBridgeConversationEntity - core routing logic
|                          RouterDecision dataclass
|                          _check_with_routers(), _classify_with_router()
|                          _apply_router_decision()
|                          _process_ollama(), _process_existing_agent(), _process_local_ha()
|
+-- config_flow.py         NeuralBridgeOptionsFlowHandler - all UI configuration steps
|                          Menu: default_prompt, add_agent, manage_agents,
|                                configure_guard_rails, advanced_settings,
|                                language_settings, done
|
+-- const.py               All constants, config keys, default values, prompts
|                          ROUTER_CLASSIFICATION_PROMPT, all ROUTER_* constants
|
+-- ollama_client.py       Async HTTP client - generate() and chat() endpoints
|
+-- guard_rail.py          GuardRailChecker, GuardRailCache - content filtering
|
+-- circuit_breaker.py     CircuitBreaker - per-agent failure tracking and bypass
|
+-- response_cache.py      ResponseCache - TTL-based cache keyed by agent+text hash
|
+-- session_memory.py      SessionMemory - per-conversation_id Ollama chat history
|
+-- statistics.py          AgentStatistics - request/success/failure/latency counters
|
+-- sensor.py              NeuralBridgeStatisticsSensor - exposes stats to HA frontend
|
+-- languages_loader.py    YAML language file loader; returns strings by key path
|
+-- strings.json           UI string keys (referenced by HA options flow)
|
+-- translations/
    +-- en.json            English translations for all UI strings
```

---

## Data Storage

All agent configuration is stored in `ConfigEntry.data` as a Python dict. Example record:

```python
{
    "agents": [
        {
            "id": "abc-123",
            "agent_type": "ollama",
            "agent_name": "Llama3 Local",
            "priority": 20,
            "ollama_url": "http://192.168.1.50:11434",
            "ollama_model": "llama3:8b",
            "timeout": 30,
            "enabled": True,
            "is_router": False,
            "system_prompt": "",
            "agent_cache_enabled": True,
            "guard_rail_enabled_for_agent": True,
            "max_retries": 2,
            "retry_base_delay": 1.0,
        }
    ],
    "default_prompt": "You are a Home Assistant voice assistant...",
    "guard_rail_enabled": False,
    "response_cache_enabled": True,
    "response_cache_ttl": 300,
    "language": "en_gb",
}
```

---

## Runtime State (`hass.data[DOMAIN][entry_id]`)

| Key | Type | Purpose |
|---|---|---|
| `"statistics"` | `AgentStatistics` | Per-agent request/success/failure/latency counters |
| `"response_cache"` | `ResponseCache` | TTL-based response cache |
| `"session_memory"` | `SessionMemory` | Per-conversation Ollama chat history |
| `"circuit_breaker"` | `CircuitBreaker` | Per-agent failure tracking and bypass state |

---

## Reliability Features

| Feature | Implementation |
|---|---|
| Timeout | `asyncio.timeout(agent_timeout)` per agent call |
| Retry | Exponential back-off on `aiohttp.ClientError`; configurable attempts and base delay |
| Circuit breaker | Agent skipped for `cooldown` seconds after `threshold` consecutive failures |
| Fail-open routing | Router failure → proceed with configured fallback; never silences assistant |
| Graceful fallback | All agents exhausted → `FALLBACK_RESPONSE` constant returned |
| Response cache | TTL-based cache per agent; avoids duplicate calls for identical queries |
| Session memory | Per-conversation Ollama history with TTL and max-turns eviction |

---

## Extension Points

- ML-based routing optimisation using historical statistics data from `AgentStatistics`
- Usage analytics dashboard in HA Lovelace using the statistics sensors
- Additional guard rail backends (custom plugins)
- Additional agent types beyond the current three
