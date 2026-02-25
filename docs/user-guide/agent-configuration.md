# Agent Configuration Guide

## Overview

NeuralBridge uses a **priority-based routing system**. You configure multiple AI agents, each
with a priority score. Requests are routed through agents in ascending priority order until one
succeeds — the first success wins.

---

## Priority System

| Priority | Role |
|---|---|
| **0** | Router agent (Ollama only). Runs before all processing agents. |
| **1–20** | High priority. Fast, local agents. Tried first. |
| **21–50** | Medium priority. |
| **51–100** | Low priority / fallback. Cloud agents or expensive models. |

Multiple agents can share the same priority — they are tried in the order they were added.

---

## Agent Types

### Home Assistant (Built-in)

Uses Home Assistant's native conversation agent for local intent matching.

**Best for:** Device control, home automation commands

**Configuration fields:**

| Field | Default | Description |
|---|---|---|
| Agent Name | — | Friendly label (e.g. "Local Intents") |
| Priority | 50 | 1–100 |
| Timeout | 5 s | 5–120 seconds |
| Enabled | true | Enable or disable without deleting |
| Assist Mode | false | Only accept `action_done` responses; fall through on all others |

**Assist Mode:** When enabled, NeuralBridge only accepts a response from this agent if HA
successfully matched and executed a device-control intent (speech output type `action_done`).
Any other response (including "I don't understand") causes NeuralBridge to fall through to the
next agent. This lets HA handle device commands while Ollama or cloud handles everything else.

---

### Existing Integration

Routes requests to a conversation agent already configured in Home Assistant — Gemini,
ChatGPT, Claude, and others.

**Best for:** Complex reasoning, general knowledge, cloud AI

**Prerequisites:** The HA integration must be fully configured under
Settings → Devices & Services before it appears in the entity selector.

**Configuration fields:**

| Field | Default | Description |
|---|---|---|
| Agent Name | — | Friendly label (e.g. "Gemini Flash") |
| Entity ID | — | Conversation agent entity selector |
| Priority | 50 | 1–100 |
| Timeout | 5 s | 5–120 seconds |
| Enabled | true | Enable or disable without deleting |

---

### Ollama (Self-Hosted)

Connects to a locally running Ollama server via HTTP API.

**Best for:** Privacy-first setups, router agents, cost savings, multi-turn conversations

**Prerequisites:** Ollama must be running and the model must be pulled before adding this agent.
NeuralBridge validates the URL and model name during setup.

**Configuration fields:**

| Field | Default | Description |
|---|---|---|
| Agent Name | — | Friendly label (e.g. "Llama3 Local") |
| Ollama URL | `http://localhost:11434` | Server URL |
| Model | — | Model name (e.g. `llama3:8b`) — validated live |
| Priority | 50 | 0–100 (0 = router agent) |
| Timeout | 5 s | 5–120 seconds |
| Enabled | true | Enable or disable without deleting |
| System Prompt | *(empty)* | Per-agent system prompt. Falls back to the global default prompt when empty. |
| Max Retries | 2 | Retry attempts on failure (exponential back-off) |
| Retry Base Delay | 1.0 s | Initial delay between retries (doubles each attempt) |
| Is Router | false | Enable JSON classification mode (set automatically when priority is 0) |
| Enable Response Cache | true | Cache responses for this agent |

**On Home Assistant OS:** Use the LAN IP of the machine running Ollama, not `localhost`.

---

## Router Agents (Priority 0, Ollama Only)

Router agents are optional. When configured, they run before all processing agents and
classify each request, returning a JSON routing decision:

```json
{"local_ha": true, "complexity": 15}
```

| Field | Type | Description |
|---|---|---|
| `local_ha` | boolean | `true` when the request is a home-automation command — pins routing to LOCAL_HA agents |
| `complexity` | integer (0–100) | `0` = block the request outright. Higher values prefer more capable agents. |

NeuralBridge uses a built-in classification prompt automatically. You can override it with a
**custom router prompt** per router agent.

### Router-Specific Fields

| Field | Default | Description |
|---|---|---|
| Router Timeout | 5 s | Separate, shorter timeout for the routing decision |
| Fallback | `default_complexity` | Behaviour when the router errors or times out (see below) |
| Log Level | `none` | How much routing detail to log (see below) |
| Custom Prompt | *(empty)* | Override the built-in classification prompt |

### Router Fallback Options

| Value | Behaviour |
|---|---|
| `default_complexity` | Fail-open: treat as complexity 50, pass all agents through |
| `skip_routing` | Skip routing entirely — pass all processing agents unchanged |
| `block` | Block the request when the router fails |

### Router Log Levels

| Value | What is logged |
|---|---|
| `none` | Nothing (default) |
| `complexity_only` | Complexity score and local_ha flag |
| `debug_info` | Full routing decision details |
| `debug_with_query` | Full details including the user's query text (⚠ logs PII) |

### Recommended Router Models

| Model | Size | Speed |
|---|---|---|
| `qwen2.5:0.5b` | 0.5B | Very fast — best all-round choice |
| `tinyllama:1.1b` | 1.1B | Extremely fast — minimal resources |
| `phi2:2.7b` | 2.7B | More capable — still fast |

### Example Router Configuration

```
Agent Name:    Qwen Router
Agent Type:    Ollama
Priority:      0
URL:           http://192.168.1.100:11434
Model:         qwen2.5:0.5b
Router Timeout: 5 s
Fallback:      default_complexity
Log Level:     complexity_only
```

---

## Guard Rails

Guard rails check content before it reaches the agent (input) and before it is returned to
the user (output). A `neuralbridge_guard_rail_triggered` HA event is fired on every flag.

| Field | Default | Description |
|---|---|---|
| Enabled | false | Toggle guard rails for this agent |
| Action | `notify_ask` | What to do when content is flagged |
| AI Threshold | 0.7 | Confidence (0.5–1.0) required to flag |
| Use Detoxify | false | Opt-in ML model for enhanced toxicity detection |
| Detoxify Threshold | 0.7 | Confidence threshold for Detoxify |
| Rules — Harmful | true | Check for harmful content |
| Rules — Privacy | true | Check for privacy violations |
| Rules — Security | true | Check for security risks |
| Rules — Inappropriate | true | Check for inappropriate material |

### Actions

| Action | Behaviour |
|---|---|
| `block` | Response suppressed; user receives a blocked-content message |
| `warn` | Response returned with a warning prefix |
| `notify_ask` | Response held; user asked to confirm before it is shown |

---

## Global Settings

Accessible from **Configure → menu options**:

### Default Prompt (Ollama Global Fallback)

A global system prompt used by any Ollama agent whose per-agent system prompt is empty.

**Default value:**
```
You are a voice assistant for Home Assistant.
Answer questions about the world truthfully.
Answer in the style of a witty British butler, answer only in plain text; keep it
simple, to the point, and avoid swearing.

Answer with time in 24-hour format and state the current timezone.
When saying a date use the format day month year eg 5th of January 2025.
```

### Response Cache

| Field | Default | Description |
|---|---|---|
| Cache Enabled | true | Toggle the response cache globally |
| Cache TTL | 300 s | How long cached responses are kept |

To purge the cache immediately: **Configure → Advanced Settings → Purge Cache Now**

### Language

Sets the language used for NeuralBridge's own UI strings and response messages.
Default: `en_gb`.

---

## Example Configurations

### Privacy-First: All Local

```
Priority 10 │ Home Assistant  │ assist_mode=true
Priority 20 │ Llama3 8B       │ system_prompt="you are a helpful assistant"
Priority 30 │ Mixtral 8x7B    │ fallback for complex queries
```

### Hybrid: Local + Cloud

```
Priority  0 │ Qwen2.5 0.5B      │ router — local_ha + complexity
Priority 10 │ Home Assistant    │ assist_mode=true
Priority 20 │ Llama3 8B         │ privacy-safe general queries
Priority 30 │ Gemini Flash      │ complex reasoning
Priority 40 │ ChatGPT           │ final fallback
```

### Cost-Optimised

```
Priority  0 │ Qwen2.5 0.5B   │ router — filter + classify
Priority 10 │ Home Assistant │ free, local
Priority 20 │ Llama3 8B      │ free, local
Priority 30 │ Gemini Flash   │ paid — only reached for hard queries
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Ollama connection fails | Check URL, run `ollama list`, verify port 11434 |
| Cloud agent not in selector | Configure the HA integration first |
| HA agent falls through unexpectedly | Disable Assist Mode or check intent matching |
| Router always blocks | Set log level to `complexity_only`, check model output format |
| Guard rails false-positive | Raise threshold, switch to `warn`, or disable a rule category |
| Agent appears in list but gets skipped | Check if circuit breaker has tripped (look for WARNING in HA logs) |
