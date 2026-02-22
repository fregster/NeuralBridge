# Usage Guide

## Basic Usage

Once configured with at least one agent, NeuralBridge works transparently through Home
Assistant's Assist pipeline — speak or type as normal and routing happens automatically.

---

## Understanding Routing

NeuralBridge tries agents in **priority order** (lowest number first). The first agent to
return a successful response wins; the others are skipped.

### Router Agent (Priority 0, Optional)

If a router agent is configured (a small Ollama model at priority 0), it runs first and
classifies each request using a JSON response:

```json
{"local_ha": true, "complexity": 15}
```

- **`local_ha: true`** — pins the request to LOCAL_HA (Home Assistant) agents
- **`complexity: 0`** — blocks the request outright with a polite refusal
- **`complexity: 1–100`** — routes to agents appropriate for the estimated complexity

Router agents are fast (under 1 second with a small model like `qwen2.5:0.5b`). When a
router errors or times out, the configured **fallback** behaviour applies (default:
fail-open with complexity 50).

### Processing Agents (Priority 1–100)

Tried in ascending priority order:

| Priority | Typical agent | Use case |
|---|---|---|
| 1–20 | Home Assistant (local) | Device control, fast local intents |
| 21–50 | Ollama (self-hosted) | General questions, privacy-focused |
| 51–100 | Cloud integration | Complex reasoning, fallback |

**Local device control examples** (typically handled by Home Assistant):
- "Turn on the kitchen lights"
- "Set the thermostat to 20 degrees"
- "Lock the front door"

**Complex queries** (typically escalate to Ollama or cloud):
- "Explain how solar panels work"
- "Write me a poem about rainy days"
- "What should I cook for dinner tonight?"

---

## Failover Behaviour

If an agent fails (timeout, error, or service unavailable), NeuralBridge automatically
tries the next agent. You will see failover in the HA logs at `WARNING` level:

```
WARNING  custom_components.neuralbridge: Agent 'Llama3 Local' (priority 20) failed: timeout
INFO     custom_components.neuralbridge: Agent 'ChatGPT' (priority 30) handled the request
```

If all agents fail, the user receives:
> "I'm having trouble connecting to my AI agents right now."

---

## Assist Mode (Home Assistant Agent)

When **Assist Mode** is enabled on a Home Assistant agent, it only accepts responses where
HA successfully matched and executed a device-control intent (`action_done`). Unrecognised
queries fall through to the next agent.

This enables a clean hybrid:
- HA handles device commands locally
- Unrecognised speech escalates to Ollama or cloud

---

## Circuit Breaker

Each agent has a circuit breaker that tracks consecutive failures. When an agent exceeds
the failure threshold (default: 3), its circuit trips and it is skipped for the cooldown
period (default: 60 seconds). This prevents timeout accumulation when a service is
known-down.

After the cooldown expires the circuit resets and one retry is allowed (half-open). A
successful call always resets the failure count.

---

## Response Cache

NeuralBridge caches responses from each agent by default (TTL: 5 minutes). Identical
queries return instantly from cache without calling the backend.

To disable caching for a specific agent (useful for real-time data queries):
- **Configure → Manage Agents → [agent] → Enable Response Cache → off**

To clear the entire cache immediately:
- **Configure → Advanced Settings → Purge Cache Now**

---

## Conversation History (Ollama Only)

Ollama agents maintain per-conversation message history within a session. Multi-turn
exchanges work naturally:

> Turn 1: "Who wrote Dune?"
> Turn 2: "What else did they write?"

Session memory expires after 30 minutes of inactivity and resets on HA restart.
To clear it manually:

```yaml
service: neuralbridge.clear_conversation
data:
  conversation_id: "your-session-id"
```

---

## Guard Rails

When guard rails are enabled for an agent, content is checked before it reaches the agent
(input) and before it is returned to the user (output).

| Action | Behaviour |
|---|---|
| `block` | Response suppressed; user receives a blocked-content message |
| `warn` | Response returned with a warning prefix |
| `notify_ask` | Response held; user asked to confirm before it is shown |

A `neuralbridge_guard_rail_triggered` HA event is fired on every flag — use this in
automations to track or alert on flagged requests.

---

## Statistics Sensors

Each agent exposes a statistics sensor:

**Entity:** `sensor.neuralbridge_<agent_name>_statistics`

**Attributes:** `request_count`, `success_count`, `failure_count`, `timeout_count`,
`block_count`, `average_latency_ms`, `success_rate`, `queries_per_hour`

Use these in Lovelace cards or automations to monitor agent health and routing patterns.

---

## Privacy

- **Local agents** (Home Assistant, Ollama) — data never leaves your network
- **Cloud agents** (Gemini, ChatGPT, etc.) — your text is sent to the cloud provider's API
- **Priority ordering** is your primary tool for controlling what reaches the cloud — put
  local agents at lower priority numbers so they are always tried first

Enable debug logging to see exactly which agent handled each request:

```yaml
# configuration.yaml
logger:
  default: warning
  logs:
    custom_components.neuralbridge: debug
```
