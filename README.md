# NeuralBridge

> Priority-based AI agent routing for Home Assistant's Assist pipeline

[![HACS Custom](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/hacs/integration)
[![HA Version](https://img.shields.io/badge/Home%20Assistant-2026.1%2B-blue.svg)](https://www.home-assistant.io/)
[![License](https://img.shields.io/github/license/pfrye/NeuralBridge)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/pfrye/NeuralBridge/releases)

NeuralBridge turns Home Assistant's Assist pipeline into an intelligent, multi-agent routing system. Configure any number of AI agents — local, cloud, or self-hosted — and NeuralBridge automatically tries them in priority order, failing over gracefully when one is unavailable.

---

## Contents

- [How It Works](#how-it-works)
- [Agent Types](#agent-types)
- [Priority System](#priority-system)
- [Router Agents](#router-agents)
- [Guard Rails](#guard-rails)
- [Circuit Breaker](#circuit-breaker)
- [Response Cache](#response-cache)
- [Session Memory](#session-memory)
- [Statistics Sensors](#statistics-sensors)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Example Setups](#example-setups)
- [Ollama Setup](#ollama-setup)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Contributing](#contributing)

---

## How It Works

Every request to the Assist pipeline flows through NeuralBridge in three stages:

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 1: Router Agents (Priority 0)    │
│  Small Ollama models classify the       │
│  request. Returns local_ha flag +       │
│  complexity score (0=block, 1-100).     │
│  Optional — skipped if none configured. │
└─────────────────┬───────────────────────┘
                  │ RouterDecision
                  ▼
┌─────────────────────────────────────────┐
│  Stage 2: Processing Agents (1–100)     │
│  Tried in priority order (lowest first) │
│  until one succeeds.  local_ha flag     │
│  pins routing to LOCAL_HA agents when   │
│  set. First success wins.               │
└─────────────────┬───────────────────────┘
                  │ Response
                  ▼
┌─────────────────────────────────────────┐
│  Stage 3: Guard Rails (Optional)        │
│  Output checked for harmful content.    │
│  Action: block / warn / notify_ask      │
└─────────────────────────────────────────┘
                  │
                  ▼
              Response
```

---

## Agent Types

NeuralBridge supports three agent types that can be mixed freely in a single configuration.

### Home Assistant (Built-in)

Uses Home Assistant's native conversation agent for local intent matching.

| Property | Value |
|---|---|
| Latency | < 100 ms |
| Privacy | 100% local |
| Best for | Device control, automations |

**Examples:** *"Turn off the kitchen lights"*, *"Set the thermostat to 20 degrees"*, *"Lock the front door"*

**Assist Mode** (optional): When enabled, only responses where HA successfully matched and executed a device-control intent (`action_done`) are accepted. Unrecognised queries fall through to the next agent. This creates a clean hybrid: HA handles device commands; everything else escalates to Ollama or cloud.

### Existing Integration

Connects to any conversation agent already configured in Home Assistant — Gemini, ChatGPT, Claude, and others.

| Property | Value |
|---|---|
| Latency | 1–3 seconds |
| Privacy | Data sent to cloud provider |
| Best for | Complex reasoning, general knowledge |

Requires the cloud integration to be already set up under Settings → Devices & Services.

### Ollama (Self-Hosted)

Communicates with a locally running [Ollama](https://ollama.com) server via HTTP API.

| Property | Value |
|---|---|
| Latency | 0.5–10 seconds (model dependent) |
| Privacy | 100% local |
| Best for | Privacy-first setups, router agents, cost savings |

NeuralBridge validates the Ollama URL and model name during setup. A per-agent **system prompt** can be configured; when left blank, the global default prompt is used.

---

## Priority System

Every agent is assigned a priority from **0 to 100**.

| Priority | Role | Behaviour |
|---|---|---|
| **0** | Router / Filter | Runs first. Classifies the request and returns a routing decision. Use small, fast Ollama models here. |
| **1–20** | High priority | Tried first among processing agents. Ideal for fast local agents. |
| **21–50** | Medium priority | Tried after high-priority agents fail. |
| **51–100** | Low priority / Fallback | Last resort. Cloud agents or expensive models fit here. |

**Rules:**
- Multiple agents can share the same priority — they are tried in the order they were added.
- If an agent times out or returns an error, NeuralBridge automatically tries the next one.
- The first agent to return a successful response wins; remaining agents are skipped.
- If all agents fail, a graceful fallback message is returned.

---

## Router Agents

Router agents (priority 0) are Ollama agents that run before all processing agents. They classify each incoming request and return a JSON routing decision:

```json
{"local_ha": true, "complexity": 15}
```

| Field | Type | Description |
|---|---|---|
| `local_ha` | boolean | `true` when the request is a home-automation / device-control command — pins routing to LOCAL_HA agents |
| `complexity` | integer (0–100) | Estimated complexity. `0` = block the request outright. Higher values prefer more capable agents. |

The built-in classification prompt is used automatically. Each router agent can override it with a **custom router prompt**.

### Router Options

| Option | Default | Description |
|---|---|---|
| Router timeout | 5 s | Separate, shorter timeout for the routing decision |
| Log level | `none` | Controls how much routing detail is logged. Options: `none`, `complexity_only`, `debug_info`, `debug_with_query` (⚠ logs user text / PII) |
| Fallback | `default_complexity` | What to do when the router errors or times out: `default_complexity` (score 50, fail-open), `skip_routing` (pass all agents unchanged), `block` |
| Custom prompt | *(empty)* | Override the built-in classification prompt |

### Recommended Router Models

| Model | Size | Speed | Notes |
|---|---|---|---|
| `qwen2.5:0.5b` | 0.5B | Very fast | Excellent classifier |
| `tinyllama:1.1b` | 1.1B | Extremely fast | Minimal resource use |
| `phi2:2.7b` | 2.7B | Fast | More capable classifier |

---

## Guard Rails

Guard rails provide content filtering on a per-agent basis. Input is checked before it reaches the agent; output is checked before it is returned to the user. A `neuralbridge_guard_rail_triggered` event is fired in Home Assistant on every flag.

### Configuration

| Setting | Default | Description |
|---|---|---|
| Enabled | `false` | Toggle guard rails on/off for the agent |
| Action | `notify_ask` | What to do when content is flagged |
| AI Threshold | `0.7` | Confidence level (0.5–1.0) required to flag content |
| Use Detoxify | `false` | Opt-in ML model for enhanced toxicity detection |
| Detoxify Threshold | `0.7` | Confidence threshold for the Detoxify model |
| Rules | all enabled | Categories to check: harmful, privacy, security, inappropriate |

### Actions

| Action | Behaviour |
|---|---|
| `block` | Response is suppressed; user receives a blocked-content message |
| `warn` | Response is returned with a warning prefix |
| `notify_ask` | Response is held; user is asked to confirm before it is shown |

---

## Circuit Breaker

Each agent has a circuit breaker that automatically skips it during a cooldown period when it has failed too many times in a row.

| Setting | Default | Description |
|---|---|---|
| Failure threshold | 3 | Consecutive failures before the circuit trips |
| Cooldown | 60 s | Time before the agent is retried |

When a circuit trips, NeuralBridge skips that agent and moves to the next — preventing timeout accumulation when a service is known-down. The circuit resets automatically after the cooldown.

---

## Response Cache

NeuralBridge caches identical queries per agent to avoid redundant backend calls.

| Setting | Default | Description |
|---|---|---|
| Cache enabled (global) | `true` | Toggle the response cache globally |
| Cache TTL | 300 s (5 min) | How long cached responses are kept |
| Cache enabled (per-agent) | `true` | Override cache on/off for a specific agent |

To purge the cache: **Configure → Advanced Settings → Purge Cache Now**

---

## Session Memory

Ollama agents maintain per-conversation message history so multi-turn exchanges work naturally:

> Turn 1: "Who wrote Dune?"
> Turn 2: "What else did they write?"

| Setting | Default | Description |
|---|---|---|
| Max turns per session | 20 pairs | Oldest turns are dropped when exceeded |
| Session TTL | 30 min | Sessions expire after this period of inactivity |
| Max sessions | 100 | Oldest session evicted when the limit is reached |

Session memory resets on Home Assistant restart. To clear a session manually:

```yaml
service: neuralbridge.clear_conversation
data:
  conversation_id: "your-session-id"
```

---

## Statistics Sensors

Each agent exposes a statistics sensor in Home Assistant:

**Entity:** `sensor.neuralbridge_<agent_name>_statistics`

| Attribute | Description |
|---|---|
| `request_count` | Total requests processed |
| `success_count` | Successful responses |
| `failure_count` | Failures and errors |
| `timeout_count` | Timed-out requests |
| `block_count` | Requests blocked by guard rails |
| `average_latency_ms` | Average latency (successful calls only) |
| `success_rate` | Fraction of requests that succeeded (0.0–1.0) |
| `queries_per_hour` | Estimated request rate since first request |

---

## Requirements

- **Home Assistant** 2026.1 or newer
- **Python** 3.11+ (provided by Home Assistant)
- **aiohttp** 3.8.0+ (provided by Home Assistant)
- **better-profanity** 0.7.0+ (installed automatically)
- **Ollama** (optional — only required if using Ollama agents)

---

## Installation

### HACS (Recommended)

1. Open HACS in Home Assistant
2. Go to **Integrations**
3. Click the three-dot menu → **Custom repositories**
4. Add `https://github.com/pfrye/NeuralBridge` with category **Integration**
5. Click **Install** on the NeuralBridge card
6. Restart Home Assistant

### Manual

1. Download the [latest release](https://github.com/pfrye/NeuralBridge/releases/latest)
2. Extract and copy the `custom_components/neuralbridge/` folder into your Home Assistant `config/custom_components/` directory
3. Restart Home Assistant

---

## Configuration

### Step 1 — Add the Integration

1. Go to **Settings → Devices & Services**
2. Click **+ Add Integration**
3. Search for **NeuralBridge**
4. Click **Submit** — no initial configuration required

NeuralBridge installs with zero agents. It is fully operational once you add at least one.

### Step 2 — Add an Agent

1. Click **Configure** on the NeuralBridge card
2. Select **Add New Agent**
3. Choose the agent type and fill in the details:

**Home Assistant**

| Field | Description |
|---|---|
| Name | Friendly label (e.g. "Local Intents") |
| Priority | 1–100 (lower = higher priority) |
| Timeout | 5–120 seconds |
| Assist Mode | Fall through on non-intent responses (optional) |

**Existing Integration**

| Field | Description |
|---|---|
| Name | Friendly label |
| Entity | Conversation agent entity selector |
| Priority | 1–100 |
| Timeout | 5–120 seconds |

**Ollama**

| Field | Description |
|---|---|
| Name | Friendly label (e.g. "Llama3 Local") |
| URL | Ollama server URL (default: `http://localhost:11434`) |
| Model | Model name (e.g. `llama3:8b`) — validated live |
| Priority | 0–100 (0 = router agent) |
| Timeout | 5–120 seconds |
| System Prompt | Per-agent prompt (falls back to global default if empty) |
| Max Retries | Retry attempts on failure (default: 2) |
| Retry Base Delay | Seconds between retries — exponential back-off (default: 1.0) |
| Is Router | Enable JSON classification mode (set automatically when priority is 0) |

4. Optionally enable **Guard Rails** and configure the action and threshold
5. Click **Submit** — NeuralBridge validates Ollama connections before saving

### Step 3 — Set as Voice Assistant

1. Go to **Settings → Voice Assistants**
2. Create or edit a voice assistant
3. Set the **Conversation agent** to **NeuralBridge**

### Step 4 — Manage Agents

From the **Configure** menu you can:
- View all configured agents
- Enable or disable individual agents without deleting them
- Delete agents that are no longer needed

### Step 5 — Use It

Talk to Home Assistant Assist as normal. NeuralBridge handles routing transparently.

```yaml
# configuration.yaml — enable debug logging
logger:
  default: warning
  logs:
    custom_components.neuralbridge: debug
```

---

## Example Setups

### Beginner: Local + Cloud Fallback

Handles device control locally; falls back to a cloud model for everything else.

```
Priority 10 │ Home Assistant     │ Local intents
Priority 20 │ ChatGPT            │ General queries
```

### Privacy-First: All Local with Ollama

No data ever leaves the home.

```
Priority 10 │ Home Assistant     │ Device control
Priority 20 │ Llama3 8B          │ General queries
Priority 30 │ Mixtral 8x7B       │ Complex queries (fallback)
```

### Smart Router: Classify Before Processing

A tiny router model (< 1 second) classifies requests. `local_ha=true` pins the request to Home Assistant; everything else routes by complexity score.

```
Priority  0 │ Qwen2.5 0.5B       │ Router — local_ha + complexity
Priority 10 │ Home Assistant     │ Device control
Priority 20 │ Gemini Flash       │ Cloud AI
```

### Cost-Optimised: Minimise Cloud API Spend

The router filters requests and scores complexity so cheap local models handle the majority of load.

```
Priority  0 │ Qwen2.5 0.5B       │ Router — classify & filter
Priority 10 │ Home Assistant     │ Free, local
Priority 20 │ Llama3 8B          │ Free, local
Priority 30 │ Gemini Flash       │ Paid cloud (fallback only)
```

### Multi-Cloud with Redundancy

Maximum reliability using multiple cloud providers.

```
Priority 10 │ Home Assistant     │ Local intents
Priority 20 │ Gemini Flash       │ Primary cloud (fast, cheap)
Priority 30 │ ChatGPT            │ Secondary cloud
Priority 40 │ Claude             │ Tertiary cloud
```

See [docs/CONFIGURATION_EXAMPLES.md](docs/CONFIGURATION_EXAMPLES.md) for complete annotated configurations.

---

## Ollama Setup

Install Ollama on the same machine as Home Assistant (or any reachable host):

```bash
# Install Ollama (Linux / macOS)
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull qwen2.5:0.5b    # Lightweight router
ollama pull llama3:8b       # General processing
ollama pull mixtral:8x7b    # Complex queries (requires more RAM)
```

Then in NeuralBridge:
- **URL**: `http://localhost:11434` (or your server's IP)
- **Model**: the exact model name from `ollama list`

On Home Assistant OS, use the machine's LAN IP instead of `localhost`.

---

## Troubleshooting

### No response / fallback message shown

- Verify at least one agent is configured and enabled: **Configure → Manage Agents**
- Check HA logs for routing errors: `custom_components.neuralbridge`
- Confirm the agent's target service is running (Ollama, cloud integration)

### Ollama connection failed during setup

- Confirm Ollama is running: `ollama list`
- Check the URL scheme (`http://` or `https://`)
- On Home Assistant OS, use the LAN IP — not `localhost`
- Verify firewall rules allow port 11434

### Existing integration agent not appearing

- The cloud integration must be fully configured under Settings → Devices & Services first
- The entity must expose a conversation agent

### All agents failing / slow responses

- Lower agent timeouts to fail faster and try the next agent sooner
- Enable debug logging to see which agent is failing and why
- Check the statistics sensors to identify agents with high failure rates
- Cloud API keys / quotas may be exhausted

### Router blocking requests unexpectedly

- Set router log level to `complexity_only` or `debug_info` to see what the router is returning
- Set router fallback to `skip_routing` to bypass the router on error
- Adjust the blocking threshold by customising the router prompt

### Guard rails blocking expected content

- Raise the AI Threshold closer to 1.0
- Switch the action from `block` to `warn` or `notify_ask`
- Disable the specific rule category triggering false positives

See [docs/user-guide/agent-configuration.md](docs/user-guide/agent-configuration.md) for full configuration options.

---

## Development

### Prerequisites

```bash
pip install -r requirements_test.txt
```

### Running Tests

```bash
# Run all tests with coverage
pytest --cov --cov-fail-under=100

# Run a specific test file
pytest tests/unit/test_conversation.py -v

# View HTML coverage report
pytest --cov --cov-report=html
open htmlcov/index.html
```

### Code Quality

```bash
# Format
black custom_components tests

# Lint (includes security checks)
ruff check custom_components tests

# Type check
mypy custom_components
```

All checks must pass before committing. CI enforces 100% test coverage, Black formatting, Ruff linting, and Mypy strict type checking.

See [docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md) and [docs/CODING_STANDARDS.md](docs/CODING_STANDARDS.md) for full standards.

### Project Layout

```
NeuralBridge/
├── custom_components/neuralbridge/   # Integration source
│   ├── conversation.py               # Core routing logic
│   ├── config_flow.py                # UI configuration flows
│   ├── const.py                      # All constants and defaults
│   ├── circuit_breaker.py            # Per-agent failure tracking
│   ├── guard_rail.py                 # Input/output content filtering
│   ├── ollama_client.py              # Async Ollama HTTP client
│   ├── response_cache.py             # TTL response cache
│   ├── session_memory.py             # Ollama conversation history
│   ├── statistics.py                 # Request/latency counters
│   └── sensor.py                     # HA statistics sensors
├── tests/                            # Unit + integration tests
├── docs/                             # All documentation
└── config/examples/                  # Sample configurations
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests first (TDD)
4. Implement the feature
5. Ensure all checks pass: `black . && ruff check custom_components tests && mypy custom_components && pytest --cov --cov-fail-under=100`
6. Open a pull request

Please read [docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md) before contributing.

---

## Documentation

| Document | Description |
|---|---|
| [Quick Start](docs/user-guide/quick-start.md) | 5-minute setup guide |
| [Installation Guide](docs/user-guide/installation.md) | Full installation walkthrough |
| [Agent Configuration](docs/user-guide/agent-configuration.md) | Detailed agent setup and all options |
| [Usage Guide](docs/user-guide/usage.md) | Day-to-day usage and tips |
| [Configuration Examples](docs/CONFIGURATION_EXAMPLES.md) | Real-world setup scenarios |
| [Architecture](docs/architecture/README.md) | System design and component map |
| [Development Guide](docs/DEVELOPMENT_GUIDE.md) | Coding standards and workflow |
| [Coding Standards](docs/CODING_STANDARDS.md) | Quick reference for code quality rules |

---

## Support

- **Bug reports & feature requests**: [GitHub Issues](https://github.com/pfrye/NeuralBridge/issues)
- **Questions & discussion**: [GitHub Discussions](https://github.com/pfrye/NeuralBridge/discussions)

---

## License

See [LICENSE](LICENSE) for details.
