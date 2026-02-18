# NeuralBridge

> Priority-based AI agent routing for Home Assistant's Assist pipeline

[![HACS Custom](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/hacs/integration)
[![HA Version](https://img.shields.io/badge/Home%20Assistant-2024.1%2B-blue.svg)](https://www.home-assistant.io/)
[![License](https://img.shields.io/github/license/pfrye/NeuralBridge)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/pfrye/NeuralBridge/releases)

NeuralBridge turns Home Assistant's Assist pipeline into an intelligent, multi-agent routing system. Configure any number of AI agents — local, cloud, or self-hosted — and NeuralBridge automatically tries them in priority order, failing over gracefully when one is unavailable.

---

## Contents

- [How It Works](#how-it-works)
- [Agent Types](#agent-types)
- [Priority System](#priority-system)
- [Guard Rails](#guard-rails)
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
│  Small, fast models classify the        │
│  request and decide whether to allow,   │
│  block, or flag it.  (Optional)         │
└─────────────────┬───────────────────────┘
                  │ Allowed
                  ▼
┌─────────────────────────────────────────┐
│  Stage 2: Processing Agents (1–100)     │
│  Tried in priority order (lowest first) │
│  until one succeeds.  First success     │
│  wins — others are skipped.             │
└─────────────────┬───────────────────────┘
                  │ Response
                  ▼
┌─────────────────────────────────────────┐
│  Stage 3: Guard Rails (Optional)        │
│  Output checked for harmful content.   │
│  Action: block / warn / notify & ask    │
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

### Existing Integration

Connects to any conversation agent already configured in Home Assistant — Gemini, ChatGPT, Claude, and others.

| Property | Value |
|---|---|
| Latency | 1–3 seconds |
| Privacy | Data sent to cloud provider |
| Best for | Complex reasoning, general knowledge |

Requires the cloud integration to be already set up under Settings → Devices & Services. NeuralBridge simply routes requests to it.

### Ollama (Self-Hosted)

Communicates with a locally running [Ollama](https://ollama.com) server via HTTP API.

| Property | Value |
|---|---|
| Latency | 0.5–10 seconds (model dependent) |
| Privacy | 100% local |
| Best for | Privacy-first setups, router agents, cost savings |

NeuralBridge validates the Ollama URL and model name during setup, so misconfigured agents are caught before they're saved.

---

## Priority System

Every agent is assigned a priority from **0 to 100**.

| Priority | Role | Behaviour |
|---|---|---|
| **0** | Router / Filter | Runs first. Classifies the request and decides whether to allow it through. Use small, fast models here. |
| **1–20** | High priority | Tried first among processing agents. Ideal for fast local agents. |
| **21–50** | Medium priority | Tried after high-priority agents fail. |
| **51–100** | Low priority / Fallback | Last resort. Cloud agents or expensive models fit here. |

**Rules:**
- Multiple agents can share the same priority — they are tried in the order they were added.
- If an agent times out or returns an error, NeuralBridge automatically tries the next one.
- The first agent to return a successful response wins; remaining agents are skipped.
- If all agents fail, a graceful fallback message is returned.

### Recommended Router Models

Small models used at priority 0 are cheap to run and give results in under a second:

| Model | Size | Speed | Notes |
|---|---|---|---|
| `qwen2.5:0.5b` | 0.5B | Very fast | Excellent classifier |
| `tinyllama:1.1b` | 1.1B | Extremely fast | Minimal resource use |
| `phi2:2.7b` | 2.7B | Fast | More capable classifier |

---

## Guard Rails

Guard rails provide output filtering on a per-agent basis. When enabled, NeuralBridge checks the agent's response before returning it to the user.

### Configuration

| Setting | Default | Description |
|---|---|---|
| Enabled | `false` | Toggle guard rails on/off for the agent |
| Action | `notify_ask` | What to do when content is flagged |
| AI Threshold | `0.7` | Confidence level (0.5–1.0) required to flag content |

### Actions

| Action | Behaviour |
|---|---|
| `block` | Response is suppressed; user receives a blocked-content message |
| `warn` | Response is returned with a warning prefix |
| `notify_ask` | Response is held; user is asked to confirm before it is shown |

### Flagged Categories

Guard rails check for: harmful content, privacy violations, security risks, and inappropriate material.

---

## Requirements

- **Home Assistant** 2024.1 or newer
- **Python** 3.11+ (provided by Home Assistant)
- **aiohttp** 3.8.0+ (provided by Home Assistant)
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
| Priority | 0–100 (lower = higher priority) |
| Timeout | 5–120 seconds |

**Existing Integration**

| Field | Description |
|---|---|
| Name | Friendly label |
| Entity | Conversation agent entity selector |
| Priority | 0–100 |
| Timeout | 5–120 seconds |

**Ollama**

| Field | Description |
|---|---|
| Name | Friendly label (e.g. "Llama3 Local") |
| URL | Ollama server URL (default: `http://localhost:11434`) |
| Model | Model name (e.g. `llama3:8b`) — validated live |
| Priority | 0–100 |
| Timeout | 5–120 seconds |

4. Optionally enable **Guard Rails** and configure the action and threshold
5. Click **Submit** — NeuralBridge validates Ollama connections before saving

### Step 3 — Manage Agents

From the **Configure** menu you can:
- View all configured agents
- Delete agents that are no longer needed

### Step 4 — Use It

Talk to Home Assistant Assist as normal. NeuralBridge handles routing transparently. Enable debug logging to see routing decisions in the HA logs.

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

### Smart Router: Filter Before Cloud

A tiny router model (< 1 second) classifies requests. Simple queries stay local; complex ones escalate to cloud.

```
Priority  0 │ Qwen2.5 0.5B       │ Router — classify & filter
Priority 10 │ Home Assistant     │ Device control
Priority 20 │ Gemini Pro         │ Cloud AI
```

### Cost-Optimized: Minimize Cloud API Spend

The router filters unnecessary cloud calls. Local models handle most load; cloud is a last resort.

```
Priority  0 │ Qwen2.5 0.5B       │ Router — skip trivial requests
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

Ollama must be accessible from the Home Assistant host. On Home Assistant OS, use the machine's LAN IP instead of `localhost`.

---

## Troubleshooting

### No response / fallback message shown

- Verify at least one agent is configured: **Configure → Manage Agents**
- Check HA logs for routing errors: `custom_components.neuralbridge`
- Confirm the agent's target service is running (Ollama, cloud integration)

### Ollama connection failed during setup

- Confirm Ollama is running: `ollama list`
- Check the URL scheme (`http://` or `https://`)
- On Home Assistant OS, use the LAN IP of the machine running Ollama — not `localhost`
- Verify firewall rules allow port 11434

### Existing integration agent not appearing

- The cloud integration (e.g. Gemini, ChatGPT) must be fully configured under Settings → Devices & Services first
- The entity must expose a conversation agent

### All agents failing / slow responses

- Lower agent timeouts to fail faster and try the next agent sooner
- Enable debug logging to see exactly which agent is failing and why
- Check that cloud API keys / quotas are valid

### Guard rails blocking expected content

- Raise the AI Threshold closer to 1.0 to require higher confidence before flagging
- Switch the action from `block` to `warn` or `notify_ask` to avoid hard blocks
- Disable guard rails on that specific agent if filtering is not needed

See [docs/user-guide/agent-configuration.md](docs/user-guide/agent-configuration.md) for detailed troubleshooting.

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
ruff check .

# Type check
mypy custom_components
```

All checks must pass before committing. CI enforces 100% test coverage, Black formatting, Ruff linting, and Mypy strict type checking.

See [docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md) and [docs/CODING_STANDARDS.md](docs/CODING_STANDARDS.md) for full standards.

### Project Layout

```
NeuralBridge/
├── custom_components/neuralbridge/   # Integration source
├── tests/                            # Unit + integration tests
├── docs/                             # All documentation
├── config/examples/                  # Sample configurations
└── .github/                          # CI/CD workflows
```

See [docs/STRUCTURE.md](docs/STRUCTURE.md) for the complete layout.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests first (TDD)
4. Implement the feature
5. Ensure all checks pass: `black . && ruff check . && mypy custom_components && pytest --cov --cov-fail-under=100`
6. Open a pull request

Please read [docs/DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md) before contributing.

---

## Documentation

| Document | Description |
|---|---|
| [Quick Start](docs/user-guide/quick-start.md) | 30-second setup guide |
| [Installation Guide](docs/user-guide/installation.md) | Full installation walkthrough |
| [Agent Configuration](docs/user-guide/agent-configuration.md) | Detailed agent setup and options |
| [Configuration Examples](docs/CONFIGURATION_EXAMPLES.md) | Real-world setup scenarios |
| [Usage Guide](docs/user-guide/usage.md) | Day-to-day usage and tips |
| [Architecture](docs/architecture/README.md) | System design and routing logic |
| [API Reference](docs/api/conversation_entity.md) | Developer API documentation |
| [Development Guide](docs/DEVELOPMENT_GUIDE.md) | Coding standards and workflow |

---

## Support

- **Bug reports & feature requests**: [GitHub Issues](https://github.com/pfrye/NeuralBridge/issues)
- **Questions & discussion**: [GitHub Discussions](https://github.com/pfrye/NeuralBridge/discussions)

---

## License

See [LICENSE](LICENSE) for details.
