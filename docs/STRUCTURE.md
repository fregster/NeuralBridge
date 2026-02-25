# NeuralBridge Directory Structure

## Root Directory

```
NeuralBridge/
├── .github/
│   ├── .copilot-instructions.md     Project development guidelines
│   └── workflows/
│       ├── test.yml                 Testing pipeline
│       └── validate.yml             HACS/Hassfest validation
├── custom_components/
│   └── neuralbridge/                Integration package
│       ├── __init__.py              Integration setup and teardown
│       ├── manifest.json            HACS and Home Assistant metadata
│       ├── const.py                 All constants, defaults, and prompt templates
│       ├── conversation.py          Core routing logic (NeuralBridgeConversationEntity)
│       ├── config_flow.py           UI configuration flows
│       ├── circuit_breaker.py       Per-agent failure tracking and bypass
│       ├── guard_rail.py            Input/output content filtering
│       ├── ollama_client.py         Async Ollama HTTP client
│       ├── response_cache.py        TTL-based response cache
│       ├── session_memory.py        Ollama per-conversation history
│       ├── statistics.py            Per-agent request/latency counters
│       ├── sensor.py                Home Assistant statistics sensors
│       ├── languages_loader.py      YAML language file loader
│       ├── strings.json             UI string keys
│       └── translations/
│           └── en.json              English UI translations
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  Shared fixtures
│   ├── unit/                        Unit tests (one file per module)
│   │   ├── test_circuit_breaker.py
│   │   ├── test_config_flow.py
│   │   ├── test_conversation.py
│   │   ├── test_guard_rail.py
│   │   ├── test_hacs_manifest.py
│   │   ├── test_init.py
│   │   ├── test_languages_loader.py
│   │   ├── test_ollama_client.py
│   │   ├── test_response_cache.py
│   │   ├── test_sensor.py
│   │   ├── test_session_memory.py
│   │   ├── test_statistics.py
│   │   └── test_translations.py
│   ├── integration/
│   │   ├── test_full_flow.py
│   │   └── test_guard_rail_integration.py
│   └── fixtures/
│       └── conversation_samples.py
├── docs/
│   ├── CODING_STANDARDS.md          Code quality quick reference
│   ├── CONFIGURATION_EXAMPLES.md    Annotated real-world setups
│   ├── DEVELOPMENT_GUIDE.md         Full development standards
│   ├── STRUCTURE.md                 This file
│   ├── architecture/
│   │   └── README.md                System design and component map
│   ├── user-guide/
│   │   ├── quick-start.md           5-minute setup guide
│   │   ├── installation.md          Full installation walkthrough
│   │   ├── agent-configuration.md   All agent options and settings
│   │   └── usage.md                 Day-to-day usage guide
│   └── api/
│       └── conversation_entity.md   Developer API reference
├── config/
│   └── examples/
│       ├── basic_configuration.yaml
│       ├── advanced_configuration.yaml
│       └── alexa_integration.yaml
├── CLAUDE.md                        Claude Code instructions (root — required)
├── LICENSE
├── README.md                        Project overview (root — required by HACS/GitHub)
├── hacs.json                        HACS metadata
├── pytest.ini                       Pytest configuration
├── pyproject.toml                   Black / Ruff / Mypy configuration
└── requirements_test.txt            Testing dependencies
```

## Key Directories

### `/custom_components/neuralbridge/`

The main integration code. This is what gets installed by users via HACS or manually.

**Core files:**
- `conversation.py` — the `NeuralBridgeConversationEntity` that handles all routing
- `config_flow.py` — the options flow UI for adding and managing agents
- `const.py` — all configuration keys, defaults, and the built-in router classification prompt

**Feature modules:**
- `circuit_breaker.py` — skips failing agents during cooldown
- `guard_rail.py` — input/output content filtering with configurable actions
- `ollama_client.py` — async HTTP client for Ollama `generate()` and `chat()` endpoints
- `response_cache.py` — TTL-based cache keyed by agent ID + query hash
- `session_memory.py` — per-session Ollama message history with TTL and eviction
- `statistics.py` — tracks requests, success/failure, latency per agent
- `sensor.py` — exposes `AgentStatistics` as HA sensor entities

### `/tests/`

100% coverage is required. Tests follow the Arrange / Act / Assert pattern with async
test functions using `pytest-homeassistant-custom-component` fixtures.

### `/docs/`

All user and developer documentation. No platform-specific generated files belong here.

### `/config/examples/`

Sample YAML snippets demonstrating real-world NeuralBridge configurations for reference.
These are not deployed or loaded automatically.
