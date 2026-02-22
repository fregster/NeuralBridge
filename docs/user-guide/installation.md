# Installation Guide

## Prerequisites

- Home Assistant 2026.1 or newer
- HACS installed (recommended) or ability to copy files manually
- At least one AI backend — Ollama, or a cloud integration (Gemini, ChatGPT, etc.) already configured in HA

## HACS Installation (Recommended)

1. Open HACS in your Home Assistant instance
2. Click **Integrations**
3. Click the three-dot menu in the top right
4. Select **Custom repositories**
5. Add the repository:
   - URL: `https://github.com/pfrye/NeuralBridge`
   - Category: Integration
6. Click **Install** on the NeuralBridge card
7. Restart Home Assistant

## Manual Installation

1. Download the latest release from [GitHub Releases](https://github.com/pfrye/NeuralBridge/releases)
2. Extract the archive
3. Copy the `custom_components/neuralbridge/` folder into your Home Assistant `config/custom_components/` directory
4. Restart Home Assistant

## Initial Setup

NeuralBridge installs with no agents configured — no setup wizard is shown.

1. Go to **Settings → Devices & Services**
2. Click **+ Add Integration**
3. Search for **NeuralBridge**
4. Click **Submit** (no options required at this step)
5. The integration is now active but has no agents yet

## Adding Your First Agent

1. Click **Configure** on the NeuralBridge integration card
2. Select **Add New Agent** from the menu
3. Choose the agent type:
   - **Home Assistant** — uses the built-in HA conversation agent (no network required)
   - **Existing Integration** — routes to a cloud agent already configured in HA
   - **Ollama** — connects to a self-hosted Ollama instance
4. Fill in the details and click **Submit**
5. For Ollama agents, the URL and model name are validated before saving

## Setting NeuralBridge as Your Voice Assistant

1. Go to **Settings → Voice Assistants**
2. Create or edit a voice assistant
3. Set the **Conversation agent** to **NeuralBridge**

## Verification

With at least one agent configured, test with a simple query in the Assist chat panel.
Try: "What time is it?" or "Turn on the lights."

## Enabling Debug Logging

```yaml
# configuration.yaml
logger:
  default: warning
  logs:
    custom_components.neuralbridge: debug
```

Restart Home Assistant after adding this. Routing decisions, agent selections, circuit
breaker events, and cache hits/misses will appear in the HA logs.

## Troubleshooting

| Problem | Solution |
|---|---|
| Ollama connection fails during setup | Check the URL (`http://host:11434`), ensure Ollama is running |
| Cloud agent not in entity selector | The HA integration (Gemini, ChatGPT, etc.) must be configured first |
| No response from NeuralBridge | Verify at least one agent is configured and enabled |
| "No AI agents are configured" message | Add an agent via Configure → Add New Agent |
