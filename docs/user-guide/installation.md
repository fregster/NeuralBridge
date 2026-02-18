# Installation Guide

## Prerequisites

- Home Assistant 2024.1 or newer
- HACS installed (recommended) or manual installation capability

## HACS Installation (Recommended)

1. Open HACS in your Home Assistant instance
2. Click on **Integrations**
3. Click the three dots menu in the top right
4. Select **Custom repositories**
5. Add the repository:
   - URL: `https://github.com/pfrye/NeuralBridge`
   - Category: Integration
6. Click **Install**
7. Restart Home Assistant

## Manual Installation

1. Download the latest release from GitHub
2. Extract the `custom_components/neuralbridge` folder
3. Copy it to your Home Assistant `config/custom_components/` directory
4. Restart Home Assistant

## Configuration

1. Go to **Settings** → **Devices & Services**
2. Click **Add Integration**
3. Search for **NeuralBridge**
4. Follow the setup wizard:
   - Select your Tier 1 agent (usually `conversation.home_assistant`)
   - Enable/disable Tier 2 filtering
   - Select your Tier 3 cloud agent (Gemini, OpenAI, etc.)

## Verification

To verify the installation:

1. Go to **Settings** → **Voice assistants**
2. Select **NeuralBridge** as your conversation agent
3. Try a simple command: "What time is it?"

## Troubleshooting

See [Troubleshooting Guide](troubleshooting.md) for common issues.
