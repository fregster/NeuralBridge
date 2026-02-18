# Usage Guide

## Basic Usage

Once configured, NeuralBridge works transparently with Home Assistant's Assist pipeline.

### Voice Commands

Simply speak to your Home Assistant voice assistant as normal. NeuralBridge will automatically route your commands through the appropriate tier.

### Text Commands

In the Home Assistant Assist interface, type your commands. NeuralBridge handles routing automatically.

## Understanding Routing

### When Tier 1 Handles Your Request

Tier 1 (local) handles direct home automation commands:
- "Turn on the kitchen lights"
- "Set the thermostat to 68 degrees"
- "What's the temperature in the bedroom?"

You'll notice these are fast (sub-second response).

### When Tier 3 Handles Your Request

Tier 3 (cloud) handles complex queries:
- "What's the weather forecast for the weekend?"
- "Tell me a joke"
- "Explain how solar panels work"

These take 1-3 seconds as they involve cloud processing.

## Best Practices

1. **Start Specific:** For home automation, use specific commands
2. **Privacy Conscious:** Remember Tier 3 sends data to cloud providers
3. **Check Logs:** Enable debug logging to see routing decisions

## Advanced Features

### Custom Filtering

Configure Tier 2 filtering rules to control what gets sent to cloud providers.

### Multiple Cloud Providers

Set up fallback cloud providers for improved reliability.

## Limitations

- Tier 3 requires internet connectivity
- Cloud providers may have rate limits
- Response time varies based on tier
