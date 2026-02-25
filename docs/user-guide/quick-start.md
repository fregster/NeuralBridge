# NeuralBridge Quick Start

## 5-Minute Setup

1. **Install via HACS**
   - HACS → Integrations → Custom repositories
   - Add: `https://github.com/pfrye/NeuralBridge`
   - Install and restart Home Assistant

2. **Add the integration**
   - Settings → Devices & Services → Add Integration
   - Search: "NeuralBridge"
   - Click Submit (no initial configuration required)

3. **Add your first agent**
   - Click **Configure** on the NeuralBridge card
   - Select **Add New Agent**
   - Choose agent type, fill in details, click Submit

4. **Set as voice assistant**
   - Settings → Voice Assistants → create or edit an assistant
   - Set **Conversation agent** to **NeuralBridge**

---

## Priority Quick Reference

```
Priority 0       = Router agent (Ollama only, optional)
Priority 1–20    = High priority processing agents
Priority 21–50   = Medium priority
Priority 51–100  = Low priority / fallback
```

---

## Agent Types

| Type | Best for | Example |
|---|---|---|
| **Home Assistant** | Local device control | "Turn on the lights" |
| **Existing Integration** | Cloud AI (Gemini, ChatGPT) | Complex questions |
| **Ollama** | Self-hosted models | Privacy-first setups |

---

## Common Setups

**Beginner: Local + Cloud**
```
Priority 10 │ Home Assistant
Priority 20 │ ChatGPT
```

**Privacy: All Local**
```
Priority 10 │ Home Assistant
Priority 20 │ Llama3 8B via Ollama
```

**Smart Router**
```
Priority  0 │ Qwen2.5 0.5B (router — classifies local_ha + complexity)
Priority 10 │ Home Assistant
Priority 20 │ Cloud AI
```

---

## Ollama Quick Setup

```bash
curl -fsSL https://ollama.com/install.sh | sh

ollama pull qwen2.5:0.5b   # Router (fast)
ollama pull llama3:8b      # Main processing
```

Then in NeuralBridge — Agent Type: Ollama, URL: `http://localhost:11434`, Model: `llama3:8b`.

On Home Assistant OS use the LAN IP of the machine running Ollama, not `localhost`.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| No agents configured | Normal — add agents via Configure |
| Ollama connection fails | Check URL and run `ollama list` |
| All agents failing | Verify config and check HA logs |
| Too slow | Use smaller models or lower timeouts |
| Router blocking requests | Set log level to `complexity_only` to diagnose |

---

## Debug Logging

```yaml
# configuration.yaml
logger:
  default: warning
  logs:
    custom_components.neuralbridge: debug
```

---

## Next Steps

- [Full installation guide](installation.md)
- [Agent configuration reference](agent-configuration.md)
- [Usage guide](../user-guide/usage.md)
- [Configuration examples](../CONFIGURATION_EXAMPLES.md)
- [GitHub Issues](https://github.com/pfrye/NeuralBridge/issues)
