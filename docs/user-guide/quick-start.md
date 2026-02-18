# NeuralBridge Quick Start Guide

## 🚀 30-Second Setup

1. **Install via HACS**
   - HACS → Integrations → Custom Repositories
   - Add: `https://github.com/pfrye/NeuralBridge`
   - Install & Restart HA

2. **Add Integration**
   - Settings → Devices & Services → Add Integration
   - Search: "NeuralBridge"
   - Submit (no config needed!)

3. **Add Your First Agent**
   - Click "Configure" on NeuralBridge
   - "Add New Agent" → Choose type
   - Fill details & Save

## 📋 Quick Reference

### Priority System

```
Priority 0     = Router/Filter (optional)
Priority 1-20  = High priority processing
Priority 21-50 = Medium priority
Priority 51-100= Low priority/fallback
```

### Agent Types

| Type | Best For | Example |
|------|----------|---------|
| **Home Assistant** | Local device control | "Turn on lights" |
| **Existing Integration** | Cloud AI (Gemini, ChatGPT) | "Explain physics" |
| **Ollama** | Self-hosted models | Privacy-first setup |

### Common Setups

**Beginner: Local + Cloud**
```
Priority 10: Home Assistant
Priority 20: ChatGPT
```

**Privacy: All Local**
```
Priority 10: Home Assistant
Priority 20: Llama3 via Ollama
```

**Advanced: Smart Router**
```
Priority 0:  Qwen 0.5B (router)
Priority 10: Home Assistant
Priority 20: Cloud AI
```

## 🛠️ Ollama Quick Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull qwen2.5:0.5b   # Router (fast)
ollama pull llama3:8b      # Main processing
```

Then in NeuralBridge:
- Agent Type: Ollama
- URL: `http://localhost:11434`
- Model: `llama3:8b`
- Priority: 20

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| No agents configured | Normal! Add agents via "Configure" |
| Ollama won't connect | Check URL & run `ollama list` |
| All agents failing | Verify config & check HA logs |
| Too slow | Use smaller models or lower timeouts |

## 📚 More Help

- **Detailed Config**: [agent-configuration.md](agent-configuration.md)
- **Examples**: [CONFIGURATION_EXAMPLES.md](../CONFIGURATION_EXAMPLES.md)
- **Issues**: [GitHub Issues](https://github.com/pfrye/NeuralBridge/issues)

## 💡 Pro Tips

1. **Start simple** - Add one agent at a time
2. **Test each agent** - Verify before adding more
3. **Use appropriate timeouts** - Fast models = short timeout
4. **Check logs** - Enable debug logging to see routing
5. **Router agents are optional** - Only needed for filtering

## 🎯 Success Checklist

- [ ] Integration installed
- [ ] At least one agent configured
- [ ] Agent tested with simple query
- [ ] Priority makes sense
- [ ] Timeout appropriate for agent

**Need more details?** See the full [README.md](../../README.md) and [documentation](../).
