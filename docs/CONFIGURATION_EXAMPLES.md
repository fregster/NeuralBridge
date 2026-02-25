# NeuralBridge Configuration Examples

## Example 1: Simple — Local + Cloud Fallback

**Use case:** Home automation with cloud backup.

**Agents:**
1. Home Assistant (Priority 10) — local device control
2. ChatGPT (Priority 20) — cloud fallback

**Setup:**
1. Add Integration → NeuralBridge
2. Configure → Add New Agent
   - Type: Home Assistant
   - Name: "Home Assistant"
   - Priority: 10, Timeout: 5 s, Assist Mode: on
3. Configure → Add New Agent
   - Type: Existing Integration
   - Name: "ChatGPT"
   - Priority: 20, Entity: `conversation.chatgpt`, Timeout: 30 s

**Result:** Local commands handled fast (< 100 ms); complex queries go to ChatGPT.

---

## Example 2: Privacy-First — All Local with Ollama

**Use case:** Everything runs locally; no cloud.

**Prerequisites:**
```bash
ollama pull llama3:8b
ollama pull mixtral:8x7b
```

**Agents:**
1. Home Assistant (Priority 10) — local intents
2. Llama3 8B (Priority 20) — main processing
3. Mixtral 8x7B (Priority 30) — complex queries

**Setup:**
1. Add Integration → NeuralBridge
2. Add Agent: Home Assistant (Priority 10, Timeout 5 s, Assist Mode on)
3. Add Agent: Ollama
   - Name: "Llama3 Main", Priority: 20, URL: `http://localhost:11434`
   - Model: `llama3:8b`, Timeout: 30 s
4. Add Agent: Ollama
   - Name: "Mixtral Complex", Priority: 30, URL: `http://localhost:11434`
   - Model: `mixtral:8x7b`, Timeout: 60 s

**Result:** All processing stays local.

---

## Example 3: Smart Router — Classify Before Processing

**Use case:** A small model classifies each request before sending to larger models.

**Prerequisites:**
```bash
ollama pull qwen2.5:0.5b
```

**Agents:**
1. Qwen Router (Priority 0) — JSON classification
2. Home Assistant (Priority 10) — local intents
3. Gemini Flash (Priority 20) — cloud AI

**Setup:**
1. Add Agent: Ollama
   - Name: "Qwen Router", Priority: **0** (this automatically enables router mode)
   - URL: `http://localhost:11434`, Model: `qwen2.5:0.5b`
   - Router Timeout: 5 s, Fallback: `default_complexity`, Log Level: `complexity_only`
2. Add Agent: Home Assistant (Priority 10, Timeout 5 s, Assist Mode on)
3. Add Agent: Existing Integration
   - Name: "Gemini Flash", Priority: 20
   - Entity: `conversation.google_generative_ai_conversation`, Timeout: 30 s

**How it works:**
- Router returns `{"local_ha": true, "complexity": 5}` → request pinned to Home Assistant
- Router returns `{"local_ha": false, "complexity": 60}` → request routed to Gemini Flash
- Router returns `{"complexity": 0}` → request blocked immediately
- Router errors or times out → fallback to complexity 50 (fail-open)

**Result:** Fast local filter that minimises unnecessary cloud calls.

---

## Example 4: Cost-Optimised — Minimise Cloud API Spend

**Use case:** Minimise cloud API costs while keeping Assist responsive.

**Prerequisites:**
```bash
ollama pull qwen2.5:0.5b
ollama pull llama3:8b
```

**Agents:**
1. Qwen Router (Priority 0) — classify and filter
2. Home Assistant (Priority 10) — free, local
3. Llama3 8B (Priority 20) — free, local
4. Gemini Flash (Priority 30) — paid cloud (fallback only)

**Result:** Most queries handled free locally; cloud only reached for hard queries.

---

## Example 5: Multi-Cloud with Redundancy

**Use case:** Maximum reliability using multiple cloud providers.

**Agents:**
1. Home Assistant (Priority 10) — local intents
2. Gemini Flash (Priority 20) — fast, cheap cloud
3. ChatGPT (Priority 30) — secondary cloud
4. Claude (Priority 40) — tertiary cloud

**Setup:** Ensure all cloud integrations are configured under Settings → Devices & Services.

**Result:** Local first, then cloud providers tried in order if previous ones fail.

---

## Advanced Configurations

### Assist Mode for Clean Hybrid Routing

Enable **Assist Mode** on the Home Assistant agent so only successfully recognised device
commands are accepted. All other queries fall through to Ollama or cloud automatically.

```
Priority 10 │ Home Assistant   │ assist_mode=true — "turn on lights" only
Priority 20 │ Llama3 8B        │ all other queries
```

### Multi-Region Ollama for Redundancy

```
Priority 10 │ Home Assistant                     │ local intents
Priority 20 │ Ollama (server1:11434) llama3:8b   │ primary Ollama
Priority 21 │ Ollama (server2:11434) llama3:8b   │ failover Ollama
Priority 30 │ Cloud integration                  │ final fallback
```

Priorities 20 and 21 are tried in sequence — if server1 is down the circuit breaker trips
after 3 failures and server2 takes over immediately.

### Per-Agent System Prompts

Each Ollama agent can have its own system prompt, or fall back to the global default:

```
Configure → Default Prompt   → "You are a voice assistant for Home Assistant..."
Agent: Llama3 Main           → system_prompt="" (uses global default)
Agent: Code Helper           → system_prompt="You are a Python coding expert..."
```

---

## Monitoring Your Configuration

Enable debug logging to see routing decisions in real time:

```yaml
# configuration.yaml
logger:
  default: warning
  logs:
    custom_components.neuralbridge: debug
```

Monitor per-agent health using the statistics sensors:

**Entity:** `sensor.neuralbridge_<agent_name>_statistics`

**Attributes:** `success_rate`, `average_latency_ms`, `failure_count`, `timeout_count`

---

## Common Issues

| Issue | Solution |
|---|---|
| All queries go to cloud | Add local Ollama agent with lower priority number |
| Ollama too slow | Use smaller models, lower timeout, or enable GPU |
| Router blocking valid requests | Set router log level to `complexity_only` to inspect decisions |
| HA agent not handling intents | Enable Assist Mode and verify HA intent configuration |
| Agent skipped unexpectedly | Check for circuit breaker trips in HA logs at WARNING level |
