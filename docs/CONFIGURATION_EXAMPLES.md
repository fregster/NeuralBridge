# NeuralBridge Configuration Examples

## Quick Start Examples

### Example 1: Simple Home Assistant + Cloud Fallback

**Use Case**: Home automation with cloud backup

**Agents:**
1. Home Assistant (Priority 10) - Local device control
2. ChatGPT (Priority 20) - Cloud fallback

**Setup Steps:**
1. Add Integration → NeuralBridge
2. Configure → Add New Agent
   - Type: Home Assistant (Built-in)
   - Name: "Home Assistant"
   - Priority: 10
   - Timeout: 5 seconds
3. Configure → Add New Agent
   - Type: Existing Integration
   - Name: "ChatGPT"
   - Priority: 20
   - Entity: conversation.chatgpt
   - Timeout: 30 seconds

**Result**: Local commands handled fast, complex queries go to ChatGPT.

---

### Example 2: Privacy-First with Ollama

**Use Case**: Everything runs locally, no cloud

**Agents:**
1. Home Assistant (Priority 10) - Built-in intents
2. Llama3 8B (Priority 20) - Main processing
3. Mixtral 8x7B (Priority 30) - Complex queries

**Prerequisites:**
```bash
# Install and start Ollama
ollama pull llama3:8b
ollama pull mixtral:8x7b
```

**Setup Steps:**
1. Add Integration → NeuralBridge
2. Add Agent: Home Assistant (Priority 10, Timeout 5s)
3. Add Agent: Ollama
   - Name: "Llama3 Main"
   - Priority: 20
   - URL: http://localhost:11434
   - Model: llama3:8b
   - Timeout: 30 seconds
4. Add Agent: Ollama
   - Name: "Mixtral Complex"
   - Priority: 30
   - URL: http://localhost:11434
   - Model: mixtral:8x7b
   - Timeout: 60 seconds

**Result**: All processing stays local, scaled by complexity.

---

### Example 3: Smart Router with Filtering

**Use Case**: Use small model to route/filter before expensive cloud calls

**Agents:**
1. Qwen Router (Priority 0) - Classification/filtering
2. Home Assistant (Priority 10) - Local intents
3. Gemini Pro (Priority 20) - Cloud processing

**Prerequisites:**
```bash
ollama pull qwen2.5:0.5b
```

**Setup Steps:**
1. Add Integration → NeuralBridge
2. Add Agent: Ollama
   - Name: "Qwen Router"
   - Priority: 0 (IMPORTANT!)
   - URL: http://localhost:11434
   - Model: qwen2.5:0.5b
   - Timeout: 3 seconds
3. Add Agent: Home Assistant (Priority 10, Timeout 5s)
4. Add Agent: Existing Integration
   - Name: "Gemini Pro"
   - Priority: 20
   - Entity: conversation.google_generative_ai_conversation
   - Timeout: 30 seconds

**Result**: Fast local router checks requests before sending to cloud.

---

### Example 4: Multi-Cloud with Local Priority

**Use Case**: Multiple cloud providers with local priority

**Agents:**
1. Home Assistant (Priority 10) - Local intents
2. Gemini Flash (Priority 20) - Fast cloud
3. ChatGPT (Priority 30) - Balanced
4. Claude (Priority 40) - Fallback

**Setup Steps:**
1. Ensure all cloud integrations are configured in HA
2. Add Integration → NeuralBridge
3. Add each agent with appropriate priorities
4. Test with various queries

**Result**: Local first, then tries cloud providers in order.

---

### Example 5: Cost-Optimized Setup

**Use Case**: Minimize cloud API costs

**Agents:**
1. Qwen Filter (Priority 0) - Classify/filter
2. Home Assistant (Priority 10) - Free local
3. Llama3 8B (Priority 20) - Free local processing
4. Gemini Flash (Priority 30) - Cheap cloud ($0.00015/1k chars)

**Monthly Cost Estimate**: ~$2-5 for heavy use

**Prerequisites:**
```bash
ollama pull qwen2.5:0.5b
ollama pull llama3:8b
```

**Setup Steps:**
1. Configure Gemini integration (get free API key)
2. Add NeuralBridge integration
3. Add Qwen filter (Priority 0)
4. Add Home Assistant (Priority 10)
5. Add Llama3 via Ollama (Priority 20)
6. Add Gemini Flash (Priority 30)

**Result**: Most queries handled free locally, rare expensive cloud calls.

---

## Advanced Configurations

### Multi-Region Ollama Setup

Run Ollama on multiple machines for load balancing:

**Agents:**
1. Home Assistant (Priority 10)
2. Ollama Server 1 - Llama3 (Priority 20)
   - URL: http://server1:11434
3. Ollama Server 2 - Llama3 (Priority 21)
   - URL: http://server2:11434
4. Cloud Fallback (Priority 30)

### Specialized Model Routing

Use different models for different query types:

**Agents:**
1. Home Assistant (Priority 10) - Device control
2. CodeLlama (Priority 15) - Code questions
   - Model: codellama:13b
3. Llama3 (Priority 20) - General queries
   - Model: llama3:8b
4. Gemini (Priority 30) - Complex reasoning

**Note**: Actual routing logic requires custom implementation.

### Development/Testing Setup

Separate dev and prod agents:

**Agents (Dev):**
1. Home Assistant (Priority 10)
2. TinyLlama (Priority 20) - Fast iteration
   - Model: tinyllama:1.1b
   - Timeout: 5s

**Agents (Prod):**
1. Qwen Router (Priority 0)
2. Home Assistant (Priority 10)
3. Llama3 70B (Priority 20)
4. ChatGPT (Priority 30)

---

## Configuration Validation

### Test Your Configuration

After setup, test with these queries:

1. **Local Intent**: "Turn on the kitchen lights"
   - Should be handled by Home Assistant (Priority 10)

2. **General Knowledge**: "What's the weather forecast?"
   - Should escalate to cloud/Ollama agent

3. **Complex Reasoning**: "Explain quantum computing"
   - Should use your highest-capability agent

### Monitor Routing Decisions

Enable debug logging in Home Assistant:

```yaml
logger:
  default: info
  logs:
    custom_components.neuralbridge: debug
```

Check logs to see which agent handled each request.

---

## Troubleshooting Common Setups

### Issue: All queries go to cloud (expensive!)

**Solution**: Add local Ollama agent with priority < cloud priority

### Issue: Ollama too slow

**Solutions**:
- Use smaller models (llama3:8b instead of 70b)
- Decrease timeout to fail faster
- Add GPU acceleration to Ollama host
- Use quantized models (Q4, Q5)

### Issue: Router blocking valid requests

**Solution**:
- Remove router agent (Priority 0) for now
- Implement proper router logic
- Use a better-trained classification model

### Issue: Home Assistant not handling local intents

**Solutions**:
- Ensure HA agent has low priority number (10-20)
- Verify HA conversation agent is working standalone
- Check HA intent configuration

---

## Migration from Legacy Tier System

If migrating from the old Tier 1/2/3 system:

**Old Tier 1** → Priority 10 (Home Assistant)
**Old Tier 2** → Priority 0 (Router/Filter)
**Old Tier 3** → Priority 20+ (Cloud/Ollama processing agents)

The new system is more flexible - add as many agents as needed!
