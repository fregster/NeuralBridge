# Agent Configuration Guide

## Overview

NeuralBridge uses a **priority-based routing system** where you configure multiple AI agents, each with a priority score. The system routes requests through agents in priority order until one successfully handles the request.

## Priority System

### Priority Levels

- **Priority 0**: Router/Filter agents
  - Small, fast models (TinyLlama, Qwen 0.5b-2b)
  - Used for classification and filtering
  - Determine if a request should be processed
  - Run BEFORE processing agents

- **Priority 1-100**: Processing agents
  - Lower numbers = higher priority (tried first)
  - Models that actually answer questions
  - Tried in order until one succeeds

### Example Priority Setup

```
Priority 0:  Qwen2.5:0.5b (Router/Filter)
Priority 10: Home Assistant (Local intents)
Priority 20: Ollama Llama3:8b (Medium queries)
Priority 30: ChatGPT (Complex reasoning)
Priority 40: Gemini Pro (Fallback)
```

## Agent Types

### 1. Home Assistant (Built-in)

Uses Home Assistant's built-in conversation agent for local intent matching.

**Best for:**
- Device control ("turn on the lights")
- Home automation queries
- Fast, local-only processing

**Configuration:**
- Agent Name: Friendly name (e.g., "Home Assistant")
- Priority: Typically 10-20 (high priority for local control)
- Timeout: 5-10 seconds

### 2. Existing Integration

Connects to existing Home Assistant conversation agents (Gemini, ChatGPT, Claude, etc.)

**Best for:**
- Using integrations you already have configured
- Cloud-based AI services
- Complex reasoning and general knowledge

**Configuration:**
- Agent Name: Friendly name (e.g., "ChatGPT Main")
- Priority: 20-50 depending on use case
- Entity ID: Select from dropdown
- Timeout: 30-60 seconds

### 3. Ollama (External)

Connects to an Ollama instance running locally or on your network.

**Best for:**
- Privacy-focused deployments
- Custom models
- Router/filter agents (small models)
- Cost-effective processing

**Configuration:**
- Agent Name: Friendly name (e.g., "Qwen Router", "Llama3 Main")
- Priority: 0 for routers, 10-50 for processing
- Ollama URL: http://localhost:11434 or network URL
- Model Name: e.g., `qwen2.5:0.5b`, `llama3:8b`
- Timeout: 10-60 seconds

## Configuration Workflow

### Initial Setup

1. **Install Integration**
   - Go to Settings → Devices & Services
   - Click "+ Add Integration"
   - Search for "NeuralBridge"
   - Click Submit (no agents configured yet)

2. **Add First Agent**
   - Click "Configure" on the NeuralBridge integration
   - Select "Add New Agent"
   - Choose agent type and configure

### Adding Router Agent (Priority 0)

Router agents are optional but recommended for:
- Safety filtering
- Request classification
- Determining if cloud processing is needed

**Recommended Models:**
- Qwen2.5:0.5b (very fast, good at classification)
- TinyLlama:1.1b (extremely fast)
- Phi-2:2.7b (small but capable)

**Example Configuration:**
```
Agent Name: Qwen Safety Filter
Agent Type: Ollama
Priority: 0
Ollama URL: http://localhost:11434
Model: qwen2.5:0.5b
Timeout: 5 seconds
```

### Adding Processing Agents

Start with local/fast agents and add progressively more capable (and slower) agents.

**Example: Local First**
```
Agent Name: Home Assistant
Agent Type: Home Assistant (Built-in)
Priority: 10
Timeout: 5 seconds
```

**Example: Medium Capability**
```
Agent Name: Llama3 Local
Agent Type: Ollama
Priority: 20
Ollama URL: http://localhost:11434
Model: llama3:8b
Timeout: 30 seconds
```

**Example: High Capability Cloud**
```
Agent Name: ChatGPT Fallback
Agent Type: Existing Integration
Priority: 30
Entity ID: conversation.chatgpt
Timeout: 60 seconds
```

## How Routing Works

### Step 1: Router Agents (Priority 0)

If you have any Priority 0 agents configured:
1. Request is sent to router agent
2. Router determines if request should be processed
3. If blocked, return error message
4. If allowed, continue to Step 2

### Step 2: Processing Agents (Priority 1+)

Agents are tried in priority order (lowest first):
1. Send request to agent with lowest priority number
2. If agent succeeds, return response immediately
3. If agent fails or times out, try next agent
4. Continue until an agent succeeds or all fail

### Fallback

If all agents fail:
- Return standard error message
- Log failure for debugging
- User sees: "I'm having trouble connecting to my AI agents right now."

## Example Configurations

### Scenario 1: Privacy-First (All Local)

```
Priority 0:  Qwen2.5:0.5b via Ollama (Router)
Priority 10: Home Assistant (Local intents)
Priority 20: Llama3:8b via Ollama (General queries)
Priority 30: Mistral:7b via Ollama (Fallback)
```

### Scenario 2: Hybrid (Local + Cloud)

```
Priority 10: Home Assistant (Local intents)
Priority 20: Llama3:8b via Ollama (Privacy-safe queries)
Priority 30: Gemini Pro (Complex reasoning)
Priority 40: ChatGPT (Fallback)
```

### Scenario 3: Cloud-Heavy (Fast Response)

```
Priority 10: Home Assistant (Device control only)
Priority 20: Gemini Flash (Fast cloud)
Priority 30: ChatGPT (General queries)
Priority 40: Claude (Fallback)
```

### Scenario 4: Cost-Optimized

```
Priority 0:  Qwen2.5:0.5b via Ollama (Filter unnecessary requests)
Priority 10: Home Assistant (Free local)
Priority 20: Llama3:8b via Ollama (Free local)
Priority 30: Gemini Flash (Low-cost cloud)
```

## Managing Agents

### Viewing Configured Agents

1. Go to NeuralBridge integration settings
2. Click "Configure"
3. Select "Manage Existing Agents"
4. View list of all agents with their priorities

### Deleting an Agent

1. Go to "Manage Existing Agents"
2. Select the agent to delete
3. Choose "Delete Agent" action
4. Confirm deletion

### Modifying an Agent

Currently, agents cannot be edited after creation. To modify:
1. Delete the existing agent
2. Add a new agent with updated configuration

## Troubleshooting

### Agent Not Responding

- Check timeout settings (increase if needed)
- For Ollama: Verify Ollama is running and accessible
- For Existing Integration: Ensure the integration is configured and working
- Check Home Assistant logs for error messages

### All Agents Failing

- Verify at least one agent is configured
- Check network connectivity for cloud agents
- Ensure Ollama models are pulled: `ollama pull model-name`
- Review agent priorities (ensure processing agents have priority > 0)

### Router Blocking Everything

- Router agents (priority 0) might be too restrictive
- Consider removing router agents for testing
- Implement proper router logic (see developer docs)

## Best Practices

1. **Start Simple**: Begin with 1-2 agents and add more as needed
2. **Test Each Agent**: Add agents one at a time and test
3. **Use Appropriate Timeouts**: Fast models = short timeout, slow models = longer
4. **Monitor Logs**: Enable debug logging to see routing decisions
5. **Balance Cost/Speed/Privacy**: Consider your priorities when choosing agents

## Advanced: Router Agent Logic

Router agents (priority 0) currently allow all requests by default. To implement custom filtering:

1. Use a small model trained for classification
2. Implement custom logic in `conversation.py`
3. Example use cases:
   - Block inappropriate requests
   - Filter out queries better handled locally
   - Classify request type and route accordingly
   - Privacy filtering (keep sensitive data local)

See the developer documentation for implementation details.
