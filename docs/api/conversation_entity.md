# Conversation Entity API

## NeuralBridgeAgent

The main conversation agent that implements Home Assistant's `ConversationEntity`.

### Class Definition

```python
class NeuralBridgeAgent(ConversationEntity):
    """NeuralBridge conversation agent."""
```

### Methods

#### `async_process(user_input: ConversationInput) -> ConversationResult`

Process a user input through the tiered routing system.

**Parameters:**
- `user_input` (ConversationInput): The user's input containing text, context, and conversation_id

**Returns:**
- `ConversationResult`: The response with speech output and metadata

**Raises:**
- `HomeAssistantError`: If all tiers fail to process the request

**Example:**

```python
result = await agent.async_process(
    ConversationInput(
        text="turn on the lights",
        context=context,
        conversation_id="abc123",
        language="en"
    )
)
```

### Routing Logic

1. **Tier 1 Attempt:** Calls configured local agent
2. **Tier 2 Filter:** If Tier 1 fails, applies safety filter
3. **Tier 3 Cloud:** If filter passes, routes to cloud LLM

### Configuration

The agent reads configuration from the ConfigEntry:
- `tier1_agent`: Entity ID of local agent
- `tier3_agent`: Entity ID of cloud agent
- `enable_tier2_filter`: Boolean to enable filtering

## Events

### `neuralbridge_tier_routed`

Fired when a request is routed to a different tier.

**Event Data:**
```python
{
    "tier": 1 | 2 | 3,
    "success": bool,
    "response_time_ms": int
}
```
