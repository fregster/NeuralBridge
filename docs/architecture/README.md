# NeuralBridge Architecture

## Overview

NeuralBridge implements a three-tier routing system for Home Assistant's Assist pipeline.

## Tier System

### Tier 1: Local Intent Matching
- **Purpose:** Handle home automation commands locally
- **Implementation:** Routes to `conversation.home_assistant`
- **Examples:** "Turn on lights", "Set thermostat to 72"
- **Latency:** <100ms
- **Privacy:** All data stays local

### Tier 2: Logic & Safety Filter
- **Purpose:** Evaluate prompt safety and appropriateness
- **Implementation:** Rule-based filtering system
- **Actions:**
  - Pass to Tier 3 if appropriate
  - Block and return error if inappropriate
- **Privacy:** All filtering happens locally

### Tier 3: Cloud LLM
- **Purpose:** Handle complex reasoning and general knowledge
- **Implementation:** Routes to configured cloud agents (Gemini, OpenAI, etc.)
- **Examples:** "What's the weather forecast?", "Tell me a joke"
- **Latency:** 1-3 seconds
- **Privacy:** Data sent to cloud provider

## Data Flow

```
User Input
    ↓
[Tier 1: Local HA Agent]
    ↓
Success? → Return Response
    ↓ (No)
[Tier 2: Safety Filter]
    ↓
Safe? → Block/Return Error
    ↓ (Yes)
[Tier 3: Cloud LLM]
    ↓
Response
```

## Error Handling

- **Tier 1 Failure:** Automatically escalate to Tier 2/3
- **Tier 2 Block:** Return appropriate error message
- **Tier 3 Unavailable:** Return fallback message

## Extension Points

Future enhancements may include:
- Custom Tier 2 filter plugins
- Multiple Tier 3 providers with automatic failover
- Usage analytics and routing optimization
