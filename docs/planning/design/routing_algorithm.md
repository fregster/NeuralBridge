# Routing Algorithm Design

## Overview

This document describes the routing algorithm used by NeuralBridge to process conversation inputs.

## Algorithm Flow

```python
async def route_conversation(input: ConversationInput) -> ConversationResult:
    """
    Main routing algorithm.

    1. Attempt Tier 1 (Local)
    2. On failure, check Tier 2 (Filter)
    3. If passes filter, attempt Tier 3 (Cloud)
    4. Return result or fallback
    """
```

## Detailed Steps

### Step 1: Tier 1 Processing

```python
try:
    result = await tier1_agent.async_process(input)
    if result.success:
        log_routing("tier1_success")
        return result
except Exception as e:
    log_routing("tier1_failed", error=e)
    # Continue to Tier 2
```

**Decision Criteria:**
- If Tier 1 returns a successful match (intent recognized)
- Return immediately, do not escalate

### Step 2: Tier 2 Filtering

```python
if not tier2_filter.should_allow(input.text):
    log_routing("tier2_blocked")
    return ConversationResult(
        response=Speech(text="I cannot process that request"),
        unmatched=True
    )
```

**Filter Rules:**
- Check for personally identifiable information
- Check for inappropriate content
- Check for requests outside scope
- All checks are local/rule-based

### Step 3: Tier 3 Processing

```python
try:
    result = await tier3_agent.async_process(input)
    log_routing("tier3_success")
    return result
except Exception as e:
    log_routing("tier3_failed", error=e)
    return fallback_response()
```

**Decision Criteria:**
- Only reached if Tier 1 failed and Tier 2 allowed
- Timeout after configured duration
- Return fallback if unavailable

## Optimization Strategies

### Parallel Processing
For certain queries, we could attempt Tier 1 and Tier 3 in parallel:
- Start both simultaneously
- Return first successful result
- Cancel the other request

**Pros:** Lower latency for cloud queries
**Cons:** Unnecessary cloud calls, privacy concerns

**Recommendation:** Implement as optional feature, disabled by default

### Caching
Cache Tier 3 responses for common queries:
- Store hash of input text
- Return cached response if available
- TTL-based expiration

**Pros:** Faster responses, reduced cloud costs
**Cons:** Stale data, cache management complexity

**Recommendation:** Implement in future version

### Smart Routing
Use ML model to predict which tier will succeed:
- Train on historical routing data
- Skip Tier 1 for queries likely to need cloud
- Still maintain privacy guarantees

**Pros:** Optimized latency
**Cons:** Complexity, potential privacy issues

**Recommendation:** Future research project

## Error Handling

### Timeout Handling
```python
async with asyncio.timeout(tier1_timeout):
    result = await tier1_agent.async_process(input)
```

### Circuit Breaker Pattern
```python
if tier3_failure_count > threshold:
    # Skip Tier 3 temporarily
    return fallback_response()
```

## Metrics Collection

Track the following for optimization:
- Tier 1 success rate
- Tier 2 block rate
- Tier 3 response time
- Overall routing path distribution
