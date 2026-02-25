# Claude Code Instructions for NeuralBridge

## Project-Specific Guidelines

For detailed project context, technical requirements, and coding standards, refer to:
- **[.github/.copilot-instructions.md](.github/.copilot-instructions.md)** - Complete project vision, architecture, and development guidelines

## Quick Reference

**Project**: NeuralBridge - A priority-based AI routing service for Home Assistant's Assist pipeline

> **PYTHON 3.13 IS THE ONLY SUPPORTED RUNTIME — NO EXCEPTIONS**
> Minimum HA version is 2026.1, which requires Python 3.13+. Python 3.12 has an
> incompatible `homeassistant` package (e.g. `ConversationInput` lacks `satellite_id`).
> **ALWAYS** run tests via `make check` or `.venv/bin/python -m pytest`.
> **NEVER** use the system `pytest`, `python3`, `ruff`, `black`, or `mypy` directly.

**Critical Principles** (In Order of Importance):
1. **Security First** - Security > Performance (always)
2. **Privacy Default** - Privacy > Convenience (always)
3. **Quality Matters** - 100% test coverage required
4. **Async-First** - Never block the event loop
5. **Type Safety** - Full type hints, mypy strict mode

## Code Quality Standards (NON-NEGOTIABLE)

### Testing Requirements
- **Coverage Target**: 100% (no exceptions)
- **Unit Tests**: Every function, every path, every edge case
- **Integration Tests**: End-to-end flows
- **Test ALL**: Success paths, error paths, timeouts, edge cases
- **CI Fail**: If coverage < 100%

### SonarQube Compliance
- **Line Length**: 100 chars (target), 115 max (hard limit)
- **Function Length**: 50 lines max (target 20-30)
- **Complexity**: Max 15 per function (target 10)
- **Nesting**: Max 4 levels deep
- **Security**: OWASP Top 10 compliant

### Security Rules (CRITICAL)
- ❌ **NEVER** log secrets, API keys, tokens, passwords
- ❌ **NEVER** log full user conversations (PII)
- ❌ **NEVER** hardcode credentials
- ✅ **ALWAYS** validate all external input
- ✅ **ALWAYS** sanitize error messages
- ✅ **ALWAYS** use parameterized queries/templates
- ✅ **ALWAYS** fail securely (closed by default)

### Privacy Rules (DEFAULT)
- **Local First**: Try local agents before cloud
- **Data Minimization**: Only send what's necessary to cloud
- **User Control**: Clear indication of cloud usage
- **Transparent**: User knows where data goes

### Code Style
- **Formatting**: Black (line-length=100)
- **Linting**: Ruff (includes security checks)
- **Type Checking**: Mypy strict mode
- **Imports**: Organized (stdlib, third-party, HA, local)
- **Naming**: PEP 8 (snake_case functions, PascalCase classes)

## Pre-Commit Checklist

Before ANY code commit, **run `make check`** (this is the single source of truth):
- [ ] `make check` passes completely — that's it, one command

If you need to run steps individually (all via `.venv/bin/python -m <tool>` or `make <target>`):
- [ ] Ruff passes (`make lint`) — **run BEFORE tests**
- [ ] Black formatted (`make format`)
- [ ] Mypy passes (`make type`)
- [ ] All tests pass with 100% coverage (`make test`)
- [ ] No security issues
- [ ] Line length ≤ 115
- [ ] Function complexity ≤ 15
- [ ] All public functions have docstrings
- [ ] No secrets in code or logs
- [ ] Proper error handling
- [ ] Resource cleanup implemented

## Common Patterns

### Good Function Example
```python
async def process_agent(
    agent_config: dict[str, Any],
    user_input: ConversationInput,
    timeout: int = 30,
) -> ConversationResult | None:
    """Process input with a single agent.

    Args:
        agent_config: Agent configuration dictionary.
        user_input: User's conversation input.
        timeout: Timeout in seconds (default: 30).

    Returns:
        Conversation result if successful, None if failed.

    Raises:
        ValueError: If agent_config is invalid.
    """
    if not agent_config:
        raise ValueError("agent_config cannot be empty")

    agent_type = agent_config.get("type")
    if not agent_type:
        _LOGGER.error("Agent config missing type")
        return None

    try:
        async with asyncio.timeout(timeout):
            if agent_type == AGENT_TYPE_OLLAMA:
                return await _process_ollama(agent_config, user_input)
            # ... other types
    except asyncio.TimeoutError:
        _LOGGER.warning("Agent timed out after %d seconds", timeout)
        return None
    except Exception as err:
        _LOGGER.error("Unexpected error: %s", err)
        raise  # Re-raise unexpected errors

    return None
```

### Test Example
```python
async def test_process_agent_success(hass, mock_config_entry):
    """Test successful agent processing."""
    # Arrange
    agent_config = {"type": AGENT_TYPE_OLLAMA, "url": "http://localhost"}
    user_input = ConversationInput(text="test")

    # Act
    result = await process_agent(agent_config, user_input)

    # Assert
    assert result is not None
    assert result.response is not None

async def test_process_agent_timeout(hass, mock_config_entry):
    """Test agent processing with timeout."""
    # Test timeout scenario

async def test_process_agent_invalid_config(hass, mock_config_entry):
    """Test agent processing with invalid config."""
    with pytest.raises(ValueError, match="agent_config cannot be empty"):
        await process_agent({}, ConversationInput(text="test"))
```

## When Writing Code

1. **Start with Tests**: Write test first (TDD)
2. **Validate Input**: Check all external input
3. **Handle Errors**: Catch specific exceptions
4. **Log Safely**: Never log secrets or PII
5. **Clean Up**: Always close resources
6. **Keep Simple**: Low complexity, short functions
7. **Document**: Docstrings on public functions

## Repository Context

This is a **standalone git repository** within `/Users/pfrye/git/`. When making changes:
- All changes should be scoped to this NeuralBridge repository only
- Do not assume similarities with other repositories in the parent directory
- Confirm scope before making modifications

## Summary

**Remember the Three Pillars:**
1. **Security First**: Defensive coding, input validation, secure defaults
2. **Quality Matters**: 100% test coverage, low complexity, clean code
3. **Privacy Default**: Local-first, data minimization, user control

**When in Doubt:**
- Security > Performance
- Privacy > Convenience
- Tested > Untested
- Simple > Complex

See [.github/.copilot-instructions.md](.github/.copilot-instructions.md) for complete details.
