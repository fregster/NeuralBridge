# NeuralBridge Coding Standards

## Overview

This document provides a quick reference for NeuralBridge coding standards. For comprehensive details, see [docs/DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) and [.github/.copilot-instructions.md](../.github/.copilot-instructions.md).

## The Three Pillars (In Order of Importance)

1. **Security First**: Security > Performance (always)
2. **Privacy Default**: Privacy > Convenience (always)
3. **Quality Matters**: 100% test coverage required

## Quick Reference Card

### Code Quality Metrics

| Metric | Target | Hard Limit | Tool |
|--------|--------|------------|------|
| Test Coverage | 100% | 100% | pytest-cov |
| Line Length | 100 chars | 115 chars | black, ruff |
| Function Length | 20-30 lines | 50 lines | manual review |
| Complexity | 10 | 15 | ruff (C90) |
| Nesting Depth | 2-3 levels | 4 levels | manual review |

### Security Rules (NON-NEGOTIABLE)

❌ **NEVER**:
- Log secrets, API keys, tokens, passwords
- Log full user conversations (PII)
- Hardcode credentials
- Use string concatenation for URLs/queries
- Expose stack traces to users
- Trust external input without validation

✅ **ALWAYS**:
- Validate ALL external input
- Sanitize error messages
- Use parameterized queries/templates
- Fail securely (closed by default)
- Follow OWASP Top 10 guidelines
- Prefer local processing before cloud

### Testing Requirements

**Target: 100% Coverage (Enforced)**

Test ALL:
- ✅ Success paths
- ✅ Error/exception paths
- ✅ Timeout scenarios
- ✅ Validation logic
- ✅ Conditional branches
- ✅ Loop iterations
- ✅ Edge cases (None, empty, invalid)
- ✅ Async operations
- ✅ Cleanup/teardown

### Type Hints (Required)

```python
# Good: Full type hints
async def process_agent(
    config: dict[str, Any],
    user_input: ConversationInput,
    timeout: int = 30,
) -> ConversationResult | None:
    """Process input with agent."""
    ...

# Bad: No type hints
async def process_agent(config, user_input, timeout=30):
    ...
```

### Docstrings (Required)

All public functions must have docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Short one-line summary.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is invalid.
    """
```

## Pre-Commit Checklist

Before EVERY commit:
- [ ] 100% test coverage (`pytest --cov --cov-fail-under=100`)
- [ ] All tests pass (`pytest`)
- [ ] Black formatted (`black .`)
- [ ] Ruff passes (`ruff check .`)
- [ ] Mypy passes (`mypy custom_components`)
- [ ] No secrets in code or logs
- [ ] Line length ≤ 115
- [ ] Function complexity ≤ 15
- [ ] Public functions documented
- [ ] Proper error handling
- [ ] Resource cleanup

## Quick Commands

```bash
# Format code
black custom_components tests

# Lint (includes security checks)
ruff check .

# Fix auto-fixable issues
ruff check . --fix

# Type check (strict mode)
mypy custom_components

# Test with coverage
pytest --cov --cov-fail-under=100

# All checks at once
black . && ruff check . && mypy custom_components && pytest --cov --cov-fail-under=100

# View coverage report
pytest --cov --cov-report=html
open htmlcov/index.html
```

## Common Patterns

### Input Validation

```python
# Good: Validate everything
def validate_url(url: str) -> str:
    """Validate URL is safe."""
    if not url:
        raise ValueError("URL cannot be empty")
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must be http or https")
    return url.rstrip("/")
```

### Error Handling

```python
# Good: Specific exceptions
try:
    result = await risky_operation()
except aiohttp.ClientError as err:
    _LOGGER.error("Connection failed: %s", err)
    return None
except asyncio.TimeoutError:
    _LOGGER.warning("Operation timed out")
    return None
```

### Logging Safety

```python
# Good: Never log secrets
_LOGGER.debug("Connecting to %s", sanitize_url(url))

# Bad: Logs credentials
_LOGGER.debug("Request: %s", full_url_with_api_key)  # NEVER!
```

### Async with Timeout

```python
# Good: Always use timeout
async with asyncio.timeout(30):
    result = await external_call()
```

### Resource Cleanup

```python
# Good: Proper cleanup
async def async_will_remove_from_hass(self) -> None:
    """Clean up resources."""
    for client in self._clients.values():
        await client.close()
    self._clients.clear()
```

## Configuration

### pyproject.toml

```toml
[tool.black]
line-length = 100

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "C90", "B", "S", "T20", "SIM", "PL", "RUF"]

[tool.ruff.lint.mccabe]
max-complexity = 15

[tool.mypy]
strict = true
```

### pytest.ini

```ini
[pytest]
addopts =
    --cov=custom_components.neuralbridge
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=100
```

## Priority Order

When making decisions, follow this priority order:

1. **Security** > Performance
2. **Privacy** > Convenience
3. **Correctness** > Speed
4. **Tested** > Untested
5. **Simple** > Complex
6. **Explicit** > Implicit
7. **Readable** > Clever

## Resources

- **Development Guide**: [docs/DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)
- **Copilot Instructions**: [.github/.copilot-instructions.md](../.github/.copilot-instructions.md)
- **Claude Instructions**: [CLAUDE.md](../CLAUDE.md)
- **Home Assistant Dev Docs**: https://developers.home-assistant.io/
- **OWASP Top 10**: https://owasp.org/www-project-top-ten/
- **Ruff Rules**: https://docs.astral.sh/ruff/rules/
- **Pytest Docs**: https://docs.pytest.org/

## Summary

**Remember the Three Pillars:**
1. Security First
2. Privacy Default
3. Quality Matters

All code must meet these standards. No exceptions.
