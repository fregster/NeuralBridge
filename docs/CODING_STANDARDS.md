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
| Class File Size | 200-300 lines | 500 lines (soft) | manual review |
| Public Methods per Class | 5-10 | 15 | manual review |
| Instance Variables per Class | 3-7 | 10 | manual review |
| Concrete classes per file | 1 | 2 | manual review |

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

## Class Design & Pluggable Architecture

### Preventing God Classes

A *god class* accumulates too many responsibilities into a single class, making it difficult to test, extend, and maintain. NeuralBridge enforces class size and design rules to prevent this anti-pattern.

**Soft limit: 500 lines per class file.**

Exceeding 500 lines is a design smell. It is not a hard block, but **MUST** trigger a design review. Any contribution that pushes an existing file beyond 500 lines requires explicit justification in the PR description. The goal is for most class files to remain well under 300 lines.

| Signal | Recommended Action |
|--------|--------------------|
| File approaching 400 lines | Plan a split before adding more logic |
| File over 500 lines | Refactor is required before the next feature addition |
| Class has > 10 instance variables | Introduce value objects or sub-components |
| Class has > 15 public methods | Extract a collaborator or strategy |
| Class responsibility requires > 1 sentence to describe | Split into focused classes |

### Framework-First Design

All **new classes** MUST follow the *Framework-First* pattern: define a lightweight interface or abstract contract first, then provide one or more concrete implementations. This keeps the architecture pluggable and each class focused.

**The three steps:**

1. **Define the interface (the framework)** — Write a `Protocol` or abstract base class (`ABC`) that captures the minimal contract. No implementation logic here — only method signatures with docstrings.
2. **Provide a default implementation (the extension)** — Create a concrete class that fulfils the interface for the common case. Keep it focused on a single transport, backend, or use-case.
3. **Register or inject, never hard-wire** — Callers depend on the interface, not the concrete class. New backends are added by implementing the interface, not by modifying existing classes.

```python
# Step 1 — Define the protocol (the "framework")
class SearchProvider(Protocol):
    """Minimal interface for any search backend."""

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Execute a search and return ranked results."""
        ...


# Step 2 — Concrete implementation (the "extension")
class BraveSearchProvider:
    """Brave Search implementation of SearchProvider."""

    def __init__(self, api_key: str, session: aiohttp.ClientSession) -> None:
        """Initialise with credentials and a shared HTTP session."""
        self._api_key = api_key
        self._session = session

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Execute a Brave Search API query."""
        ...


# Step 3 — Test stub is trivial because the interface is minimal
class StubSearchProvider:
    """In-memory stub for unit tests."""

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Return a fixed stub result."""
        return [SearchResult(title="stub", url="http://example.com")]
```

### Strategies for Splitting a Class That Is Too Large

When a class approaches the 500-line soft limit, apply one or more of these patterns **before** writing more code into it:

| Strategy | When to Apply |
|----------|---------------|
| **Extract a collaborator** | A group of methods share private state unrelated to the class's primary role |
| **Introduce a Strategy** | A conditional chain (`if agent_type == ...`) selects different algorithms |
| **Separate I/O from logic** | Mix of HTTP/DB calls and business logic in the same class |
| **Use composition** | The class inherits behaviour it could instead delegate to a held reference |
| **Split by lifecycle** | Construction/config, runtime operation, and teardown can each be their own object |

### Checklist for New Classes

- [ ] A `Protocol` or `ABC` exists that defines the minimal interface
- [ ] The new class file starts below 300 lines
- [ ] The class satisfies the **Single Responsibility Principle** — it has exactly one reason to change
- [ ] The class responsibility can be described in one sentence
- [ ] The file contains **at most one primary concrete class** (small dataclasses, value objects, and private helpers are permitted alongside it)
- [ ] If a `Protocol` and its sole concrete implementation share a file, a comment marks the intent to split the file when a second implementation is added
- [ ] Concrete classes are injected (not instantiated) by callers where possible
- [ ] A corresponding stub or fake exists in `tests/` for the interface

---

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
- [ ] No class file exceeds 500 lines without documented justification
- [ ] New classes have a corresponding `Protocol` or `ABC` interface defined
- [ ] New class responsibility can be described in a single sentence (Single Responsibility Principle)
- [ ] Each new file contains at most one primary concrete class
- [ ] If a `Protocol` and its implementation share a file, a comment documents the split intent

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
