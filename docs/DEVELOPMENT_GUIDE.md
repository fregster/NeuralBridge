# NeuralBridge Development Guide

## Overview

This guide outlines the development standards, testing requirements, and best practices for NeuralBridge development.

## Core Principles

### The Three Pillars (In Order)

1. **Security First**: Security > Performance (always)
2. **Privacy Default**: Privacy > Convenience (always)
3. **Quality Matters**: 100% test coverage required

## Code Quality Standards

### Testing Requirements (NON-NEGOTIABLE)

**Target: 100% Unit Test Coverage**

```bash
# Run tests with coverage
pytest --cov --cov-fail-under=100

# View HTML coverage report
pytest --cov --cov-report=html
open htmlcov/index.html
```

**What Must Be Tested:**
- ✅ All functions and methods
- ✅ All success paths
- ✅ All error/exception paths
- ✅ All timeout scenarios
- ✅ All conditional branches (if/else/elif)
- ✅ All loop iterations (empty, single, multiple)
- ✅ All edge cases (None, empty, invalid data)
- ✅ All validation logic
- ✅ All cleanup/teardown logic

**Test Structure:**
```python
async def test_function_name_success(hass, mock_config_entry):
    """Test function with successful input."""
    # Arrange: Set up test data and mocks
    agent_config = {"type": "ollama", "url": "http://localhost"}

    # Act: Call the function being tested
    result = await process_agent(agent_config, user_input)

    # Assert: Verify expected results
    assert result is not None
    assert result.response.text == "expected response"

async def test_function_name_error(hass, mock_config_entry):
    """Test function handles errors correctly."""
    # Test error handling

async def test_function_name_edge_case(hass, mock_config_entry):
    """Test function with edge case input."""
    # Test boundary conditions
```

### SonarQube Compliance

**Complexity Limits:**
- **Functions**: Max complexity 15 (target 10)
- **Cyclomatic Complexity**: Max 10 paths per function
- **Nesting**: Max 4 levels deep

```bash
# Check complexity with ruff
ruff check . --select C90
```

**Line Length:**
- **Target**: 100 characters (soft limit)
- **Maximum**: 115 characters (hard limit)

**Function Length:**
- **Maximum**: 50 lines per function (target 20-30)
- **If too long**: Extract helper functions

**Reducing Complexity:**
```python
# Good: Low complexity with early returns
async def process_agent(config: dict) -> Result | None:
    """Process with single agent."""
    if not config:
        return None

    agent_type = config.get("type")
    if agent_type == "ollama":
        return await _process_ollama(config)
    if agent_type == "existing":
        return await _process_existing(config)

    _LOGGER.error("Unknown type: %s", agent_type)
    return None

# Bad: High complexity with nested conditions
async def process_agent(config: dict) -> Result | None:
    """Don't do this - too complex!"""
    if config:
        agent_type = config.get("type")
        if agent_type:
            if agent_type == "ollama":
                try:
                    if config.get("url"):
                        # Too deep!
```

### Security Standards (CRITICAL)

**OWASP Top 10 Compliance**

1. **Injection Prevention**
   ```python
   # Good: Parameterized
   url = f"{base_url}/api/generate"
   data = {"model": model, "prompt": sanitized_prompt}

   # Bad: Concatenation
   url = base_url + "/api/" + user_input  # Dangerous!
   ```

2. **Input Validation**
   ```python
   # Good: Validate everything
   def validate_url(url: str) -> str:
       """Validate URL is safe."""
       if not url:
           raise ValueError("URL cannot be empty")
       if not url.startswith(("http://", "https://")):
           raise ValueError("URL must be http or https")
       if "@" in url:
           raise ValueError("URL cannot contain credentials")
       return url.rstrip("/")

   # Bad: No validation
   def use_url(url: str) -> str:
       return url  # Accepts anything!
   ```

3. **Sensitive Data Protection**
   ```python
   # Good: Never log secrets
   _LOGGER.debug("Connecting to %s", sanitize_url(url))

   # Bad: Logs credentials
   _LOGGER.debug("Request: %s", full_url_with_api_key)  # Dangerous!
   ```

4. **Error Handling**
   ```python
   # Good: Safe error messages
   except ValueError as err:
       _LOGGER.error("Invalid configuration: %s", str(err))
       return "Configuration error. Please check your settings."

   # Bad: Exposes internals
   except ValueError as err:
       return f"Error: {err}\nStack: {traceback.format_exc()}"  # TMI!
   ```

**Security Checklist:**
- [ ] No SQL injection vulnerabilities
- [ ] No command injection vulnerabilities
- [ ] No path traversal vulnerabilities
- [ ] No hardcoded secrets
- [ ] No logging of passwords/API keys
- [ ] No logging of PII/user conversations
- [ ] Input validation on all external data
- [ ] Sanitized error messages
- [ ] Secure defaults (fail closed)

### Privacy Standards (DEFAULT)

**Privacy > Convenience**

```python
# Good: Privacy-first routing
async def route_query(query: str) -> str:
    """Route query with privacy priority."""
    # Try local agents first
    if result := await try_local_agents(query):
        return result

    # Ask before going to cloud
    if not user_consented_to_cloud:
        return "I can only process that with cloud AI. Enable in settings?"

    # Only then use cloud
    return await try_cloud_agents(query)

# Bad: Cloud-first
async def route_query(query: str) -> str:
    """Privacy not considered."""
    return await try_cloud_agents(query)  # Always cloud!
```

**Privacy Principles:**
1. **Data Minimization**: Only send necessary data to cloud
2. **Local First**: Prefer local processing
3. **User Control**: Clear indication of cloud usage
4. **Transparent**: User knows where data goes

## Code Style

### Formatting & Linting

```bash
# Format code with black (100 char line length)
black custom_components tests

# Lint with ruff (includes security checks)
ruff check .

# Fix auto-fixable issues
ruff check . --fix

# Type check with mypy (strict mode)
mypy custom_components
```

### Type Hints (Required)

```python
# Good: Full type hints
from typing import Any

async def process_agent(
    config: dict[str, Any],
    user_input: ConversationInput,
    timeout: int = 30,
) -> ConversationResult | None:
    """Process input with agent."""
    ...

# Bad: No type hints
async def process_agent(config, user_input, timeout=30):
    """Missing types!"""
    ...
```

### Docstrings (Required)

```python
def calculate_priority(
    agent_config: dict[str, Any],
    default: int = 50,
) -> int:
    """Calculate agent priority from configuration.

    Priority 0 (or `is_router=True`) indicates a router/filter agent. Priority 1-100
    indicates processing agents, with lower values having higher
    priority.

    Args:
        agent_config: Agent configuration dictionary containing
            priority and other settings.
        default: Default priority if not specified (default: 50).

    Returns:
        The agent's priority value (0-100).

    Raises:
        ValueError: If priority is outside valid range (0-100).
    """
    priority = agent_config.get("priority", default)

    if not 0 <= priority <= 100:
        raise ValueError(f"Priority must be 0-100, got {priority}")

    return priority
```

**When to Document:**
- ✅ All public functions/methods (required)
- ✅ All classes (required)
- ✅ Complex private functions
- ✅ Non-obvious logic
- ❌ Don't document obvious code

### Import Organization

```python
"""Module docstring."""
from __future__ import annotations  # Always first

import asyncio  # Standard library
import logging
from typing import Any

import aiohttp  # Third-party
import voluptuous as vol

from homeassistant.core import HomeAssistant  # Home Assistant
from homeassistant.helpers import config_validation as cv

from .const import DOMAIN  # Local (relative imports)
from .ollama_client import OllamaClient
```

## Error Handling

### Exception Handling

```python
# Good: Specific exception handling
async def fetch_data(url: str) -> dict[str, Any] | None:
    """Fetch data from URL."""
    try:
        async with asyncio.timeout(30):
            async with session.get(url) as response:
                return await response.json()
    except aiohttp.ClientError as err:
        _LOGGER.error("Connection failed: %s", err)
        return None
    except asyncio.TimeoutError:
        _LOGGER.warning("Request timed out")
        return None
    except ValueError as err:
        _LOGGER.error("Invalid JSON: %s", err)
        return None
    # Let unexpected errors bubble up

# Bad: Catch-all
async def fetch_data(url: str) -> dict[str, Any] | None:
    """Fetch data from URL."""
    try:
        return await session.get(url).json()
    except Exception:  # Too broad! Hides bugs
        return None
```

**When to Catch:**
- ✅ Expected errors (network, timeout, validation)
- ✅ External service failures
- ✅ User input errors
- ❌ Programming errors (let them raise)
- ❌ Unexpected errors (let them bubble)

### Logging

```python
# Good: Appropriate log levels
_LOGGER.debug("Trying agent: %s (priority: %d)", name, priority)
_LOGGER.info("Agent %s handled request successfully", name)
_LOGGER.warning("Agent %s timed out, trying next", name)
_LOGGER.error("Failed to connect to %s: %s", sanitize_url(url), err)
_LOGGER.exception("Unexpected error in routing")  # Includes traceback

# Bad: Wrong levels or unsafe
_LOGGER.error("Trying agent")  # Not an error!
_LOGGER.debug("Failed completely!")  # Should be WARNING/ERROR
_LOGGER.info("API key: %s", api_key)  # NEVER log secrets!
```

## Development Workflow

### 1. Setup Development Environment

```bash
# Clone repository
git clone https://github.com/pfrye/NeuralBridge.git
cd NeuralBridge

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements_test.txt
```

### 2. Write Tests First (TDD)

```bash
# Create test file
touch tests/unit/test_new_feature.py

# Write failing test
# Implement feature
# Make test pass
```

### 3. Check Code Quality

```bash
# Format
black custom_components tests

# Lint
ruff check .

# Type check
mypy custom_components

# Run tests with coverage
pytest --cov --cov-fail-under=100

# All checks in one command
black . && ruff check . && mypy custom_components && pytest --cov --cov-fail-under=100
```

### 4. Pre-Commit Checklist

Before committing:
- [ ] 100% test coverage
- [ ] All tests pass
- [ ] Black formatted
- [ ] Ruff passes (including security checks)
- [ ] Mypy passes (strict mode)
- [ ] No secrets in code
- [ ] No sensitive data in logs
- [ ] Line length ≤ 115
- [ ] Function complexity ≤ 15
- [ ] Public functions have docstrings

### 5. Commit

```bash
# Run all checks
make check  # Or run checks manually

# Commit
git add .
git commit -m "feat: descriptive commit message"

# Commit message format:
# feat: new feature
# fix: bug fix
# docs: documentation
# test: test updates
# refactor: code refactoring
# style: formatting
# chore: maintenance
```

## Common Patterns

### Async Best Practices

```python
# Good: Proper async with timeout
async def fetch_with_timeout(url: str, timeout: int = 30) -> str | None:
    """Fetch URL with timeout."""
    try:
        async with asyncio.timeout(timeout):
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.text()
    except asyncio.TimeoutError:
        _LOGGER.warning("Request timed out after %d seconds", timeout)
        return None

# Bad: No timeout
async def fetch(url: str) -> str:
    """Fetch URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()  # Can hang forever!
```

### Resource Cleanup

```python
# Good: Proper cleanup
class MyClient:
    """Client with cleanup."""

    def __init__(self) -> None:
        """Initialize client."""
        self._session: aiohttp.ClientSession | None = None

    async def close(self) -> None:
        """Close client resources."""
        if self._session:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> MyClient:
        """Enter context manager."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit context manager."""
        await self.close()

# Usage
async with MyClient() as client:
    result = await client.fetch(url)
```

### Configuration Validation

```python
import voluptuous as vol
from homeassistant.helpers import config_validation as cv

# Good: Schema validation
AGENT_SCHEMA = vol.Schema({
    vol.Required(CONF_NAME): cv.string,
    vol.Required(CONF_PRIORITY): vol.All(
        vol.Coerce(int),
        vol.Range(min=0, max=100),
    ),
    vol.Optional(CONF_TIMEOUT, default=30): cv.positive_int,
    vol.Optional(CONF_URL): cv.url,
})

# Validate
try:
    validated = AGENT_SCHEMA(user_input)
except vol.Invalid as err:
    _LOGGER.error("Invalid config: %s", err)
    raise
```

## Troubleshooting

### Tests Failing

```bash
# Run with verbose output
pytest -vv

# Run specific test
pytest tests/unit/test_conversation.py::test_specific_function -vv

# Run with print statements visible
pytest -s

# Debug test
pytest --pdb  # Drop into debugger on failure
```

### Coverage Not 100%

```bash
# See what's not covered
pytest --cov --cov-report=term-missing

# Generate HTML report for details
pytest --cov --cov-report=html
open htmlcov/index.html
```

### Ruff Errors

```bash
# See all errors
ruff check .

# Fix auto-fixable issues
ruff check . --fix

# Ignore specific rule (last resort)
# Add to pyproject.toml under [tool.ruff.lint] ignore list
```

### Mypy Errors

```bash
# Check specific file
mypy custom_components/neuralbridge/conversation.py

# Show error context
mypy --show-error-context custom_components
```

## Resources

- **Home Assistant Dev Docs**: https://developers.home-assistant.io/
- **Python Type Hints**: https://docs.python.org/3/library/typing.html
- **Pytest Docs**: https://docs.pytest.org/
- **Ruff Rules**: https://docs.astral.sh/ruff/rules/
- **OWASP Top 10**: https://owasp.org/www-project-top-ten/

## Summary

**Remember:**
1. **Security First**: Never compromise security for performance
2. **Privacy Default**: Local processing before cloud
3. **100% Coverage**: Every line, every path, every edge case
4. **Quality Matters**: Low complexity, clean code, proper types
5. **Document**: Public functions need docstrings

**When in Doubt:**
- Security > Performance
- Privacy > Convenience
- Tested > Untested
- Simple > Complex
- Explicit > Implicit
