# Python 3.14 Upgrade Planning

**Target HA release:** 2026.3 (est. ~March 2026)
**Python version change:** 3.13 → 3.14
**Document created:** 2026-02-26
**Status:** IN PROGRESS

---

## Summary

Home Assistant 2026.3 is expected to bump the minimum Python runtime from 3.13 to 3.14.
Python 3.14 was released on 7 October 2025. The changes relevant to NeuralBridge fall into
four categories: one **already fixed**, two **pending action**, and several **monitoring items**.

---

## Already Fixed

### `asyncio.get_event_loop()` → `asyncio.get_running_loop()` (guard_rail.py)

`asyncio.get_event_loop()` emits `DeprecationWarning` when called from within a running event
loop context (where `get_running_loop()` is the correct API since Python 3.7). Under Python 3.14
the deprecation warning cadence increases. Fixed in the same commit as this document.

**File:** `custom_components/neuralbridge/guard_rail.py`
**Change:** `asyncio.get_event_loop()` → `asyncio.get_running_loop()`

---

## Pending — Must Complete Before HA 2026.3 Merges

### 1. Bump toolchain target versions in pyproject.toml

All three tool configurations are hardcoded to Python 3.13.

| Setting | File | Current | Target |
|---|---|---|---|
| `[tool.black] target-version` | pyproject.toml | `['py313']` | `['py314']` |
| `[tool.ruff] target-version` | pyproject.toml | `"py313"` | `"py314"` |
| `[tool.mypy] python_version` | pyproject.toml | `"3.13"` | `"3.14"` |

Also update the minimum-version declaration in:
- `CLAUDE.md` header (`PYTHON 3.13 IS THE ONLY SUPPORTED RUNTIME`)
- `README.md` (any version badges / requirements section)
- `requirements_test.txt` (`homeassistant>=2026.1.0` → `homeassistant>=2026.3.0`)

**Trigger:** Do this in the same PR that bumps `homeassistant` in requirements once HA 2026.3
RC is available.

### 2. Run the full test suite against Python 3.14 + HA 2026.3

Before merging any 3.14-targeted PR, verify:

```bash
python3.14 -m venv .venv314
source .venv314/bin/activate
pip install -r requirements_test.txt  # constrained to homeassistant>=2026.3
make check
```

All 100 % coverage and zero mypy errors must be maintained.

---

## Monitoring — Third-Party Dependency Wheel Availability

Several dependencies ship C extensions and must publish Python 3.14 wheels independently.
Check PyPI for `cp314` wheels before HA 2026.3 release date.

| Package | Type | Risk | Status |
|---|---|---|---|
| `aiohttp` | C extension | HIGH — core HA transport | Monitor |
| `orjson` | Rust/C extension | HIGH — used by HA core | Monitor |
| `homeassistant` | meta-package | Owned by HA team | Ships with HA 2026.3 |
| `pytest-homeassistant-custom-component` | pure Python + test harness | MEDIUM | Monitor |
| `better-profanity` | pure Python | LOW | Likely fine |
| `mypy` | C extension (mypyc) | MEDIUM | Monitor |
| `ruff` | Rust binary | LOW — not Python ABI | Fine |
| `black` | pure Python | LOW | Fine |

**Check command:**
```bash
pip index versions aiohttp orjson mypy 2>/dev/null | grep -E "^(aiohttp|orjson|mypy)"
```

---

## Low-Risk / No Action Required

### `from __future__ import annotations` (all source files)

All 18 source files and all test files carry `from __future__ import annotations`. In Python
3.14 this import is formally deprecated (PEP 649/749 — annotations are deferred by default
without it). However:

- It still works identically in 3.14 with no behaviour change
- Removal is scheduled no earlier than after Python 3.13 EOL (2029)
- No `DeprecationWarning` is currently emitted for the `__future__` import itself

**Decision:** Defer removal. The ~40-file churn is unjustified risk for a 2-week window.
Revisit when dropping support for Python < 3.14 entirely.

### `NotImplemented` return in `RouterDecision.__eq__` (conversation.py:412)

Python 3.14 makes `bool(NotImplemented)` raise `TypeError`. The existing code returns
`NotImplemented` from `__eq__` as a sentinel to Python's comparison protocol — this is
**correct** and is not coerced to bool by the runtime. No change needed. The corresponding
test also only checks `sentinel is NotImplemented` (identity, not truthiness).

### Incremental GC (`gc.collect()` behaviour change)

Python 3.14 switched to incremental garbage collection. `gc.collect(1)` now performs a GC
increment instead of collecting generation 1. NeuralBridge does not call `gc.collect()`
directly. No impact.

### `asyncio.iscoroutinefunction()` deprecation

Deprecated in 3.14, scheduled for removal in 3.16. NeuralBridge does not use it. No action.

### `asyncio` policy system deprecation

`asyncio.AbstractEventLoopPolicy` and related classes deprecated in 3.14, removed in 3.16.
NeuralBridge does not set or reference asyncio policies. No action.

---

## New Opportunities in Python 3.14

### Asyncio introspection (`asyncio` task tree printing)

Python 3.14 adds rich asyncio introspection: task IDs, coroutine stacks, awaiter chains.
Relevant for `agent_benchmark.py` and `statistics.py` — could enable more detailed async
profiling output. Track for a future enhancement.

### Template strings (t-strings, PEP 750)

The new `t"..."` syntax (like f-strings but returns a `Template` instead of `str`) could be
used for safe prompt construction in `prompts_loader.py` — user-controlled fragments would be
kept separate from static prompt text without manual escaping. Low priority but worth
exploring post-upgrade.

---

## Checklist

- [x] Fix `asyncio.get_event_loop()` → `asyncio.get_running_loop()` in guard_rail.py
- [ ] Bump `target-version` / `python_version` in pyproject.toml to 3.14
- [ ] Update minimum version statement in CLAUDE.md
- [ ] Update minimum version statement in README.md
- [ ] Bump `homeassistant>=2026.3.0` in requirements_test.txt
- [ ] Verify `aiohttp` 3.14 wheel available on PyPI
- [ ] Verify `orjson` 3.14 wheel available on PyPI
- [ ] Verify `mypy` supports Python 3.14 target
- [ ] Verify `pytest-homeassistant-custom-component` supports HA 2026.3
- [ ] Run full `make check` under Python 3.14 venv — must be 100 % pass
- [ ] Open and merge upgrade PR before HA 2026.3 stable release

---

## References

- [Python 3.14 What's New](https://docs.python.org/3.14/whatsnew/3.14.html)
- [PEP 649 — Deferred Evaluation of Annotations](https://peps.python.org/pep-0649/)
- [PEP 749 — Implementing PEP 649](https://peps.python.org/pep-0749/)
- [HA 2026.3 release milestone](https://github.com/home-assistant/core/milestones)
