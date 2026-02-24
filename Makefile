# NeuralBridge — developer convenience targets
#
# Run `make check` before every commit to mirror exactly what the
# pre-commit hooks and CI validate.  The order matches the checklist
# in CLAUDE.md / .github/.copilot-instructions.md.
#
#   make check    — full pre-commit checklist (lint → format → type → test)
#   make lint     — ruff only (fast, run frequently while developing)
#   make format   — black + ruff --fix (auto-format in place)
#   make type     — mypy strict
#   make test     — pytest with 100% coverage gate
#   make clean    — remove build/coverage artefacts

.DEFAULT_GOAL := check

# ── Tools ─────────────────────────────────────────────────────────────────────
PYTHON     ?= python3
BLACK      ?= black
RUFF       ?= ruff
MYPY       ?= mypy
PYTEST     ?= pytest

SRC        := custom_components tests

# ── Targets ───────────────────────────────────────────────────────────────────

.PHONY: check
check: lint format-check type test  ## Run the full pre-commit checklist

.PHONY: lint
lint:  ## Ruff static analysis (no auto-fix)
	$(RUFF) check $(SRC)

.PHONY: format
format:  ## Auto-format: black + ruff --fix
	$(BLACK) .
	$(RUFF) check --fix $(SRC)

.PHONY: format-check
format-check:  ## Check formatting without modifying files
	$(BLACK) --check .

.PHONY: type
type:  ## Mypy strict type checking
	$(MYPY) custom_components

.PHONY: test
test:  ## Run pytest with 100% coverage gate
	$(PYTEST)

.PHONY: clean
clean:  ## Remove coverage and cache artefacts
	rm -rf htmlcov .coverage .mypy_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
