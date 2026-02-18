# NeuralBridge Directory Structure

This document provides an overview of the project's folder structure and organization.

## Root Directory

```
NeuralBridge/
├── .github/                    # GitHub configuration
│   ├── .copilot-instructions.md
│   └── workflows/              # CI/CD workflows
│       ├── test.yml            # Testing pipeline
│       └── validate.yml        # HACS/Hassfest validation
├── custom_components/          # Main application code
│   └── neuralbridge/           # Integration package
│       ├── __init__.py         # Integration setup
│       ├── manifest.json       # Integration metadata
│       ├── const.py            # Constants and configuration
│       ├── conversation.py     # Conversation agent implementation
│       └── config_flow.py      # Configuration UI
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py             # Test fixtures
│   ├── unit/                   # Unit tests
│   │   ├── test_init.py
│   │   └── test_conversation.py
│   ├── integration/            # Integration tests
│   │   └── test_full_flow.py
│   └── fixtures/               # Test data
│       └── conversation_samples.py
├── docs/                       # All documentation (keep docs here, not root)
│   ├── STRUCTURE.md            # This file
│   ├── CODING_STANDARDS.md     # Coding standards quick reference
│   ├── DEVELOPMENT_GUIDE.md    # Comprehensive development guide
│   ├── CONFIGURATION_EXAMPLES.md # Configuration examples
│   ├── IMPLEMENTATION_SUMMARY.md # Implementation summary
│   ├── PROJECT_SETUP.md        # Project setup reference
│   ├── architecture/           # Technical architecture
│   │   └── README.md
│   ├── user-guide/             # User documentation
│   │   ├── installation.md
│   │   ├── usage.md
│   │   ├── agent-configuration.md
│   │   └── quick-start.md
│   ├── api/                    # API reference
│   │   └── conversation_entity.md
│   └── planning/               # Project planning
│       ├── requirements/       # Requirements specifications
│       │   ├── functional_requirements.md
│       │   └── technical_requirements.md
│       ├── design/             # Design documents
│       │   └── routing_algorithm.md
│       └── milestones/         # Release milestones
│           ├── v0.1.0_mvp.md
│           └── v0.2.0_enhanced_filtering.md
├── config/                     # Configuration examples
│   └── examples/
│       ├── basic_configuration.yaml
│       └── advanced_configuration.yaml
├── .gitignore                  # Git ignore rules
├── CLAUDE.md                   # Claude Code instructions (root only - required)
├── LICENSE                     # Project license
├── README.md                   # Project overview (root only - required by HACS/GitHub)
├── hacs.json                   # HACS metadata
├── pytest.ini                  # Pytest configuration
├── pyproject.toml              # Python project configuration
└── requirements_test.txt       # Testing dependencies
```

## Directory Purposes

### `/custom_components/neuralbridge/` - Application Code
The main Home Assistant integration code. This is what gets installed by users.

**Key Files:**
- `__init__.py`: Integration entry point and lifecycle management
- `manifest.json`: HACS and Home Assistant metadata
- `const.py`: Configuration constants and default values
- `conversation.py`: Core tiered routing implementation
- `config_flow.py`: Configuration UI flows

### `/tests/` - Test Suite
Comprehensive test coverage for the integration.

**Structure:**
- `unit/`: Fast, isolated tests for individual components
- `integration/`: End-to-end tests with Home Assistant
- `fixtures/`: Reusable test data and sample inputs

### `/docs/` - Documentation
All user and developer documentation. **All new docs must go here.**

**Structure:**
- `architecture/`: Technical design and routing logic
- `user-guide/`: Installation and usage instructions (including `quick-start.md`)
- `api/`: API reference for developers
- `planning/`: Requirements, design decisions, and milestone roadmaps
- Root `docs/*.md` files: Development and project-level guides

### `/config/` - Configuration Examples
Example configuration files for users.

**Structure:**
- `examples/`: Sample YAML configurations

### `/.github/` - GitHub Configuration
GitHub Actions workflows and repository configuration.

**Structure:**
- `workflows/`: CI/CD pipelines for testing and validation

## File Naming Conventions

- **Python files**: `snake_case.py`
- **Test files**: `test_*.py`
- **Markdown docs**: `lowercase_with_underscores.md`
- **Config files**: `descriptive_name.yaml`

## Development Workflow

1. **Planning**: Update documents in `/docs/planning/`
2. **Implementation**: Write code in `/custom_components/neuralbridge/`
3. **Testing**: Add tests in `/tests/`
4. **Documentation**: Update `/docs/` as needed (never add docs to root)
5. **CI/CD**: GitHub Actions automatically run tests

## HACS Installation

HACS installs only the `/custom_components/neuralbridge/` directory to the user's Home Assistant instance. All other directories are for development only.
