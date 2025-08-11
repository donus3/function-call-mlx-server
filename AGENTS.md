# AGENTS.md

## Build, Lint, and Test Commands

### Build
```bash
uv pip install -e .
uv build
```

### Lint
```bash
uv run ruff check .
uv run ruff format .
```

### Test
```bash
uv run pytest tests/ -v
```

## Code Style Guidelines

### Imports
- Standard library first, then third-party, then local.

### Formatting
- 4 spaces, PEP8, 88 char limit, double quotes.

### Types
- Type hints, Optional, Union.

### Naming
- snake_case, PascalCase, UPPER_CASE.

### Error Handling
- Specific exceptions, log, graceful.

### Organization
- Short functions, docstrings, logical import order.