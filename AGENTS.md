# AGENTS.md

## Build, Lint, and Test Commands

### Build
```bash
# Install dependencies
uv pip install -e .

# Build the package
uv build
```

### Lint
```bash
# Run linters (if configured)
uv run ruff check .
uv run ruff format .
```

### Test
```bash
# Run tests (if any)
uv run pytest tests/ -v
```

## Code Style Guidelines

### Imports
- Standard library imports first, followed by third-party imports, then local imports
- Use `import` for modules and `from module import` for specific functions/classes
- Group imports by standard library, third-party, and local imports

### Formatting
- Use 4 spaces for indentation (not tabs)
- Follow PEP 8 style guidelines
- Maximum line length of 88 characters (per Black formatting)
- Use double quotes for strings

### Types
- Use type hints for all function parameters and return values
- Use `Optional` for parameters that can be None
- Use `Union` for parameters that can be multiple types

### Naming Conventions
- Use `snake_case` for variables and functions
- Use `PascalCase` for classes
- Use `UPPER_CASE` for constants
- Use descriptive names that clearly indicate the purpose

### Error Handling
- Use specific exception types instead of bare `except:`
- Log errors appropriately with context information
- Handle errors gracefully and provide meaningful error messages
- Use try/except blocks for operations that might fail

### Code Organization
- Keep functions short and focused on a single responsibility
- Group related functions in classes when appropriate
- Use docstrings for all public functions and classes
- Organize imports in a logical, readable order