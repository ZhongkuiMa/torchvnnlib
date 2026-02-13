# Contributing to TorchVNNLIB

We welcome contributions! Please follow these guidelines to ensure smooth collaboration.

## Development Setup

### Setting Up Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/torchvnnlib.git
cd torchvnnlib

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify setup
pytest tests/ -v
ruff check src/torchvnnlib tests
```

## Development Workflow

### Branch Naming Conventions

Use descriptive branch names with prefixes:
- `feature/` - New features (e.g., `feature/add-type6-processor`)
- `fix/` - Bug fixes (e.g., `fix/parser-tokenization-issue`)
- `refactor/` - Code refactoring (e.g., `refactor/simplify-backend`)
- `docs/` - Documentation updates (e.g., `docs/improve-api-examples`)
- `test/` - Test improvements (e.g., `test/add-edge-cases`)

### Commit Message Format

Write clear, concise commit messages:

```
<type>: <short summary in present tense>

<optional detailed description>

<optional footer with issue references>
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code refactoring (no behavior change)
- `test:` - Add or update tests
- `docs:` - Documentation changes
- `style:` - Code style/formatting (ruff, whitespace)
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks (dependencies, config)

**Examples:**
```
feat: Add Type 6 processor for complex OR patterns

Implements optimized processing for Type 6 VNN-LIB specifications
with nested OR groups and complex constraint structures.

Fixes #42
```

```
fix: Correct tokenization for negative floating-point constants

The tokenizer was incorrectly parsing negative floats as separate
tokens, causing parse failures for constraints with negative bounds.
```

### Push Workflow

```bash
# Create feature branch
git checkout -b feature/my-new-feature

# Make changes and commit
git add <files>
git commit -m "feat: Add my new feature"

# Run pre-push checks (recommended)
pytest tests/ -v
ruff check src/torchvnnlib tests
python -m mypy

# Push to your fork
git push origin feature/my-new-feature

# Create PR on GitHub
```

## Code Quality Standards

### Linting
```bash
ruff check src/torchvnnlib tests
```

### Formatting
```bash
ruff format src/torchvnnlib tests
```

### Type Checking
```bash
python -m mypy
```

### Code Style

TorchVNNLIB follows strict code quality standards:

- **Formatter**: `ruff format` (100 char line length)
- **Linter**: `ruff check` (comprehensive ruleset)
- **Type checker**: `mypy`
- **Docstrings**: PEP 257 style with type hints

## Pull Request Guidelines

1. **Before creating a PR:**
   ```bash
   # Run all tests
   pytest tests/ -v
   pytest tests/test_benchmarks/ -v

   # Run linting
   ruff check src/torchvnnlib tests
   ruff format src/torchvnnlib tests

   # Run type checking
   python -m mypy
   ```

2. **Create PR with:**
   - **Clear title**: Follow commit message format
   - **Description**: Explain what changes and why
   - **Tests**: Add tests for new features/fixes
   - **Documentation**: Update README/docstrings if needed

3. **Review process:**
   - Maintainers will review within 48-72 hours
   - Address feedback by pushing to your PR branch
   - Once approved, maintainers will merge

## Testing

Run the test suite to verify functionality:

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/torchvnnlib --cov-report=term-missing -v

# Run specific test file
pytest tests/test_units/test_ast_parse.py -v
```

## CI/CD

### Running CI Locally

Before pushing, run the same checks as GitHub Actions:

```bash
# Full CI simulation
pytest tests/ --cov=src/torchvnnlib --cov-report=term-missing -v
ruff check src/torchvnnlib tests
ruff format --check src/torchvnnlib tests
python -m mypy
```

## Documentation

- All public classes/functions must have docstrings
- Use reStructuredText format (Sphinx-compatible)
- Include type hints for all parameters and return values

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bug reports**: Open an Issue with reproducible example
- **Feature requests**: Open an Issue with use case description

Thank you for contributing to TorchVNNLIB!
