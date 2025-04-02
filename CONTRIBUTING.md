# Contributing to DecadalClimate

Thank you for your interest in contributing to DecadalClimate! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

Before submitting a bug report:
- Check the issue tracker to avoid duplicate reports
- Collect relevant information about the bug

When submitting a bug report, please include:
- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- System information (OS, Python version, etc.)
- Any relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! When submitting:
- Use a clear and descriptive title
- Provide a detailed description of the proposed enhancement
- Explain why this enhancement would be useful

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure they pass
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

1. Clone your fork of the repository
2. Set up the Python environment:
   ```bash
   ./scripts/setup_env.sh --dev
   source ./scripts/activate_env.sh
   ```
3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Coding Standards

This project uses several tools to maintain code quality:

- **Black**: For code formatting
- **isort**: For sorting imports
- **flake8**: For linting
- **mypy**: For static type checking

Pre-commit hooks are configured to run these tools automatically when you commit.

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Write docstrings in NumPy format
- Include type hints where possible

### Tests

- Write tests for all new functionality
- Ensure existing tests pass before submitting a PR
- Run tests using pytest:
  ```bash
  pytest
  ```

## Documentation

Good documentation is essential. When contributing:

- Update or add docstrings for all functions and classes
- Update the README.md if necessary
- Add examples for new features
- Ensure the documentation is clear and understandable

## Git Workflow

1. Create a branch from `main`
2. Make your changes
3. Keep your branch updated with `main`
4. Submit a PR to `main`

### Commit Messages

Write clear, concise commit messages that describe the changes made:
- Start with a verb in the present tense
- Keep the first line under 72 characters
- Reference issues or PRs when relevant

Example:
```
Add function to reshape multiple files at once

This adds a new function that can process multiple files in batch mode,
which improves processing efficiency for large datasets.

Closes #123
```

## Thank You!

Your contributions help make this project better. We appreciate your time and effort!
