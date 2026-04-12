# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`dataset-builder` is a Python 3.12 project managed with `pyproject.toml`. It is in early development with no dependencies yet declared.

## Commands

```bash
# Run the application
uv run python main.py

# Add a dependency
uv add <package>

# Sync the environment
uv sync
```

## Structure

- `main.py` — entry point; contains the `main()` function
- `pyproject.toml` — project metadata and dependency declarations (PEP 517/518)
- `.python-version` — pins Python to 3.12 (used by pyenv/uv)

## Skills & Patterns

Read `.claude/skills/python-project-builder.md` for the established project structure pattern used across this codebase — covers Lambda layout, dependency injection, AWS client wrappers, service layer, configuration, and local testing conventions.
