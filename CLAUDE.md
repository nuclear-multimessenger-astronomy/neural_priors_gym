# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## IMPORTANT GUIDELINES

**Contributing guide**: `CONTRIBUTING.md` at the repo root is the authoritative reference for development workflow and PR requirements. Always consult it when working on new features or before opening a PR.

**Run tests**: After a major change in the source code, run tests locally to ensure nothing is broken before pushing. Use `uv run pytest -v tests/` to run all tests.

**Testing Philosophy**: When tests fail, investigate root causes rather than modifying tests to pass.

**Documentation Style**: Write clear, concise documentation in full sentences as if by a human researcher. Avoid LLM-like verbosity.

**Sphinx warnings are errors**: CI runs `sphinx-build -W --keep-going`, which turns every Sphinx warning into a build failure that blocks merging. Always verify locally before committing.

**Math Formatting in Docstrings**: All mathematical expressions in docstrings must use Sphinx/reStructuredText formatting:
- Use `:math:` role for inline math: `:math:`\Gamma(x)``
- Use `.. math::` directive for display equations
- Always use raw strings (`r"""`) for docstrings containing LaTeX

**File Operations**: Use proper tools (Write, Edit, Read) instead of bash heredocs or cat redirection.

---

## Project Overview
