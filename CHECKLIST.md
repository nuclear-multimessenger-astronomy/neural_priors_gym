# New Package Checklist

Steps to follow every time you start a new package from this template.

---

## 1. Initialize the template

```bash
make init
```

Renames all `neural_priors_gym` / `nuclear-multimessenger-astronomy` / `YOUR_NAME` placeholders in one pass. Run this before doing anything else.

---

## 2. Smoke-test the skeleton

```bash
make smoketest
```

Installs the package, runs tests, and type-checks. Everything should pass before you write a line of actual code.

---

## 3. Fill in the blanks

DONE

---

## 4. GitHub repository setup

- [ ] Enable **GitHub Pages** (Settings → Pages → Source: GitHub Actions)

---

## 5. Pre-commit hooks

```bash
uv run pre-commit install
uv run pre-commit run --all-files   # verify clean on first run
```

---

## 6. First real commit

```bash
git add -A
git commit -m "chore: initialize from package-template"
git push -u origin main
```

Check that CI goes green before adding any substantive code.

---

## 7. Ongoing

- Write tests *before* (or alongside) new features — see `CONTRIBUTING.md`.
- Keep `CITATION.cff` updated when papers are published.
- Bump the version in `pyproject.toml` and `src/<package>/__init__.py` before each release.
