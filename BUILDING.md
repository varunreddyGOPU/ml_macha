# KFP ML Library — Build, Package & Distribute Guide

> Complete documentation for building wheel files, publishing to GitHub,
> setting up CI/CD with GitHub Actions, and managing the project with
> **Poetry** and **uv**.

---

## Table of Contents

1. [Quick Reference (Cheat Sheet)](#1-quick-reference-cheat-sheet)
2. [Building a Wheel File (setuptools)](#2-building-a-wheel-file-setuptools)
3. [Publishing to GitHub](#3-publishing-to-github)
4. [GitHub Actions CI/CD](#4-github-actions-cicd)
5. [Poetry — Full Guide](#5-poetry--full-guide)
6. [uv — Full Guide](#6-uv--full-guide)
7. [Choosing Between setuptools / Poetry / uv](#7-choosing-between-setuptools--poetry--uv)
8. [Publishing to PyPI](#8-publishing-to-pypi)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Quick Reference (Cheat Sheet)

| Task | setuptools | Poetry | uv |
|---|---|---|---|
| Install deps | `pip install -e ".[dev]"` | `poetry install --with dev` | `uv sync --all-extras` |
| Build wheel | `python -m build` | `poetry build` | `uv build` |
| Run tests | `pytest tests/` | `poetry run pytest tests/` | `uv run pytest tests/` |
| Publish PyPI | `twine upload dist/*` | `poetry publish` | `uv publish` |
| Lock deps | `pip freeze > lock.txt` | `poetry lock` | `uv lock` |
| Add dep | edit requirements.txt | `poetry add <pkg>` | `uv add <pkg>` |

---

## 2. Building a Wheel File (setuptools)

### 2.1 Prerequisites

```bash
# Python 3.9+ required
python --version

# Install build tools
pip install --upgrade pip setuptools wheel build
```

### 2.2 Project Files Required

Your project already has these files:

| File | Purpose |
|---|---|
| `setup.py` | Package metadata, dependencies, extras |
| `pyproject.toml` | Build-system declaration + tool configs |
| `requirements.txt` | Core dependency list |
| `kfp_ml_library/__init__.py` | Version string + public API |

### 2.3 Build the Wheel

```bash
# Navigate to project root
cd d:\kfp_pipelien

# Build both wheel (.whl) and source distribution (.tar.gz)
python -m build
```

Output appears in `dist/`:

```
dist/
├── kfp_ml_library-1.0.0-py3-none-any.whl
└── kfp-ml-library-1.0.0.tar.gz
```

### 2.4 Verify the Wheel

```bash
# Check wheel contents
python -m zipfile -l dist/kfp_ml_library-1.0.0-py3-none-any.whl

# Install locally to test
pip install dist/kfp_ml_library-1.0.0-py3-none-any.whl

# Verify import
python -c "import kfp_ml_library; print(kfp_ml_library.__version__)"
# → 1.0.0
```

### 2.5 Install with Extras

```bash
# Core only
pip install dist/kfp_ml_library-1.0.0-py3-none-any.whl

# With TensorFlow support
pip install "dist/kfp_ml_library-1.0.0-py3-none-any.whl[tensorflow]"

# With PyTorch support
pip install "dist/kfp_ml_library-1.0.0-py3-none-any.whl[pytorch]"

# With GCP support
pip install "dist/kfp_ml_library-1.0.0-py3-none-any.whl[gcp]"

# Everything
pip install "dist/kfp_ml_library-1.0.0-py3-none-any.whl[all]"

# Dev tools
pip install "dist/kfp_ml_library-1.0.0-py3-none-any.whl[dev]"
```

### 2.6 Editable Install (Development)

```bash
# Install in editable mode — changes to source are reflected immediately
pip install -e ".[dev]"
```

### 2.7 Clean Previous Builds

```bash
# Windows PowerShell
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue

# Linux / macOS
rm -rf dist/ build/ *.egg-info/
```

---

## 3. Publishing to GitHub

### 3.1 Create a GitHub Repository

1. Go to **https://github.com/new**
2. Repository name: `kfp-ml-library`
3. Description: *A comprehensive Kubeflow Pipelines ML model deployment library*
4. Visibility: **Private** or **Public**
5. Do **NOT** initialize with README (we already have one)
6. Click **Create repository**

### 3.2 Initialize Git and Push

```bash
cd d:\kfp_pipelien

# Initialize git
git init
git branch -M main

# Add all files
git add .

# Verify what will be committed
git status

# Commit
git commit -m "feat: initial commit — KFP ML Library v1.0.0"

# Add remote (replace YOUR-USERNAME)
git remote add origin https://github.com/YOUR-USERNAME/kfp-ml-library.git

# Push
git push -u origin main
```

### 3.3 Update setup.py URL

Open `setup.py` and replace:

```python
url="https://github.com/your-org/kfp-ml-library",
```

with your actual GitHub URL:

```python
url="https://github.com/YOUR-USERNAME/kfp-ml-library",
```

### 3.4 Create a Release with a Tag

```bash
# Tag the current commit
git tag -a v1.0.0 -m "Release v1.0.0 — initial public release"

# Push the tag (this triggers the GitHub Actions workflow)
git push origin v1.0.0
```

### 3.5 Recommended Branch Protection

Go to **Settings → Branches → Add rule** for `main`:

- [x] Require pull request reviews before merging
- [x] Require status checks to pass (select `lint`, `test`, `build`)
- [x] Require branches to be up to date before merging

---

## 4. GitHub Actions CI/CD

Three workflow files are provided in `.github/workflows/`:

| Workflow | File | When to use |
|---|---|---|
| **setuptools** | `build-wheel.yml` | Default — pip + setuptools |
| **Poetry** | `build-wheel-poetry.yml` | If using Poetry |
| **uv** | `build-wheel-uv.yml` | If using uv |

> **Pick one.** Delete the other two workflows you don't need.

### 4.1 Workflow: `build-wheel.yml` (setuptools)

**Triggers:**
- Push/PR to `main` → lint + test + build (no publish)
- Push tag `v*` → lint + test + build + publish to PyPI + GitHub Release

**Jobs:**

```
lint  ────────┐
              ├──→  build  ──→  publish-pypi
test (matrix) ┘              └→  github-release
```

**What each job does:**

| Job | Description |
|---|---|
| `lint` | Runs `ruff check` and `black --check` |
| `test` | Tests on Python 3.9, 3.10, 3.11 |
| `build` | `python -m build`, uploads artifacts |
| `publish-pypi` | Publishes to PyPI (tag only) |
| `github-release` | Creates GitHub Release with wheel attached (tag only) |

### 4.2 Required GitHub Secrets

Go to **Settings → Secrets and variables → Actions → New repository secret**:

| Secret Name | Value | Required For |
|---|---|---|
| `PYPI_API_TOKEN` | Your PyPI API token | Publishing to PyPI |

**How to get a PyPI API token:**
1. Go to https://pypi.org/manage/account/token/
2. Create token with scope "Entire account" or project-specific
3. Copy the token (starts with `pypi-...`)
4. Add it as a GitHub Secret named `PYPI_API_TOKEN`

### 4.3 Triggering the Workflow

```bash
# For CI only (no publish)
git push origin main

# For CI + publish
git tag v1.0.0
git push origin v1.0.0
```

### 4.4 Downloading Built Wheel from GitHub Actions

1. Go to **Actions** tab in your repository
2. Click the latest **Build & Publish Wheel** run
3. Scroll down to **Artifacts**
4. Download `kfp-ml-library-wheel`

---

## 5. Poetry — Full Guide

### 5.1 What is Poetry?

[Poetry](https://python-poetry.org/) is a modern Python packaging and dependency management tool that:
- Manages virtual environments automatically
- Resolves dependencies deterministically with a lock file
- Builds wheel/sdist packages
- Publishes to PyPI
- Replaces setup.py, setup.cfg, requirements.txt, and MANIFEST.in

### 5.2 Install Poetry

```bash
# Recommended: official installer (all platforms)
curl -sSL https://install.python-poetry.org | python3 -

# Or on Windows PowerShell:
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Or via pipx (if you have pipx):
pipx install poetry

# Verify
poetry --version
# → Poetry (version 1.8.x)
```

### 5.3 Configure Poetry

```bash
# Create virtual environments inside the project folder (recommended)
poetry config virtualenvs.in-project true

# Verify config
poetry config --list
```

### 5.4 Set Up This Project with Poetry

```bash
cd d:\kfp_pipelien

# OPTION A: Use the provided Poetry config
copy pyproject.poetry.toml pyproject.toml      # Windows
# cp pyproject.poetry.toml pyproject.toml      # Linux/macOS

# Remove setup.py (Poetry replaces it)
del setup.py                                    # Windows
# rm setup.py                                  # Linux/macOS

# Install all dependencies (creates .venv/ and poetry.lock)
poetry install --with dev

# Install with extras
poetry install --with dev --extras "gcp"
poetry install --with dev --extras "tensorflow"
poetry install --with dev --extras "all"
```

### 5.5 Poetry — Daily Workflow

#### Add a new dependency

```bash
# Add a core dependency
poetry add httpx

# Add a dev dependency
poetry add --group dev pre-commit

# Add an optional dependency
poetry add --optional "dask>=2024.1.0"
```

#### Remove a dependency

```bash
poetry remove httpx
```

#### Update dependencies

```bash
# Update all deps
poetry update

# Update a specific package
poetry update pandas

# Show outdated packages
poetry show --outdated
```

#### Run commands in the venv

```bash
# Run any command inside the Poetry venv
poetry run python -c "import kfp_ml_library; print(kfp_ml_library.__version__)"

# Run tests
poetry run pytest tests/ -v

# Run linter
poetry run ruff check kfp_ml_library/
poetry run black --check kfp_ml_library/

# Run type checking
poetry run mypy kfp_ml_library/

# Start a shell inside the venv
poetry shell
```

#### Show dependency tree

```bash
poetry show --tree
```

### 5.6 Poetry — Build Wheel

```bash
# Build wheel + sdist
poetry build

# Output:
# Building kfp-ml-library (1.0.0)
#   - Building sdist
#   - Built kfp_ml_library-1.0.0.tar.gz
#   - Building wheel
#   - Built kfp_ml_library-1.0.0-py3-none-any.whl
```

### 5.7 Poetry — Publish to PyPI

```bash
# First time: configure PyPI credentials
poetry config pypi-token.pypi pypi-YOUR-TOKEN-HERE

# Publish
poetry publish --build

# Or publish to a private registry
poetry config repositories.private https://your-private-pypi.com/simple/
poetry publish --repository private --build
```

### 5.8 Poetry — Publish to TestPyPI

```bash
# Configure TestPyPI
poetry config repositories.testpypi https://test.pypi.org/legacy/
poetry config pypi-token.testpypi pypi-YOUR-TEST-TOKEN

# Publish to TestPyPI
poetry publish --repository testpypi --build

# Install from TestPyPI to verify
pip install --index-url https://test.pypi.org/simple/ kfp-ml-library
```

### 5.9 Poetry — Version Management

```bash
# Bump patch: 1.0.0 → 1.0.1
poetry version patch

# Bump minor: 1.0.0 → 1.1.0
poetry version minor

# Bump major: 1.0.0 → 2.0.0
poetry version major

# Set exact version
poetry version 2.1.0

# Show current version
poetry version
```

### 5.10 Poetry — Lock File

```bash
# Generate / update lock file
poetry lock

# Install exactly what's in the lock file (CI-friendly)
poetry install --no-interaction

# Export to requirements.txt (for Docker or legacy systems)
poetry export -f requirements.txt --output requirements.txt --without-hashes
poetry export -f requirements.txt --output requirements-dev.txt --with dev --without-hashes
```

### 5.11 Poetry — Environment Management

```bash
# Show Python environments Poetry knows about
poetry env list

# Use a specific Python version
poetry env use python3.11

# Remove a venv
poetry env remove python3.11

# Show environment info
poetry env info
```

### 5.12 Poetry — pyproject.toml Explanation

Here's what each section of `pyproject.poetry.toml` does:

```toml
[tool.poetry]                      # ← Package metadata
name = "kfp-ml-library"           # ← PyPI package name
version = "1.0.0"                 # ← Version (single source of truth)
packages = [{ include = "..." }]  # ← Which folders to include

[tool.poetry.dependencies]         # ← Required runtime dependencies
python = ">=3.9,<3.13"            # ← Python version constraint

[tool.poetry.extras]               # ← Optional dependency groups
tensorflow = ["tensorflow"]        # ← pip install kfp-ml-library[tensorflow]

[tool.poetry.group.dev.dependencies]  # ← Dev-only deps (not shipped)

[build-system]                     # ← MUST use poetry-core
requires = ["poetry-core>=1.9.0"]
build-backend = "poetry.core.masonry.api"
```

---

## 6. uv — Full Guide

### 6.1 What is uv?

[uv](https://docs.astral.sh/uv/) is an extremely fast Python package manager by Astral (creators of Ruff). It:
- Installs packages **10-100x faster** than pip
- Replaces pip, pip-tools, virtualenv, and pyenv
- Manages Python versions
- Builds wheels
- Publishes to PyPI
- Uses standard PEP 621 `pyproject.toml` (no vendor lock-in)

### 6.2 Install uv

```bash
# On Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip (any platform)
pip install uv

# Or via pipx
pipx install uv

# Verify
uv --version
# → uv 0.5.x
```

### 6.3 Set Up This Project with uv

```bash
cd d:\kfp_pipelien

# OPTION A: Use the provided uv config
copy pyproject.uv.toml pyproject.toml        # Windows
# cp pyproject.uv.toml pyproject.toml        # Linux/macOS

# Remove setup.py (uv reads pyproject.toml directly)
del setup.py                                  # Windows
# rm setup.py                                # Linux/macOS

# Install a specific Python version (if needed)
uv python install 3.11

# Create venv and install all deps
uv sync --all-extras

# Or install specific extras
uv sync --extra gcp
uv sync --extra tensorflow
```

### 6.4 uv — Daily Workflow

#### Add a new dependency

```bash
# Add a core dependency
uv add httpx

# Add a dev dependency
uv add --dev pre-commit

# Add with version constraint
uv add "dask>=2024.1.0"
```

#### Remove a dependency

```bash
uv remove httpx
```

#### Update dependencies

```bash
# Update all
uv lock --upgrade

# Update specific package
uv lock --upgrade-package pandas
```

#### Run commands in the venv

```bash
# Run any command
uv run python -c "import kfp_ml_library; print(kfp_ml_library.__version__)"

# Run tests
uv run pytest tests/ -v

# Run linter
uv run ruff check kfp_ml_library/
uv run black --check kfp_ml_library/

# Run type checking
uv run mypy kfp_ml_library/
```

#### Install (pip compatibility mode)

```bash
# uv can be a drop-in replacement for pip
uv pip install pandas
uv pip install -r requirements.txt
uv pip install -e ".[dev]"
uv pip list
uv pip show kfp-ml-library
```

### 6.5 uv — Build Wheel

```bash
# Build wheel + sdist
uv build

# Output appears in dist/
# dist/
# ├── kfp_ml_library-1.0.0-py3-none-any.whl
# └── kfp_ml_library-1.0.0.tar.gz

# Build only wheel
uv build --wheel

# Build only sdist
uv build --sdist
```

### 6.6 uv — Publish to PyPI

```bash
# Publish (will prompt for token)
uv publish --token pypi-YOUR-TOKEN

# Or set token via environment variable
$env:UV_PUBLISH_TOKEN = "pypi-YOUR-TOKEN"    # PowerShell
export UV_PUBLISH_TOKEN="pypi-YOUR-TOKEN"     # bash
uv publish
```

### 6.7 uv — Publish to TestPyPI

```bash
# Publish to TestPyPI
uv publish --publish-url https://test.pypi.org/legacy/ --token pypi-YOUR-TEST-TOKEN

# Install from TestPyPI to verify
uv pip install --index-url https://test.pypi.org/simple/ kfp-ml-library
```

### 6.8 uv — Python Version Management

```bash
# List available Python versions
uv python list

# Install a specific version
uv python install 3.11
uv python install 3.12

# Pin project to a version
uv python pin 3.11

# Use a specific version for venv
uv venv --python 3.11
```

### 6.9 uv — Lock File

```bash
# Generate lock file
uv lock

# Install exactly what's in the lock file (CI-friendly)
uv sync --frozen

# Export to requirements.txt
uv export --format requirements-txt > requirements.txt
```

### 6.10 uv — Virtual Environment Management

```bash
# Create venv (default: .venv/)
uv venv

# Create with specific Python
uv venv --python 3.11

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/macOS)
source .venv/bin/activate

# uv run auto-activates — you rarely need manual activation
uv run python -m pytest
```

### 6.11 uv — pyproject.toml Explanation

```toml
[build-system]                        # ← uv works with any PEP 517 backend
requires = ["hatchling>=1.21.0"]      # ← hatchling is a good default
build-backend = "hatchling.build"

[project]                             # ← PEP 621 standard metadata
name = "kfp-ml-library"
version = "1.0.0"
dependencies = [...]                  # ← Runtime deps

[project.optional-dependencies]       # ← Extras (standard PEP 621)
tensorflow = ["tensorflow>=2.15.0"]

[tool.uv]                            # ← uv-specific settings
dev-dependencies = [...]              # ← Dev deps managed by uv

[tool.hatch.build.targets.wheel]     # ← Tell hatchling what to include
packages = ["kfp_ml_library"]
```

### 6.12 uv — Speed Comparison

Typical installation times for this project (~15 packages):

| Tool | Cold install | Warm install (cached) |
|---|---|---|
| pip | ~45s | ~12s |
| Poetry | ~30s | ~8s |
| **uv** | **~3s** | **~0.5s** |

---

## 7. Choosing Between setuptools / Poetry / uv

| Criteria | setuptools | Poetry | uv |
|---|---|---|---|
| **Maturity** | 20+ years | 6+ years | 1+ year |
| **Speed** | Moderate | Fast | **Blazing fast** |
| **Lock file** | No (manual) | Yes | Yes |
| **Dependency resolver** | Basic | Strong | Strong |
| **Virtual env management** | No | Yes | Yes |
| **Python version management** | No | No | Yes |
| **PEP 621 standard** | Partial | No (custom) | **Yes** |
| **PyPI publishing** | Via twine | Built-in | Built-in |
| **Learning curve** | Low | Medium | Low |
| **Ecosystem support** | Universal | Wide | Growing fast |
| **Recommendation** | Legacy / existing projects | Teams wanting full toolchain | **New projects / speed priority** |

### Decision Matrix

- **Choose setuptools** if: you have existing CI/CD built around pip, or need maximum compatibility.
- **Choose Poetry** if: you want a battle-tested all-in-one tool with excellent documentation.
- **Choose uv** if: you want the fastest possible installs, standard PEP 621 compliance, and a modern toolchain.

---

## 8. Publishing to PyPI

### 8.1 TestPyPI First (Recommended)

Always test on TestPyPI before publishing to production PyPI.

**Create accounts:**
1. https://test.pypi.org/account/register/ (TestPyPI)
2. https://pypi.org/account/register/ (PyPI)

**Create API tokens:**
1. TestPyPI: https://test.pypi.org/manage/account/token/
2. PyPI: https://pypi.org/manage/account/token/

### 8.2 Publish with twine (setuptools)

```bash
pip install twine

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Verify it works
pip install --index-url https://test.pypi.org/simple/ kfp-ml-library

# Upload to production PyPI
twine upload dist/*
```

### 8.3 Automated Publishing via GitHub Actions

The provided workflows publish automatically when you push a version tag:

```bash
# Bump version in setup.py / pyproject.toml
# Then:
git add .
git commit -m "chore: bump version to 1.1.0"
git tag v1.1.0
git push origin main --tags
```

The workflow will:
1. Run linters
2. Run tests across Python 3.9, 3.10, 3.11
3. Build wheel + sdist
4. Upload to PyPI
5. Create a GitHub Release with the wheel attached

### 8.4 Version Bumping Workflow

```bash
# 1. Update version in source files:
#    - kfp_ml_library/__init__.py  →  __version__ = "1.1.0"
#    - setup.py                    →  version="1.1.0"
#    - pyproject.toml              →  version = "1.1.0"
#
# 2. Commit and tag
git add .
git commit -m "chore: release v1.1.0"
git tag -a v1.1.0 -m "Release v1.1.0 — <description>"
git push origin main --tags
```

---

## 9. Troubleshooting

### 9.1 Common Build Errors

**Error: `ModuleNotFoundError: No module named 'setuptools'`**

```bash
pip install --upgrade setuptools wheel build
```

**Error: `Invalid build-backend`**

Make sure `pyproject.toml` has:
```toml
build-backend = "setuptools.build_meta"    # ← NOT setuptools.backends._legacy:_Backend
```

**Error: `No module named 'build'`**

```bash
pip install build
```

**Error: `error: package directory 'kfp_ml_library' does not exist`**

Ensure you're running the build command from the project root directory.

### 9.2 Common Poetry Errors

**Error: `Poetry could not find a pyproject.toml file`**

Ensure you've copied `pyproject.poetry.toml` to `pyproject.toml`.

**Error: `The current project's Python requirement (>=3.9,<3.13) is not compatible with...`**

```bash
poetry env use python3.11    # Switch to a compatible Python
```

**Error: `SolverProblemError`**

```bash
poetry cache clear --all pypi
poetry lock --no-update
```

### 9.3 Common uv Errors

**Error: `No Python interpreters found`**

```bash
uv python install 3.11
```

**Error: `No solution found when resolving dependencies`**

```bash
uv lock --upgrade          # Allow all deps to float
```

**Error: `Failed to build kfp-ml-library`**

Ensure `[tool.hatch.build.targets.wheel]` is correctly configured.

### 9.4 GitHub Actions Debugging

**Workflow not triggering:**
- Check file is in `.github/workflows/` (exact path)
- Check YAML syntax: `yamllint .github/workflows/build-wheel.yml`
- Ensure branch name matches (`main` vs `master`)

**Tests passing locally but failing in CI:**
```bash
# Reproduce the CI environment locally with Docker
docker run --rm -v ${PWD}:/app -w /app python:3.10-slim bash -c "
  pip install -e '.[dev]' && pytest tests/ -v
"
```

**Publish step failing:**
- Verify `PYPI_API_TOKEN` is set in repo Secrets
- Ensure the version hasn't already been published (PyPI won't accept duplicate versions)

---

## Appendix A — Complete File Listing

After setup, your project should look like this:

```
kfp-ml-library/
├── .github/
│   └── workflows/
│       ├── build-wheel.yml            ← setuptools CI/CD
│       ├── build-wheel-poetry.yml     ← Poetry CI/CD
│       └── build-wheel-uv.yml        ← uv CI/CD
├── .gitignore
├── kfp_ml_library/
│   ├── __init__.py
│   ├── components/
│   ├── configs/
│   ├── frameworks/
│   ├── pipelines/
│   └── utils/
├── tests/
├── dist/                              ← Built wheel (gitignored)
├── Dockerfile
├── README.md
├── requirements.txt
├── setup.py                           ← setuptools config
├── pyproject.toml                     ← Active config (choose one)
├── pyproject.poetry.toml              ← Poetry variant (reference)
├── pyproject.uv.toml                  ← uv variant (reference)
├── use.md                             ← Usage documentation
└── BUILDING.md                        ← This file
```

## Appendix B — Complete Workflow: From Zero to Published

```bash
# ===== 1. Clone / set up =====
git clone https://github.com/YOUR-USERNAME/kfp-ml-library.git
cd kfp-ml-library

# ===== 2. Choose your tool =====
# Option A: pip + setuptools (already set up)
pip install -e ".[dev]"

# Option B: Poetry
copy pyproject.poetry.toml pyproject.toml
poetry install --with dev

# Option C: uv
copy pyproject.uv.toml pyproject.toml
uv sync --all-extras

# ===== 3. Make changes =====
# Edit source code in kfp_ml_library/

# ===== 4. Lint and test =====
ruff check kfp_ml_library/    # or: poetry run / uv run
black kfp_ml_library/
pytest tests/ -v

# ===== 5. Bump version =====
# Edit __init__.py, setup.py (or pyproject.toml)

# ===== 6. Build =====
python -m build     # or: poetry build / uv build

# ===== 7. Test install =====
pip install dist/kfp_ml_library-*.whl

# ===== 8. Commit and tag =====
git add .
git commit -m "feat: release v1.1.0"
git tag v1.1.0
git push origin main --tags

# ===== 9. GitHub Actions takes over =====
# → lint → test → build → publish to PyPI → GitHub Release
```

---

*Generated for kfp-ml-library v1.0.0*
