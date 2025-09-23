# 🚀 UV Cheat Sheet

**UV** is an extremely fast Python package and project manager, written in Rust.
It manages Python projects, dependencies, and virtual environments.

🔗 Official docs: [docs.astral.sh/uv](https://docs.astral.sh/uv/)

______________________________________________________________________

## 📦 Core Concepts

- **`uv init`** → Start a new Python project (`pyproject.toml` is created).
- **Dependencies** → Declared in `pyproject.toml`, locked in `uv.lock`.
- **Virtual environments** → Automatically created and managed.
- **Commands** → Always run inside the correct environment and Python version.

______________________________________________________________________

## 🛠️ Common Commands

### Project Setup

```sh
uv init                      # create a new project
uv python install 3.12       # install/manage Python versions
```

### Dependencies

```sh
uv add fastapi "pydantic>=2" # add dependencies (with specifiers)
uv remove <package>          # remove a dependency
uv sync                      # sync env with lockfile (uv.lock)
uv lock                      # (re)generate lockfile
uv tree                      # show dependency tree
uv export -o requirements.txt # export lockfile to requirements.txt
```

### Running Code

```sh
uv run <command>             # run inside project environment
uv run python main.py        # run a script with correct env
```

______________________________________________________________________

## ⚡ Workflow Example

```sh
uv init myproject            # create new project
cd myproject
uv add fastapi "pydantic>=2" # add dependencies
uv run python main.py        # run your app
uv lock                      # lock dependencies
uv sync                      # sync environment
```

______________________________________________________________________

## 🔍 Help & Exploration

```sh
uv --help        # see all available commands
uv <command> -h  # detailed help for a command
```

______________________________________________________________________

✅ **Pro tip**: `uv run` will *auto-install missing dependencies* before executing.
