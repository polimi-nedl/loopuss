# Installation

## Requirements

- Python 3.10 or later
- NumPy ≥ 2.0, SciPy ≥ 1.11 and Matplotlib ≥ 3.8 (installed automatically)
- [LTspice](https://www.analog.com/en/resources/design-tools-and-calculators/ltspice-simulator.html)
  XVII or later, needed only for the SPICE solvers

## Install

### Using loopus²

Install the package from GitHub into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install git+https://github.com/polimi-nedl/loopuss
```

### Developing loopus²

To modify the code, run the tests or build the documentation, clone the
repository and install it in editable mode with the development tools (pytest,
ruff):

```bash
git clone https://github.com/polimi-nedl/loopuss
cd loopuss
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Configuring LTspice

loopus² looks for the LTspice executable in this order:

1. the `LOOPUSS_SPICE_PATH` environment variable;
2. a `config.json` file in the installed `loopuss` package directory
   (`src/loopuss/` in a clone), containing `{"spicePath": "<path to the executable>"}`;
3. the default install locations: on Windows
   `C:\Program Files\ADI\LTspice\LTspice.exe` and the LTspice XVII paths, on
   macOS `/Applications/LTspice.app/Contents/MacOS/LTspice`.

On macOS the native LTspice app runs directly; if it is installed elsewhere,
point `LOOPUSS_SPICE_PATH` at the `LTspice` binary inside the app bundle
(`<app>/Contents/MacOS/LTspice`).

On Linux, LTspice runs through Wine, so point `LOOPUSS_SPICE_PATH` at the Windows
executable. Headless sessions (no `DISPLAY`) also need Xvfb. To check the setup:

```python
from loopuss.spice import spice_available, spice_path

spice_available()  # True if an executable was found
spice_path()  # its path
```

## Running the tests

From a development install:

```bash
pytest
```

Tests that need LTspice skip themselves when it is not found. Check the skipped
count in the summary before relying on SPICE results.

## Linting

Formatting and lint rules are configured in `pyproject.toml` and checked with
ruff:

```bash
ruff format .
ruff check .
```

## Building this documentation

From a development install:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html`.
