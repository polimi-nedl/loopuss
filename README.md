[![Banner](docs/sfx/logo_white_small.png)](https://github.com/polimi-nedl/loopuss)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-8-orange.svg?style=flat-square)](#contributors) ![Version-shield](https://img.shields.io/badge/version-v1.0.0-blue) [![License-shield](https://img.shields.io/badge/license-AGPLv3-yellow.svg)](LICENSE)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

> Overview & full documentation available on https://github.com/polimi-nedl/loopuss

**loopus²** is a Python framework for the behavioral simulation and architectural benchmarking of open- and closed-loop analog in-memory computing circuits. It provides compact models of memory arrays, operational amplifiers, data converters, programming circuits, and other system components, together with interchangeable analytical and SPICE simulation backends.

The framework supports:
- **Analytical circuit simulation**, including fast operating-point, transient, and noise analyses based on compact behavioral models.
- **Nonlinear feedback simulation**, including supply-limited operational amplifiers handled through operating-region detection and local linearization.
- **Crossbar nonideality modeling**, including line resistance, finite conductance resolution, and stochastic programming variability.
- **Automated LTspice simulation**, including netlist generation, parallel execution, and direct parsing of simulation results.
- **Architectural benchmarking** of area, latency, energy, power, and computational performance for memory arrays, peripheral circuits, and complete analog computing systems.
- **Mapping and post-processing utilities** for constructing common linear-algebra and analog computing workloads.

---

## Installation

The package requires Python 3.8+. LTspice XVII or later is required only for SPICE-based analyses.

1. Clone the repository:
   ```bash
   git clone https://github.com/polimi-nedl/loopuss
   cd loopuss
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows Powershell:
   .venv\Scripts\activate
   # Windows Command Prompt:
   .venv\Scripts\activate.bat
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. Install the package in editable mode with development dependencies (for testing and linting):
   ```bash
   pip install -e ".[dev]"
   ```

4. **LTSpice Configuration**:
   The framework attempts to resolve the LTSpice path automatically. If LTSpice is not installed at the default location, set the environment variable `LOOPUSS_SPICE_PATH` or configure it in `src/loopuss/config.json`:
   ```json
   {
       "spicePath": "C:\\Program Files\\LTC\\LTspiceXVII\\XVII.exe"
   }
   ```

---

## Usage Example

```python
import numpy as np
from loopuss import CrossBar, LoopUS, VoltageSignal, OAModel, ADCModel

# Parameters
N = 10
G0 = 1.0 / 10e3
V0 = 1.0

# 1. Setup crossbar conductances
A = np.random.rand(N, N)
Gin = CrossBar(np.empty((N, 0)))  # Input array
Gf  = CrossBar(A * G0)            # Feedback array

# 2. Instantiate LoopUS circuit with peripheral models
circ = LoopUS(
    Gin, Gf,
    OAModel=OAModel('1P'),
    ADCModel=ADCModel('Fan19'),
    benchEnable=True,
    verbose=True
)

# 3. Apply inputs and run transient simulation
y = np.random.rand(N)
circ.set_inputs(VoltageSignal.ytov(y, V0))

time = np.concatenate([[0.0], np.logspace(-10, -5, 1000)])
Vout, t = circ.tran("model", time)

# Get benchmark report
report = circ.benchmark()
print(f"Total Latency: {report.latency['Total'] * 1e6:.3f} us")
print(f"Total Energy: {report.energy['Total'] * 1e9:.3f} nJ")
```

---

## Running Tests

Tests are managed using `pytest`. Run the entire suite with:
```bash
pytest
```

---

## Code Formatting & Linting

We use `ruff` to enforce formatting and import sorting rules (configured in `pyproject.toml`):
```bash
# Format codebase
ruff format .

# Check lint rules
ruff check .
```

---

## Project Information

### Funding
This project has received funding from the European Research Council through the ANIMATE project (grant agreement n° 101054098). 

### Citation
The reference publication describing loopus² is forthcoming. Until a formal publication citation becomes available, please cite the software:

> P. Mannocci et al., *loopus²*, version 1.0.0, 2026. Available: https://github.com/polimi-nedl/loopuss

### Development and Maintenance
loopus² was initiated by [Piergiulio Mannocci](mailto:piergiulio.mannocci@polimi.it) and is currently maintained by [Piergiulio Mannocci](mailto:piergiulio.mannocci@polimi.it) and [Eugenio Montinaro](mailto:eugenio.montinaro@mail.polimi.it).
See the [Contributors](#contributors) section for the complete list of developers and contributors.

### Contact
For administrative or project-related questions, please contact [Prof. Daniele Ielmini](mailto:daniele.ielmini@polimi.it).
For technical questions, bug reports, or feature requests, please contact [Piergiulio Mannocci](mailto:piergiulio.mannocci@polimi.it) or [Eugenio Montinaro](mailto:eugenio.montinaro@mail.polimi.it).

## Contributors
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table align="center">
  <tr>
    <td align="center"><a href="https://github.com/jackphen"><img src="https://avatars.githubusercontent.com/u/22836767?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Piergiulio Mannocci</b></sub></a><br /><a href="https://github.com/polimi-nedl/loopuss/commits?author=jackphen" title="Code">💻</a> <a href="https://github.com/polimi-nedl/loopuss/issues?q=author%3Ajackphen" title="Bug reports">🐛</a> <a href="https://github.com/polimi-nedl/loopuss/commits?author=jackphen" title="Documentation">📖</a> <a href="https://github.com/polimi-nedl/loopuss/commits?author=jackphen" title="User Testing">📓</a> <a href="https://github.com/polimi-nedl/loopuss/commits?author=jackphen" title="Tutorials">✅</a></td>
    <td align="center"><a href="https://github.com/eumonti"><img src="https://avatars.githubusercontent.com/u/62185102?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Eugenio Montinaro</b></sub></a><br /><a href="https://github.com/polimi-nedl/loopuss/commits?author=eumonti" title="Code">💻</a> <a href="https://github.com/polimi-nedl/loopuss/issues?q=author%3Aeumonti" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/OldHenry96"><img src="https://avatars.githubusercontent.com/u/61509046?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Enrico Melacarne</b></sub></a><br /><a href="https://github.com/polimi-nedl/loopuss/commits?author=OldHenry96" title="Code">💻</a> <a href="https://github.com/polimi-nedl/loopuss/issues?q=author%3AOldHenry96" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/Giuseppe-Falcone"><img src="https://avatars.githubusercontent.com/u/101051093?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Giuseppe Falcone</b></sub></a><br /><a href="https://github.com/polimi-nedl/loopuss/commits?author=Giuseppe-Falcone" title="Tests">⚠️</a></td>
   </tr>
   <tr>
     <td align="center"><a href="https://github.com/andreapezzoli"><img src="https://avatars.githubusercontent.com/u/100782614?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Andrea Pezzoli</b></sub></a><br /><a href="https://github.com/polimi-nedl/loopuss/commits?author=andreapezzoli" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/irene-andreoli"><img src="https://avatars.githubusercontent.com/u/161730429?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Irene Andreoli</b></sub></a><br /><a href="https://github.com/polimi-nedl/loopuss/commits?author=irene-andreoli" title="Code">💻</a> <a href="https://github.com/polimi-nedl/loopuss/commits?author=irene-andreoli" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/Carlo-Zucchelli"><img src="https://avatars.githubusercontent.com/u/159053517?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Carlo Zucchelli</b></sub></a><br /><a href="https://github.com/polimi-nedl/loopuss/commits?author=Carlo-Zucchelli" title="Code">💻</a> <a href="https://github.com/polimi-nedl/loopuss/commits?author=Carlo-Zucchelli" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/ElisabettaGiannone"><img src="https://avatars.githubusercontent.com/u/73816646?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Elisabetta Giannone</b></sub></a><br /><a href="https://github.com/polimi-nedl/loopuss/commits?author=ElisabettaGiannone" title="Code">💻</a> <a href="https://github.com/polimi-nedl/loopuss/issues?q=author%3AElisabettaGiannone" title="Bug reports">🐛</a></td>
  </tr>
</table>
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
