# loopus²

**loopus²** is a Python framework for the behavioral simulation and
architectural benchmarking of open- and closed-loop analog in-memory computing
circuits.

It models crossbar arrays of resistive memories, including line resistance,
finite precision and programming variability. It also models the circuitry
around them: op-amps with finite gain, bandwidth and supply rails, data
converters, programming peripherals and digital post-processing. The same
circuit description runs on two backends:

- **`model`**: analytical operating-point, transient, pole-zero and noise
  solvers that account for op-amp saturation;
- **`SPICE/fast`, `SPICE/accurate`, `SPICE/extreme`**: LTspice netlists that
  are generated, run and parsed automatically.

With benchmarking enabled, every simulation also accumulates latency, energy,
power and area, which can be compared against digital baselines.

```python
import numpy as np
from loopuss import CrossBar, CurrentSignal, LoopUS

N, G0 = 10, 1e-4
A = np.random.rand(N, N)
A = A.T @ A + N * np.eye(N)
A /= A.max()

# Feedback array A, no input array: the circuit solves A x = -y.
circ = LoopUS(CrossBar(np.empty((N, 0))), CrossBar(A * G0))
y = np.random.rand(N)
circ.set_inputs(CurrentSignal.ytoi(y, I0=G0))
Vout, t = circ.tran("model", 1e-4)

print(np.linalg.norm(Vout[:, -1] + np.linalg.solve(A, y)))
```

```{toctree}
:maxdepth: 2
:hidden:

installation
usage
examples/index
presets
api
license
```

## Citing

A reference publication is forthcoming. Until then, please cite the software
itself; the repository's `CITATION.cff` has the metadata:

> P. Mannocci et al., *loopus²*, 2026. <https://github.com/polimi-nedl/loopuss>

## Funding

This project has received funding from the European Research Council through
the ANIMATE project (grant agreement n° 101054098).
