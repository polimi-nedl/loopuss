# User guide

## The circuit

A {class}`~loopuss.LoopUS` circuit has `n` op-amps. Row `i` of the `n × n`
*feedback* array and row `i` of the `n × p` *input* array meet at a junction wired
to one input of op-amp `i`, whose output drives column `i` of the feedback
array. Inputs are voltages on the input-array columns and currents injected at
the junctions. With ideal op-amps and ideal lines, the circuit settles on

$$ G_f \, V_{out} = -\left(I_{in} + G_{in} V_{in}\right). $$

Both arrays are {class}`~loopuss.CrossBar` objects built from a conductance
matrix:

```python
import numpy as np
from loopuss import CrossBar, LoopUS

N, G0 = 10, 1e-4  # unit conductance: 10 kΩ
A = np.random.rand(N, N)
A = A.T @ A + N * np.eye(N)
A /= A.max()

Gin = CrossBar(np.empty((N, 0)))  # no input columns: current inputs only
Gf = CrossBar(A * G0)
circ = LoopUS(Gin, Gf)
```

`LoopUS` also takes peripheral models (`oa_model`, `adc_model`, `voltage_dac`,
`current_dac`, `mem_write_model`, `dsp_model`) and wiring options (`sign_vector`,
`output_mask`, `input_mask_i`, `input_mask_v`). All indices are 0-based. See
{class}`~loopuss.LoopUS` for the full list.

A `CrossBar` models non-idealities too: `line_resistance` adds IR drop,
`memory_model` clamps and quantizes the conductances to a device (see
{doc}`presets`), and a {class}`~loopuss.MemWriteModel` with `SigmaG > 0` adds
programming variability.

## Inputs

Inputs are {class}`~loopuss.VoltageSignal` or {class}`~loopuss.CurrentSignal`
waveforms: DC, STEP, PULSE, SINE or PWL. {meth}`CurrentSignal.ytoi
<loopuss.CurrentSignal.ytoi>` and {meth}`VoltageSignal.ytov
<loopuss.VoltageSignal.ytov>` turn a vector into one source per channel, and
{meth}`~loopuss.LoopUS.set_inputs` loads them into the DACs:

```python
from loopuss import CurrentSignal

y = np.random.rand(N)
circ.set_inputs(CurrentSignal.ytoi(y, I0=G0))  # steps from 0 to y * I0 at t = 0
```

Current signals go to the rows in `input_mask_i`, and voltage signals to the
input-array columns in `input_mask_v`. Each signal is delayed by its DAC's
latency.

## Solvers

Every analysis takes the solver name as its first argument:

| Solver | Backend |
|---|---|
| `"model"` | analytical solvers |
| `"SPICE/fast"` (or `"spice"`) | LTspice, `reltol = 1e-8` |
| `"SPICE/accurate"` | LTspice, `reltol = 1e-11` |
| `"SPICE/extreme"` | LTspice, `reltol = 1e-13` |

The SPICE solvers need LTspice (see {doc}`installation`). Their netlists and
`.raw` files go to `simfolder/<pid>/` under the working directory, which is
deleted when the process exits. Pass `file_path` to `LoopUS` to keep them.

## Operating point

```python
from loopuss import relerr
from loopuss.plotting import corrplot

Vout, _ = circ.op("model")
ref = -np.linalg.solve(A, y)
print(relerr(Vout, ref))
corrplot(Vout, ref)
```

## Transient analysis

```python
from loopuss.plotting import tranplot

time = np.concatenate([[0.0], np.logspace(-10, -4, 1000)])
Vout, t = circ.tran("model", time)
tranplot(t, Vout)
```

The outputs pass through the ADC and DSP models, and `t` is shifted by the ADC
latency. With `"SPICE/fast"` the same call simulates from `time[0]` to
`time[-1]` and returns LTspice's time grid; `tstep=` and `dtmax=` control its
timestep. `return_report=True` adds a {class}`~loopuss.TranReport` with the
voltages and currents of every op-amp, plus the dissipated power and energy.

## Non-linear circuits

Op-amps with finite `Vsupply` rails saturate, and the model solvers track which
op-amps are saturated. Pass a list to use a different op-amp on each row. For
example, the LASSO circuit of {func}`~loopuss.lassoreg` needs clamping op-amps
on its last rows:

```python
from loopuss import OAModel, lassoreg

m, n = 10, 5
L, S = lassoreg(1.0, 0.1, np.random.rand(m, n))
oa = [OAModel("GM") for _ in range(m + n)]
oa += [OAModel("1P", Vsupply=[-0.1, 0.1]) for _ in range(n)]
lasso = LoopUS(None, CrossBar(L * G0), sign_vector=S, oa_model=oa)
```

## Stability and accuracy

- {meth}`~loopuss.LoopUS.pz` returns the poles of the linearized circuit, and
  {func}`~loopuss.plotting.reimplot` draws them.
- {meth}`~loopuss.LoopUS.tts` estimates the time-to-solution from the slowest
  pole.
- {meth}`~loopuss.LoopUS.errbound` gives the worst-case relative error caused by
  finite op-amp gain, and the input that produces it.

```python
poles, *_ = circ.pz()
tts, tts_min, tts_max = circ.tts()
maxerr, worst_input, *_ = circ.errbound()
```

## Noise analysis

Enable the thermal noise of the cells with `noiseless=False`, and give the
op-amps noise densities:

```python
noisy = LoopUS(
    Gin,
    CrossBar(A * G0, noiseless=False),
    oa_model=OAModel("1P", noise_voltage=1e-9, noise_current=[1e-12, 1e-12]),
)
Sout, freq, report = noisy.noise("model", 100e9)  # sweep up to 100 GHz
```

Unlike the other analyses, `noise` defaults to `"SPICE/fast"`. Its remaining
arguments are `Fmax[, Fmin, Fstep, Fsweep]` or a
{class}`~loopuss.spice.SpiceAnalysis`. {func}`~loopuss.plotting.psdplot` plots
`Sout`.

## Benchmarking

With `bench_enable=True`, every `tran` and `reprogram` call adds to the
circuit's latency, energy, power and operation count.
{meth}`~loopuss.LoopUS.benchmark` returns them as a
{class}`~loopuss.BenchReport`. Meaningful numbers need realistic peripherals,
which the {doc}`presets` provide:

```python
from loopuss import ADCModel, CurrentDAC, MemoryModel, MemWriteModel

rram = MemoryModel("Woo16")
bench = LoopUS(
    None,
    CrossBar(A * rram.LRS, memory_model=rram),
    oa_model=OAModel("Feinberg21", Vsupply=[-1, 1]),
    adc_model=ADCModel("Fan19"),
    current_dac=CurrentDAC("Ishraqul17"),
    mem_write_model=MemWriteModel("IGVVA10"),
    bench_enable=True,
)
bench.set_inputs(CurrentSignal.ytoi(y, I0=rram.LRS))
bench.tran("model", np.concatenate([[0.0], np.logspace(-10, -7, 1000)]))

report = bench.benchmark()
print(report.throughput, report.energy_efficiency)
report.bdplot()
```

Reports can be combined arithmetically. `A + B` runs `B` after `A`
({meth}`~loopuss.BenchReport.combine`), and `k * A` runs `k` copies in
parallel ({meth}`~loopuss.BenchReport.scale`).
`BenchReport.barplot([A, B], labels=["RRAM", "SRAM"])` compares several
reports. {meth}`GPUModel.benchmark <loopuss.GPUModel.benchmark>` produces a
report for a digital baseline running the same workload.

## Mapping problems onto circuits

The mappers ({func}`~loopuss.linreg`, {func}`~loopuss.ridgereg`,
{func}`~loopuss.lassoreg`, {func}`~loopuss.eigdecomp`) build a block matrix `M`
and a sign vector `S`. The circuit settles on `M v = -i`. Matrices with negative
entries go through {func}`~loopuss.posneg`, which splits them into non-negative
conductances plus inverting buffer rows:

```python
from loopuss import posneg, ridgereg

m, n, lam = 10, 4, 1.0
X = np.random.rand(m, n)
M, S = ridgereg(np.sqrt(lam), np.sqrt(lam), X)
M, S, *_ = posneg(M, 1.0, sign_vector=S)

ridge = LoopUS(None, CrossBar(M * G0), sign_vector=S, output_mask=m + np.arange(n))
ridge.set_inputs(CurrentSignal.ytoi(np.random.rand(m), I0=G0))
x, _ = ridge.tran("model", time)  # x[:, -1] ≈ -(XᵀX + λI)⁻¹ Xᵀ y
```

The {doc}`examples/index` apply these steps to complete problems.
