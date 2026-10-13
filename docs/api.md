# API reference

Most classes and functions can be imported from the top-level package, for
example `from loopuss import LoopUS`. The exceptions are the names below that
carry a submodule prefix: `plotting`, `analysis`, `reports`, `spice` and the
internals.

## Circuit

```{eval-rst}
.. currentmodule:: loopuss

.. autosummary::
   :toctree: generated
   :nosignatures:

   LoopUS
```

## Memory arrays and devices

```{eval-rst}
.. currentmodule:: loopuss

.. autosummary::
   :toctree: generated
   :nosignatures:

   CrossBar
   MemoryModel
   TechModel
```

## Peripherals

```{eval-rst}
.. currentmodule:: loopuss

.. autosummary::
   :toctree: generated
   :nosignatures:

   PeriphCirc
   ConverterModel
   OAModel
   ADCModel
   VoltageDAC
   CurrentDAC
   MemWriteModel
   DSPModel
```

## Signals

```{eval-rst}
.. currentmodule:: loopuss

.. autosummary::
   :toctree: generated
   :nosignatures:

   Signal
   VoltageSignal
   CurrentSignal
```

## Reports and baselines

```{eval-rst}
.. currentmodule:: loopuss

.. autosummary::
   :toctree: generated
   :nosignatures:

   BenchReport
   TranReport
   NoiseReport
   reports.sum_reports
   reports.mean_reports
   GPUModel
```

## Problem mappers

```{eval-rst}
.. currentmodule:: loopuss

.. autosummary::
   :toctree: generated
   :nosignatures:

   linreg
   ridgereg
   lassoreg
   eigdecomp
   posneg
```

## Analysis and plotting

```{eval-rst}
.. currentmodule:: loopuss

.. autosummary::
   :toctree: generated
   :nosignatures:

   relerr
   analysis.natsortpoles
   plotting.tranplot
   plotting.corrplot
   plotting.matplot
   plotting.reimplot
   plotting.psdplot
```

## SPICE backend

```{eval-rst}
.. currentmodule:: loopuss

.. autosummary::
   :toctree: generated
   :nosignatures:

   spice.SpiceAnalysis
   spice.spice_path
   spice.spice_available
   spice.run_spice
   spice.raw_parse
   spice.read_log_elapsed
```

## Internals

These are called by {class}`~loopuss.LoopUS`; they are listed here for
contributors.

```{eval-rst}
.. currentmodule:: loopuss

.. autosummary::
   :toctree: generated
   :nosignatures:

   solver.nlop_model
   solver.nltran_model
   noise.noise_model
   noise.noise_spice
   netlist.build_netlist
   utils.GroupedDisplay
   utils.si_format
```
