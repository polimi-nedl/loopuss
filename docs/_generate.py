"""Pages generated at build time from the code they describe.

``examples/`` gets one page per script in the repository's ``examples/``
folder, and ``presets.md`` tabulates every model preset by instantiating it, so
neither can drift from the package.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np

from loopuss import (
    ADCModel,
    CurrentDAC,
    GPUModel,
    MemoryModel,
    MemWriteModel,
    OAModel,
    TechModel,
    VoltageDAC,
    adc_model,
    dac,
    gpu_model,
    mem_write_model,
    memory_model,
    oa_model,
    tech_model,
)


def write_pages(docs: Path) -> None:
    _write_examples(docs)
    _write_presets(docs)


def _write(path: Path, text: str) -> None:
    # Rewriting unchanged files would make Sphinx re-read them on every build.
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


# --- examples ---
_EXAMPLE_PAGE = """\
# {title}

```{{eval-rst}}
.. currentmodule:: loopuss

{body}
```

Run it from the repository root with `python examples/{stem}.py`.

```{{literalinclude}} ../../examples/{stem}.py
:language: python
:lines: {first_line}-
```
"""

_EXAMPLE_INDEX = """\
# Examples

Each page shows one script from the repository's `examples/` folder. The
scripts open matplotlib figures and use the analytical `model` solver, with two
exceptions: `linear_system_PD_positive_IRdrop.py` also runs LTspice and needs
it installed, and `noise_analysis.py` adds an LTspice run when LTspice is
available.

```{{toctree}}
:maxdepth: 1

{entries}
```
"""


def _write_examples(docs: Path) -> None:
    out = docs / "examples"
    out.mkdir(exist_ok=True)
    stems = []
    for script in sorted((docs.parent / "examples").glob("*.py")):
        tree = ast.parse(script.read_text(encoding="utf-8"))
        doc = ast.get_docstring(tree) or script.stem
        title, _, body = doc.partition("\n")
        # Show the code without repeating the docstring rendered above it.
        first_line = tree.body[0].end_lineno + 1 if ast.get_docstring(tree) else 1
        page = _EXAMPLE_PAGE.format(
            title=title.rstrip("."),
            body=body.strip(),
            stem=script.stem,
            first_line=first_line,
        )
        _write(out / f"{script.stem}.md", page)
        stems.append(script.stem)
    _write(out / "index.md", _EXAMPLE_INDEX.format(entries="\n".join(stems)))


# --- presets ---
_PREFIXES = {-18: "a", -15: "f", -12: "p", -9: "n", -6: "µ", -3: "m", 0: "", 3: "k", 6: "M"}
_PREFIXES.update({9: "G", 12: "T"})


def _si(value, unit: str = "") -> str:
    """Engineering notation; ``[lo, hi]`` pairs render as ranges."""
    arr = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    if arr.size == 0 or np.all(np.isnan(arr)):
        return "–"
    if arr.size == 2:
        lo, hi = arr
        if np.isinf(lo) and np.isinf(hi):
            return "–"
        if lo == -hi:
            return "±" + _si(hi, unit)
        return f"{_si(lo, unit)} … {_si(hi, unit)}"
    if arr.size > 2:
        return ", ".join(_si(v, unit) for v in arr)
    v = float(arr[0])
    if math.isinf(v):
        return "∞" if v > 0 else "−∞"
    if v == 0:
        return f"0 {unit}".strip()
    exp = min(12, max(-18, math.floor(math.log10(abs(v)) / 3) * 3))
    return f"{v / 10**exp:.4g} {_PREFIXES[exp]}{unit}".strip()


def _area(a: float) -> str:
    if a == 0:
        return "0"
    return f"{a / 1e-6:.4g} mm²" if a >= 1e-9 else f"{a / 1e-12:.4g} µm²"


def _write_energy(o) -> str:
    # Some presets charge energy per bit of cell precision.
    de = o.dynamic_energy
    return f"{_si(de(1), 'J')}/bit" if callable(de) else _si(de, "J")


_NODE = ("Node", lambda o: _si(o.technology.node, "m"))
_SUPPLY = ("Supply", lambda o: _si(o.Vsupply, "V"))
_LEAKAGE = ("Leakage", lambda o: _si(o.leakage_power, "W"))
_AREA = ("Area", lambda o: _area(o.area))
_CONVERTER = [
    ("Bits", lambda o: f"{o.Nbit:g}"),
    ("Channels", lambda o: str(o.Nchannel)),
    ("Latency", lambda o: _si(o.latency, "s")),
    ("Dyn. energy", lambda o: _si(o.dynamic_energy, "J")),
    _LEAKAGE,
    _SUPPLY,
    _AREA,
    _NODE,
]

# DOIs of the publications the presets are taken from.
_REFS = {
    "OAModel": {
        "Valero12": "10.1109/ISCAS.2012.6271661",
        "Gupta16": "10.1109/ICEEOT.2016.7755591",
        "Kouhalvandi17": "10.1109/UBMK.2017.8093422",
        "Nidhi19": "10.1007/978-981-13-0776-8_43",
        "Parthipan19": "10.1109/ICCMC.2019.8819641",
        "Rahaman19": "10.1109/JEDS.2019.2923208",
        "AlQaysi20": "10.11591/IJEECS.v20.i2.pp670-679",
        "Padma20": "10.1109/VDAT50263.2020.9190288",
        "Feinberg21": "10.1109/HPCA51647.2021.00069",
        "Sancandi21": "10.1038/s41928-025-01549-1",
    },
    "ADCModel": {
        "Sandner05": "10.1109/JSSC.2005.847215",
        "Pang09": "10.1109/ASSCC.2009.5357200",
        "Ahmadi13": "10.1109/CICC.2013.6658425",
        "Brenna14": "10.1007/s10470-014-0384-z",
        "Kadetotad15": "10.1109/JETCAS.2015.2426495",
        "Wang18": "10.1109/ISCAS.2018.8351064",
        "Shi18": "10.1016/j.vlsi.2018.03.010",
        "Fan19": "10.1109/LSSC.2019.2934351",
        "Zhang21": "10.1109/ISNE48910.2021.9493612",
        "Tang21": "10.1109/TNS.2021.3096162",
        "Bermak21": "10.3390/electronics10222856",
        "Sancandi21": "10.1038/s41928-025-01549-1",
    },
    "VoltageDAC": {
        "Reaz19": "10.1109/JEEIT.2019.8717383",
        "Sancandi21": "10.1038/s41928-025-01549-1",
    },
    "CurrentDAC": {
        "Tseng14": "10.1109/VLSIC.2014.6858435",
        "Ishraqul17": "10.1109/ICRTEECT.2017.30",
        "Mao18": "10.1109/TVLSI.2018.2791462",
        "Bandali18": "10.1142/S021812661950172X",
        "Chang18": "10.1587/elex.15.20180509",
        "Kim19": "10.1109/TCSI.2019.2913174",
        "Kommangunta21": "10.1109/ICEIC51217.2021.9369809",
        "Kumar22": "10.1016/j.jestch.2021.06.015",
        "Sancandi21_1X": "10.1038/s41928-025-01549-1",
    },
    "MemoryModel": {
        "RRAM": "10.1063/5.0136403",
        "STT-MRAM": "10.1063/5.0136403",
        "PCM": "10.1063/5.0136403",
        "FEFET": "10.1063/5.0136403",
        "Jo10": "10.1021/nl904092h",
        "Park13": "10.1109/IEDM.2013.6724692",
        "Gao15": "10.1088/0957-4484/26/45/455204",
        "Woo16": "10.1109/LED.2016.2582859",
        "Kuzum11": "10.1021/nl201040y",
        "Jerry17": "10.1109/IEDM.2017.8268338",
        "Oh17": "10.1109/LED.2017.2698083",
        "SRAM": "10.3390/mi13081332",
        "Sancandi21": "10.1038/s41928-025-01549-1",
    },
    "MemWriteModel": {
        "SRAM": "10.3390/mi13081332",
        "Sancandi21": "10.1038/s41928-025-01549-1",
        "Gul22_6T": "10.3390/mi13081332",
        "Zahoor20_SRAM": "10.1109/TED.2021.3089995",
        "Zahoor20_RRAM": "10.1109/TED.2021.3089995",
        "IGVVA10": "10.1109/TED.2021.3089995",
    },
}

_OA_TOPOLOGIES = ("1P", "GM", "2P", "2P1Z")

# (heading, class, preset names, columns)
_SECTIONS = [
    (
        "Operational amplifiers",
        OAModel,
        [k for k in oa_model._PRESETS if k not in _OA_TOPOLOGIES],
        [
            ("A0", lambda o: f"{20 * math.log10(o.A0):.1f} dB"),
            ("GBWP", lambda o: _si(o.GBWP, "Hz")),
            ("Vos", lambda o: _si(o.Vos, "V")),
            ("Rout", lambda o: _si(o.Rout, "Ω")),
            ("Iout", lambda o: _si(o.Iout, "A")),
            _SUPPLY,
            _LEAKAGE,
            _AREA,
            _NODE,
        ],
    ),
    ("ADCs", ADCModel, list(adc_model._PRESETS), _CONVERTER),
    (
        "Voltage DACs",
        VoltageDAC,
        list(dac._VDAC_PRESETS),
        [*_CONVERTER, ("Vref", lambda o: _si(o.Vref, "V"))],
    ),
    (
        "Current DACs",
        CurrentDAC,
        list(dac._CDAC_PRESETS),
        [*_CONVERTER, ("Iref", lambda o: _si(o.Iref, "A"))],
    ),
    (
        "Memory devices",
        MemoryModel,
        list(memory_model._PRESETS),
        [
            ("Type", lambda o: o.type),
            ("LRS", lambda o: _si(o.LRS, "S")),
            ("HRS", lambda o: _si(o.HRS, "S")),
            ("ZRS", lambda o: _si(o.ZRS, "S")),
            ("Window", lambda o: f"{o.window:.3g}"),
            ("Bits", lambda o: f"{o.Nbit:g}"),
            _AREA,
            _NODE,
        ],
    ),
    (
        "Programming peripherals",
        MemWriteModel,
        list(mem_write_model._PRESETS),
        [
            ("Target", lambda o: o.type),
            ("Latency", lambda o: _si(o.latency, "s")),
            ("Dyn. energy", _write_energy),
            _LEAKAGE,
            ("SigmaG", lambda o: _si(o.SigmaG, "S")),
            _AREA,
            _NODE,
        ],
    ),
    (
        "Digital baselines",
        GPUModel,
        list(gpu_model._PRESETS),
        [
            ("Bits", lambda o: f"{o.Nbit:g}"),
            ("Throughput", lambda o: _si(o.Throughput, "OPS")),
            ("Memory", lambda o: _si(o.MemorySize, "B")),
            ("Bandwidth", lambda o: _si(o.MemoryBandwidth, "B/s")),
            ("Power", lambda o: _si(o.Power, "W")),
            ("Area", lambda o: _area(o.Area)),
            _NODE,
        ],
    ),
    (
        "Technology nodes",
        TechModel,
        list(tech_model._PRESETS),
        [
            ("Node", lambda o: _si(o.node, "m")),
            ("Rswn", lambda o: f"{o.Rswn:g} Ω/□"),
            ("Rswp", lambda o: f"{o.Rswp:g} Ω/□"),
            ("Rsh", lambda o: f"{o.Rsh:g} Ω/□"),
            ("Wr", lambda o: f"{o.Wr:g}"),
            ("Vdd", lambda o: _si(o.Vdd, "V")),
        ],
    ),
]

_PRESETS_INTRO = """\
# Presets

Device, peripheral and baseline models can be built from named presets, for
example `ADCModel("Fan19")` or `MemoryModel("Woo16")`. Keyword arguments given
together with a preset override single values. For peripherals, a
`technology` with a different node first rescales area and power (see
{class}`~loopuss.PeriphCirc`).

The tables are generated from the preset dictionaries when the documentation
is built. Each one lists the values the constructor ends up with, after its
consistency checks.

Besides the literature presets, {class}`~loopuss.OAModel` accepts the topologies
`1P`, `GM`, `2P` and `2P1Z`, and {class}`~loopuss.DSPModel` has the single preset
`ADSP2148-450MHz`.
"""


def _table(cls, names, columns) -> str:
    refs = _REFS.get(cls.__name__)
    header = ["Preset", *(name for name, _ in columns)] + (["Reference"] if refs else [])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name in names:
        obj = cls(name)
        row = [f"`{name}`", *(get(obj) for _, get in columns)]
        if refs:
            doi = refs.get(name)
            row.append(f"[{doi}](https://doi.org/{doi})" if doi else "")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _write_presets(docs: Path) -> None:
    parts = [_PRESETS_INTRO]
    for heading, cls, names, columns in _SECTIONS:
        parts.append(f"## {heading}\n\n{{class}}`~loopuss.{cls.__name__}`\n")
        parts.append(_table(cls, names, columns) + "\n")
    _write(docs / "presets.md", "\n".join(parts))
