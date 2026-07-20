"""LTSpice netlist generation.

`build_netlist` renders a `LoopUS` circuit into one or more LTSpice `.net`
files (one for tran/op, one per probed output for noise) and returns the netlist
and expected `.raw` paths. Running the netlists and parsing their output are
`spice.py`'s job.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import spice as _spice
from .signal import VoltageSignal

if TYPE_CHECKING:
    from .loopus import LoopUS


def _vto_supply_net(V: float) -> str | None:
    """Net name for a supply voltage.

    Returns None for infinite (unspecified) supplies. E.g. 1.2 -> '+1V200',
    -0.8 -> '-0V800'.
    """
    if not np.isfinite(V):
        return None
    sign = "-" if V < 0 else "+"
    a = abs(V)
    return f"{sign}{int(np.floor(a))}V{round(1e3 * (a - np.floor(a)))}"


def build_netlist(circ: LoopUS, solver_type: str, sp_analysis: _spice.SpiceAnalysis):
    """Generate LTSpice netlist(s) for the circuit.

    Returns ``(net_paths, raw_paths)`` as lists (one element for tran/op,
    one per output for noise).
    """
    n = circ.feedback_array.M
    p = circ.input_array.N

    # --- node families (1-based node names) ---
    Vout = [f"Vout_{i + 1}" for i in range(n)]
    Vjunc = [f"Vjunc_{i + 1}" for i in range(n)]
    VinOA = [f"VinOA_{i + 1}" for i in range(n)]

    oa_ref_zero = circ.oa_reference.vtoy() == 0
    in_ref_zero = circ.input_reference.vtoy() == 0
    VrefOA = ["0" if oa_ref_zero else f"VrefOA_{i + 1}" for i in range(n)]
    Vrefin = ["0" if in_ref_zero else f"Vrefin_{j + 1}" for j in range(p)]

    Vin = []
    for j in range(p):
        loaded = j < len(circ.voltage_dac) and circ.voltage_dac[j].is_loaded()
        Vin.append(f"Vin_{j + 1}" if loaded else "0")

    VinOAp = list(VrefOA)
    if np.all(circ.feedback_array.line_resistance == 0):
        Vjunc = list(VinOA)

    comp: list[str] = []

    # --- reference sources ---
    if not oa_ref_zero:
        for i in range(n):
            comp.append(circ.oa_reference.to_spice(f"REFOA{i + 1}", VrefOA[i], "0"))
    if not in_ref_zero:
        for j in range(p):
            comp.append(circ.input_reference.to_spice(f"REFIN{j + 1}", Vrefin[j], "0"))

    # --- input sources ---
    for j in range(p):
        comp += circ.voltage_dac[j].to_spice(f"IN{j + 1}", (Vin[j], Vrefin[j]))
    for i in range(n):
        comp += circ.current_dac[i].to_spice(f"IN{i + 1}", ("0", Vjunc[i]))

    # --- OA supplies ---
    supplies = sorted(
        {
            float(v)
            for m in circ.oa_models
            for v in np.asarray(m.Vsupply, float).ravel()
            if np.isfinite(v)
        }
    )
    for k, v in enumerate(supplies, start=1):
        net = _vto_supply_net(v)
        comp.append(VoltageSignal("DC", dcvalue=v).to_spice(f"supply{k}", net, "0"))

    # --- operational amplifiers ---
    for i, m in enumerate(circ.oa_models):
        vsup = np.asarray(m.Vsupply, float).ravel()
        if circ.sign_vector[i] > 0:
            inn, inp = VinOA[i], VinOAp[i]
        else:
            inn, inp = VinOAp[i], VinOA[i]
        comp.append(
            m.to_spice(i + 1, inn, inp, Vout[i], _vto_supply_net(vsup[0]), _vto_supply_net(vsup[1]))
        )

    # --- input matrix (orientation xbar_orient[0]) ---
    io = circ.xbar_orient
    teu, ted = (None, Vin) if io[0][0].lower() == "bottom" else (Vin, None)
    bel, ber = (Vjunc, None) if io[0][1].lower() == "left" else (None, Vjunc)
    comp += circ.input_array.to_spice("RIN", bel, ted, ber, teu)

    # --- feedback matrix (orientation xbar_orient[1]) ---
    teu, ted = (None, Vout) if io[1][0].lower() == "bottom" else (Vout, None)
    bel, ber = (Vjunc, VinOA) if io[1][1].lower() == "left" else (VinOA, Vjunc)
    comp += circ.feedback_array.to_spice("RF", bel, ted, ber, teu)

    # --- input noise reference source ---
    if sp_analysis.is_noise():
        comp.append(VoltageSignal("DC", dcvalue=1.0).to_spice("NOISE", "0", "Vnoise"))

    comp = [c for c in comp if c]
    comp_str = "\r\n".join(comp) + "\r\n"

    # --- library includes ---
    for lib in _spice.LIB_FILES:
        comp_str += f".lib {_spice._LIB_DIR / lib}\r\n"

    # --- analysis + save + footer ---
    if sp_analysis.is_tran() or sp_analysis.is_op():
        analysis = f"\r\n* SPICE Analysis\r\n{sp_analysis.to_spice()}\r\n"
        save = ".save I(I*) V(Vin_*) V(VinOA_*) V(Vout_*) Ix(oa*) I(V*) V(Vjunc_*)\r\n"
        if sp_analysis.is_tran() and circ.initial_condition is not None:
            ic = ".ic"
            for i, v in enumerate(np.asarray(circ.initial_condition).ravel()):
                ic += f" V(XOA{i + 1}:NX01)={v:.16g}"
            analysis += ic + "\r\n"
        elif solver_type != "SPICE/extreme":
            analysis += (
                ".option ptrantau=0\r\n.option topologycheck=0\r\n"
                ".option gminsteps=0\r\n.option itl6=0\r\n"
            )
        analyses = [analysis]
        save_str = save
        file_ids = [""]
    else:  # noise - one netlist per output
        analyses, file_ids = [], []
        for k in circ.output_mask:
            args = (f"V(Vout_{int(k) + 1})", "VNOISE")
            analyses.append(f"\r\n* SPICE Analysis\r\n{sp_analysis.to_spice(args)}\r\n")
            file_ids.append(f"_V{int(k) + 1}")
        # NB: LTSpice ignores .save for .noise — it always writes every
        # device's noise contribution to the .raw (hundreds of MB for a
        # large array). The raw is instead deleted right after parsing in
        # run_spice(); only V(onoise) is read back in _noise_spice.
        save_str = ""

    footer = _spice._SPICE_FOOTERS[solver_type]

    # --- write netlist file(s) ---
    out_dir = Path.cwd() / circ.file_path
    out_dir.mkdir(parents=True, exist_ok=True)
    header = f"* {circ.file_path}//{circ.file_name}.asc\r\n"

    net_paths, raw_paths = [], []
    for analysis, fid in zip(analyses, file_ids):
        net = out_dir / f"{circ.file_name}{fid}.net"
        raw = out_dir / f"{circ.file_name}{fid}.raw"
        net.write_text(header + comp_str + save_str + analysis + footer)
        net_paths.append(str(net))
        raw_paths.append(str(raw))
    return net_paths, raw_paths
