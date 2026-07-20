"""SPICE co-simulation orchestration: LTSpice analysis description, netlist
invocation, and ``.raw`` parsing.

This uses **LTSpice** as the simulation engine via a ``netlist`` ->
``run`` -> ``.raw`` parse flow. The LTSpice ``.lib`` subcircuit library lives
in ``spice_lib/``.

The ``.raw`` parser handles the subset used by loopuss:
real-valued ``.tran``, ``.op`` and ``.noise`` results in binary or ASCII form.
LTspice's lossy 2nd-order time compression is *not* handled here; netlists are
generated with ``.option plotwinsize=0`` so the time vector is uncompressed.
"""

from __future__ import annotations

import ctypes
import json
import os
import queue as _queue
import re
import select
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import GroupedDisplay

_LIB_DIR = Path(__file__).resolve().parent / "spice_lib"

#: Library files included into every netlist (order matters).
LIB_FILES = ("arraycells.lib", "nloa_stages.lib", "nloa.lib", "niccells.lib", "noisegen.lib")

# Per-solver-accuracy LTSpice .option footers. Membership in this dict also
# defines which solver_type strings are SPICE modes (versus the fast ODE path).
# plotwinsize=0 is added to all modes so the .raw time vector is uncompressed
# (the parser does not implement LTSpice 2nd-order decompression).
# numdgt=15 is set in all modes so LTSpice writes a double-precision .raw;
# without it (default numdgt=6) fast/accurate store single-precision (float32),
# flooring their steady-state error at ~float32 epsilon regardless of tolerance.
_SPICE_FOOTERS = {
    "SPICE/fast": (
        ".option plotwinsize=0\r\n.option numdgt=15\r\n.option measdgt=15\r\n"
        ".option abstol=1e-8 \r\n.option vntol=1e-6 \r\n"
        ".option reltol=1e-8 \r\n.backanno\r\n.end\r\n"
    ),
    "SPICE/accurate": (
        ".option numdgt=15\r\n.option measdgt=15\r\n.option abstol=1e-11\r\n.option vntol=1e-9 \r\n"
        ".option reltol=1e-11\r\n.option plotwinsize=0\r\n.backanno\r\n.end\r\n"
    ),
    "SPICE/extreme": (
        ".option numdgt=15\r\n.option measdgt=15\r\n.option abstol=1e-13\r\n"
        ".option vntol=1e-11\r\n.option reltol=1e-13\r\n.option plotwinsize=0\r\n"
        ".backanno\r\n.end\r\n"
    ),
}


# --- SpiceAnalysis ---
@dataclass(repr=False)
class SpiceAnalysis(GroupedDisplay):
    """Description of a SPICE analysis.

    ``type`` is one of ``'tran'``, ``'op'`` or ``'noise'``.
    """

    type: str
    Tstart: float = 0.0
    Tstop: float = 1.0
    Tstep: float | None = None
    dTmax: float = 1e3
    Fmin: float = 1.0
    Fmax: float = 10e9
    Fstep: float = 100
    Fsweep: str = "dec"

    def __post_init__(self):
        if self.Tstep is None:
            self.Tstep = (self.Tstop - self.Tstart) / 100.0
        self._validate()

    def _validate(self):
        if self.Fmax < self.Fmin:
            self.Fmax = 10 * self.Fmin
        if self.type.lower() not in ("tran", "op", "noise"):
            raise ValueError(f"SPICE analysis {self.type} unsupported.")

    def is_op(self) -> bool:
        return self.type.lower() == "op"

    def is_tran(self) -> bool:
        return self.type.lower() == "tran"

    def is_noise(self) -> bool:
        return self.type.lower() == "noise"

    # --- display: type-dependent property groups ---
    def _property_groups(self):
        if self.is_tran():
            fields = {
                "type": self.type,
                "Tstart": self.Tstart,
                "Tstop": self.Tstop,
                "Tstep": self.Tstep,
                "dTmax": self.dTmax,
            }
        elif self.is_noise():
            fields = {
                "type": self.type,
                "Fmin": self.Fmin,
                "Fmax": self.Fmax,
                "Fstep": self.Fstep,
                "Fsweep": self.Fsweep,
            }
        else:  # op
            fields = {"type": self.type}
        return [("", fields)]

    def to_spice(self, args=None) -> str:
        """SPICE directive string."""
        t = self.type.lower()
        if t == "op":
            return ".op"
        if t == "tran":
            return f".tran {self.Tstep:.16g} {self.Tstop:.16g} {self.Tstart:.16g} {self.dTmax:.16g}"
        if t == "noise":
            return (
                f".noise {args[0]} {args[1]} {self.Fsweep} "
                f"{int(self.Fstep)} {self.Fmin:.16g} {self.Fmax:.16g}"
            )
        raise ValueError(self.type)

    def to_double(self) -> np.ndarray:
        """Numeric sweep vector."""
        t = self.type.lower()
        if t == "tran":
            step = min(self.Tstep, self.dTmax)
            return np.arange(self.Tstart, self.Tstop + step / 2, step)
        if t == "op":
            return np.array([0.0])
        if t == "noise":
            if self.Fsweep.lower() == "dec":
                n = np.floor(np.log10(self.Fmax)) - np.ceil(np.log10(self.Fmin))
                return 10.0 ** np.linspace(
                    np.log10(self.Fmin), np.log10(self.Fmax), int(n * self.Fstep) + 1
                )
            if self.Fsweep.lower() == "oct":
                n = np.floor(np.log2(self.Fmax)) - np.ceil(np.log2(self.Fmin))
                return 2.0 ** np.linspace(
                    np.log2(self.Fmin), np.log2(self.Fmax), int(n * self.Fstep) + 1
                )
            if self.Fsweep.lower() == "lin":
                n = np.floor(self.Fmax) - np.ceil(self.Fmin)
                return np.linspace(self.Fmin, self.Fmax, int(n) + 1)
        raise ValueError(self.type)


# --- LTSpice path ---
_DEFAULT_SPICE_LOCATIONS = (
    r"C:\Program Files\ADI\LTspice\LTspice.exe",
    r"C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe",
    r"C:\Program Files\LTC\LTspiceXVII\XVIIx86.exe",
    r"C:\Program Files (x86)\LTC\LTspiceXVII\XVIIx64.exe",
)


def spice_path() -> str:
    """Path to the LTSpice executable.

    Resolution order: ``LOOPUSS_SPICE_PATH`` env var, then ``config.json`` in
    the package directory (``{"spicePath": "..."}``), then known default install
    locations, with env/auto-detect fallbacks.
    """
    env = os.environ.get("LOOPUSS_SPICE_PATH")
    if env and Path(env).exists():
        return env

    cfg = Path(__file__).resolve().parent / "config.json"
    if cfg.exists():
        p = json.loads(cfg.read_text()).get("spicePath")
        if p and Path(p).exists():
            return p

    for loc in _DEFAULT_SPICE_LOCATIONS:
        if Path(loc).exists():
            return loc

    raise FileNotFoundError(
        "LTSpice executable not found. Set the LOOPUSS_SPICE_PATH environment "
        'variable or create loopuss/config.json with {"spicePath": "<path>"}.'
    )


def spice_available() -> bool:
    """True if an LTSpice executable can be located (used to skip SPICE tests)."""
    try:
        spice_path()
        return True
    except FileNotFoundError:
        return False


# --- run ---
#: How many times to (re)launch LTSpice per netlist before giving up. LTSpice's
#: exit code is unreliable, so success is judged by a parseable ``.raw``; a
#: transient miss (e.g. an ``xvfb-run -a`` display-number collision when several
#: runs start at once) is retried rather than raised, so concurrent runs
#: (parallel scripts / pytest workers) are robust.
_SPICE_RETRIES = 3


def _die_with_parent():
    """Ask the kernel to SIGTERM this child when its parent dies.

    _xvfb_pool reaps its servers in a ``finally``, but nothing runs on SIGKILL —
    and a killed sweep would then strand one live Xvfb per worker, each holding a
    display number and a ``/tmp/.X*-lock``. Enough of those accumulate and new
    runs can no longer get a display at all. PR_SET_PDEATHSIG (=1) with SIGTERM
    (=15) makes the leak impossible rather than merely unlikely.
    """
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, 15, 0, 0, 0)
    except OSError:
        pass  # best-effort: the finally in _xvfb_pool still covers us


#: Cached probe of ``setarch -R`` (see _no_aslr_prefix).
_setarch_ok: bool | None = None


def _no_aslr_prefix() -> list[str]:
    """``setarch -R`` prefix to disable ASLR for LTSpice, or [] if unavailable.

    LTspice 26's JIT (LLVM RuntimeDyld) emits IMAGE_REL_AMD64_ADDR32NB
    relocations, which reach only a 32-bit displacement. Under wine, ASLR can
    scatter its allocations further apart than that, whereupon LTspice aborts
    with "relocation requires an ordered section layout", exits 9, and leaves an
    empty ``.raw``. Fixing the layout makes the JIT reproducible: on this box the
    same netlist went 8/8 successful with this prefix and 0/8 without, so the
    retries in _run_one cannot paper over it — every attempt loses equally.
    """
    global _setarch_ok
    if _setarch_ok is None:
        try:
            _setarch_ok = (
                subprocess.run(
                    ["setarch", "-R", "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                ).returncode
                == 0
            )
        except OSError:
            _setarch_ok = False
    return ["setarch", "-R"] if _setarch_ok else []


def _start_xvfb(timeout: float = 20.0):
    """Start one Xvfb server; return ``(proc, display)`` or ``(None, None)``.

    ``-displayfd`` has Xvfb pick a free display itself and report it back once
    it is ready to accept connections. Picking a number ourselves would mean a
    check-then-bind race against every other starting server, and would also
    have to reason about stale ``/tmp/.X*-lock`` files left by killed servers —
    Xvfb already handles both internally.
    """
    exe = shutil.which("Xvfb")
    if exe is None:
        return None, None
    r, w = os.pipe()
    try:
        proc = subprocess.Popen(
            [exe, "-displayfd", str(w), "-screen", "0", "640x480x8", "-nolisten", "tcp"],
            pass_fds=(w,),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=_die_with_parent,
        )
    except OSError:
        os.close(r)
        os.close(w)
        return None, None
    os.close(w)  # only Xvfb should hold the write end, so reads see EOF

    num, deadline = b"", time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if not select.select([r], [], [], 0.1)[0]:
                if proc.poll() is not None:
                    break  # exited without reporting a display
                continue
            chunk = os.read(r, 64)
            if not chunk:
                break  # EOF
            num += chunk
            if num.strip().isdigit() and num.endswith((b"\n", b"\r")):
                return proc, f":{num.strip().decode()}"
    finally:
        os.close(r)

    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=10)
    return None, None


@contextmanager
def _null_ctx():
    """No-op stand-in for _xvfb_pool: children inherit the caller's environment."""
    yield None


@contextmanager
def _xvfb_pool(n: int):
    """Yield a queue of ``n`` private DISPLAYs, or None if Xvfb is unavailable.

    One X server per *worker*: not one per netlist, and not one shared by all.
    Two LTSpice instances on the same display trample each other's windows (the
    ADI build maps one even under ``-b``) and die with a BadWindow X error, so
    workers cannot share. The alternative of ``xvfb-run -a`` per netlist stands
    up a server per run and races on the display number, and it leaks a live
    Xvfb plus a ``/tmp/.X*-lock`` whenever a run is killed — leftovers that
    accumulate until fresh runs can no longer get a display. Owning the servers
    means we also reliably reap them. Callers fall back to ``xvfb-run -a`` on
    None.
    """
    procs, displays = [], _queue.Queue()
    try:
        for _ in range(n):
            proc, display = _start_xvfb()
            if display is None:
                break  # no Xvfb binary, or none would start
            procs.append(proc)
            displays.put(display)
        yield displays if displays.qsize() else None
    finally:
        for proc in procs:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def _run_on_display(args, displays, ctl=None):
    """Run one netlist, borrowing a DISPLAY for the duration if pooled.

    The queue doubles as the mutual exclusion: a display is checked out for the
    length of the run and only then returned, so it is never in use twice.
    """
    if displays is None:
        return _run_one(*args, env=None, ctl=ctl)
    display = displays.get()
    try:
        return _run_one(*args, env=dict(os.environ, DISPLAY=display), ctl=ctl)
    finally:
        displays.put(display)


class _Cancelled(Exception):
    """Raised in a worker whose run was cancelled; never reaches the caller."""


class _Cancellation:
    """Kill-switch shared by a sweep's workers, so Ctrl-C stops it promptly.

    Without one, interrupting a sweep means waiting out the in-flight wave --
    ~50 min per netlist at N=256 - because the LTSpice children keep solving
    and the retry loop cheerfully relaunches them. Cancelling kills every live
    child so its worker unblocks at once, and latches a flag the retry loop
    checks so nothing is restarted on the way out.
    """

    def __init__(self):
        self._flag = threading.Event()
        self._procs: set[subprocess.Popen] = set()
        self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        return self._flag.is_set()

    def register(self, proc):
        with self._lock:
            if self._flag.is_set():
                proc.kill()  # cancelled between the check and the spawn
            else:
                self._procs.add(proc)

    def unregister(self, proc):
        with self._lock:
            self._procs.discard(proc)

    def cancel(self):
        with self._lock:
            self._flag.set()
            for proc in list(self._procs):
                proc.kill()


def _run_one(net, raw, cmd_prefix, exe, keep_raw, keep_vars, env, ctl=None):
    """Run LTSpice on a single netlist and return the parsed ``.raw``."""
    net, raw = Path(net), Path(raw)
    # LTspice under wine ignores an absolute Unix netlist path (it exits 0
    # without writing the .raw); run from the netlist's directory and pass
    # the bare filename so the path resolves under both wine and native.
    cmd = cmd_prefix + [exe, "-b", net.name]

    # Success is judged solely by a parseable .raw, never by the exit code:
    # LTspice exits 0 even when it fails to write output, and (under wine) it
    # can also exit non-zero *after* writing a perfectly good .raw. A stale
    # .raw is removed before each attempt so a previous run's file can't be
    # mistaken for this one's. Transient misses are retried.
    res, last_rc, last_err = None, None, ""
    for _ in range(_SPICE_RETRIES):
        if ctl is not None and ctl.cancelled:
            raise _Cancelled()  # never start (or retry) on the way out
        try:
            raw.unlink()
        except FileNotFoundError:
            pass
        proc = subprocess.Popen(
            cmd, cwd=str(net.parent), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env
        )
        if ctl is not None:
            ctl.register(proc)
        try:
            _, stderr = proc.communicate()
        except BaseException:  # incl. KeyboardInterrupt on the serial path
            proc.kill()
            proc.wait()
            raise
        finally:
            if ctl is not None:
                ctl.unregister(proc)
        if ctl is not None and ctl.cancelled:
            raise _Cancelled()  # the .raw is half-written; don't parse it
        last_rc = proc.returncode
        last_err = (stderr or b"").decode("utf-8", "replace").strip()
        if raw.exists():
            try:
                res = _natsort_result(raw_parse(str(raw), keep_vars))
                if not res.get("num_data_pnts"):
                    res = None  # parsed but empty (0 data points) -> retry
                else:
                    break
            except Exception:
                res = None  # truncated / garbled .raw -> retry

    if res is None:
        log = net.with_suffix(".log").name
        msg = (
            f"LTspice produced no usable output for {net.name!r} after "
            f"{_SPICE_RETRIES} attempt(s) (last exit code {last_rc}). "
            f"Check {log!r} beside the netlist."
        )
        if last_err:
            msg += f"\nLast stderr:\n{last_err[-800:]}"
        raise RuntimeError(msg)

    # The .raw is parsed into ``res`` and no longer needed; drop it (and the
    # operating-point .op.raw byproduct) so a long sweep doesn't pile up.
    if not keep_raw:
        for scratch in (raw, raw.with_suffix(".op.raw")):
            try:
                scratch.unlink()
            except FileNotFoundError:
                pass
    return res


def run_spice(net_paths, raw_paths, keep_raw=False, *, keep_vars=None, max_workers=None):
    """Run LTSpice on each netlist and parse the resulting ``.raw`` files.

    Invokes LTSpice in batch mode, parses each ``.raw``, and natural-sorts
    variables (LTSpice may reorder outputs).
    Returns a single dict for one netlist, or a list of dicts otherwise.

    Unless ``keep_raw`` is set, each ``.raw`` (and its ``.op.raw`` sibling) is
    deleted immediately after it parses successfully. The parsed data is in the
    returned dict, so the file is pure scratch; this matters for ``.noise`` runs
    where LTSpice ignores ``.save`` and writes every device's contribution
    (hundreds of MB per output for a large array). Deleting per file caps peak
    disk at a single ``.raw`` instead of letting a sweep accumulate to many GB.

    ``keep_vars`` is forwarded to raw_parse. ``max_workers`` runs the netlists
    concurrently; a single netlist or a single worker stays serial rather than
    pay the setup cost. Threads suffice because each task is
    dominated by waiting on the LTSpice subprocess. Results stay in netlist
    order, which callers rely on to map them back onto outputs.

    Ctrl-C (or any exception) kills the in-flight LTSpice children and drops the
    queued netlists, rather than waiting out a wave that can run ~50 min per
    netlist at N=256.
    """
    if isinstance(net_paths, (str, Path)):
        net_paths = [net_paths]
    if isinstance(raw_paths, (str, Path)):
        raw_paths = [raw_paths]
    if len(net_paths) != len(raw_paths):
        raise ValueError("Netlists and raw files should have a 1-1 correspondence.")

    exe = spice_path()
    unix = sys.platform not in ("win32", "darwin")
    # Linux runs LTspice via wine. The ADI build creates a
    # window even in -b batch mode; under a headless session (no DISPLAY) wine
    # can't, and LTspice exits 0 without writing the .raw — hence a virtual X
    # server. With DISPLAY already set, nothing is needed.
    needs_xvfb = unix and not os.environ.get("DISPLAY")
    workers = min(max_workers or 1, len(net_paths))

    with _xvfb_pool(workers) if needs_xvfb else _null_ctx() as displays:
        prefix = _no_aslr_prefix() + ["wine"] if unix else []
        if needs_xvfb and displays is None and shutil.which("xvfb-run"):
            # Xvfb itself is missing; fall back to a server per run.
            prefix = ["xvfb-run", "-a"] + prefix

        args = [
            (net, raw, prefix, exe, keep_raw, keep_vars) for net, raw in zip(net_paths, raw_paths)
        ]
        ctl = _Cancellation()
        if workers > 1:
            # Not `with ThreadPoolExecutor(...)`: its __exit__ waits for running
            # futures *and* still runs the queued ones. On the way out we want
            # neither, so shutdown is driven explicitly.
            pool = ThreadPoolExecutor(max_workers=workers)
            try:
                futures = [pool.submit(_run_on_display, a, displays, ctl) for a in args]
                results = [f.result() for f in futures]
            except BaseException:
                ctl.cancel()  # unblocks the workers before we join them
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            pool.shutdown()
        else:
            results = [_run_on_display(a, displays, ctl) for a in args]

    return results[0] if len(results) == 1 else results


def read_log_elapsed(path) -> float | None:
    """LTSpice's reported 'Total elapsed time' [s] from the .log beside the
    given .net/.raw path, or None if unavailable.

    This is the solver/integration time as measured inside LTSpice; it excludes
    the OS process startup, the Python-side netlist writing, and the .raw parse.
    """
    log = Path(path).with_suffix(".log")
    if not log.exists():
        return None
    m = re.search(r"Total elapsed time:\s*([0-9.eE+-]+)\s*seconds", log.read_text(errors="ignore"))
    return float(m.group(1)) if m else None


def _natsort_result(res: dict) -> dict:
    """Reorder variables by natural sort of their names."""
    names = res["variable_name_list"]
    order = sorted(range(len(names)), key=lambda i: _natkey(names[i]))
    res["variable_name_list"] = [names[i] for i in order]
    res["variable_type_list"] = [res["variable_type_list"][i] for i in order]
    if res.get("variable_mat") is not None and len(res["variable_mat"]):
        res["variable_mat"] = res["variable_mat"][order, :]
    return res


def _natkey(s: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", s)]


# --- .raw parser ---
def _spice_node_matrix(res: dict, names: list[str]) -> np.ndarray:
    """Extract rows of res['variable_mat'] for the given LTSpice variable names.

    Matching is case-insensitive (LTSpice lowercases node names).
    """
    lut = {nm.lower(): i for i, nm in enumerate(res["variable_name_list"])}
    mat = np.asarray(res["variable_mat"])
    npts = mat.shape[1] if mat.ndim == 2 else 1
    out = np.empty((len(names), npts))
    for r, nm in enumerate(names):
        i = lut.get(nm.lower())
        if i is None:
            raise KeyError(f"Variable {nm!r} not found in .raw output.")
        out[r, :] = np.real(mat[i, :])
    return out


def raw_parse(filename: str, keep_vars=None) -> dict:
    """Parse an LTSpice ``.raw`` file.

    Supports real ``.tran`` / ``.op`` / ``.noise`` analyses in binary or ASCII.
    Returns a dict with keys: ``title, date, plotname, flags,
    variable_name_list, variable_type_list, variable_mat`` (nvars x npts), and
    ``time_vect`` (.tran/.op) or ``freq_vect`` (.noise).

    ``keep_vars`` restricts the result to the named variables (case-insensitive),
    which for ``.noise`` is the difference between materialising every device's
    contribution and materialising the one column the caller wants. LTSpice
    ignores ``.save`` for ``.noise``, so an NxN array writes N^2 columns; at
    N=256 that is a ~577 MB float64 matrix built to keep a single ~9 kB column.
    Names absent from the file are silently dropped, leaving the lookup in the
    caller to report them precisely. ``None`` (default) parses everything.
    """
    path = Path(filename)
    if not path.exists() and Path(filename + ".raw").exists():
        path = Path(filename + ".raw")

    raw = path.read_bytes()

    # Detect UTF-16-LE (LTspice XVII) vs ASCII (LTspice IV) by NUL bytes in head.
    utf16 = b"\x00" in raw[:64]

    # Locate the data marker ("Binary:" or "Values:") in the text header.
    marker_bin = "Binary:".encode("utf-16-le") if utf16 else b"Binary:"
    marker_asc = "Values:".encode("utf-16-le") if utf16 else b"Values:"
    i_bin = raw.find(marker_bin)
    i_asc = raw.find(marker_asc)
    if i_bin == -1 and i_asc == -1:
        raise ValueError(f'Format error in LTspice file "{path}": no data marker.')
    if i_asc == -1 or (i_bin != -1 and i_bin < i_asc):
        file_format, marker_pos, marker = "binary", i_bin, marker_bin
    else:
        file_format, marker_pos, marker = "ascii", i_asc, marker_asc

    header_bytes = raw[:marker_pos]
    header = header_bytes.decode("utf-16-le" if utf16 else "latin-1")

    tags, var_names, var_types = _parse_header(header)

    plotname = tags.get("plotname", "").lower()
    if "transient" in plotname:
        sim = "tran"
    elif "noise spectral density" in plotname:
        sim = "noise"
    elif "operating point" in plotname:
        sim = "op"
    else:
        raise ValueError(f"Unsupported LTspice analysis: {tags.get('plotname')!r}")

    flags = tags.get("flags", "").lower()
    if "fastaccess" in flags:
        raise ValueError("Fast Access format is not supported.")
    if "complex" in flags:
        raise ValueError("Complex (.ac) data is not supported by this parser.")

    n_vars_total = int(round(float(tags["no. variables"])))  # includes time/freq
    n_pts = int(round(float(tags["no. points"])))
    double_flag = "double" in flags

    # For .tran/.noise the first variable is the time/freq axis and is excluded
    # from the variable lists. For .op there is no axis variable -> keep all.
    if sim == "op":
        name_list, type_list = list(var_names), list(var_types)
    else:
        name_list, type_list = var_names[1:], var_types[1:]

    # ``sel`` indexes into name_list (i.e. excludes the axis for .tran/.noise);
    # the readers map it back onto the payload's own column numbering.
    sel = None
    if keep_vars is not None:
        pos = {nm.lower(): i for i, nm in enumerate(name_list)}
        sel = [pos[v.lower()] for v in keep_vars if v.lower() in pos]
        name_list = [name_list[i] for i in sel]
        type_list = [type_list[i] for i in sel]

    out = {
        "title": tags.get("title", ""),
        "date": tags.get("date", ""),
        "plotname": tags.get("plotname", ""),
        "flags": tags.get("flags", ""),
        "variable_name_list": name_list,
        "variable_type_list": type_list,
    }

    if file_format == "binary":
        # Binary payload starts right after the marker line terminator.
        start = marker_pos + len(marker)
        # skip the rest of the marker line (newline)
        nl = "\n".encode("utf-16-le") if utf16 else b"\n"
        nlpos = raw.find(nl, start)
        data_start = nlpos + len(nl)
        # memoryview, not a bytes slice: the latter copies the whole payload
        # (~289 MB per .noise raw at N=256) for no reason.
        payload = memoryview(raw)[data_start:]
        axis, mat = _read_binary(payload, n_vars_total, n_pts, double_flag, sim, sel)
    else:
        text = raw[marker_pos + len(marker) :]
        text = text.decode("utf-16-le" if utf16 else "latin-1")
        axis, mat = _read_ascii(text, n_vars_total, n_pts, sim, sel)

    if sim == "noise":
        out["freq_vect"] = axis
    else:
        out["time_vect"] = axis
    out["variable_mat"] = mat
    out["num_variables"] = n_vars_total - 1
    out["num_data_pnts"] = mat.shape[1] if mat is not None else 0
    return out


def _parse_header(header: str):
    """Extract scalar tags and the variable table from the text header."""
    lines = header.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    tags: dict[str, str] = {}
    var_names: list[str] = []
    var_types: list[str] = []
    in_vars = False
    for line in lines:
        if not line.strip():
            continue
        if line.lower().startswith("variables:"):
            in_vars = True
            continue
        if in_vars and (line.startswith("\t") or line.startswith(" ")):
            parts = line.split()
            # parts: <idx> <name> <type>
            if len(parts) >= 3:
                var_names.append(parts[1])
                var_types.append(parts[2])
            continue
        in_vars = False
        if ":" in line:
            key, _, val = line.partition(":")
            tags[key.strip().lower()] = val.strip()
    return tags, var_names, var_types


def _read_binary(
    payload: bytes, n_vars_total: int, n_pts: int, double_flag: bool, sim: str, sel=None
):
    """Read a real-valued binary payload into (axis, variable_mat).

    ``sel`` (see raw_parse) keeps only the listed variables, indexed as in the
    caller's name list: for .tran/.noise variable i sits at payload column i+1,
    the axis occupying column 0.
    """
    if sim == "op":
        # Single point, no time axis. With the 'double' flag every var is
        # float64; otherwise LTSpice stores the first var double (8B) and the
        # rest float32 (4B) (numdgt<=6).
        if double_flag:
            vals = np.frombuffer(payload[: 8 * n_vars_total], dtype="<f8").copy()
        else:
            vals = np.empty(n_vars_total)
            vals[0] = np.frombuffer(payload[:8], dtype="<f8")[0]
            if n_vars_total > 1:
                vals[1:] = np.frombuffer(payload[8 : 8 + 4 * (n_vars_total - 1)], dtype="<f4")
        # .op has no axis column, so sel indexes vals directly.
        if sel is not None:
            vals = vals[sel]
        return np.array([0.0]), vals.reshape(-1, 1)

    if double_flag:
        # axis + all vars stored as float64.
        block = np.frombuffer(payload[: 8 * n_vars_total * n_pts], dtype="<f8")
        block = block.reshape(n_pts, n_vars_total).T
        rows = slice(1, None) if sel is None else [1 + i for i in sel]
        return block[0, :].copy(), block[rows, :].copy()

    # Standard real: per point = [axis float64][var1..varN-1 float32].
    if sel is not None:
        # Only some columns are wanted, so skip the full nvars x npts matrix.
        # The record is 8 + 4*(nvars-1) bytes, a multiple of 4, so the payload
        # views as an f4 grid whose columns can be picked directly: the axis
        # occupies words 0-1 (one f8) and variable i lives at word 2 + i.
        words = 2 + (n_vars_total - 1)
        buf = np.frombuffer(payload[: 4 * words * n_pts], dtype="<f4").reshape(n_pts, words)
        axis = np.abs(np.ascontiguousarray(buf[:, :2]).view("<f8").ravel())
        mat = buf[:, [2 + i for i in sel]].T.astype(np.float64)
        return axis, mat

    dtype = np.dtype([("axis", "<f8")] + [(f"v{i}", "<f4") for i in range(n_vars_total - 1)])
    rec = np.frombuffer(payload[: dtype.itemsize * n_pts], dtype=dtype)
    axis = np.abs(rec["axis"].astype(np.float64))  # |.| guards compression sign
    mat = np.empty((n_vars_total - 1, n_pts))
    for i in range(n_vars_total - 1):
        mat[i, :] = rec[f"v{i}"]
    return axis, mat


def _read_ascii(text: str, n_vars_total: int, n_pts: int, sim: str, sel=None):
    """Read a real-valued ASCII 'Values:' section into (axis, variable_mat).

    ``sel`` filters after the fact rather than during the scan: the ASCII path
    has to tokenise the whole block anyway, and LTSpice writes binary in
    practice, so the memory win here would not pay for the complexity.
    """
    nums = np.array(text.replace(",", " ").split(), dtype=float)
    if sim == "op":
        # Layout: point_index, value0 ... value(N-1). Drop the index only.
        vals = nums[1 : n_vars_total + 1]
        if sel is not None:
            vals = vals[sel]
        return np.array([0.0]), vals.reshape(-1, 1)
    # Each block: point_index, axis, var1..varN-1  -> (n_vars_total + 1) per point
    per = n_vars_total + 1
    block = nums[: per * n_pts].reshape(n_pts, per).T
    axis = np.abs(block[1, :])
    rows = slice(2, None) if sel is None else [2 + i for i in sel]
    mat = block[rows, :].copy()
    return axis, mat
