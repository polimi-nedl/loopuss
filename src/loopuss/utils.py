"""Utility helpers."""

from __future__ import annotations

import math

import numpy as np


def _ci_get(kwargs: dict, key: str, default=None):
    """Case-insensitive dict lookup."""
    key_lower = key.lower()
    for k, v in kwargs.items():
        if k.lower() == key_lower:
            return True, v
    return False, default


def vparse(kwargs: dict, key: str, default=None):
    """Return kwargs[key] (case-insensitive) or default."""
    _, v = _ci_get(kwargs, key, default)
    return v


_SI = [
    (1e15, "P"),
    (1e12, "T"),
    (1e9, "G"),
    (1e6, "M"),
    (1e3, "k"),
    (1.0, ""),
    (1e-3, "m"),
    (1e-6, "u"),
    (1e-9, "n"),
    (1e-12, "p"),
    (1e-15, "f"),
]


def si_format(value: float, unit: str = "") -> str:
    """Format a number with an SI prefix."""
    if value == 0:
        return f"0 {unit}".strip()
    abs_val = abs(value)
    for scale, prefix in _SI:
        if abs_val >= scale:
            return f"{value / scale:.4g} {prefix}{unit}".strip()
    scale, prefix = _SI[-1]
    return f"{value / scale:.4g} {prefix}{unit}".strip()


def si_scale(x: float) -> int:
    """SI exponent (multiple of 3, clamped to [-15, 15]) for ``x``.

    Computes ``min(15, max(-15, floor(log10(x)/3)*3))``.
    """
    ax = abs(float(x))
    if ax == 0.0:
        return -15
    if math.isinf(ax) or math.isnan(ax):
        return 15
    return int(min(15, max(-15, math.floor(math.log10(ax) / 3.0) * 3)))


_SI_PREFIX = {
    -15: "f",
    -12: "p",
    -9: "n",
    -6: "µ",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
    15: "P",
}


def si_prefix(n: int) -> str:
    """SI prefix string for exponent ``n`` (a multiple of 3)."""
    return _SI_PREFIX[int(n)]


# --- grouped display ---
#
# The display is split into two concerns: the rendering engine (this mixin) and
# the per-class ``_property_groups()`` that says *which* properties go in
# *which* labelled group. Subclasses implement ``_property_groups()`` and this
# mixin renders it for the terminal (__repr__) and for Jupyter/IPython
# (_repr_html_).


def fmt_value(v) -> str:
    """Render one property value compactly for command-window display.

    Arrays show as ``[rows x cols dtype]``, structs (dicts) as their key set and
    object lists as ``[N-elem TypeName]``, rather than dumping the whole contents.
    Strings/preformatted values pass through unchanged.
    """
    if v is None:
        return "[]"
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return "[]"
        if v.size == 1:
            return fmt_value(v.item())
        return f"[{'x'.join(str(s) for s in v.shape)} {v.dtype}]"
    if isinstance(v, dict):
        return "{" + ", ".join(map(str, v.keys())) + "}"
    if isinstance(v, (list, tuple)):
        if not v:
            return "[]"
        if all(isinstance(x, (int, float, str, bool)) for x in v) and len(v) <= 6:
            return repr(list(v))
        return f"[{len(v)}-elem {type(v[0]).__name__}]"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class GroupedDisplay:
    """Mixin giving objects a grouped, labelled repr (see module note above).

    Subclasses implement ``_property_groups()`` returning
    ``[(title, {label: value}), …]``. A falsy ``title`` renders the fields with
    no group header.
    Override ``_display_header()`` to customise the first line.
    """

    def _display_header(self) -> str:
        return type(self).__name__

    def _property_groups(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        lines = [self._display_header()]
        for title, fields in self._property_groups():
            if not fields:
                continue
            indent = "    "
            if title:
                lines.append(f"  {title}:")
            else:
                indent = "  "
            w = max((len(k) for k in fields), default=0)
            for k, v in fields.items():
                lines.append(f"{indent}{k:>{w}} : {fmt_value(v)}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        parts = [f"<b>{_html_escape(self._display_header())}</b>"]
        for title, fields in self._property_groups():
            if not fields:
                continue
            rows = "".join(
                "<tr>"
                f"<td style='text-align:right;padding-right:8px;font-weight:600'>{k}</td>"
                f"<td>{_html_escape(fmt_value(v))}</td></tr>"
                for k, v in fields.items()
            )
            if title:
                parts.append(f"<b>{_html_escape(title)}:</b>")
            parts.append(f"<table style='margin:0 0 6px 12px'>{rows}</table>")
        return "".join(parts)


def relerr(xcomp, xexac, n: int | float = 2):
    """Relative error between ``xcomp`` and ``xexac`` in the p-norm ``n``.

    Norms are taken along the first axis, so passing an ``(m, k)`` ``xcomp``
    against an ``(m, 1)`` reference returns the per-column relative error.
    """
    xcomp = np.asarray(xcomp, dtype=float)
    xexac = np.asarray(xexac, dtype=float)
    num = np.linalg.norm(xcomp - xexac, ord=n, axis=0)
    den = np.linalg.norm(xexac, ord=n, axis=0)
    return num / den
