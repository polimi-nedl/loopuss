"""Checks that the package ships the data files it needs at runtime.

These assert against the imported package rather than the repo, so they only bite
when the suite runs against a non-editable install (a built wheel in a clean
venv). Under an editable install they pass trivially, since the package resolves
back into the source tree.
"""

from pathlib import Path

import pytest

from loopuss import spice


def test_lib_dir_exists():
    assert spice._LIB_DIR.is_dir(), (
        f"spice_lib/ missing from the installed package at {spice._LIB_DIR}; "
        "check [tool.setuptools.package-data] in pyproject.toml"
    )


@pytest.mark.parametrize("name", spice.LIB_FILES)
def test_lib_file_shipped(name):
    """Every library a netlist .lib-includes must exist and be non-empty."""
    path: Path = spice._LIB_DIR / name
    assert path.is_file(), f"{name} missing from the installed package at {path}"
    assert path.stat().st_size > 0, f"{name} is empty"
