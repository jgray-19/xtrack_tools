"""Tests for line helpers."""

from __future__ import annotations

import numpy as np
import pytest
import xpart as xp
import xtrack as xt

from xtrack_tools.line import make_element_thin


@pytest.fixture
def thick_line() -> xt.Line:
    """A short line whose ``tkicker`` is a thick quadrupole at a known centre."""
    line = xt.Line(
        elements=[
            xt.Drift(length=2.0),
            xt.Quadrupole(length=1.0, k1=0.2),
            xt.Drift(length=2.0),
        ],
        element_names=["drift1", "tkicker", "drift2"],
    )
    line.particle_ref = xt.Particles(mass=xp.PROTON_MASS_EV, energy0=450e9)
    return line


def test_make_element_thin_replaces_thick_element_with_centred_marker(thick_line: xt.Line):
    """Test a thick element becomes a zero-length marker at its original centre."""
    original_length = thick_line.get_length()
    original_centre = float(thick_line.get_table()["s_center", "tkicker"])

    result = make_element_thin(thick_line, "tkicker")

    assert result == "tkicker"
    assert isinstance(thick_line["tkicker"]._get_viewed_object(), xt.Marker)
    assert np.isclose(thick_line.get_length(), original_length)
    assert np.isclose(thick_line.get_table()["s_center", "tkicker"], original_centre)


def test_make_element_thin_is_case_insensitive_and_returns_canonical_name(thick_line: xt.Line):
    """Test the upper-case MAD-X name resolves to the canonical (lower-case) name."""
    result = make_element_thin(thick_line, "TKICKER")

    assert result == "tkicker"
    assert isinstance(thick_line["tkicker"]._get_viewed_object(), xt.Marker)


def test_make_element_thin_leaves_already_thin_element_untouched(thick_line: xt.Line):
    """Test a zero-length element is returned unchanged without inserting a body drift."""
    thick_line.insert(thick_line.env.place(thick_line.env.new("mk", xt.Marker), at=1.0))
    original_names = tuple(thick_line.element_names)

    result = make_element_thin(thick_line, "mk")

    assert result == "mk"
    assert tuple(thick_line.element_names) == original_names
    assert not any("__thinned_body" in name for name in thick_line.element_names)


def test_make_element_thin_raises_for_unknown_element(thick_line: xt.Line):
    """Test thinning a missing element raises ValueError."""
    with pytest.raises(ValueError, match="not found in the line"):
        make_element_thin(thick_line, "does_not_exist")
