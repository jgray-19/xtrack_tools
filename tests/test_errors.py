"""Tests for the seeded magnet-error injection helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xtrack as xt

if TYPE_CHECKING:
    from pathlib import Path

from xtrack_tools.env import create_xsuite_environment
from xtrack_tools.errors import (
    apply_quad_tilt,
    apply_relative_bend_field_errors,
    apply_vertical_quad_misalignment,
    correct_closed_orbit,
)

PSB_SEQ_NAME = "psb3"
PSB_KINETIC_ENERGY_GEV = 0.160
# The 16 powered PSB main bends are two half-bends each, named ``br.bhz*``.
MAIN_BEND_PREFIX = "br.bhz"
QUAD_PREFIX = "br.q"


@pytest.fixture
def psb_line(seq_psb: Path) -> xt.Line:
    """A fresh PSB ring-3 line built from the test sequence."""
    env = create_xsuite_environment(
        sequence_file=seq_psb,
        kinetic_energy=PSB_KINETIC_ENERGY_GEV,
        seq_name=PSB_SEQ_NAME,
        json_file=seq_psb.parent / f"{seq_psb.stem}.json",
    )
    return env[PSB_SEQ_NAME].copy()


def _bend_geometry(line: xt.Line, prefix: str) -> dict[str, float]:
    """Return ``{name: angle/length}`` for every powered bend under *prefix*."""
    out: dict[str, float] = {}
    for name in line.element_names:
        if not str(name).lower().startswith(prefix):
            continue
        element = line[name]
        angle = float(getattr(element, "angle", 0.0))
        length = float(getattr(element, "length", 0.0))
        if angle != 0.0 and length != 0.0:
            out[str(name)] = angle / length
    return out


# ---------------------------------------------------------------------------
# Bend field errors
# ---------------------------------------------------------------------------


def test_bend_field_errors_are_reproducible(psb_line: xt.Line, seq_psb: Path):
    """The same seed produces identical ``k0`` values on a fresh line."""
    other = create_xsuite_environment(
        sequence_file=seq_psb,
        kinetic_energy=PSB_KINETIC_ENERGY_GEV,
        seq_name=PSB_SEQ_NAME,
        json_file=seq_psb.parent / f"{seq_psb.stem}.json",
    )[PSB_SEQ_NAME].copy()

    k0_a = apply_relative_bend_field_errors(psb_line, rms=8e-4, seed=7, name_prefix=MAIN_BEND_PREFIX)
    k0_b = apply_relative_bend_field_errors(other, rms=8e-4, seed=7, name_prefix=MAIN_BEND_PREFIX)

    assert k0_a
    assert k0_a.keys() == k0_b.keys()
    for name, value in k0_a.items():
        assert value == pytest.approx(k0_b[name], rel=0, abs=0)


def test_bend_field_errors_keep_geometry_and_scale(psb_line: xt.Line):
    """``k0`` is scaled by ``(1 + rel)`` while ``h = angle/length`` is unchanged."""
    geometry_before = _bend_geometry(psb_line, MAIN_BEND_PREFIX)
    rms = 8e-4

    new_k0 = apply_relative_bend_field_errors(
        psb_line, rms=rms, seed=3, name_prefix=MAIN_BEND_PREFIX
    )
    geometry_after = _bend_geometry(psb_line, MAIN_BEND_PREFIX)

    assert set(new_k0) == set(geometry_before)
    # Geometry (angle/length) is untouched.
    for name, h in geometry_before.items():
        assert geometry_after[name] == pytest.approx(h)
    # The relative field change has the requested scale.
    rel = np.array([new_k0[name] / geometry_before[name] - 1.0 for name in new_k0])
    assert np.any(rel != 0.0)
    assert np.std(rel) == pytest.approx(rms, rel=0.6)


def test_bend_field_errors_prefix_filters(psb_line: xt.Line):
    """Only elements matching the prefix are perturbed."""
    new_k0 = apply_relative_bend_field_errors(
        psb_line, rms=8e-4, seed=1, name_prefix=MAIN_BEND_PREFIX
    )
    assert new_k0
    assert all(name.lower().startswith(MAIN_BEND_PREFIX) for name in new_k0)


def test_bend_field_errors_distort_horizontal_orbit(psb_line: xt.Line):
    """A field error opens up a non-zero horizontal closed orbit."""
    nominal = psb_line.twiss(method="4d")
    assert float(np.abs(nominal.x).max()) < 1e-6

    apply_relative_bend_field_errors(psb_line, rms=8e-4, seed=7, name_prefix=MAIN_BEND_PREFIX)
    distorted = psb_line.twiss(method="4d")
    assert float(np.abs(distorted.x).max()) > 1e-4


# ---------------------------------------------------------------------------
# Vertical quad misalignment
# ---------------------------------------------------------------------------


def test_quad_misalignment_sets_shift_y_on_quads_only(psb_line: xt.Line):
    """Only quadrupoles receive a ``shift_y``; the returned names are all quads."""
    shifts = apply_vertical_quad_misalignment(psb_line, rms=2e-4, seed=5, name_prefix=QUAD_PREFIX)

    assert shifts
    for name, value in shifts.items():
        element = psb_line[name]
        assert isinstance(element, xt.Quadrupole)
        assert float(element.shift_y) == pytest.approx(value)


def test_quad_misalignment_is_reproducible(psb_line: xt.Line, seq_psb: Path):
    """The same seed produces identical vertical offsets on a fresh line."""
    other = create_xsuite_environment(
        sequence_file=seq_psb,
        kinetic_energy=PSB_KINETIC_ENERGY_GEV,
        seq_name=PSB_SEQ_NAME,
        json_file=seq_psb.parent / f"{seq_psb.stem}.json",
    )[PSB_SEQ_NAME].copy()

    shifts_a = apply_vertical_quad_misalignment(psb_line, rms=2e-4, seed=5)
    shifts_b = apply_vertical_quad_misalignment(other, rms=2e-4, seed=5)

    assert shifts_a.keys() == shifts_b.keys()
    for name, value in shifts_a.items():
        assert value == pytest.approx(shifts_b[name], rel=0, abs=0)


def test_quad_misalignment_has_expected_scale(psb_line: xt.Line):
    """The vertical offsets have the requested RMS."""
    rms = 2e-4
    shifts = apply_vertical_quad_misalignment(psb_line, rms=rms, seed=5, name_prefix=QUAD_PREFIX)
    values = np.array(list(shifts.values()))
    assert values.size > 5
    assert np.std(values) == pytest.approx(rms, rel=0.6)


def test_quad_misalignment_distorts_vertical_orbit(psb_line: xt.Line):
    """A vertical misalignment opens up a non-zero vertical closed orbit."""
    nominal = psb_line.twiss(method="4d")
    assert float(np.abs(nominal.y).max()) < 1e-6

    apply_vertical_quad_misalignment(psb_line, rms=2e-4, seed=5, name_prefix=QUAD_PREFIX)
    distorted = psb_line.twiss(method="4d")
    assert float(np.abs(distorted.y).max()) > 1e-4


# ---------------------------------------------------------------------------
# Quad tilt
# ---------------------------------------------------------------------------


def test_quad_tilt_sets_rot_s_rad_on_quads_only(psb_line: xt.Line):
    """Only quadrupoles are rolled, with the requested RMS."""
    rms = 5e-4
    tilts = apply_quad_tilt(psb_line, rms=rms, seed=11, name_prefix=QUAD_PREFIX)

    assert len(tilts) > 5
    for name, value in tilts.items():
        element = psb_line[name]
        assert isinstance(element, xt.Quadrupole)
        assert float(element.rot_s_rad) == pytest.approx(value)
    assert np.std(list(tilts.values())) == pytest.approx(rms, rel=0.6)


def test_quad_tilt_creates_vertical_dispersion_and_coupling(psb_line: xt.Line):
    """Rolled quads couple Dx into Dy and open |c-| without moving the orbit."""
    nominal = psb_line.twiss(method="4d")
    assert float(np.abs(nominal.dy).max()) < 1e-9
    assert nominal.c_minus < 1e-6

    apply_quad_tilt(psb_line, rms=5e-4, seed=11, name_prefix=QUAD_PREFIX)
    tilted = psb_line.twiss(method="4d")

    # Pure skew-quad errors on a flat orbit do not steer the beam.
    assert float(np.abs(tilted.y).max()) < 1e-9
    # Dy ~ theta * k1l * Dx * beta: a few mm for PSB at 0.5 mrad.
    assert float(np.abs(tilted.dy).max()) > 1e-3
    assert tilted.c_minus > 1e-4


# ---------------------------------------------------------------------------
# Closed orbit correction
# ---------------------------------------------------------------------------

BPM_PREFIX = "br3.bpm"
HCORR_PREFIX = "br3.dhz"
VCORR_PREFIX = "br3.dvt"


def test_correct_closed_orbit_flattens_orbit(psb_line: xt.Line):
    """Correcting a distorted orbit reduces the BPM RMS in both planes."""
    apply_relative_bend_field_errors(psb_line, rms=8e-4, seed=7, name_prefix=MAIN_BEND_PREFIX)
    apply_vertical_quad_misalignment(psb_line, rms=2e-4, seed=5, name_prefix=QUAD_PREFIX)

    rms = correct_closed_orbit(
        psb_line,
        monitor_prefix=BPM_PREFIX,
        corrector_prefix_x=HCORR_PREFIX,
        corrector_prefix_y=VCORR_PREFIX,
    )

    assert rms["x_before"] > 1e-4
    assert rms["y_before"] > 1e-4
    assert rms["x_after"] < 0.2 * rms["x_before"]
    assert rms["y_after"] < 0.2 * rms["y_before"]
    # The returned numbers are the real orbit at the BPMs.
    tws = psb_line.twiss(method="4d")
    bpms = [n for n in tws.name if str(n).startswith(BPM_PREFIX)]
    assert np.sqrt(np.mean(tws.rows[bpms].x ** 2)) == pytest.approx(rms["x_after"])


def test_correct_closed_orbit_rejects_bad_prefix(psb_line: xt.Line):
    """A prefix that matches nothing raises instead of silently doing nothing."""
    with pytest.raises(ValueError, match="check the prefixes"):
        correct_closed_orbit(
            psb_line,
            monitor_prefix="no_such_bpm",
            corrector_prefix_x=HCORR_PREFIX,
            corrector_prefix_y=VCORR_PREFIX,
        )
