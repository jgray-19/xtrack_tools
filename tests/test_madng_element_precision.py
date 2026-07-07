"""Per-element numerical-precision comparison between xsuite and MAD-NG.

For each LHC element type the test builds a minimal single-element MAD-X
sequence, tracks one particle through it with xsuite and with MAD-NG
(``method=6``, a thick map), and asserts that the exit phase-space coordinates
agree to close to machine precision.

Motivation
----------
When xsuite quadrupoles are configured with ``mat-kick-mat`` (as
:func:`xtrack_tools.env._configure_line_models` does), the two codes disagree by
~1e-9 *relative* per element. ``mat-kick-mat`` uses the expanded (paraxial)
Hamiltonian: its error scales as the cube of the transverse angle and does *not*
shrink with more slices, because it is a formulation error, not an integration
error. Across a full ring this accumulates into a ~5e-7 tune difference and a
~1e-8 orbit difference, dominated by the plane with the largest angles (the
crossing plane through the triplet).

MAD-NG integrates the *exact* Hamiltonian. To reproduce its thick map to machine
precision, xsuite must use:

* ``drift-kick-drift-exact`` for quadrupoles / sextupoles / octupoles /
  multipoles, with enough slices (``num_multipole_kicks``) to converge to the
  thick map, and
* ``bend-kick-bend`` for bends (already exact to ~3e-14; the ``expanded`` /
  ``drift-kick-drift-exact`` bend cores are the *paraxial* ones and disagree at
  ~1e-9).

This test pins that matching configuration so regressions are caught.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xpart as xp
import xtrack as xt

pytest.importorskip("pymadng", reason="pymadng is required for the MAD-NG comparison")
from pymadng import MAD  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

# Beam / reference-particle definition shared by both codes.
KINETIC_ENERGY_GEV = 6800.0
TOTAL_ENERGY_GEV = KINETIC_ENERGY_GEV + xp.PROTON_MASS_EV / 1e9
SEQ_NAME = "lhcb1"
COORDS = ("x", "px", "y", "py")

# Initial condition with an offset *and* a finite angle in both planes, so that
# the paraxial-vs-exact Hamiltonian difference (which scales with the angle) is
# exercised. A pure on-axis particle would hide it.
INITIAL = {"x": 1.0e-3, "px": 1.0e-4, "y": 1.0e-3, "py": 1.0e-4}

# Slices used for the sliced (drift-kick-drift-exact) thick elements. MAD-NG uses
# a single thick map, so xsuite needs enough slices to converge to it.
NUM_SLICES = 256

# The two codes agree to floating-point precision once the models match. The
# floor is ~3e-14 (bends, near the float64 limit for these magnitudes); the
# non-bend elements sit at ~1e-18. This bound keeps a wide margin while still
# failing loudly if the paraxial ``mat-kick-mat`` model creeps back in (which
# would push the disagreement to ~1e-9).
ATOL = 1.0e-11


def _configure_exact_models(line: xt.Line) -> None:
    """Configure *line* with the models that reproduce MAD-NG's thick maps.

    Quadrupoles and higher multipoles use ``drift-kick-drift-exact`` (exact
    Hamiltonian) with many slices; bends use ``bend-kick-bend``; drifts are
    exact.
    """
    line.configure_drift_model(model="exact")
    table = line.get_table()
    for etype in ("Quadrupole", "Sextupole", "Octupole", "Multipole"):
        rows = table.rows[table.element_type == etype]
        if len(rows.name):
            line.set(
                rows,
                model="drift-kick-drift-exact",
                integrator="yoshida4",
                num_multipole_kicks=NUM_SLICES,
            )
    if len(table.rows[table.element_type == "Bend"].name):
        line.configure_bend_model(
            core="bend-kick-bend",
            edge="full",
            integrator="yoshida4",
            num_multipole_kicks=NUM_SLICES,
        )


# Each case: (id, MAD-X element definition placed at s=4.0 inside a 10 m region).
ELEMENT_CASES: list[tuple[str, str]] = [
    ("quadrupole", "q: quadrupole, l:=2.0, k1:=9.0e-3, at=4.0;"),
    ("quadrupole_skew", "q: quadrupole, l:=2.0, k1s:=9.0e-3, at=4.0;"),
    ("sbend", "b: sbend, l:=5.0, angle:=8.0e-3, at=4.0;"),
    ("sbend_edges", "b: sbend, l:=5.0, angle:=8.0e-3, e1:=4.0e-3, e2:=4.0e-3, at=4.0;"),
    ("sbend_tilt", "b: sbend, l:=5.0, angle:=8.0e-3, tilt:=0.3, at=4.0;"),
    ("sextupole", "s: sextupole, l:=1.0, k2:=5.0e-2, at=4.0;"),
    ("octupole", "o: octupole, l:=1.0, k3:=1.0e-1, at=4.0;"),
    ("hkicker", "h: hkicker, l:=1.0, kick:=1.5e-5, at=4.0;"),
    ("vkicker", "v: vkicker, l:=1.0, kick:=-1.7e-5, at=4.0;"),
    ("thin_hkicker", "h: hkicker, kick:=1.5e-5, at=4.0;"),
]


def _write_sequence(tmp_path: Path, case_id: str, element_def: str) -> Path:
    """Write a minimal single-element MAD-X sequence to *tmp_path*."""
    seq_file = tmp_path / f"single_{case_id}.seq"
    seq_file.write_text(
        "\n".join(
            [
                f"{SEQ_NAME}: sequence, l=10.0, refer=entry;",
                "  start: marker, at=0.0;",
                f"  {element_def}",
                "  finish: marker, at=10.0;",
                "endsequence;",
                "",
            ]
        )
    )
    return seq_file


def _xsuite_exit(seq_file: Path) -> np.ndarray:
    """Track one particle through *seq_file* in xsuite and return exit coords."""
    env = xt.load(file=seq_file, _rbend_correct_k0=True, format="madx")
    line = env[SEQ_NAME]
    line.particle_ref = xt.Particles(
        mass=xp.PROTON_MASS_EV, kinetic_energy0=KINETIC_ENERGY_GEV * 1e9
    )
    _configure_exact_models(line)

    particle = line.build_particles(zeta=0.0, delta=0.0, **INITIAL)
    line.track(particle)
    return np.array([particle.x[0], particle.px[0], particle.y[0], particle.py[0]], dtype=float)


def _madng_exit(seq_file: Path, tmp_path: Path) -> np.ndarray:
    """Track one particle through *seq_file* in MAD-NG (method 6) and return exit coords."""
    mad = MAD(stdout=str(tmp_path / "madng.log"))
    mad.send("MADX.option.rbarc = false")
    mad.send(f'MADX:load("{seq_file}", "{tmp_path / "single.mad"}", {{rbarc=false}})')
    mad.send(f"loaded_sequence = MADX.{SEQ_NAME}")
    mad.send(f'loaded_sequence.beam = beam {{ particle="proton", energy={TOTAL_ENERGY_GEV:.15e} }}')
    mad.send(
        """
x0 = py:recv()
tbl, flw = track { sequence=loaded_sequence, X0=x0, nturn=1, observe=0, method=6 }
py:send(flw.tpar == flw.npar)
"""
    ).send(
        [
            {
                "x": INITIAL["x"],
                "px": INITIAL["px"],
                "y": INITIAL["y"],
                "py": INITIAL["py"],
                "t": 0.0,
                "pt": 0.0,
            }
        ]
    )
    if not mad.recv():
        raise RuntimeError("MAD-NG lost the particle while tracking a single element")
    row = mad.tbl.to_df(force_pandas=True).iloc[-1]
    return np.array([row["x"], row["px"], row["y"], row["py"]], dtype=float)


@pytest.mark.parametrize(
    ("case_id", "element_def"),
    ELEMENT_CASES,
    ids=[case[0] for case in ELEMENT_CASES],
)
def test_madng_matches_xsuite_per_element(
    tmp_path: Path,
    case_id: str,
    element_def: str,
) -> None:
    """Each LHC element type transports one particle identically in both codes."""
    seq_file = _write_sequence(tmp_path, case_id, element_def)

    xsuite_exit = _xsuite_exit(seq_file)
    madng_exit = _madng_exit(seq_file, tmp_path)

    diff = madng_exit - xsuite_exit
    assert np.max(np.abs(diff)) < ATOL, (
        f"{case_id}: xsuite/MAD-NG exit coordinates differ by "
        f"{np.max(np.abs(diff)):.3e} (>{ATOL:.0e})\n"
        + "\n".join(
            f"  {c}: xsuite={xsuite_exit[i]:+.9e} madng={madng_exit[i]:+.9e} diff={diff[i]:+.3e}"
            for i, c in enumerate(COORDS)
        )
    )
