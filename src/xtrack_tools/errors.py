"""Seeded magnet-error injection for xsuite lines.

These helpers add reproducible imperfections to an :class:`xtrack.Line` so that
tracking studies can exercise a distorted closed orbit. They are deliberately
minimal and side-effecting (they mutate the line in place) and return the applied
values so the *same* imperfection can be mirrored onto a matching model (e.g. a
MAD-NG twiss model) when required.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xtrack as xt

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


def _iter_named_elements(line: xt.Line, name_prefix: str | None) -> Iterator[tuple[str, object]]:
    """Yield ``(name, element)`` pairs, optionally filtered by a name prefix."""
    lowered = name_prefix.lower() if name_prefix is not None else None
    for name in line.element_names:
        if lowered is not None and not str(name).lower().startswith(lowered):
            continue
        yield str(name), line[name]


def apply_relative_bend_field_errors(
    line: xt.Line,
    *,
    rms: float,
    seed: int,
    name_prefix: str | None = None,
) -> dict[str, float]:
    """Apply a seeded relative dipole field error to every powered bend in place.

    A field error scales the dipole field ``k0`` by ``(1 + rel_error)`` while
    leaving the reference geometry ``h = angle / length`` unchanged — the physical
    meaning of, say, a 0.08% bend error. Powered bends are those with a non-zero
    ``angle``; the geometry is read from ``angle`` / ``length`` and the native
    ``k0`` field is set to ``(angle / length) * (1 + rel_error)``.

    Perturbing the native ``k0`` (rather than adding a separate ``knl[0]``
    multipole) is what makes an xsuite line and a matching MAD-NG model agree to
    the codes' floor (~1e-8): a stand-alone dipole multipole inside a *curved*
    element is transported differently by the two codes (~0.1% orbit
    disagreement), whereas the native dipole field is handled identically.

    Args:
        line: Line to modify in place.
        rms: Standard deviation of the relative field error (e.g. ``8e-4``).
        seed: Seed for the ``numpy`` random generator (reproducible).
        name_prefix: If given, only elements whose name starts with this prefix
            (case-insensitive) are considered.

    Returns:
        ``{name: k0_after}`` for every perturbed bend, so the identical absolute
        ``k0`` values can be written onto a matching model.
    """
    rng = np.random.default_rng(seed)
    new_k0: dict[str, float] = {}
    for name, element in _iter_named_elements(line, name_prefix):
        try:
            angle = float(getattr(element, "angle", 0.0))
            length = float(getattr(element, "length", 0.0))
        except (TypeError, ValueError):
            continue
        if angle == 0.0 or length == 0.0:
            continue
        k0 = (angle / length) * (1.0 + float(rng.normal(0.0, rms)))
        element.k0 = k0
        new_k0[name] = k0
    logger.info(
        "Applied relative bend field error (rms=%.2e, seed=%d) to %d bends",
        rms, seed, len(new_k0),
    )
    return new_k0


def apply_vertical_quad_misalignment(
    line: xt.Line,
    *,
    rms: float,
    seed: int,
    name_prefix: str | None = None,
) -> dict[str, float]:
    """Apply a seeded vertical misalignment to every quadrupole in place.

    Each :class:`xtrack.Quadrupole` is shifted vertically by an independent draw
    from ``N(0, rms)`` via its ``shift_y`` attribute, distorting the vertical
    closed orbit.

    Args:
        line: Line to modify in place.
        rms: Standard deviation of the vertical offset in metres (e.g. ``2e-4``
            for 0.2 mm).
        seed: Seed for the ``numpy`` random generator (reproducible).
        name_prefix: If given, only elements whose name starts with this prefix
            (case-insensitive) are considered.

    Returns:
        ``{name: shift_y}`` for every misaligned quadrupole.
    """
    rng = np.random.default_rng(seed)
    shifts: dict[str, float] = {}
    for name, element in _iter_named_elements(line, name_prefix):
        if not isinstance(element, xt.Quadrupole):
            continue
        shift_y = float(rng.normal(0.0, rms))
        element.shift_y = shift_y
        shifts[name] = shift_y
    logger.info(
        "Applied vertical quad misalignment (rms=%.2e, seed=%d) to %d quadrupoles",
        rms, seed, len(shifts),
    )
    return shifts


def apply_quad_tilt(
    line: xt.Line,
    *,
    rms: float,
    seed: int,
    name_prefix: str | None = None,
) -> dict[str, float]:
    """Apply a seeded roll about the ``s`` axis to every quadrupole in place.

    Each :class:`xtrack.Quadrupole` is rolled by an independent draw from
    ``N(0, rms)`` via its ``rot_s_rad`` attribute. A rolled quadrupole carries a
    skew component ``~ 2 * theta * k1l``, which couples the horizontal dispersion
    into the vertical plane. This is the cheapest physical way to give a tracking
    line a genuine vertical dispersion that a nominal (untilted) model does not
    know about -- unlike :func:`apply_vertical_quad_misalignment`, which kicks the
    vertical *orbit* but leaves ``Dy`` essentially untouched.

    Args:
        line: Line to modify in place.
        rms: Standard deviation of the roll angle in radians (e.g. ``5e-4`` for
            0.5 mrad).
        seed: Seed for the ``numpy`` random generator (reproducible).
        name_prefix: If given, only elements whose name starts with this prefix
            (case-insensitive) are considered.

    Returns:
        ``{name: rot_s_rad}`` for every tilted quadrupole.
    """
    rng = np.random.default_rng(seed)
    tilts: dict[str, float] = {}
    for name, element in _iter_named_elements(line, name_prefix):
        if not isinstance(element, xt.Quadrupole):
            continue
        tilt = float(rng.normal(0.0, rms))
        element.rot_s_rad = tilt
        tilts[name] = tilt
    logger.info(
        "Applied quadrupole tilt (rms=%.2e rad, seed=%d) to %d quadrupoles",
        rms,
        seed,
        len(tilts),
    )
    return tilts


def correct_closed_orbit(
    line: xt.Line,
    *,
    monitor_prefix: str,
    corrector_prefix_x: str,
    corrector_prefix_y: str,
    n_singular_values: int | None = None,
) -> dict[str, float]:
    """Steer the closed orbit flat with the machine's own orbit correctors.

    Real error lattices are steered by operators before measurement, leaving a
    few mm of residual orbit. Note the correctors are dipoles, so the corrected
    line carries their dispersion even though its closed orbit is flat.

    Args:
        line: Line to correct in place.
        monitor_prefix: Case-insensitive name prefix selecting the BPMs.
        corrector_prefix_x: Case-insensitive name prefix selecting the horizontal
            correctors.
        corrector_prefix_y: As above, vertical.
        n_singular_values: Truncate the correction SVD to this rank. ``None``
            uses every mode.

    Returns:
        ``{"x_before": .., "x_after": .., "y_before": .., "y_after": ..}`` closed
        orbit RMS at the BPMs in metres.
    """
    monitors, correctors_x, correctors_y = (
        [name for name, _ in _iter_named_elements(line, prefix)]
        for prefix in (monitor_prefix, corrector_prefix_x, corrector_prefix_y)
    )
    if not (monitors and correctors_x and correctors_y):
        raise ValueError(
            f"orbit correction found monitors={len(monitors)} "
            f"correctors_x={len(correctors_x)} correctors_y={len(correctors_y)}; "
            "check the prefixes"
        )

    def _orbit_rms(stage: str) -> dict[str, float]:
        tws = line.twiss(method="4d").rows[monitors]
        return {f"{p}_{stage}": float(np.sqrt(np.mean(tws[p] ** 2))) for p in ("x", "y")}

    rms = _orbit_rms("before")
    line.correct_trajectory(
        monitor_names_x=monitors,
        monitor_names_y=monitors,
        corrector_names_x=correctors_x,
        corrector_names_y=correctors_y,
        n_singular_values=n_singular_values,
    )
    rms |= _orbit_rms("after")
    logger.info("Orbit correction (%d BPMs): %s", len(monitors), rms)
    return rms
