from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xtrack as xt
from xobjects import ContextCpu as Context

from .action_angle import _build_coords_from_action_angle
from .env import create_xsuite_environment
from .line import get_element_s_centre, resolve_element_name
from .monitors import get_monitor_names_at_pattern, process_tracking_data
from .tracking import run_tracking

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

logger = logging.getLogger(__name__)


def state_marker_names(acd_marker: str) -> tuple[str, str]:
    """Return the ``(before, after)`` state-marker names for an AC-dipole marker."""
    return f"{acd_marker}_before", f"{acd_marker}_after"


def insert_ac_dipole(
    line: xt.Line,
    tws: xt.TwissTable,
    acd_marker: str,
    acd_ramp: int,
    total_turns: int,
    driven_tunes: list[float],
    lag: float = 0.0,
    insert_state_markers: bool = False,
) -> xt.Line:
    """Insert horizontal and vertical AC dipoles at the AC dipole marker.

    Args:
        line: Base line to copy and modify.
        tws: Twiss table used for beta functions at the AC dipole.
        acd_marker: Name of the marker element at which to place the AC dipoles.
        acd_ramp: Number of ramp turns.
        total_turns: Total tracking turns at flat top.
        driven_tunes: Driven tunes (horizontal, vertical).
        lag: Phase lag for the AC dipoles.
        insert_state_markers: If ``True``, also insert zero-length markers
            ``{acd_marker}_before`` and ``{acd_marker}_after`` a tiny step (1e-8 m)
            up- and downstream of the AC-dipole kicks, so they unambiguously
            record the true pre-kick and post-kick momenta.

    Returns:
        A new line with AC dipole elements inserted.
    """
    line = line.copy()
    betxac = tws.rows[acd_marker]["betx"]
    betyac = tws.rows[acd_marker]["bety"]
    logger.info(
        f"Inserting AC dipole at {acd_marker} with betx={betxac}, bety={betyac} and driven tunes {driven_tunes}"
    )

    driven_tunes = [q % 1 for q in driven_tunes]
    qxd_qx = driven_tunes[0] - tws["qx"] % 1
    qyd_qy = driven_tunes[1] - tws["qy"] % 1
    logger.info(f"Qxd/Qx: {qxd_qx}, Qyd/Qy: {qyd_qy}")
    pbeam = line.particle_ref.p0c / 1e9

    line.env.elements[f"{acd_marker}_x"] = xt.ACDipole(
        plane="x",
        volt=2 * 0.042 * pbeam * abs(qxd_qx) / np.sqrt(180.0 * betxac),
        freq=driven_tunes[0],
        lag=lag,
        ramp=[0, acd_ramp, total_turns, total_turns + acd_ramp],
    )
    line.env.elements[f"{acd_marker}_y"] = xt.ACDipole(
        plane="y",
        volt=2 * 0.042 * pbeam * abs(qyd_qy) / np.sqrt(177.0 * betyac),
        freq=driven_tunes[1],
        lag=lag,
        ramp=[0, acd_ramp, total_turns, total_turns + acd_ramp],
    )
    placement = get_element_s_centre(line, acd_marker)
    if insert_state_markers:
        before_name, after_name = state_marker_names(acd_marker)
        line.env.elements[before_name] = xt.Marker()
        line.env.elements[after_name] = xt.Marker()
        # Offset the markers by a tiny step so they sit unambiguously up- and
        # downstream of the kicks (a 1e-8 m drift is negligible for the state).
        line.insert(before_name, at=placement - 1e-8)
        line.insert(after_name, at=placement + 1e-8)
    line.insert(f"{acd_marker}_x", at=placement)
    line.insert(f"{acd_marker}_y", at=placement)
    return line


def prepare_acd_line_with_monitors(
    line: xt.Line,
    tws: xt.TwissTable | None,
    acd_marker: str,
    ramp_turns: int,
    flattop_turns: int,
    driven_tunes: list[float],
    lag: float,
    bpm_pattern: str,
    insert_state_markers: bool = False,
) -> tuple[xt.Line, int, list[str]]:
    """Insert AC dipole and prepare multi-element BPM monitoring.

    Args:
        line: Base line to copy and modify.
        tws: Optional twiss table; computed if not provided.
        acd_marker: Name of the marker element at which to place the AC dipoles.
        ramp_turns: Number of ramp turns.
        flattop_turns: Number of flat-top turns.
        driven_tunes: Driven tunes (horizontal, vertical).
        lag: Phase lag for the AC dipoles.
        bpm_pattern: Regex pattern for BPM locations.

    Returns:
        A tuple of ``(tracked_line, total_turns, monitor_names)``.
    """
    logger.info(
        "Preparing AC-dipole tracking line with ramp_turns=%d flattop_turns=%d bpm_pattern='%s'",
        ramp_turns,
        flattop_turns,
        bpm_pattern,
    )
    total_turns = ramp_turns + flattop_turns
    if tws is None:
        logger.info("No Twiss table provided, computing one from the input line")
        tws = line.twiss(method="4d")
    tracked_line = insert_ac_dipole(
        line=line,
        tws=tws,
        acd_marker=acd_marker,
        acd_ramp=ramp_turns,
        total_turns=total_turns,
        driven_tunes=driven_tunes,
        lag=lag,
        insert_state_markers=insert_state_markers,
    )
    monitor_names = get_monitor_names_at_pattern(tracked_line, bpm_pattern)
    if insert_state_markers:
        monitor_names = [*monitor_names, *state_marker_names(acd_marker)]
    logger.info(
        "Prepared AC-dipole line with %d total turns and %d monitor points",
        total_turns,
        len(monitor_names),
    )
    return tracked_line, total_turns, monitor_names


def run_acd_twiss(
    line: xt.Line, acd_marker: str, dpp: float, driven_tunes: list[float]
) -> xt.TwissTable:
    """Run a twiss calculation with AC dipole elements inserted.

    Args:
        line: Base line to copy and modify.
        acd_marker: Name of the marker element at which to place the AC dipoles.
        dpp: Momentum deviation for twiss calculation.
        driven_tunes: Driven tunes (horizontal, vertical).

    Returns:
        A Twiss table including AC dipole elements.

    Raises:
        ValueError: If the AC dipole marker is not found.
    """
    if acd_marker not in line.element_names:
        raise ValueError(f"AC dipole marker '{acd_marker}' not found in the line.")
    line_acd = line.copy()
    before_acd_tws = line_acd.twiss(method="4d", delta0=dpp)

    bet_at_acdipole = before_acd_tws.rows[acd_marker]
    logger.info(
        f"Running twiss with AC dipole at {acd_marker} with betx={bet_at_acdipole['betx']}, bety={bet_at_acdipole['bety']}"
    )

    line_acd.env.elements[f"{acd_marker}_x"] = xt.ACDipole(
        plane="x",
        natural_q=before_acd_tws["qx"] % 1,
        freq=driven_tunes[0],
        beta_at_acdipole=bet_at_acdipole["betx"],
        twiss_mode=True,
    )
    line_acd.env.elements[f"{acd_marker}_y"] = xt.ACDipole(
        plane="y",
        natural_q=before_acd_tws["qy"] % 1,
        freq=driven_tunes[1],
        beta_at_acdipole=bet_at_acdipole["bety"],
        twiss_mode=True,
    )

    placement = get_element_s_centre(line_acd, acd_marker)
    line_acd.insert(f"{acd_marker}_x", at=placement)
    line_acd.insert(f"{acd_marker}_y", at=placement)
    return line_acd.twiss(method="4d", delta0=dpp)


def run_acd_track(
    sequence_file: Path,
    acd_marker: str,
    sequence_name: str,
    kinetic_energy: float = 6800,
    delta_p: float = 0.0,
    ramp_turns: int = 1000,
    flattop_turns: int = 100,
    driven_tunes: list[float] | None = None,
    bpm_pattern: str = r"(?i)bpm.*",
    json_path: Path | None = None,
    add_variance_columns: bool = True,
    replace_thick_monitors_with_thin: bool = True,
    state_markers: bool = False,
) -> tuple[pd.DataFrame, xt.TwissTable, xt.Line]:
    """Run AC dipole tracking for a sequence file and return tracking data.

    Args:
        sequence_file: Path to the MAD-X sequence file.
        acd_marker: Marker name for the AC dipole element.
        sequence_name: Name of the sequence in the environment.
        kinetic_energy: Kinetic energy of the particle in GeV.
        delta_p: Momentum deviation for the tracked particle.
        kinetic_energy: Beam kinetic energy in GeV used to set the particle reference.
        ramp_turns: Number of ramp turns.
        flattop_turns: Number of flat-top turns.
        driven_tunes: Driven tunes (horizontal, vertical). Defaults to a typical pair.
        bpm_pattern: Regex pattern for BPM locations.
        json_path: Optional JSON cache path.
        add_variance_columns: Whether to add default variance columns to the output.
        replace_thick_monitors_with_thin: If ``True``, replace thick monitored
            elements with thin monitor points before tracking. Thin monitors are
            left unchanged.
        state_markers: If ``True``, insert and monitor ``{acd_marker}_before`` and
            ``{acd_marker}_after`` markers bracketing the AC-dipole kicks, so the
            tracked data carries the true pre-kick and post-kick momenta.

    Returns:
        Tuple of ``(tracking_df, twiss_table, monitored_line)``.
    """
    if driven_tunes is None:
        driven_tunes = [0.27, 0.322]
    logger.info(
        "Running AC-dipole tracking from sequence %s for element %s with delta_p=%s, using acd_marker='%s' with driven_tunes=%s",
        sequence_file,
        acd_marker,
        delta_p,
        acd_marker,
        driven_tunes,
    )

    env = create_xsuite_environment(
        json_file=json_path,
        sequence_file=sequence_file,
        kinetic_energy=kinetic_energy,
        seq_name=sequence_name,
    )
    baseline_line: xt.Line = env[sequence_name].copy()
    acd_marker = resolve_element_name(baseline_line, acd_marker)
    tws_input: xt.TwissTable = baseline_line.twiss4d()

    qx = float(tws_input.qx % 1)
    qy = float(tws_input.qy % 1)
    logger.info(f"Natural tunes: Qx = {qx:.6f}, Qy = {qy:.6f}")
    if (
        not (np.isclose(qx, 0.28, atol=1e-3) and np.isclose(qy, 0.31, atol=1e-3))
        and "lhc" in sequence_name.lower()
    ):
        logger.warning(f"Tunes (Qx={qx:.6f}, Qy={qy:.6f}) differ from expected (0.28, 0.31)")

    monitored_line, total_turns, monitor_names = prepare_acd_line_with_monitors(
        line=baseline_line,
        tws=tws_input,
        acd_marker=acd_marker,
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
        driven_tunes=driven_tunes,
        lag=0.0,
        bpm_pattern=bpm_pattern,
        insert_state_markers=state_markers,
    )

    ctx = Context()
    particles: xt.Particles = monitored_line.build_particles(
        _context=ctx,
        x=0,
        y=0,
        px=0,
        py=0,
        delta=delta_p,
    )

    logger.info(f"Tracking {total_turns} turns with AC dipole")
    monitored_line = run_tracking(
        line=monitored_line,
        particles=particles,
        nturns=total_turns,
        monitor_names=monitor_names,
        replace_thick_monitors_with_thin=replace_thick_monitors_with_thin,
    )

    tracking_df = process_tracking_data(
        monitored_line, ramp_turns, flattop_turns, add_variance_columns
    )

    return tracking_df, tws_input, monitored_line

def run_ac_dipole_tracking_with_particles(
    line: xt.Line,
    acd_marker: str,
    sequence_name: str,
    tws: xt.TwissTable | None = None,
    ramp_turns: int = 1000,
    flattop_turns: int = 100,
    driven_tunes: list[float] | None = None,
    lag: float = 0.0,
    bpm_pattern: str = r"bpm.*[^k]",
    particle_coords: dict[str, list[float]] | None = None,
    action_list: list[float] | None = None,
    angle_list: list[float] | None = None,
    use_diagonal_kicks: bool = True,
    start_marker: str | None = None,
    delta_values: list[float] | None = None,
    replace_thick_monitors_with_thin: bool = True,
    state_markers: bool = False,
) -> xt.Line:
    """Track multiple particles with an AC dipole using explicit or action-angle inputs.

    Args:
        line: Base line to copy and modify.
        tws: Optional twiss table; computed if not provided.
        acd_marker: Marker name for the AC dipole element.
        sequence_name: Name of the sequence in the environment.
        ramp_turns: Number of ramp turns.
        flattop_turns: Number of flat-top turns.
        driven_tunes: Driven tunes (horizontal, vertical). Defaults to a typical pair.
        lag: Phase lag for the AC dipoles.
        bpm_pattern: Regex pattern for BPM locations.
        particle_coords: Explicit coordinates for each particle.
        action_list: Action values for initial conditions.
        angle_list: Angle values for initial conditions.
        use_diagonal_kicks: If ``True``, initialize both planes.
        start_marker: Optional marker name to set as the first element.
        delta_values: Optional momentum offsets per particle.
        replace_thick_monitors_with_thin: If ``True``, replace thick monitored
            elements with thin monitor points before tracking. Thin monitors are
            left unchanged.
        state_markers: If ``True``, insert and monitor ``{acd_marker}_before`` and
            ``{acd_marker}_after`` markers bracketing the AC-dipole kicks, so the
            tracked data carries the true pre-kick and post-kick momenta.

    Returns:
        The monitored line after tracking.

    Raises:
        ValueError: If neither explicit coordinates nor action-angle inputs are provided.
    """
    if driven_tunes is None:
        driven_tunes = [0.27, 0.322]
    acd_marker = resolve_element_name(line, acd_marker)
    logger.info(
        "Running AC-dipole particle tracking with ramp_turns=%d flattop_turns=%d bpm_pattern='%s'",
        ramp_turns,
        flattop_turns,
        bpm_pattern,
    )

    if particle_coords is not None:
        num_particles = len(particle_coords["x"])
        xs = particle_coords["x"]
        pxs = particle_coords["px"]
        ys = particle_coords["y"]
        pys = particle_coords["py"]
        deltas = particle_coords.get("delta", [0.0] * num_particles)
    elif action_list is not None and angle_list is not None:
        working_line = line.copy()
        if start_marker is not None:
            working_line.cycle(name_first_element=start_marker.lower(), inplace=True)
            start_elem = working_line.element_names[0].upper()
        else:
            start_elem = working_line.element_names[0].upper()

        if tws is None:
            tws = working_line.twiss4d()
        xs, pxs, ys, pys = _build_coords_from_action_angle(
            action_list,
            angle_list,
            tws,
            use_diagonal_kicks,
            start_elem,
        )
        num_particles = len(xs)
        deltas = delta_values if delta_values is not None else [0.0] * num_particles
    else:
        raise ValueError("Provide particle_coords or both action_list and angle_list")

    monitored_line, total_turns, monitor_names = prepare_acd_line_with_monitors(
        line=line,
        tws=tws,
        acd_marker=acd_marker,
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
        driven_tunes=driven_tunes,
        lag=lag,
        bpm_pattern=bpm_pattern,
        insert_state_markers=state_markers,
    )

    ctx = Context()
    particles = monitored_line.build_particles(
        _context=ctx,
        x=xs,
        px=pxs,
        y=ys,
        py=pys,
        delta=deltas,
    )

    logger.info(f"Tracking {num_particles} particles for {total_turns} turns")
    return run_tracking(
        line=monitored_line,
        particles=particles,
        nturns=total_turns,
        monitor_names=monitor_names,
        replace_thick_monitors_with_thin=replace_thick_monitors_with_thin,
    )
