from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from xobjects import ContextCpu as Context

from xtrack_tools.coordinates import create_initial_conditions

from .action_angle import _build_coords_from_action_angle
from .monitors import (
    _normalise_recorded_multi_element_monitor_names,
    get_monitor_names_at_pattern,
    replace_thick_monitors_with_thin_markers,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import tfs
    import xtrack as xt


def run_tracking(
    line: xt.Line,
    particles: xt.Particles,
    nturns: int,
    monitor_names: list[str] | None = None,
    replace_thick_monitors_with_thin: bool = True,
) -> xt.Line:
    """Track particles through a copied line for a fixed number of turns.

    Args:
        line: The xsuite line to copy and track through.
        particles: Particles to track.
        nturns: Number of turns to track.
        monitor_names: Element names to pass to ``multi_element_monitor_at``.
        replace_thick_monitors_with_thin: If ``True``, insert thin markers at
            the centres of thick monitored elements before tracking while
            keeping the recorded monitor names mapped back to the originals.
            Thin monitors are left unchanged.

    Returns:
        A tracked copy of ``line``. The input line is not modified.

    Raises:
        RuntimeError: If tracking ends with non-successful particle state.
    """
    logger.info(
        "Starting tracking for %d turns with %d particles and %d requested monitor points",
        nturns,
        len(particles.x),
        0 if monitor_names is None else len(monitor_names),
    )
    tracked_line = line.copy()
    monitor_name_aliases: dict[str, str] | None = None
    if monitor_names:
        if replace_thick_monitors_with_thin:
            logger.info("Checking whether requested monitor elements need thick-to-thin conversion")
            monitor_names, monitor_name_aliases = replace_thick_monitors_with_thin_markers(
                tracked_line, monitor_names
            )
        else:
            logger.info("Tracking with requested monitor elements without thick-to-thin conversion")
        tracked_line.track(
            particles,
            num_turns=nturns,
            with_progress=True,
            multi_element_monitor_at=monitor_names,
        )
        _normalise_recorded_multi_element_monitor_names(
            tracked_line, alias_map=monitor_name_aliases
        )
    else:
        logger.info("Tracking without multi-element monitor recording")
        tracked_line.track(particles, num_turns=nturns, with_progress=True)
    if np.all(np.asarray(particles.state) == 1):
        logger.info("Tracking completed successfully")
        return tracked_line
    raise RuntimeError("Tracking failed. Please check the input parameters.")


def run_tracking_without_ac_dipole(
    line: xt.Line,
    tws: xt.TwissTable,
    flattop_turns: int,
    bpm_pattern: str = r"bpm.*[^k]",
    particle_coords: dict[str, list[float]] | None = None,
    action_list: list[float] | None = None,
    angle_list: list[float] | None = None,
    use_diagonal_kicks: bool = True,
    start_marker: str | None = None,
    deltas: list[float] | float | None = None,
    context: Context | None = None,
    replace_thick_monitors_with_thin: bool = True,
) -> xt.Line:
    """Track particles without an AC dipole using a multi-element BPM monitor.

    Args:
        line: Base line to copy and track.
        tws: Twiss data used to convert action-angle to coordinates.
        flattop_turns: Number of turns to track at flat top.
        bpm_pattern: Regex pattern for BPM element names.
        particle_coords: Explicit particle coordinate arrays.
        action_list: Action values used with ``angle_list`` to build coordinates.
        angle_list: Angle values used with ``action_list`` to build coordinates.
        use_diagonal_kicks: If ``True``, use action-angle in both planes.
        start_marker: Optional marker name to set as the first element.
        deltas: Optional momentum offsets per particle.
        context: Optional xobjects context.
        replace_thick_monitors_with_thin: If ``True``, replace thick monitored
            elements with thin monitor points before tracking. Thin monitors are
            left unchanged.

    Returns:
        The monitored line after tracking.

    Raises:
        ValueError: If neither explicit coordinates nor action-angle inputs are provided.
    """

    logger.info(
        "Running tracking without AC dipole for %d turns using bpm_pattern='%s'",
        flattop_turns,
        bpm_pattern,
    )
    working_line = line.copy()
    if start_marker is not None:
        working_line.cycle(name_first_element=start_marker.lower(), inplace=True)
        logger.info("Cycled working line to start at marker %s", start_marker)

    start_elem = working_line.element_names[0].upper()
    logger.debug("Using default starting element %s", start_elem)
    if deltas is None:
        deltas = 0.0


    if particle_coords is not None:
        num_particles = len(particle_coords["x"])
        xs = particle_coords["x"]
        pxs = particle_coords["px"]
        ys = particle_coords["y"]
        pys = particle_coords["py"]
        logger.info("Using explicit particle coordinates for %d particles", num_particles)
    elif action_list is not None and angle_list is not None:
        xs, pxs, ys, pys = _build_coords_from_action_angle(
            action_list,
            angle_list,
            tws,
            use_diagonal_kicks,
            start_elem,
        )
        num_particles = len(xs)
        logger.info("Built particle coordinates from action-angle data for %d particles", num_particles)
    else:
        raise ValueError("Provide particle_coords or both action_list and angle_list")

    monitor_names = get_monitor_names_at_pattern(working_line, bpm_pattern)

    ctx = context or Context()
    particles = working_line.build_particles(
        _context=ctx,
        x=xs,
        px=pxs,
        y=ys,
        py=pys,
        delta=deltas,
    )

    return run_tracking(
        line=working_line,
        particles=particles,
        nturns=flattop_turns,
        monitor_names=monitor_names,
        replace_thick_monitors_with_thin=replace_thick_monitors_with_thin,
    )


def start_tracking_xsuite_batch(
    env: xt.Environment,
    batch_start: int,
    batch_end: int,
    action_list: list[float],
    angle_list: list[float],
    twiss_data: tfs.TfsDataFrame,
    use_diagonal_kicks: bool,
    flattop_turns: int,
    progress_interval: int,
    num_tracks: int,
    true_deltap: float,
    seq_name: str,
    replace_thick_monitors_with_thin: bool = True,
) -> xt.Line:
    """Track a batch of particles defined by action-angle coordinates.

    Args:
        env: xsuite environment containing the target sequence.
        batch_start: Inclusive start index for the batch.
        batch_end: Exclusive end index for the batch.
        action_list: Action values for initial conditions.
        angle_list: Angle values for initial conditions.
        twiss_data: Twiss data table for coordinate conversion.
        use_diagonal_kicks: If ``True``, initialize both planes.
        flattop_turns: Number of turns to track.
        progress_interval: Logging interval for batch progress.
        num_tracks: Total number of tracks for progress reporting.
        true_deltap: Momentum deviation applied to all particles.
        seq_name: Sequence name in the environment.
        replace_thick_monitors_with_thin: If ``True``, replace thick monitored
            elements with thin monitor points before tracking. Thin monitors are
            left unchanged.

    Returns:
        The tracked line with multi-element monitor data recorded.
    """

    logger.info(
        "Starting xsuite batch tracking for tracks [%d, %d) in sequence '%s'",
        batch_start,
        batch_end,
        seq_name,
    )
    line: xt.Line = env[seq_name]

    logger.info(
        "Building initial conditions for batch tracks [%d, %d) using action-angle coordinates",
        batch_start,
        batch_end,
    )
    actions = np.asarray(action_list[batch_start:batch_end], dtype=float)
    angles = np.asarray(angle_list[batch_start:batch_end], dtype=float)
    starting_bpm = line.element_names[0].upper()

    for ntrk in range(batch_start, batch_end):
        if ntrk % progress_interval == 0:
            logger.info(
                f"Starting tracking command for process {ntrk}/{num_tracks - 1} "
                f"({ntrk / num_tracks * 100:.1f}%)"
            )

    planes = ("xy",) if use_diagonal_kicks else ("x", "y")
    coord_blocks = [
        create_initial_conditions(
            actions,
            angles,
            twiss_data,
            kick_plane=plane,
            starting_bpm=starting_bpm,
        )
        for plane in planes
    ]
    x_list = np.concatenate(
        [np.atleast_1d(np.asarray(block["x"], dtype=float)) for block in coord_blocks]
    )
    px_list = np.concatenate(
        [np.atleast_1d(np.asarray(block["px"], dtype=float)) for block in coord_blocks]
    )
    y_list = np.concatenate(
        [np.atleast_1d(np.asarray(block["y"], dtype=float)) for block in coord_blocks]
    )
    py_list = np.concatenate(
        [np.atleast_1d(np.asarray(block["py"], dtype=float)) for block in coord_blocks]
    )

    num_particles = len(x_list)
    deltas = [true_deltap] * num_particles
    logger.info("Built batch particle set with %d particles and delta=%s", num_particles, true_deltap)

    monitor_names = get_monitor_names_at_pattern(line, "bpm.*[^k]")

    ctx = Context(32)
    particles = line.build_particles(
        _context=ctx,
        x=x_list.tolist(),
        px=px_list.tolist(),
        y=y_list.tolist(),
        py=py_list.tolist(),
        delta=deltas,
    )

    return run_tracking(
        line=line,
        particles=particles,
        nturns=flattop_turns,
        monitor_names=monitor_names,
        replace_thick_monitors_with_thin=replace_thick_monitors_with_thin,
    )
