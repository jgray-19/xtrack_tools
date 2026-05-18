from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import xtrack as xt
from xobjects import ContextCpu as Context

from .env import create_xsuite_environment
from .monitors import get_monitor_names_at_pattern, process_tracking_data
from .tracking import run_tracking

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

logger = logging.getLogger(__name__)

KickPlane = Literal["horizontal", "vertical", "diagonal"]


def _knl_ksl(kick_strength: float, plane: KickPlane) -> tuple[float, float]:
    if plane == "horizontal":
        return kick_strength, 0.0
    if plane == "vertical":
        return 0.0, kick_strength
    # diagonal
    return kick_strength, kick_strength


def _insert_exciter_at(
    line: xt.Line,
    name: str,
    s: float,
    knl: float,
    ksl: float,
    frev: float,
    start_turn: int,
) -> None:
    """Insert an ``xt.Exciter`` with a single sample at position s."""
    exciter = xt.Exciter(
        samples=[1.0],
        sampling_frequency=frev*1e6,
        frev=frev,
        start_turn=start_turn,
        duration=1/frev/2, # half a revolution, to ensure the kick is applied for exactly one turn
        knl=[knl],
        ksl=[ksl],
    )
    line.env.elements[name] = exciter
    line.insert(line.env.place(name, at=s))
    logger.info(
        "Inserted exciter '%s' at s=%.3f m (knl=%g, ksl=%g, start_turn=%d)",
        name, s, knl, ksl, start_turn,
    )


def plot_kicker_tracking(
    df: pd.DataFrame,
    tws: xt.TwissTable,
    s_kicker: float,
    kick_turn: int,
    save_path: str | None = "kicker_test.png",
) -> None:
    """Plot x and y at each BPM as one continuous unrolled line across all turns.

    The x-axis is the unrolled longitudinal position: BPM s-coordinate plus
    ``(turn - 1) * circumference``.  Before the kick the particle rides the
    closed orbit; after the kick it oscillates.  The kick event is shown as a
    red dashed vertical line at the kicker s-position on the turn it fires.

    Args:
        df: Tracking DataFrame with columns ``name``, ``turn``, ``x``, ``y``.
        tws: Twiss table providing element s-positions, circumference, and closed orbit.
        s_kicker: s-position of the kicker [m].
        kick_turn: Turn index (0-based, matching xtrack convention) on which the
            kick fires, used to position the red vertical line.
        save_path: File path to save the figure, or ``None`` to skip saving.
    """
    # Only import matplotlib here since it's only needed for this optional plotting function
    import matplotlib.pyplot as plt
    import pandas as pd

    circumference = float(tws["circumference"])
    tws_df = pd.DataFrame({
        "name": [str(n).upper() for n in tws.name],
        "s": tws.s,
        "x_co": tws.x,
        "y_co": tws.y,
    })
    tws_map = tws_df.set_index("name")

    rows = []
    for bpm in df["name"].unique():
        if bpm.upper() not in tws_map.index:
            logger.warning("BPM '%s' not found in Twiss table — skipping", bpm)
            continue
        bpm_rows = df[df["name"] == bpm].copy()
        bpm_rows["s_bpm"] = tws_map.loc[bpm.upper(), "s"]
        rows.append(bpm_rows)

    plot_df = pd.concat(rows)
    plot_df["s_lin"] = plot_df["s_bpm"] + (plot_df["turn"] - 1) * circumference
    plot_df = plot_df.sort_values("s_lin")

    nturns = int(plot_df["turn"].max())

    # Closed orbit repeated for every turn
    co_bpms = tws_df[tws_df["name"].isin(plot_df["name"].str.upper().unique())].sort_values("s")
    co_s_lin = np.concatenate([co_bpms["s"].values + t * circumference for t in range(nturns)])
    co_x = np.tile(co_bpms["x_co"].values, nturns)
    co_y = np.tile(co_bpms["y_co"].values, nturns)
    order = np.argsort(co_s_lin)

    # The kick fires at xtrack turn `kick_turn`, which is DataFrame turn `kick_turn + 1`.
    # In the unrolled axis this is at s_kicker + kick_turn * circumference.
    s_kick_lin = (s_kicker + kick_turn * circumference) / 1e3

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax1.plot(plot_df["s_lin"] / 1e3, plot_df["x"] * 1e3, marker="o", ms=4, lw=1, label="tracked")
    ax1.plot(co_s_lin[order] / 1e3, co_x[order] * 1e3, ls="--", lw=1, color="gray", label="closed orbit")
    ax2.plot(plot_df["s_lin"] / 1e3, plot_df["y"] * 1e3, marker="o", ms=4, lw=1, color="tab:orange", label="tracked")
    ax2.plot(co_s_lin[order] / 1e3, co_y[order] * 1e3, ls="--", lw=1, color="gray", label="closed orbit")

    ax1.axvline(s_kick_lin, color="tab:red", lw=1.0, ls="--", label="kicker")
    ax2.axvline(s_kick_lin, color="tab:red", lw=1.0, ls="--", label="kicker")

    for turn in range(1, nturns):
        ax1.axvline(turn * circumference / 1e3, color="k", lw=0.5, ls=":", alpha=0.3)
        ax2.axvline(turn * circumference / 1e3, color="k", lw=0.5, ls=":", alpha=0.3)

    ax1.set_ylabel("x [mm]")
    ax1.set_title("Single-turn kicker: BPM readings (unrolled)")
    ax1.legend(fontsize=8)
    ax2.set_ylabel("y [mm]")
    ax2.set_xlabel("Unrolled position [km]")
    ax2.legend(fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info("Saved plot to %s", save_path)
    plt.show()


def run_kicker_track(
    sequence_file: Path,
    nturns: int,
    tkicker_name: str,
    kick_strength: float,
    plane: KickPlane,
    kick_turn: int = 0,
    exciter_name: str = "single_turn_kicker",
    delta_p: float = 0.0,
    bpm_pattern: str = r"(?i)bpm.*",
    kinetic_energy: float = 6800,
    seq_name: str | None = None,
    json_path: Path | None = None,
    add_variance_columns: bool = False,
    replace_thick_monitors_with_thin: bool = True,
) -> tuple[pd.DataFrame, xt.TwissTable, xt.Line, float, int]:
    """Track a single particle with a single-turn kicker placed at a named tkicker.

    An ``xt.Exciter`` is inserted at the s-position of ``tkicker_name`` and fires
    a single kick on ``kick_turn`` (0-based xtrack turn counter).  The particle
    rides the closed orbit for all turns before ``kick_turn`` and then oscillates
    freely afterwards.  The line is not cycled; tracking starts from the natural
    first element of the sequence.

    Args:
        sequence_file: Path to the MAD-X sequence file.
        nturns: Total number of turns to track.
        tkicker_name: Name of the tkicker element in the sequence that sets the
            insertion point for the exciter.
        kick_strength: Kick strength in rad (scaled by the active plane).
        plane: Kick plane — ``"horizontal"``, ``"vertical"``, or ``"diagonal"``.
        kick_turn: 0-based turn on which the exciter fires (default 0 = first turn).
            Set this to e.g. 5 to see 5 turns on the closed orbit before the kick.
        exciter_name: Name given to the inserted ``xt.Exciter`` element.
        delta_p: Momentum deviation for the tracked particle.
        bpm_pattern: Regex pattern selecting monitor elements.
        kinetic_energy: Beam kinetic energy in GeV.
        seq_name: Sequence name inside the environment; defaults to the file stem.
        json_path: Optional path for the cached JSON environment.
        add_variance_columns: Whether to add default variance columns to the output.
        replace_thick_monitors_with_thin: If ``True``, replace thick monitored
            elements with thin marker points before tracking.

    Returns:
        Tuple of ``(tracking_df, twiss_table, baseline_line, s_kicker, kick_turn)``.

    Raises:
        ValueError: If ``tkicker_name`` is not found in the line.
    """
    logger.info(
        "Running single-turn kicker tracking from sequence %s for %d turns "
        "(tkicker=%s, plane=%s, strength=%g, kick_turn=%d)",
        sequence_file,
        nturns,
        tkicker_name,
        plane,
        kick_strength,
        kick_turn,
    )

    env = create_xsuite_environment(
        json_file=json_path,
        sequence_file=sequence_file,
        kinetic_energy=kinetic_energy,
        seq_name=seq_name,
    )
    seq = (seq_name or sequence_file.stem).lower()
    baseline_line: xt.Line = env[seq].copy()

    if tkicker_name not in baseline_line.element_names:
        raise ValueError(f"tkicker element '{tkicker_name}' not found in line '{seq}'.")

    tws: xt.TwissTable = baseline_line.twiss(method="4d")
    frev = float(1.0 / tws["t_rev0"])
    s_kicker = float(baseline_line.get_s_position(tkicker_name))
    logger.info("tkicker '%s' at s=%.3f m, frev=%.6f Hz", tkicker_name, s_kicker, frev)

    knl, ksl = _knl_ksl(kick_strength, plane)

    kicked_line = baseline_line.copy()
    _insert_exciter_at(kicked_line, exciter_name, s_kicker, knl, ksl, frev, start_turn=kick_turn)

    monitor_names = get_monitor_names_at_pattern(kicked_line, bpm_pattern)

    start_elem = kicked_line.element_names[0].upper()
    co_row = tws.rows[start_elem] if start_elem in tws.name else tws.rows[0]
    x0, px0 = float(co_row["x"]), float(co_row["px"])
    y0, py0 = float(co_row["y"]), float(co_row["py"])
    logger.info("Closed orbit at '%s': x=%g px=%g y=%g py=%g", start_elem, x0, px0, y0, py0)

    ctx = Context()
    particles: xt.Particles = kicked_line.build_particles(
        _context=ctx, x=x0, px=px0, y=y0, py=py0, delta=delta_p,
    )

    monitored_line = run_tracking(
        line=kicked_line,
        particles=particles,
        nturns=nturns,
        monitor_names=monitor_names,
        replace_thick_monitors_with_thin=replace_thick_monitors_with_thin,
    )

    tracking_df = process_tracking_data(
        monitored_line,
        ramp_turns=0,
        flattop_turns=nturns,
        add_variance_columns=add_variance_columns,
    )

    return tracking_df, tws, baseline_line, s_kicker, kick_turn


def run_free_track_from_kick(
    baseline_line: xt.Line,
    tws: xt.TwissTable,
    kick_df: pd.DataFrame,
    ref_bpm: str,
    s_kicker: float,
    kick_turn: int,
    bpm_pattern: str = r"(?i)bpm.*",
    replace_thick_monitors_with_thin: bool = True,
) -> pd.DataFrame:
    """Re-track freely from the particle state captured right after the kick.

    Extracts x, px, y, py at the first BPM downstream of the kicker on the turn
    immediately after the kick fires (DataFrame turn ``kick_turn + 2``), cycles
    the baseline line to start at that BPM, and tracks for the remaining turns
    without any kicker element.

    Comparing this output with the post-kick portion of the original kicker-track
    DataFrame verifies whether the exciter fired exactly once: if the trajectories
    are identical the kick did not repeat; if they diverge it fired again.

    Args:
        baseline_line: The unmodified line (no exciter) used for free tracking.
        tws: Twiss table of the baseline line.
        kick_df: Tracking DataFrame returned by ``run_kicker_track``.
        s_kicker: s-position of the kicker [m].
        kick_turn: 0-based xtrack turn on which the kick fired.
        bpm_pattern: Regex pattern selecting monitor elements.
        replace_thick_monitors_with_thin: Passed through to ``run_tracking``.

    Returns:
        Tracking DataFrame for the free-tracking segment, with ``turn`` column
        renumbered starting from 1.
    """
    # kick_turn is 0-based xtrack; DF turn = xtrack turn + 1, so the first post-kick
    # DF turn is kick_turn + 2.
    post_kick_df_turn = kick_turn + 2
    row = kick_df[(kick_df["name"] == ref_bpm.upper()) & (kick_df["turn"] == post_kick_df_turn)]
    if row.empty:
        raise ValueError(
            f"No data for BPM '{ref_bpm}' on DF turn {post_kick_df_turn}. "
            "Check kick_turn and that the BPM appears in the tracking data."
        )
    # Do not go from 64 bit to 32 bit here by using .item() or similar, as that can lose precision for large s.
    x0 = row["x"].iloc[0].item()
    px0 = row["px"].iloc[0].item()
    y0 = row["y"].iloc[0].item()
    py0 = row["py"].iloc[0].item()

    nturns_free = int(kick_df["turn"].max()) - post_kick_df_turn + 1

    free_line = baseline_line.copy()
    free_line.cycle(name_first_element=ref_bpm, inplace=True)

    monitor_names = get_monitor_names_at_pattern(free_line, bpm_pattern)

    ctx = Context()
    particles: xt.Particles = free_line.build_particles(_context=ctx, x=x0, px=px0, y=y0, py=py0)

    monitored_line = run_tracking(
        line=free_line,
        particles=particles,
        nturns=nturns_free,
        monitor_names=monitor_names,
        replace_thick_monitors_with_thin=replace_thick_monitors_with_thin,
    )

    free_df = process_tracking_data(monitored_line, ramp_turns=0, flattop_turns=nturns_free)
    logger.info("Free tracking complete: %d turns from BPM '%s'", nturns_free, ref_bpm)
    return free_df
