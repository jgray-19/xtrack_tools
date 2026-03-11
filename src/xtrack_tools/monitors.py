from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs
from turn_by_turn import convert_to_tbt

if TYPE_CHECKING:
    import xtrack as xt
    from turn_by_turn.structures import TbtData

logger = logging.getLogger(__name__)


def get_monitor_names_at_pattern(line: xt.Line, pattern: str) -> list[str]:
    """Return line element names matching a regex pattern.

    Args:
        line: Line whose element names should be searched.
        pattern: Regular expression matched against each element name.

    Returns:
        Matching element names in line order.

    Raises:
        ValueError: If the pattern matches no elements.
    """
    names = [name for name in line.element_names if re.match(pattern, name)]
    if not names:
        raise ValueError(f"No elements found matching pattern '{pattern}'.")
    logger.info("Selected %d monitor elements matching pattern '%s'", len(names), pattern)
    return names


def _normalise_recorded_multi_element_monitor_names(
    line: xt.Line,
    alias_map: dict[str, str] | None = None,
) -> None:
    """Upper-case recorded monitor names and optionally restore original aliases.

    Args:
        line: Tracked line holding ``record_multi_element_last_track``.
        alias_map: Optional mapping from temporary inserted monitor names back to
            the original element names requested by the caller.
    """
    monitor = _get_multi_element_monitor(line)
    alias_map = alias_map or {}
    if alias_map:
        logger.debug("Normalising recorded monitor names with %d aliases", len(alias_map))
    monitor.obs_names = [alias_map.get(str(name), str(name)).upper() for name in monitor.obs_names]
    monitor._name_to_index = {name: idx for idx, name in enumerate(monitor.obs_names)}


def _next_available_element_name(line: xt.Line, base_name: str) -> str:
    """Return an element name that is unused in both the line and its environment."""
    candidate = base_name
    suffix = 1
    while candidate in line.env.elements or candidate in line.element_names:
        candidate = f"{base_name}_{suffix}"
        suffix += 1
    return candidate


def replace_thick_monitors_with_thin_markers(
    line: xt.Line,
    monitor_names: list[str],
    length_tol: float = 1e-12,
) -> tuple[list[str], dict[str, str]]:
    """Insert thin markers at the centers of thick monitor elements.

    Args:
        line: Line to modify in place.
        monitor_names: Element names intended for ``multi_element_monitor_at``.
        length_tol: Elements with absolute length at or below this threshold are
            treated as already thin and left unchanged.

    Returns:
        A tuple ``(thin_monitor_names, alias_map)`` where ``thin_monitor_names``
        can be passed directly to ``multi_element_monitor_at`` and ``alias_map``
        maps any inserted thin-marker names back to the original monitor names.

    Raises:
        ValueError: If any requested monitor name is not present in the line.
    """
    import xtrack as xt

    thin_monitor_names: list[str] = []
    alias_map: dict[str, str] = {}
    insertions = []
    converted_count = 0

    for monitor_name in monitor_names:
        if monitor_name not in line.element_names:
            raise ValueError(f"Monitor element '{monitor_name}' not found in the line.")

        element_length = float(getattr(line[monitor_name], "length", 0.0))
        if abs(element_length) <= length_tol:
            thin_monitor_names.append(monitor_name)
            continue

        thin_name = _next_available_element_name(line, f"{monitor_name}__thin")
        monitor_center = float(line.get_s_position(monitor_name)) + 0.5 * element_length
        line.env.elements[thin_name] = xt.Marker()
        insertions.append(line.env.place(thin_name, at=monitor_center))
        thin_monitor_names.append(thin_name)
        alias_map[thin_name] = monitor_name
        converted_count += 1

    if insertions:
        logger.info("Replacing %d thick monitors with thin markers", converted_count)
        line.insert(insertions)
    else:
        logger.info(
            "No thick monitors detected in %d requested monitor elements", len(monitor_names)
        )

    logger.debug(
        "Monitor conversion complete: %d unchanged thin monitors, %d converted thick monitors",
        len(monitor_names) - converted_count,
        converted_count,
    )

    return thin_monitor_names, alias_map


def _get_multi_element_monitor(line: xt.Line) -> xt.MultiElementMonitor:
    """Return the multi-element monitor recorded on the line.

    Args:
        line: Tracked line expected to hold multi-element monitor data.

    Returns:
        The recorded multi-element monitor.

    Raises:
        ValueError: If the line has no recorded multi-element monitor data.
    """
    try:
        monitor = line.record_multi_element_last_track
    except RuntimeError as exc:
        raise ValueError(
            "No multi-element monitor data found on the tracked line (RuntimeError during access)."
        ) from exc
    if monitor is None:
        raise ValueError(
            "No multi-element monitor data found on the tracked line (monitor is None)."
        )
    return monitor


def tbt_data_to_dataframes(
    tbt_data: TbtData,
    tracked_line: xt.Line,
) -> list[pd.DataFrame]:
    """Convert ``TbtData`` into per-particle tracking DataFrames.

    Each returned DataFrame contains the monitor name, turn index (1-based), and
    transverse coordinates for a single particle.

    Args:
        tbt_data: Turn-by-turn data, one matrix per tracked particle.
        tracked_line: Tracked line containing multi-element monitor data for ``px``/``py``.

    Returns:
        A list of DataFrames, one per particle.
    """
    monitor = _get_multi_element_monitor(tracked_line)
    px_all = np.asarray(monitor.get("px"))
    py_all = np.asarray(monitor.get("py"))
    logger.info(
        "Converting turn-by-turn data to DataFrames for %d particles across %d monitor points",
        len(tbt_data.matrices),
        px_all.shape[2] if px_all.ndim == 3 else 0,
    )

    tracking_dataframes = []
    for pid, transverse_data in enumerate(tbt_data.matrices):
        monitor_names = [str(name).upper() for name in transverse_data.X.index]
        num_turns = transverse_data.X.shape[1]
        turns_rep = np.repeat(np.arange(num_turns), len(monitor_names))
        monitor_names_rep = np.tile(monitor_names, num_turns)

        x_values = transverse_data.X.to_numpy().T.flatten()
        y_values = transverse_data.Y.to_numpy().T.flatten()
        px_values = px_all[:, pid, :].reshape(-1)
        py_values = py_all[:, pid, :].reshape(-1)

        tracking_dataframes.append(
            pd.DataFrame(
                {
                    "name": monitor_names_rep,
                    "turn": turns_rep + 1,
                    "x": x_values,
                    "px": px_values,
                    "y": y_values,
                    "py": py_values,
                }
            )
        )
    return tracking_dataframes


def line_to_dataframes(tracked_line: xt.Line) -> list[pd.DataFrame]:
    """Convert tracked multi-element monitor data into per-particle DataFrames.

    Args:
        tracked_line: Line containing recorded multi-element monitor data.

    Returns:
        A list of DataFrames, one per tracked particle.

    Raises:
        ValueError: If the line has no recorded multi-element monitor data.
    """
    _get_multi_element_monitor(tracked_line)
    logger.info("Converting recorded multi-element monitor data on tracked line to DataFrames")
    return tbt_data_to_dataframes(convert_to_tbt(tracked_line), tracked_line=tracked_line)


def process_tracking_data(
    monitored_line: xt.Line,
    ramp_turns: int = 1000,
    flattop_turns: int = 100,
    add_variance_columns: bool = True,
) -> pd.DataFrame:
    """Trim ramp turns from tracking data and optionally add variance columns.

    Args:
        monitored_line: Line with monitor data already collected.
        ramp_turns: Number of initial turns to discard.
        flattop_turns: Number of turns to keep after the ramp.
        add_variance_columns: Whether to add default variance and kick-plane columns.

    Returns:
        A single DataFrame containing processed tracking data for the first particle.
    """
    logger.info(
        "Processing tracking data with ramp_turns=%d flattop_turns=%d add_variance_columns=%s",
        ramp_turns,
        flattop_turns,
        add_variance_columns,
    )
    tracking_df = line_to_dataframes(monitored_line)[0]
    tracking_df = tracking_df[tracking_df["turn"] > ramp_turns].copy()
    tracking_df["turn"] = tracking_df["turn"] - ramp_turns
    tracking_df = tracking_df[tracking_df["turn"] <= flattop_turns].copy()
    tracking_df = tracking_df.reset_index(drop=True)

    if add_variance_columns:
        tracking_df["var_x"] = 1e-4**2
        tracking_df["var_y"] = 1e-4**2
        tracking_df["var_px"] = 3e-6**2
        tracking_df["var_py"] = 3e-6**2

    logger.info("Processed tracking data contains %d rows", len(tracking_df))
    return tracking_df


def xsuite_tws_to_ng(tws) -> pd.DataFrame:
    """Convert an xsuite Twiss object to an NG-compatible ``tfs`` DataFrame.

    Args:
        tws: xsuite Twiss object.

    Returns:
        A ``tfs.TfsDataFrame`` with NG-compatible column names and headers.
    """
    logger.info("Converting xsuite Twiss table with %d rows to NG-compatible format", len(tws.name))
    tws_df = tws.to_pandas()
    tws_df = tfs.TfsDataFrame(tws_df)
    tws_df.columns = [c.lower() for c in tws_df.columns]
    tws_df["name"] = tws_df["name"].str.upper()  # ty:ignore[unresolved-attribute]
    tws_df = tws_df.rename(
        columns={
            "betx": "beta11",
            "bety": "beta22",
            "alfx": "alfa11",
            "alfy": "alfa22",
            "mux": "mu1",
            "muy": "mu2",
        }
    )
    tws_df.headers = {"q1": float(tws.qx % 1), "q2": float(tws.qy % 1)}
    return tws_df.set_index("name")
