from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from xtrack_tools.coordinates import create_initial_conditions

if TYPE_CHECKING:
    import xtrack as xt

logger = logging.getLogger(__name__)


def _build_coords_from_action_angle(
    action_list: list[float],
    angle_list: list[float],
    tws: xt.TwissTable,
    use_diagonal_kicks: bool,
    start_marker: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create coordinate arrays from action-angle values using Twiss data.

    Args:
        action_list: Action values for each particle.
        angle_list: Angle values for each particle.
        tws: Twiss table used for coordinate conversion.
        use_diagonal_kicks: If ``True``, kick both planes at the same time, otherwise separate kicks for x and y.
        start_marker: Optional starting element name in the Twiss table.

    Returns:
        Tuple of arrays: ``(x, px, y, py)`` per particle.

    Raises:
        ValueError: If ``action_list`` and ``angle_list`` lengths differ.
    """
    if len(action_list) != len(angle_list):
        raise ValueError(
            f"action_list and angle_list must have the same length: "
            f"{len(action_list)} != {len(angle_list)}"
        )

    logger.info(
        "Building tracking coordinates from %d action-angle pairs starting at %s",
        len(action_list),
        start_marker if start_marker is not None else "first Twiss element",
    )

    tws_df = tws.to_pandas()
    tws_df = tws_df.rename(
        columns={
            "betx": "beta11",
            "bety": "beta22",
            "alfx": "alfa11",
            "alfy": "alfa22",
        }
    )
    tws_df.set_index("name", inplace=True)
    tws_df.index = [name.upper() for name in tws_df.index]
    start_name = start_marker if start_marker else str(tws_df.index[0])
    actions = np.asarray(action_list, dtype=float)
    angles = np.asarray(angle_list, dtype=float)

    planes = ("xy",) if use_diagonal_kicks else ("x", "y")
    num_particles = len(action_list) * len(planes)
    xs = np.empty(num_particles, dtype=float)
    pxs = np.empty(num_particles, dtype=float)
    ys = np.empty(num_particles, dtype=float)
    pys = np.empty(num_particles, dtype=float)

    batch_size = len(actions)
    for i, plane in enumerate(planes):
        coords = create_initial_conditions(
            actions,
            angles,
            tws_df,
            kick_plane=plane,
            starting_bpm=start_name,
        )
        sl = slice(i * batch_size, (i + 1) * batch_size)
        for out, key in ((xs, "x"), (pxs, "px"), (ys, "y"), (pys, "py")):
            values = np.asarray(coords[key], dtype=float)
            out[sl] = values if values.ndim > 0 else values.item()

    logger.info("Built coordinate arrays for %d particles", len(xs))
    return xs, pxs, ys, pys
