from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from xtrack_tools.coordinates import create_initial_conditions

if TYPE_CHECKING:
    import xtrack as xt

logger = logging.getLogger(__name__)


def _build_coords_from_action_angle(
    action_list: list[float],
    angle_list: list[float],
    tws: xt.TwissTable,
    kick_both_planes: bool,
    start_marker: str | None,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Create coordinate arrays from action-angle values using Twiss data.

    Args:
        action_list: Action values for each particle.
        angle_list: Angle values for each particle.
        tws: Twiss table used for coordinate conversion.
        kick_both_planes: If ``True``, initialize both planes.
        start_marker: Optional starting element name in the Twiss table.

    Returns:
        Tuple of lists: ``(x, px, y, py)`` per particle.

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
    xs: list[float] = []
    pxs: list[float] = []
    ys: list[float] = []
    pys: list[float] = []
    start_name = start_marker if start_marker else str(tws_df.index[0])

    for i in range(len(action_list)):
        x0 = create_initial_conditions(
            i,
            action_list,
            angle_list,
            tws_df,
            kick_both_planes=kick_both_planes,
            starting_bpm=start_name,
        )
        xs.append(x0["x"])
        pxs.append(x0["px"])
        ys.append(x0["y"])
        pys.append(x0["py"])

    logger.info("Built coordinate arrays for %d particles", len(xs))
    return xs, pxs, ys, pys
