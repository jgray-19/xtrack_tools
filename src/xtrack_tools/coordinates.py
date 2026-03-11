"""
Coordinate generation utilities for accelerator tracking simulations.

This module provides functions for generating action-angle coordinates
and creating initial conditions for particle tracking.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def generate_action_angle_coordinates(
    num_tracks: int,
    action_range: tuple[float, float],
) -> tuple[list[float], list[float]]:
    """
    Generate action-angle coordinate pairs for tracking.

    Args:
        num_tracks: Total number of tracks needed
        action_range: Tuple of (min_action, max_action) in meters
        num_actions: Number of action values (optional, calculated if not provided)

    Returns:
        Tuple of (action_list, angle_list) where both lists have length num_tracks
    """
    logger.info(
        "Generating %d action-angle coordinates in range [%g, %g]",
        num_tracks,
        action_range[0],
        action_range[1],
    )
    # Ensure we get exactly the requested number of tracks
    action_values = np.linspace(action_range[0], action_range[1], num=num_tracks)
    angle_values = np.linspace(0.1, 2 * np.pi, num=num_tracks, endpoint=False)

    logger.info(
        f"Generated {len(action_values)} action-angle pairs for {num_tracks} tracks"
    )

    return action_values.tolist(), angle_values.tolist()


def create_initial_conditions(
    action: float | np.ndarray,
    angle: float | np.ndarray,
    twiss_data: pd.DataFrame,
    kick_plane: str = "xy",
    starting_bpm: str | int = 0,
) -> dict[str, float | np.ndarray]:
    """
    Create initial conditions for a specific track from action-angle coordinates.

    Args:
        action: Action value(s) (scalar or array)
        angle: Angle value(s) (scalar or array, same shape as action)
        twiss_data: Twiss parameters at starting point
        kick_plane: Plane to kick ("x", "y", or "xy")
        starting_bpm: Starting BPM name or index

    Returns:
        Dictionary with initial coordinates (x, px, y, py, t, pt)
    """
    # Get beta and alpha functions at starting point (first BPM)
    first_bpm = starting_bpm
    if isinstance(starting_bpm, int):
        first_bpm = twiss_data.index[starting_bpm]

    beta11 = twiss_data.loc[first_bpm, "beta11"]
    beta22 = twiss_data.loc[first_bpm, "beta22"]
    alfa11 = twiss_data.loc[first_bpm, "alfa11"]
    alfa22 = twiss_data.loc[first_bpm, "alfa22"]
    cox = twiss_data.loc[first_bpm, "x"]
    copx = twiss_data.loc[first_bpm, "px"]
    coy = twiss_data.loc[first_bpm, "y"]
    copy = twiss_data.loc[first_bpm, "py"]

    # Compute normalised coordinates from action and angle
    cos_ang = np.cos(angle)
    sin_ang = np.sin(angle)

    def _closed_orbit_like(value: float) -> float | np.ndarray:
        if np.ndim(cos_ang) == 0:
            return value
        return np.full(np.shape(cos_ang), value, dtype=float)

    # Convert to real space coordinates
    x = np.sqrt(action * beta11) * cos_ang + cox
    px = -np.sqrt(action / beta11) * (sin_ang + alfa11 * cos_ang) + copx
    y = np.sqrt(action * beta22) * cos_ang + coy
    py = -np.sqrt(action / beta22) * (sin_ang + alfa22 * cos_ang) + copy

    # Set coordinates to the closed orbit in the plane not being kicked
    if "y" not in kick_plane:
        y = _closed_orbit_like(coy)
        py = _closed_orbit_like(copy)
    if "x" not in kick_plane:
        x = _closed_orbit_like(cox)
        px = _closed_orbit_like(copx)

    return {
        "x": x,
        "px": px,
        "y": y,
        "py": py,
        "t": 0.0,
        "pt": 0.0,
    }


def get_kick_plane_category(ntrk: int, kick_both_planes: bool) -> str:
    """
    Determine kick plane category for a track.

    Args:
        ntrk: Track number
        kick_both_planes: Whether to kick both planes

    Returns:
        Kick plane category string ("xy", "x", or "y")
    """
    logger.debug(
        "Resolving kick plane category for track %d with kick_both_planes=%s",
        ntrk,
        kick_both_planes,
    )
    if kick_both_planes:
        return "xy"
    return "x" if ntrk % 2 == 0 else "y"


def validate_coordinate_generation(
    num_tracks: int, action_list: list[float], angle_list: list[float]
) -> bool:
    """
    Validate that action-angle coordinate generation produces the expected number of tracks.

    Args:
        num_tracks: Expected number of tracks
        action_list: Generated action values
        angle_list: Generated angle values

    Returns:
        True if validation passes

    Raises:
        AssertionError: If validation fails
    """
    logger.info(
        "Validating coordinate generation for %d tracks", num_tracks
    )
    assert len(action_list) == num_tracks, (
        f"Expected {num_tracks} action values, got {len(action_list)}."
    )
    assert len(angle_list) == num_tracks, (
        f"Expected {num_tracks} angle values, got {len(angle_list)}."
    )
    assert len(action_list) == len(angle_list), (
        f"Action and angle lists must have the same length, got {len(action_list)} and {len(angle_list)}."
    )
    logger.info("Coordinate generation validation passed")
    return True
