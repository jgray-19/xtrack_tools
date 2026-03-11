"""
Tests for xtrack_tools.coordinates module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tfs

from xtrack_tools.coordinates import (
    create_initial_conditions,
    generate_action_angle_coordinates,
    get_kick_plane_category,
    validate_coordinate_generation,
)


# Tests for generate_action_angle_coordinates function
@pytest.mark.parametrize("num_tracks", [1, 5, 10])
def test_generate_action_angle_coordinates_correct_number(num_tracks: int) -> None:
    """Test that function generates the requested number of coordinate pairs."""
    action_range = (1e-6, 1e-4)
    actions, angles = generate_action_angle_coordinates(num_tracks, action_range)

    assert len(actions) == num_tracks
    assert len(angles) == num_tracks


def test_generate_action_angle_coordinates_actions_span_range() -> None:
    """Test that generated actions cover the specified range."""
    action_range = (1e-6, 1e-4)
    actions, angles = generate_action_angle_coordinates(5, action_range)

    assert min(actions) == action_range[0]
    assert max(actions) == action_range[1]


def test_generate_action_angle_coordinates_angles_valid_range() -> None:
    """Test that angles are in valid range (0, 2π)."""
    actions, angles = generate_action_angle_coordinates(10, (1e-6, 1e-4))

    assert all(0 < angle < 2 * np.pi for angle in angles)


# Tests for create_initial_conditions function
@pytest.fixture
def twiss_data_multiple_bpms() -> tfs.TfsDataFrame:
    """Create Twiss DataFrame with multiple BPMs for testing starting_bpm parameter."""
    data = {
        "beta11": [10.0, 12.0],
        "beta22": [8.0, 9.0],
        "alfa11": [0.1, 0.2],
        "alfa22": [-0.05, 0.1],
        "x": [1e-6, 2e-6],
        "px": [0.0, 1e-6],
        "y": [0.5e-6, 1e-6],
        "py": [-0.2e-6, 0.1e-6],
    }
    df = pd.DataFrame(data, index=["BPM1", "BPM2"])  # ty:ignore[invalid-argument-type]
    return tfs.TfsDataFrame(df)


@pytest.mark.parametrize(("starting_bpm", "kick_plane"), [("BPM1", "xy"), ("BPM2", "x"), (0, "xy"), (1, "y")])
def test_create_initial_conditions_closed_orbit_and_starting_bpm(
    twiss_data_multiple_bpms: tfs.TfsDataFrame, starting_bpm: str | int, kick_plane: str
) -> None:
    """Test that coordinates properly include closed orbit and starting_bpm works."""
    action = 1e-6
    angle = 0.5

    if isinstance(starting_bpm, str):
        expected_bpm = starting_bpm
    else:
        expected_bpm = twiss_data_multiple_bpms.index[starting_bpm]

    twiss_data_no_co = twiss_data_multiple_bpms.copy()
    twiss_data_no_co.loc[:, ["x", "px", "y", "py"]] = 0.0

    result_with_co = create_initial_conditions(
        action,
        angle,
        twiss_data_multiple_bpms,
        kick_plane=kick_plane,
        starting_bpm=starting_bpm,
    )
    result_without_co = create_initial_conditions(
        action,
        angle,
        twiss_data_no_co,
        kick_plane=kick_plane,
        starting_bpm=starting_bpm,
    )

    required_keys = {"x", "px", "y", "py", "t", "pt"}
    assert set(result_with_co.keys()) == required_keys
    assert all(isinstance(v, (float, np.floating)) for v in result_with_co.values())

    assert result_with_co["t"] == 0.0
    assert result_with_co["pt"] == 0.0

    cox = twiss_data_multiple_bpms.loc[expected_bpm, "x"]
    copx = twiss_data_multiple_bpms.loc[expected_bpm, "px"]
    coy = twiss_data_multiple_bpms.loc[expected_bpm, "y"]
    copy = twiss_data_multiple_bpms.loc[expected_bpm, "py"]

    assert np.isclose(result_with_co["x"], result_without_co["x"] + cox)
    assert np.isclose(result_with_co["px"], result_without_co["px"] + copx)
    assert np.isclose(result_with_co["y"], result_without_co["y"] + coy)
    assert np.isclose(result_with_co["py"], result_without_co["py"] + copy)


def test_create_initial_conditions_vectorises_over_action_angle_pairs(
    twiss_data_multiple_bpms: tfs.TfsDataFrame,
) -> None:
    """Test vectorised action-angle conversion and single-plane suppression."""
    actions = np.array([1e-6, 4e-6])
    angles = np.array([0.25, 0.75])

    result = create_initial_conditions(
        actions,
        angles,
        twiss_data_multiple_bpms,
        kick_plane="x",
        starting_bpm="BPM2",
    )

    assert result["t"] == 0.0
    assert result["pt"] == 0.0
    assert np.asarray(result["x"]).shape == (2,)
    assert np.asarray(result["px"]).shape == (2,)
    assert np.asarray(result["y"]).shape == (2,)
    assert np.asarray(result["py"]).shape == (2,)
    assert np.allclose(np.asarray(result["y"]), twiss_data_multiple_bpms.loc["BPM2", "y"])
    assert np.allclose(np.asarray(result["py"]), twiss_data_multiple_bpms.loc["BPM2", "py"])


# Tests for get_kick_plane_category function
@pytest.mark.parametrize(
    "ntrk,kick_both_planes,expected",
    [
        (0, True, "xy"),
        (1, True, "xy"),
        (0, False, "x"),
        (1, False, "y"),
        (2, False, "x"),
        (3, False, "y"),
    ],
)
def test_get_kick_plane_category(
    ntrk: int, kick_both_planes: bool, expected: str
) -> None:
    """Test kick plane category determination."""
    assert get_kick_plane_category(ntrk, kick_both_planes) == expected


# Tests for validate_coordinate_generation function
def test_validate_coordinate_generation_valid_input() -> None:
    """Test that valid inputs pass validation."""
    assert validate_coordinate_generation(3, [1, 2, 3], [0.1, 0.2, 0.3]) is True


@pytest.mark.parametrize(
    "invalid_input",
    [
        (3, [1, 2], [0.1, 0.2, 0.3]),  # Wrong action length
        (3, [1, 2, 3], [0.1, 0.2]),  # Wrong angle length
        (3, [1, 2, 3], [0.1, 0.2, 0.3, 0.4]),  # Mismatched lengths
    ],
)
def test_validate_coordinate_generation_invalid_input(invalid_input) -> None:
    """Test that invalid inputs raise AssertionError."""
    with pytest.raises(AssertionError):
        validate_coordinate_generation(*invalid_input)
