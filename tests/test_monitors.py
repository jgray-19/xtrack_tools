"""Tests for monitor and Twiss conversion helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import pytest
import tfs
import xtrack as xt

from xtrack_tools.env import create_xsuite_environment
from xtrack_tools.line import get_element_s_centre
from xtrack_tools.monitors import (
    get_monitor_names_at_pattern,
    line_to_dataframes,
    process_tracking_data,
    replace_thick_monitors_with_thin_markers,
    xsuite_tws_to_ng,
)
from xtrack_tools.tracking import run_tracking


@pytest.mark.parametrize(("pattern", "expected"), [("bpm.", 4), ("quad.", 4)])
def test_get_monitor_names_at_pattern(test_line: xt.Line, pattern: str, expected: int):
    """Test selecting monitor names by regex pattern."""
    monitor_names = get_monitor_names_at_pattern(test_line, pattern)
    assert len(monitor_names) == expected
    assert all(pattern.split(".")[0] in name for name in monitor_names)


def test_get_monitor_names_at_pattern_raises_on_no_match(test_line: xt.Line):
    """Test selecting monitor names raises when nothing matches."""
    with pytest.raises(ValueError, match="No elements found matching pattern"):
        get_monitor_names_at_pattern(test_line, "sext.")


@pytest.mark.slow
def test_replace_thick_monitors_preserves_sps_centres(seq_sps: Path, tmp_path):
    """Test SPS thick monitors are sliced around thin markers at the same centre position."""
    env = create_xsuite_environment(
        sequence_file=seq_sps,
        seq_name="sps",
        kinetic_energy=450,
        json_file=tmp_path / "sps.json",
    )
    line = env.lines["sps"]
    thick_monitor_names = [
        name
        for name in get_monitor_names_at_pattern(line, r"bp.*")
        if float(getattr(line[name], "length", 0.0)) > 0
    ]

    assert thick_monitor_names
    table_before = line.get_table()
    original_centres = np.asarray(
        [table_before["s_center", name] for name in thick_monitor_names], dtype=float
    )

    working_line: xt.Line = line.copy()
    thin_names, alias_map = replace_thick_monitors_with_thin_markers(
        working_line, thick_monitor_names
    )
    inserted_positions = np.asarray(
        [get_element_s_centre(working_line, thin_name) for thin_name in thin_names], dtype=float
    )
    twiss_after = working_line.get_table()
    thin_twiss_positions = np.asarray(
        [twiss_after["s_center", thin_name] for thin_name in thin_names], dtype=float
    )
    second_slice_positions = np.asarray(
        [twiss_after["s", f"{original_name}..1"] for original_name in thick_monitor_names],
        dtype=float,
    )

    assert len(thin_names) == len(thick_monitor_names)
    assert [alias_map[thin_name] for thin_name in thin_names] == thick_monitor_names
    assert np.allclose(inserted_positions, original_centres, atol=1e-12, rtol=0)
    assert np.allclose(thin_twiss_positions, second_slice_positions, atol=1e-12, rtol=0)

    for original_name, thin_name in zip(thick_monitor_names, thin_names, strict=True):
        assert thin_name in working_line.element_names
        assert original_name not in working_line.element_names
        assert f"{original_name}..0" in working_line.element_names
        assert f"{original_name}..1" in working_line.element_names


def test_line_to_dataframes(test_line: xt.Line):
    """Test converting multi-element monitor tracking data to list of DataFrames."""
    monitored_line = test_line.copy()
    monitor_names = get_monitor_names_at_pattern(monitored_line, "bpm.")
    particles = monitored_line.build_particles(
        x=[1e-6, 2e-6],
        px=[1e-7, 2e-7],
        y=[3e-6, 4e-6],
        py=[3e-7, 4e-7],
        delta=[0.0, 0.0],
    )
    tracked_line = run_tracking(monitored_line, particles, 2, monitor_names=monitor_names)

    dataframes = line_to_dataframes(tracked_line)

    assert len(dataframes) == 2
    assert all(isinstance(df, pd.DataFrame) for df in dataframes)

    expected_columns = ["name", "turn", "x", "px", "y", "py"]
    for df in dataframes:
        assert list(df.columns) == expected_columns
        assert len(df) == 8
        assert df["name"].tolist()[:4] == ["BPM.1", "BPM.2", "BPM.3", "BPM.4"]
        assert df["turn"].tolist()[:4] == [1, 1, 1, 1]
        assert df["turn"].tolist()[4:] == [2, 2, 2, 2]
        assert df[["x", "px", "y", "py"]].notna().all().all()


def test_line_to_dataframes_no_monitors(test_line: xt.Line):
    """Test line_to_dataframes raises error when no monitors present."""
    with pytest.raises(ValueError, match="No multi-element monitor data found"):
        line_to_dataframes(test_line)


def test_process_tracking_data_removes_ramp_and_keeps_flattop_turns(test_line: xt.Line):
    """Test ramp turns are removed and remaining turns are renumbered from 1."""
    monitored_line = test_line.copy()
    monitor_names = get_monitor_names_at_pattern(monitored_line, "bpm.")
    particles = monitored_line.build_particles(
        x=[1e-6],
        px=[0.0],
        y=[0.0],
        py=[0.0],
        delta=[0.0],
    )
    tracked_line = run_tracking(monitored_line, particles, 5, monitor_names=monitor_names)

    tracking_df = process_tracking_data(
        tracked_line,
        ramp_turns=2,
        flattop_turns=3,
        add_variance_columns=True,
    )

    assert len(tracking_df) == 12
    assert tracking_df["turn"].min() == 1
    assert tracking_df["turn"].max() == 3
    assert tracking_df["turn"].tolist()[:4] == [1, 1, 1, 1]
    assert tracking_df["turn"].tolist()[-4:] == [3, 3, 3, 3]
    assert {"var_x", "var_y", "var_px", "var_py"} <= set(tracking_df.columns)


def test_replace_thick_monitors_with_thin_markers_raises_on_missing_element(test_line: xt.Line):
    """Test requesting a missing monitor raises ValueError."""
    with pytest.raises(ValueError, match="Monitor element 'bpm.99' not found"):
        replace_thick_monitors_with_thin_markers(test_line.copy(), ["bpm.99"])


def test_xsuite_tws_to_ng_converts_columns_and_headers(twiss_table: xt.TwissTable):
    """Test xsuite Twiss conversion to NG-compatible TFS format."""
    ng_twiss = xsuite_tws_to_ng(twiss_table)

    assert isinstance(ng_twiss, tfs.TfsDataFrame)
    assert ng_twiss.index[0].isupper()
    assert {"beta11", "beta22", "alfa11", "alfa22", "mu1", "mu2"} <= set(ng_twiss.columns)
    assert np.isclose(ng_twiss.headers["q1"], float(twiss_table.qx % 1))
    assert np.isclose(ng_twiss.headers["q2"], float(twiss_table.qy % 1))
