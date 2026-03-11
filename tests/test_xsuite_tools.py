"""
Tests for xsuite_tools module.

This module contains pytest tests for the xsuite_tools functions.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xpart as xp
import xtrack as xt

from xtrack_tools.acd import insert_ac_dipole, run_acd_twiss
from xtrack_tools.env import create_xsuite_environment, initialise_env
from xtrack_tools.monitors import (
    get_monitor_names_at_pattern,
    line_to_dataframes,
    process_tracking_data,
    replace_thick_monitors_with_thin_markers,
)
from xtrack_tools.tracking import run_tracking, run_tracking_without_ac_dipole

BEAM_ENERGY = 6800  # in GeV
LHCB1_SEQ_NAME = "lhcb1"


@pytest.fixture(scope="function")
def test_env():
    """Create a test environment with a simple line."""
    env = xt.Environment()

    # Create a more realistic test line with BPMs and proper optics
    test_line = xt.Line(
        elements=[
            xt.Drift(length=2.0),
            xt.Quadrupole(length=1.0, k1=0.2),
            xt.Marker(),  # bpm.1
            xt.Drift(length=2.0),
            xt.Quadrupole(length=1.0, k1=-0.2),
            xt.Marker(),  # bpm.2
            xt.Drift(length=2.0),
            xt.Quadrupole(length=1.0, k1=0.2),
            xt.Marker(),  # bpm.3
            xt.Drift(length=2.0),
            xt.Quadrupole(length=1.0, k1=-0.2),
            xt.Marker(),  # bpm.4
            xt.Marker(),  # mkqa.6l4.b1
        ],
        element_names=[
            "drift1",
            "quad.1",
            "bpm.1",
            "drift2",
            "quad.2",
            "bpm.2",
            "drift3",
            "quad.3",
            "bpm.3",
            "drift4",
            "quad.4",
            "mkqa.6l4.b1",
            "bpm.4",
        ],
    )

    # Set particle reference
    test_line.particle_ref = xt.Particles(
        mass=xp.PROTON_MASS_EV,
        energy0=450e9,  # 450 GeV
    )

    env.lines["test_line"] = test_line
    return env


@pytest.fixture(scope="function")
def test_line(test_env):
    """Get test line from environment."""
    return test_env.lines["test_line"]


@pytest.fixture
def twiss_table(test_line):
    """Compute twiss table for test line."""
    return test_line.twiss(method="4d")


@pytest.mark.slow
def test_create_xsuite_environment(tmp_path, seq_b1):
    # Test basic creation
    json_file = tmp_path / "lhcb1.json"
    json_file.unlink(missing_ok=True)
    env = create_xsuite_environment(
        sequence_file=seq_b1, seq_name=LHCB1_SEQ_NAME, json_file=json_file
    )
    # MAD-X converts sequence names to lowercase
    seq_name_lower = LHCB1_SEQ_NAME.lower()
    assert seq_name_lower in env.lines
    assert json_file.exists()
    line = env.lines[seq_name_lower]
    assert np.isclose(line.particle_ref.energy0[0], BEAM_ENERGY * 1e9, rtol=1e-10)
    assert len(line.particle_ref.energy0) == 1

    # Test rerun_madx flag
    mod_time_before = json_file.stat().st_mtime
    env2 = create_xsuite_environment(
        sequence_file=seq_b1, seq_name=LHCB1_SEQ_NAME, json_file=json_file
    )
    mod_time_after = json_file.stat().st_mtime
    assert mod_time_before == mod_time_after
    assert seq_name_lower in env2.lines

    env3 = create_xsuite_environment(
        sequence_file=seq_b1, seq_name=LHCB1_SEQ_NAME, rerun_madx=True, json_file=json_file
    )
    mod_time_after_rerun = json_file.stat().st_mtime
    assert mod_time_after_rerun > mod_time_after
    assert seq_name_lower in env3.lines

    # Test custom json and energy
    temp_json = tmp_path / "temp_xsuite.json"
    env4 = create_xsuite_environment(
        sequence_file=seq_b1, seq_name=LHCB1_SEQ_NAME, json_file=temp_json, beam_energy=450
    )
    assert seq_name_lower in env4.lines
    line4 = env4.lines[seq_name_lower]
    assert np.isclose(line4.particle_ref.energy0[0], 450e9, rtol=1e-10)
    assert len(line4.particle_ref.energy0) == 1

    # Test sequence file newer than json
    mod_time_before = json_file.stat().st_mtime
    # Make sequence file appear newer
    future_time = time.time() + 1
    os.utime(str(seq_b1), (future_time, future_time))
    env5 = create_xsuite_environment(
        sequence_file=seq_b1, seq_name=LHCB1_SEQ_NAME, json_file=json_file
    )
    mod_time_after = json_file.stat().st_mtime
    assert mod_time_after > mod_time_before
    assert seq_name_lower in env5.lines


def test_create_xsuite_environment_requires_sequence_file():
    """Ensure sequence_file is required."""
    with pytest.raises(ValueError, match="sequence_file must be provided"):
        create_xsuite_environment(sequence_file=None)


@pytest.mark.parametrize(
    "qx, qy, k1_mqy, k0_mb, k2_mcs",
    [
        (0.31, 0.32, 0.00323, 0.0003166, -1.3),
        (0.28, 0.31, 0.0025, 0.00025, -1.0),
    ],
    ids=[
        "Init test case 1",
        "Init test case 2",
    ],
)
def test_initialise_env(corrector_table, seq_b1, qx, qy, k1_mqy, k0_mb, k2_mcs, tmp_path):
    """Test initialise_env function."""
    json_file = tmp_path / "temp_xsuite.json"
    json_file.unlink(missing_ok=True)  # Ensure fresh environment
    matched_tunes = {"dqx_b1_op": qx, "dqy_b1_op": qy}
    magnet_strengths = {
        "mqy.b5l2.b1.k1": k1_mqy,
        "mb.b8r2.b1.k0": k0_mb,
        "mcs.b8r2.b1.k2": k2_mcs,
    }

    env = initialise_env(
        matched_tunes=matched_tunes,
        magnet_strengths=magnet_strengths,
        corrector_table=corrector_table,
        sequence_file=seq_b1,
        beam_energy=BEAM_ENERGY,
        seq_name=LHCB1_SEQ_NAME,
        json_file=json_file,
    )
    assert env["dqx.b1_op"] == qx
    assert env["dqy.b1_op"] == qy
    assert np.isclose(env["mqy.b5l2.b1"].k1, k1_mqy)
    assert np.isclose(env["mb.b8r2.b1"].k0, k0_mb)
    assert np.isclose(env["mcs.b8r2.b1"].k2, k2_mcs)

    for row in corrector_table.itertuples():
        assert len(env[row.ename.lower()].knl) == 1
        assert np.isclose(env[row.ename.lower()].knl[0], -row.hkick)
        assert len(env[row.ename.lower()].ksl) == 1
        assert np.isclose(env[row.ename.lower()].ksl[0], row.vkick)


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


def test_insert_ac_dipole(test_line: xt.Line, twiss_table: xt.TwissTable):
    """Test inserting AC dipole into the line."""
    beam = 1
    acd_ramp = 100
    total_turns = 1000
    driven_tunes = [0.16, 0.18]

    initial_num_elements = len(test_line.elements)
    new_line = insert_ac_dipole(test_line, twiss_table, beam, acd_ramp, total_turns, driven_tunes)

    # Check that a new line is returned (function copies)
    assert new_line is not test_line
    assert len(test_line.elements) == initial_num_elements

    # Check that AC dipoles are inserted
    acd_marker = f"mkqa.6l4.b{beam}"
    acdh_name = f"mkach.6l4.b{beam}"
    acdv_name = f"mkacv.6l4.b{beam}"

    assert acdh_name in new_line.element_names
    assert acdv_name in new_line.element_names

    # Check that they are ACDipole elements
    acdh = new_line[acdh_name]
    acdv = new_line[acdv_name]
    assert isinstance(acdh, xt.ACDipole)
    assert isinstance(acdv, xt.ACDipole)

    # Check properties
    assert acdh.plane == "h"
    assert acdv.plane == "v"
    assert acdh.freq == driven_tunes[0]
    assert acdv.freq == driven_tunes[1]

    # Check that they are at the same position as the marker
    marker_pos = new_line.get_s_position(acd_marker)
    acdh_pos = new_line.get_s_position(acdh_name)
    acdv_pos = new_line.get_s_position(acdv_name)
    assert acdh_pos == marker_pos
    assert acdv_pos == marker_pos

    # Check that the parameters are set correctly
    assert all(acdh.ramp == [0, acd_ramp, total_turns, total_turns + acd_ramp])
    assert all(acdv.ramp == [0, acd_ramp, total_turns, total_turns + acd_ramp])

    # Check that the twiss is unaffected by the insertion
    new_twiss = new_line.twiss(method="4d")
    # Remove the row corresponding to the AC dipole for comparison
    pd_twiss = new_twiss.to_pandas()
    pd_twiss = pd_twiss[pd_twiss["name"] != acdh_name]
    pd_twiss = pd_twiss[pd_twiss["name"] != acdv_name]
    pd_twiss.reset_index(drop=True, inplace=True)

    pd.testing.assert_frame_equal(twiss_table.to_pandas(), pd_twiss, check_exact=False, rtol=1e-10)


def test_run_acd_twiss(test_line: xt.Line):
    """Test running AC dipole twiss."""
    driven_tunes = [0.16, 0.18]
    beam = 1
    dpp = 0.05

    twiss_orig = test_line.twiss(method="4d")
    acd_twiss_on_mom = run_acd_twiss(test_line, beam, 0, driven_tunes)
    acd_twiss_off_mom = run_acd_twiss(test_line, beam, dpp, driven_tunes)
    assert isinstance(acd_twiss_on_mom, xt.TwissTable)
    assert isinstance(acd_twiss_off_mom, xt.TwissTable)

    assert f"mkach.6l4.b{beam}" in acd_twiss_on_mom["name"]
    assert f"mkacv.6l4.b{beam}" in acd_twiss_on_mom["name"]
    assert f"mkach.6l4.b{beam}" in acd_twiss_off_mom["name"]
    assert f"mkacv.6l4.b{beam}" in acd_twiss_off_mom["name"]

    assert np.isclose(acd_twiss_on_mom["qx"], driven_tunes[0], atol=1e-10, rtol=1e-10)
    assert np.isclose(acd_twiss_on_mom["qy"], driven_tunes[1], atol=1e-10, rtol=1e-10)
    assert np.isclose(acd_twiss_off_mom["qx"], driven_tunes[0], atol=1e-10, rtol=1e-10)
    assert np.isclose(acd_twiss_off_mom["qy"], driven_tunes[1], atol=1e-10, rtol=1e-10)

    # Check that the off and on momentum twiss are different
    assert not np.isclose(
        acd_twiss_on_mom["betx"],
        acd_twiss_off_mom["betx"],
        atol=1e-2,
        rtol=1e-2,
    ).all()

    # Check original line unaffected
    twiss_after = test_line.twiss(method="4d")
    assert np.isclose(twiss_orig["qx"], twiss_after["qx"], atol=1e-10, rtol=1e-10)
    assert np.isclose(twiss_orig["qy"], twiss_after["qy"], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    "nturns, num_particles",
    [
        (1, 1),
        (20, 5),
    ],
    ids=[
        "1_turn_1_particle",
        "20_turns_5_particles",
    ],
)
def test_run_tracking(test_line: xt.Line, nturns: int, num_particles: int):
    """Test running tracking simulation with various turns and particles."""
    # Create particles at origin
    particles = test_line.build_particles(
        x=np.zeros(num_particles),
        px=np.zeros(num_particles),
        y=np.zeros(num_particles),
        py=np.zeros(num_particles),
        delta=np.zeros(num_particles),
    )

    # Configure monitoring at BPM locations
    monitored_line = test_line.copy()
    monitor_names = get_monitor_names_at_pattern(monitored_line, "bpm.")

    # Run tracking for nturns
    tracked_line = run_tracking(monitored_line, particles, nturns, monitor_names=monitor_names)

    # Check that tracking completed successfully for all particles
    assert np.all(particles.state == 1)

    with pytest.raises(ValueError, match="No multi-element monitor data found"):
        line_to_dataframes(monitored_line)

    monitor = tracked_line.record_multi_element_last_track
    assert monitor is not None
    assert monitor.obs_names == ["BPM.1", "BPM.2", "BPM.3", "BPM.4"]
    assert np.asarray(monitor.get("x")).shape == (nturns, num_particles, 4)
    assert np.asarray(monitor.get("y")).shape == (nturns, num_particles, 4)
    assert np.asarray(monitor.get("px")).shape == (nturns, num_particles, 4)
    assert np.asarray(monitor.get("py")).shape == (nturns, num_particles, 4)


def test_run_tracking_raises_if_any_particle_is_lost():
    """Test tracking fails if any particle is lost."""
    line = xt.Line(
        elements=[
            xt.Marker(),
            xt.LimitRect(min_x=-1e-3, max_x=1e-3, min_y=-1e-3, max_y=1e-3),
            xt.Drift(length=1.0),
        ],
        element_names=["m0", "aper", "d1"],
    )
    line.particle_ref = xt.Particles(
        mass=xp.PROTON_MASS_EV,
        energy0=450e9,
    )
    particles = line.build_particles(
        x=[0.0, 2e-3],
        px=[0.0, 0.0],
        y=[0.0, 0.0],
        py=[0.0, 0.0],
        delta=[0.0, 0.0],
    )

    with pytest.raises(RuntimeError, match="Tracking failed"):
        run_tracking(line, particles, 1)


def test_run_tracking_can_replace_thick_monitors_with_thin():
    """Test replacing thick monitored elements with thin markers before tracking."""
    monitored_line = xt.Line(
        elements=[
            xt.Drift(length=2.0),
            xt.Quadrupole(length=1.0, k1=0.2),
            xt.Drift(length=0.4),
            xt.Drift(length=2.0),
            xt.Quadrupole(length=1.0, k1=-0.2),
            xt.Drift(length=0.6),
            xt.Drift(length=2.0),
        ],
        element_names=[
            "drift1",
            "quad.1",
            "bpm.1",
            "drift2",
            "quad.2",
            "bpm.2",
            "drift3",
        ],
    )
    monitored_line.particle_ref = xt.Particles(
        mass=xp.PROTON_MASS_EV,
        energy0=450e9,
    )
    original_element_names = tuple(monitored_line.element_names)

    original_table = monitored_line.get_table()
    original_length = monitored_line.get_length()
    original_centres = {
        name: float(original_table["s_center", name]) for name in ("bpm.1", "bpm.2")
    }

    particles = monitored_line.build_particles(
        x=np.zeros(2),
        px=np.zeros(2),
        y=np.zeros(2),
        py=np.zeros(2),
        delta=np.zeros(2),
    )
    monitor_names = get_monitor_names_at_pattern(monitored_line, "bpm.")

    tracked_line = run_tracking(
        monitored_line,
        particles,
        3,
        monitor_names=monitor_names,
        replace_thick_monitors_with_thin=True,
    )

    assert tuple(monitored_line.element_names) == original_element_names

    monitor = tracked_line.record_multi_element_last_track
    assert monitor is not None
    assert monitor.obs_names == ["BPM.1", "BPM.2"]
    assert np.asarray(monitor.get("x")).shape == (3, 2, 2)
    assert np.isclose(tracked_line.get_length(), original_length)

    inserted_monitor_names = [name for name in tracked_line.element_names if "__thin" in name]
    assert len(inserted_monitor_names) == 2
    for inserted_name, original_name in zip(
        inserted_monitor_names, ("bpm.1", "bpm.2"), strict=True
    ):
        assert isinstance(tracked_line[inserted_name]._get_viewed_object(), xt.Marker)
        assert np.isclose(
            tracked_line.get_s_position(inserted_name), original_centres[original_name]
        )


def test_run_tracking_without_ac_dipole_doubles_particles_for_split_plane_kicks(
    test_line: xt.Line, twiss_table: xt.TwissTable
):
    """Test one action-angle pair becomes two particles when planes are split."""
    tracked_line = run_tracking_without_ac_dipole(
        line=test_line,
        tws=twiss_table,
        flattop_turns=2,
        action_list=[1e-6],
        angle_list=[0.5],
        use_diagonal_kicks=False,
    )

    monitor = tracked_line.record_multi_element_last_track
    assert monitor is not None

    x = np.asarray(monitor.get("x"))
    y = np.asarray(monitor.get("y"))
    px = np.asarray(monitor.get("px"))
    py = np.asarray(monitor.get("py"))

    assert x.shape == (2, 2, 4)
    assert y.shape == (2, 2, 4)
    assert px.shape == (2, 2, 4)
    assert py.shape == (2, 2, 4)
    assert np.max(np.abs(x[:, 0, :])) > 0.0
    assert np.max(np.abs(py[:, 0, :])) == 0.0
    assert np.max(np.abs(y[:, 1, :])) > 0.0
    assert np.max(np.abs(px[:, 1, :])) == 0.0


@pytest.mark.slow
def test_replace_thick_monitors_preserves_sps_centres(seq_sps: Path, tmp_path):
    """Test SPS thick monitors are sliced around thin markers at the same centre position."""
    env = create_xsuite_environment(
        sequence_file=seq_sps,
        seq_name="sps",
        beam_energy=450,
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

    working_line = line.copy()
    thin_names, alias_map = replace_thick_monitors_with_thin_markers(
        working_line, thick_monitor_names
    )
    inserted_positions = np.asarray(
        [working_line.get_s_position(thin_name) for thin_name in thin_names], dtype=float
    )
    twiss_after = working_line.twiss(method="4d")
    thin_twiss_positions = np.asarray([twiss_after["s", thin_name] for thin_name in thin_names], dtype=float)
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

    # Should return one DataFrame per particle
    assert len(dataframes) == 2
    assert all(isinstance(df, pd.DataFrame) for df in dataframes)

    # Each DataFrame should have correct structure
    expected_columns = ["name", "turn", "x", "px", "y", "py"]
    for df in dataframes:
        assert list(df.columns) == expected_columns
        assert len(df) == 8  # 4 monitors × 2 turns
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
