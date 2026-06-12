"""Tests for tracking helpers."""

from __future__ import annotations

import numpy as np
import pytest
import tfs
import xpart as xp
import xtrack as xt

from xtrack_tools.monitors import get_monitor_names_at_pattern, line_to_dataframes
from xtrack_tools.tracking import (
    run_tracking,
    run_tracking_without_ac_dipole,
    start_tracking_xsuite_batch,
)


def _tracking_twiss_dataframe(twiss_table: xt.TwissTable) -> tfs.TfsDataFrame:
    """Convert an xsuite Twiss table to the column layout expected by tracking helpers."""
    twiss_df = twiss_table.to_pandas().rename(
        columns={
            "betx": "beta11",
            "bety": "beta22",
            "alfx": "alfa11",
            "alfy": "alfa22",
        }
    )
    twiss_df = twiss_df.set_index("name")
    twiss_df.index = [name.upper() for name in twiss_df.index]
    return tfs.TfsDataFrame(twiss_df)


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
    particles = test_line.build_particles(
        x=np.zeros(num_particles),
        px=np.zeros(num_particles),
        y=np.zeros(num_particles),
        py=np.zeros(num_particles),
        delta=np.zeros(num_particles),
    )

    monitored_line = test_line.copy()
    monitor_names = get_monitor_names_at_pattern(monitored_line, "bpm.")
    tracked_line = run_tracking(monitored_line, particles, nturns, monitor_names=monitor_names)

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
            tracked_line.get_table()["s_center", inserted_name], original_centres[original_name]
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


def test_run_tracking_without_ac_dipole_uses_explicit_coords_and_start_marker(
    test_line: xt.Line, twiss_table: xt.TwissTable
):
    """Test explicit particle coordinates follow the cycled line order."""
    tracked_line = run_tracking_without_ac_dipole(
        line=test_line,
        tws=twiss_table,
        flattop_turns=2,
        particle_coords={
            "x": [1e-6],
            "px": [0.0],
            "y": [2e-6],
            "py": [0.0],
        },
        start_marker="BPM.2",
    )

    monitor = tracked_line.record_multi_element_last_track
    assert monitor is not None
    assert monitor.obs_names == ["BPM.2", "BPM.3", "BPM.4", "BPM.1"]
    assert np.asarray(monitor.get("x")).shape == (2, 1, 4)


def test_run_tracking_without_ac_dipole_accepts_madng_style_twiss_dataframe(
    test_line: xt.Line, twiss_table: xt.TwissTable
):
    """Test action-angle tracking accepts a MAD-NG-style Twiss DataFrame directly."""
    tracked_line = run_tracking_without_ac_dipole(
        line=test_line,
        tws=_tracking_twiss_dataframe(twiss_table),
        flattop_turns=2,
        action_list=[1e-6],
        angle_list=[0.5],
        use_diagonal_kicks=False,
    )

    monitor = tracked_line.record_multi_element_last_track
    assert monitor is not None
    assert np.asarray(monitor.get("x")).shape == (2, 2, 4)
    assert np.asarray(monitor.get("y")).shape == (2, 2, 4)


def test_run_tracking_without_ac_dipole_requires_coordinates_or_action_angle(
    test_line: xt.Line, twiss_table: xt.TwissTable
):
    """Test missing coordinate inputs raise ValueError."""
    with pytest.raises(ValueError, match="Provide particle_coords or both action_list and angle_list"):
        run_tracking_without_ac_dipole(
            line=test_line,
            tws=twiss_table,
            flattop_turns=2,
        )


def test_start_tracking_xsuite_batch_doubles_particles_for_split_plane_kicks(
    test_env, twiss_table: xt.TwissTable
):
    """Test batched tracking doubles the particle count for split-plane kicks."""
    tracked_line = start_tracking_xsuite_batch(
        env=test_env,
        batch_start=0,
        batch_end=1,
        action_list=[1e-6],
        angle_list=[0.5],
        twiss_data=_tracking_twiss_dataframe(twiss_table),
        use_diagonal_kicks=False,
        flattop_turns=2,
        progress_interval=1,
        num_tracks=1,
        true_deltap=0.0,
        seq_name="test_line",
    )

    monitor = tracked_line.record_multi_element_last_track
    assert monitor is not None
    assert np.asarray(monitor.get("x")).shape == (2, 2, 4)
    assert np.asarray(monitor.get("y")).shape == (2, 2, 4)
