"""Tests for AC dipole helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import xpart as xp
import xtrack as xt
from scipy.signal import find_peaks

from xtrack_tools.acd import (
    insert_ac_dipole,
    prepare_acd_line_with_monitors,
    run_ac_dipole_tracking_with_particles,
    run_acd_track,
    run_acd_twiss,
)
from xtrack_tools.line import get_element_s_centre

if TYPE_CHECKING:
    from pathlib import Path

LHCB1_SEQ_NAME = "lhcb1"
ACD_MARKER_NAME = "mkqa.6l4.b1"


def test_insert_ac_dipole(test_line: xt.Line, twiss_table: xt.TwissTable):
    """Test inserting AC dipole into the line."""
    acd_ramp = 100
    total_turns = 1000
    driven_tunes = [0.16, 0.18]

    initial_num_elements = len(test_line.elements)
    new_line = insert_ac_dipole(test_line, twiss_table, ACD_MARKER_NAME, acd_ramp, total_turns, driven_tunes)

    assert new_line is not test_line
    assert len(test_line.elements) == initial_num_elements

    acdh_name = f"{ACD_MARKER_NAME}_x"
    acdv_name = f"{ACD_MARKER_NAME}_y"

    assert acdh_name in new_line.element_names
    assert acdv_name in new_line.element_names

    acdh = new_line[acdh_name]
    acdv = new_line[acdv_name]
    assert isinstance(acdh, xt.ACDipole)
    assert isinstance(acdv, xt.ACDipole)
    assert acdh.plane == "h"
    assert acdv.plane == "v"
    assert acdh.freq == driven_tunes[0]
    assert acdv.freq == driven_tunes[1]

    marker_pos = get_element_s_centre(new_line, ACD_MARKER_NAME)
    acdh_pos = get_element_s_centre(new_line, acdh_name)
    acdv_pos = get_element_s_centre(new_line, acdv_name)
    assert acdh_pos == marker_pos
    assert acdv_pos == marker_pos

    assert all(acdh.ramp == [0, acd_ramp, total_turns, total_turns + acd_ramp])
    assert all(acdv.ramp == [0, acd_ramp, total_turns, total_turns + acd_ramp])

    new_twiss = new_line.twiss(method="4d")
    pd_twiss = new_twiss.to_pandas()
    pd_twiss = pd_twiss[pd_twiss["name"] != acdh_name]
    pd_twiss = pd_twiss[pd_twiss["name"] != acdv_name]
    pd_twiss.reset_index(drop=True, inplace=True)

    pd.testing.assert_frame_equal(twiss_table.to_pandas(), pd_twiss, check_exact=False, rtol=1e-10)


def test_get_element_s_position_raises_for_missing_element(test_line: xt.Line):
    """Test requesting an unknown element position raises ValueError."""
    with pytest.raises(ValueError, match="Element 'missing.bpm' not found in the line."):
        get_element_s_centre(test_line, "missing.bpm")


def test_run_acd_twiss(test_line: xt.Line):
    """Test running AC dipole twiss."""
    driven_tunes = [0.16, 0.18]
    dpp = 0.05

    twiss_orig = test_line.twiss(method="4d")
    acd_twiss_on_mom = run_acd_twiss(test_line, ACD_MARKER_NAME, 0, driven_tunes)
    acd_twiss_off_mom = run_acd_twiss(test_line, ACD_MARKER_NAME, dpp, driven_tunes)
    assert isinstance(acd_twiss_on_mom, xt.TwissTable)
    assert isinstance(acd_twiss_off_mom, xt.TwissTable)

    assert f"{ACD_MARKER_NAME}_x" in acd_twiss_on_mom["name"]
    assert f"{ACD_MARKER_NAME}_y" in acd_twiss_on_mom["name"]
    assert f"{ACD_MARKER_NAME}_x" in acd_twiss_off_mom["name"]
    assert f"{ACD_MARKER_NAME}_y" in acd_twiss_off_mom["name"]

    assert np.isclose(acd_twiss_on_mom["qx"], driven_tunes[0], atol=1e-10, rtol=1e-10)
    assert np.isclose(acd_twiss_on_mom["qy"], driven_tunes[1], atol=1e-10, rtol=1e-10)
    assert np.isclose(acd_twiss_off_mom["qx"], driven_tunes[0], atol=1e-10, rtol=1e-10)
    assert np.isclose(acd_twiss_off_mom["qy"], driven_tunes[1], atol=1e-10, rtol=1e-10)

    assert not np.isclose(
        acd_twiss_on_mom["betx"],
        acd_twiss_off_mom["betx"],
        atol=1e-2,
        rtol=1e-2,
    ).all()

    twiss_after = test_line.twiss(method="4d")
    assert np.isclose(twiss_orig["qx"], twiss_after["qx"], atol=1e-10, rtol=1e-10)
    assert np.isclose(twiss_orig["qy"], twiss_after["qy"], atol=1e-10, rtol=1e-10)


def test_prepare_acd_line_with_monitors_computes_twiss_when_missing(test_line: xt.Line):
    """Test AC-dipole line preparation computes Twiss if none is supplied."""
    tracked_line, total_turns, monitor_names = prepare_acd_line_with_monitors(
        line=test_line,
        acd_marker=ACD_MARKER_NAME,
        tws=None,
        ramp_turns=3,
        flattop_turns=2,
        driven_tunes=[0.16, 0.18],
        lag=0.0,
        bpm_pattern="bpm.",
    )

    assert total_turns == 5
    assert monitor_names == ["bpm.1", "bpm.2", "bpm.3", "bpm.4"]
    assert f"{ACD_MARKER_NAME}_x" in tracked_line.element_names
    assert f"{ACD_MARKER_NAME}_y" in tracked_line.element_names


def test_run_acd_twiss_raises_when_marker_is_missing():
    """Test run_acd_twiss fails if the AC-dipole marker is absent."""
    line = xt.Line(
        elements=[xt.Drift(length=1.0), xt.Marker()],
        element_names=["drift1", "bpm.1"],
    )
    line.particle_ref = xt.Particles(
        mass=xp.PROTON_MASS_EV,
        energy0=450e9,
    )

    with pytest.raises(ValueError, match="AC dipole marker"):
        run_acd_twiss(line, acd_marker=ACD_MARKER_NAME, dpp=0.0, driven_tunes=[0.16, 0.18])


def test_run_ac_dipole_tracking_with_particles_supports_explicit_coords(test_line: xt.Line):
    """Test AC-dipole tracking accepts explicit particle coordinates."""
    tracked_line = run_ac_dipole_tracking_with_particles(
        line=test_line,
        tws=None,
        sequence_name=LHCB1_SEQ_NAME,
        acd_marker=ACD_MARKER_NAME,
        ramp_turns=2,
        flattop_turns=3,
        driven_tunes=None,
        particle_coords={
            "x": [1e-6, 2e-6],
            "px": [0.0, 0.0],
            "y": [0.0, 1e-6],
            "py": [0.0, 0.0],
        },
    )

    monitor = tracked_line.record_multi_element_last_track
    assert monitor is not None
    assert np.asarray(monitor.get("x")).shape == (5, 2, 4)
    assert np.asarray(monitor.get("y")).shape == (5, 2, 4)


def test_run_ac_dipole_tracking_with_particles_defaults_to_first_line_element(
    test_line: xt.Line, twiss_table: xt.TwissTable
):
    """Test missing start_marker behaves like explicitly selecting the first line element."""
    implicit_start = run_ac_dipole_tracking_with_particles(
        line=test_line,
        tws=twiss_table,
        sequence_name=LHCB1_SEQ_NAME,
        acd_marker=ACD_MARKER_NAME,
        ramp_turns=1,
        flattop_turns=2,
        driven_tunes=[0.16, 0.18],
        action_list=[1e-6],
        angle_list=[0.5],
        use_diagonal_kicks=False,
        start_marker=None,
    )
    explicit_start = run_ac_dipole_tracking_with_particles(
        line=test_line,
        tws=twiss_table,
        sequence_name=LHCB1_SEQ_NAME,
        acd_marker=ACD_MARKER_NAME,
        ramp_turns=1,
        flattop_turns=2,
        driven_tunes=[0.16, 0.18],
        action_list=[1e-6],
        angle_list=[0.5],
        use_diagonal_kicks=False,
        start_marker="drift1",
    )

    implicit_monitor = implicit_start.record_multi_element_last_track
    explicit_monitor = explicit_start.record_multi_element_last_track
    assert implicit_monitor is not None
    assert explicit_monitor is not None
    assert implicit_monitor.obs_names == explicit_monitor.obs_names
    assert np.allclose(np.asarray(implicit_monitor.get("x")), np.asarray(explicit_monitor.get("x")))
    assert np.allclose(np.asarray(implicit_monitor.get("y")), np.asarray(explicit_monitor.get("y")))


@pytest.mark.slow
def test_run_acd_track_uses_default_driven_tunes_and_returns_flattop_data(seq_b1: Path, tmp_path):
    """Test run_acd_track default tunes and processed output length."""
    tracking_df, twiss_table, monitored_line = run_acd_track(
        sequence_file=seq_b1,
        sequence_name=LHCB1_SEQ_NAME,
        acd_marker=ACD_MARKER_NAME,
        delta_p=0.0,
        ramp_turns=2,
        flattop_turns=3,
        driven_tunes=None,
        json_path=tmp_path / "acd_track.json",
    )

    assert isinstance(tracking_df, pd.DataFrame)
    assert isinstance(twiss_table, xt.TwissTable)
    assert isinstance(monitored_line, xt.Line)
    assert tracking_df["turn"].min() == 1
    assert tracking_df["turn"].max() == 3
    assert tracking_df["turn"].nunique() == 3


def test_psb_acd_tracking_records_br3_bpms(seq_psb: Path):
    """The PSB line should track with an AC dipole inserted at br3.des3l1."""
    acd_marker_name = "br3.des3l1"
    tracking_df, _, tracked_line = run_acd_track(
        sequence_file=seq_psb,
        sequence_name="psb3",
        acd_marker=acd_marker_name,
        kinetic_energy=0.16,
        ramp_turns=1000,
        flattop_turns=1000,
        driven_tunes=[0.16, 0.22],
        bpm_pattern=r"(?i)br3\.bpm.*",
        add_variance_columns=False,
    )

    assert f"{acd_marker_name}_x" in tracked_line.element_names
    assert f"{acd_marker_name}_y" in tracked_line.element_names

    # Check that we have non zero data that lasts for flattop_turns turns
    assert tracking_df["x"].abs().max() > 1e-4, "x coordinate below 0.1mm - AC dipole may not have fired"
    assert tracking_df["y"].abs().max() > 1e-4, "y coordinate below 0.1mm - AC dipole may not have fired"
    assert tracking_df["turn"].max() >= 1000, "Tracking did not last for the full flattop duration"


    # Check that the data in x and y are oscillating at the correct frequencies by looking at the FFT peaks
    # It is necessary to look at the FFT at a single BPM and then this will give the driven tunes as peaks in the FFT.
    chosen_bpm = "BR3.BPM3L3"
    df_bpm = tracking_df[tracking_df["name"] == chosen_bpm].sort_values("turn")
    x_fft = np.fft.fft(df_bpm["x"])
    y_fft = np.fft.fft(df_bpm["y"])
    freqs = np.fft.fftfreq(len(df_bpm), d=1)
    x_peaks, _ = find_peaks(np.abs(x_fft), prominence=0.4 * np.max(np.abs(x_fft)))
    y_peaks, _ = find_peaks(np.abs(y_fft), prominence=0.4 * np.max(np.abs(y_fft)))

    x_peak_freqs = freqs[x_peaks]
    y_peak_freqs = freqs[y_peaks]
    assert any(np.isclose(x_peak_freqs, 0.16, atol=0.01))
    assert any(np.isclose(y_peak_freqs, 0.22, atol=0.01))
