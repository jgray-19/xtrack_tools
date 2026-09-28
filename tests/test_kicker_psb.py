"""PSB regression tests for single-turn kicker tracking via xtrack.Exciter."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from xtrack_tools.kicker import plot_kicker_tracking, run_kicker_track

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
SEQ_FILE = Path(__file__).parent / "data" / "sequences" / "psb3_saved.seq"
TKICKER = "br3.kqm12l1"
NTURNS = 10
KICK_TURN = 3
K = 1e-5


@pytest.fixture(scope="module")
def psb_kicker_result(tmp_path_factory: pytest.TempPathFactory):
    return run_kicker_track(
        sequence_file=SEQ_FILE,
        nturns=NTURNS,
        tkicker_name=TKICKER,
        kick_strength=K,
        plane="diagonal",
        kick_turn=KICK_TURN,
        kinetic_energy=0.16,
        seq_name="psb3",
        json_path=tmp_path_factory.mktemp("xtrack-psb-kicker") / "psb3_saved.json",
        bpm_pattern=r"(?i)br3\.bpm.*",
    )


def test_psb_kicker_x_displaced(psb_kicker_result):
    """A horizontal PSB kicker should create a visible horizontal excursion."""
    df, _, _, _, _ = psb_kicker_result
    assert df["x"].abs().max() > 1e-7, (
        "PSB x coordinate unchanged after kick at br3.kqm12l1 "
        "— exciter may not have fired"
    )


def test_psb_kicker_px_displaced(psb_kicker_result):
    """A horizontal PSB kicker should create a visible px excursion."""
    df, _, _, _, _ = psb_kicker_result
    assert df["px"].abs().max() > 1e-7, (
        "PSB px coordinate unchanged after kick at br3.kqm12l1 "
        "— exciter may not have fired"
    )


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="No need to run plot tests in Github Actions.")
def test_psb_kicker_plot(psb_kicker_result, tmp_path: Path):
    """The PSB kicker tracking plot should render to a PNG file."""
    df, tws, _, s_kicker, kick_turn = psb_kicker_result
    output = tmp_path / "psb_kicker_plot.png"
    plot_kicker_tracking(df, tws, s_kicker, kick_turn, save_path=str(output))
    assert output.exists()
    assert output.stat().st_size > 0


def test_psb_kicker_off_momentum_rides_closed_orbit_before_kick(tmp_path: Path):
    """Off-momentum: before the kick the particle sits on the delta0 closed orbit."""
    delta_p = 1e-3
    df, tws, _, _, kick_turn = run_kicker_track(
        sequence_file=SEQ_FILE,
        nturns=NTURNS,
        tkicker_name=TKICKER,
        kick_strength=K,
        plane="diagonal",
        kick_turn=KICK_TURN,
        kinetic_energy=0.16,
        seq_name="psb3",
        json_path=tmp_path / "psb3_saved.json",
        bpm_pattern=r"(?i)br3\.bpm.*",
        delta_p=delta_p,
    )
    co_x = dict(zip([str(n).upper() for n in tws.name], tws.x))
    pre = df[df["turn"] <= kick_turn]
    assert not pre.empty
    residual = pre["x"].to_numpy() - pre["name"].str.upper().map(co_x).to_numpy()
    # Orbit is Dx * delta ~ mm; any betatron mismatch would be comparable.
    assert np.abs(pre["x"]).max() > 1e-4
    assert np.abs(residual).max() < 1e-8
