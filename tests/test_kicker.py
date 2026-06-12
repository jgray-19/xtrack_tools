"""Tests for single-turn kicker tracking via xtrack.Exciter."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from xtrack_tools.kicker import plot_kicker_tracking, run_free_track_from_kick, run_kicker_track

# SEQ_FILE = Path(__file__).parent / "data" / "sequences" / "sps.seq"
# TKICKER = "adkcv.32171"  # first SPS ADK tkicker

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
SEQ_FILE = Path(__file__).parent / "data" / "sequences" / "lhcb1.seq"
TKICKER = "mkqa.6l4.b1"
BPM_AFTER_KICKER = "bpmya.5l4.b1"
NTURNS = 15
KICK_TURN = 5   # particle rides CO for 5 turns, then gets kicked
K = 1e-6        # kick strength [rad]


@pytest.fixture(scope="module")
def kicker_result():
    return run_kicker_track(
        sequence_file=SEQ_FILE,
        nturns=NTURNS,
        tkicker_name=TKICKER,
        kick_strength=K,
        plane="diagonal",
        kick_turn=KICK_TURN,
    )


def test_kicker_x_displaced(kicker_result):
    """After a horizontal/diagonal kick, x should grow from the reference orbit."""
    df, _, _, _, _ = kicker_result
    assert df["x"].abs().max() > 0, "x coordinate unchanged — exciter may not have fired"


def test_kicker_y_displaced(kicker_result):
    """After a vertical/diagonal kick, y should grow from the reference orbit."""
    df, _, _, _, _ = kicker_result
    assert df["y"].abs().max() > 0, "y coordinate unchanged — exciter may not have fired"


def test_kicker_diagonal(kicker_result):
    """With a diagonal kick the maximum excursions in x and y should be comparable."""
    df, _, _, _, _ = kicker_result
    ratio = df["x"].abs().max() / df["y"].abs().max()
    assert 0.1 < ratio < 10, f"x/y ratio {ratio:.2f} — kick does not look diagonal"


def test_single_kick(kicker_result):
    """Free-tracking from the post-kick state must match the kicker track exactly.

    If the exciter fired only once, the particle's trajectory from that point is
    purely ballistic and must be identical to re-tracking from the same initial
    conditions without any kicker element.  Any discrepancy means the exciter
    fired again on a later turn.

    Only BPMs downstream of ref_bpm (s >= s_ref_bpm in the original ring) are
    compared.  Upstream BPMs appear near the end of the cycled free-track
    revolution, so free turn 1 at an upstream BPM corresponds to kick_df turn
    kick_turn+3, not kick_turn+2 — a one-turn offset that is not a physics error.
    """
    df, tws, baseline_line, s_kicker, kick_turn = kicker_result

    free_df = run_free_track_from_kick(
        baseline_line=baseline_line,
        tws=tws,
        kick_df=df,
        ref_bpm=BPM_AFTER_KICKER,
        s_kicker=s_kicker,
        kick_turn=kick_turn,
    )

    # Build a map from BPM name (upper) to s-position for downstream filtering.
    tws_s = {str(n).upper(): float(s) for n, s in zip(tws.name, tws.s)}
    s_ref = tws_s.get(BPM_AFTER_KICKER.upper(), 0.0)

    # Only compare BPMs downstream of ref_bpm in the original ring order.
    # For these BPMs, free turn 1 aligns exactly with kick_df DF turn kick_turn+2.
    common_bpms = set(free_df["name"].unique()) & set(df["name"].unique())
    downstream_bpms = {b for b in common_bpms if tws_s.get(b.upper(), 0.0) >= s_ref}
    assert downstream_bpms, "No downstream BPMs to compare."

    for bpm in sorted(downstream_bpms):
        kicked_bpm = (
            df[(df["name"] == bpm) & (df["turn"] >= kick_turn + 2)]
            .sort_values("turn")
            .reset_index(drop=True)
        )
        free_bpm = (
            free_df[free_df["name"] == bpm]
            .sort_values("turn")
            .reset_index(drop=True)
        )
        n = min(len(kicked_bpm), len(free_bpm))
        assert n > 0, f"No overlapping turns for BPM '{bpm}'."

        np.testing.assert_allclose(
            kicked_bpm["x"].values[:n],
            free_bpm["x"].values[:n],
            rtol=1e-9,
            err_msg=f"x mismatch at BPM '{bpm}' — exciter may have fired more than once",
        )
        np.testing.assert_allclose(
            kicked_bpm["y"].values[:n],
            free_bpm["y"].values[:n],
            rtol=1e-9,
            err_msg=f"y mismatch at BPM '{bpm}' — exciter may have fired more than once",
        )


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="No need to run plot tests in Github Actions.")
def test_kicker_plot(kicker_result):
    """Visual check: particle on closed orbit for KICK_TURN turns, then oscillating."""
    df, tws, _, s_kicker, kick_turn = kicker_result
    plot_kicker_tracking(df, tws, s_kicker, kick_turn, save_path="kicker_test.png")
