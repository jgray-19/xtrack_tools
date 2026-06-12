"""Tests for xtrack_tools environment helpers."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from xtrack_tools.env import create_xsuite_environment, initialise_env

BEAM_ENERGY = 6800
LHCB1_SEQ_NAME = "lhcb1"


@pytest.mark.slow
def test_create_xsuite_environment(tmp_path, seq_b1):
    json_file = tmp_path / "lhcb1.json"
    json_file.unlink(missing_ok=True)
    env = create_xsuite_environment(
        sequence_file=seq_b1, seq_name=LHCB1_SEQ_NAME, json_file=json_file
    )
    seq_name_lower = LHCB1_SEQ_NAME.lower()
    assert seq_name_lower in env.lines
    assert json_file.exists()
    line = env.lines[seq_name_lower]
    assert np.isclose(line.particle_ref.kinetic_energy0[0], BEAM_ENERGY * 1e9, rtol=1e-10)
    assert len(line.particle_ref.kinetic_energy0) == 1

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

    temp_json = tmp_path / "temp_xsuite.json"
    env4 = create_xsuite_environment(
        sequence_file=seq_b1, seq_name=LHCB1_SEQ_NAME, json_file=temp_json, kinetic_energy=450
    )
    assert seq_name_lower in env4.lines
    line4 = env4.lines[seq_name_lower]
    assert np.isclose(line4.particle_ref.kinetic_energy0[0], 450e9, rtol=1e-10)
    assert len(line4.particle_ref.kinetic_energy0) == 1

    mod_time_before = json_file.stat().st_mtime
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
    json_file.unlink(missing_ok=True)
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
        kinetic_energy=BEAM_ENERGY,
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
