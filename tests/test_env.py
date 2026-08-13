"""Tests for xtrack_tools environment helpers."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import xtrack as xt

from xtrack_tools.env import _numeric_strength, create_xsuite_environment, initialise_env

BEAM_ENERGY = 6800
LHCB1_SEQ_NAME = "lhcb1"


@pytest.fixture(scope="module")
def shared_json_file(tmp_path_factory):
    """A JSON environment cache built once and reused by tests that don't
    exercise the cache-invalidation logic itself (that's covered by
    ``test_create_xsuite_environment``), avoiding a repeat ~20s MAD-X rebuild
    of the full LHC sequence per test.
    """
    seq_b1 = Path(__file__).parent / "data" / "sequences" / "lhcb1.seq"
    json_file = tmp_path_factory.mktemp("shared_env") / "lhcb1.json"
    create_xsuite_environment(sequence_file=seq_b1, seq_name=LHCB1_SEQ_NAME, json_file=json_file)
    return json_file


@pytest.mark.slow
def test_create_xsuite_environment(tmp_path, seq_b1):
    # Work on a copy: this test bumps the sequence file's mtime below to
    # force cache invalidation, and must not touch the shared checked-out
    # seq_b1 file or it would permanently poison its on-disk JSON cache for
    # every other test that relies on mtime comparison against it.
    seq_b1_copy = tmp_path / seq_b1.name
    shutil.copy2(seq_b1, seq_b1_copy)
    seq_b1 = seq_b1_copy

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


def test_numeric_strength_resolves_bend_k0_from_h():
    """xtrack exposes derived bend k0 as 'from_h'; tools need the numeric value."""
    bend = xt.Bend(length=2.0, angle=-0.5, k0_from_h=True)

    assert bend.k0 == "from_h"
    assert np.isclose(_numeric_strength(bend, "k0"), -0.25)


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
def test_initialise_env(corrector_table, seq_b1, qx, qy, k1_mqy, k0_mb, k2_mcs, shared_json_file):
    """Test initialise_env function."""
    json_file = shared_json_file
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


def test_initialise_env_converts_integrated_dknl_to_per_length_strength(
    corrector_table, seq_b1, shared_json_file
):
    """Integrated dk*l perturbations are applied as per-length k* deltas."""
    json_file = shared_json_file
    element_name = "mqy.b5l2.b1"
    integrated_delta = 2.0e-6

    base_env = create_xsuite_environment(
        sequence_file=seq_b1,
        seq_name=LHCB1_SEQ_NAME,
        json_file=json_file,
    )
    element = base_env[element_name]
    initial_k1 = element.k1
    length = element.length

    env = initialise_env(
        matched_tunes={},
        magnet_strengths={f"{element_name}.dk1l": integrated_delta},
        corrector_table=corrector_table,
        sequence_file=seq_b1,
        kinetic_energy=BEAM_ENERGY,
        seq_name=LHCB1_SEQ_NAME,
        json_file=json_file,
    )

    assert np.isclose(env[element_name].k1, initial_k1 + integrated_delta / length)
