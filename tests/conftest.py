from __future__ import annotations

from pathlib import Path

import pytest
import tfs
import xpart as xp
import xtrack as xt


@pytest.fixture
def data_dir() -> Path:
    """Return the local tests data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def seq_b1(data_dir: Path) -> Path:
    """Path to the example sequence file for beam 1 used by tests."""
    return data_dir / "sequences" / "lhcb1.seq"


@pytest.fixture
def seq_sps(data_dir: Path) -> Path:
    """Path to the example SPS sequence file used by tests."""
    return data_dir / "sequences" / "sps.seq"

@pytest.fixture
def seq_psb(data_dir: Path) -> Path:
    """Path to the example PSB sequence file used by tests."""
    return data_dir / "sequences" / "psb3_saved.seq"


@pytest.fixture
def corrector_table(data_dir: Path) -> tfs.TfsDataFrame:
    """Load the corrector table, removing monitor elements."""
    corrector_file = data_dir / "correctors" / "corrector_table.tfs"
    corrector_table = tfs.read(corrector_file)
    return corrector_table[corrector_table["kind"] != "monitor"]  # ty:ignore[invalid-return-type]


@pytest.fixture(scope="function")
def test_env():
    """Create a test environment with a simple line."""
    env = xt.Environment()

    test_line = xt.Line(
        elements=[
            xt.Drift(length=2.0),
            xt.Quadrupole(length=1.0, k1=0.2),
            xt.Marker(),
            xt.Drift(length=2.0),
            xt.Quadrupole(length=1.0, k1=-0.2),
            xt.Marker(),
            xt.Drift(length=2.0),
            xt.Quadrupole(length=1.0, k1=0.2),
            xt.Marker(),
            xt.Drift(length=2.0),
            xt.Quadrupole(length=1.0, k1=-0.2),
            xt.Marker(),
            xt.Marker(),
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

    test_line.particle_ref = xt.Particles(
        mass=xp.PROTON_MASS_EV,
        energy0=450e9,
    )

    env.lines["test_line"] = test_line
    return env


@pytest.fixture(scope="function")
def test_line(test_env):
    """Get the shared test line from the environment."""
    return test_env.lines["test_line"]


@pytest.fixture
def twiss_table(test_line):
    """Compute twiss table for the shared test line."""
    return test_line.twiss(method="4d")
