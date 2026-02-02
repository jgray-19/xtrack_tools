from __future__ import annotations

from pathlib import Path

import pytest
import tfs


@pytest.fixture
def data_dir() -> Path:
    """Return the local tests data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def seq_b1(data_dir: Path) -> Path:
    """Path to the example sequence file for beam 1 used by tests."""
    return data_dir / "sequences" / "lhcb1.seq"


@pytest.fixture
def corrector_table(data_dir: Path) -> tfs.TfsDataFrame:
    """Load the corrector table, removing monitor elements."""
    corrector_file = data_dir / "correctors" / "corrector_table.tfs"
    corrector_table = tfs.read(corrector_file)
    return corrector_table[corrector_table["kind"] != "monitor"]  # ty:ignore[invalid-return-type]
