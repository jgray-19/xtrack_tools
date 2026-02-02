from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xpart as xp
import xtrack as xt
from xtrack import load_madx_lattice  # ty:ignore[unresolved-import]

if TYPE_CHECKING:
    from pathlib import Path

    import tfs

logger = logging.getLogger(__name__)

def create_xsuite_environment(
    beam: int | None = None,
    sequence_file: Path | None = None,
    beam_energy: float = 6800,
    seq_name: str | None = None,
    rerun_madx: bool = False,
    json_file: Path | None = None,
) -> xt.Environment:
    """Create or load an xsuite environment for a given MAD-X sequence.

    The environment is generated from a MAD-X sequence file and cached to JSON.
    If the cached JSON exists and is newer than the sequence file, it will be used
    unless ``rerun_madx`` is set to ``True``.

    Args:
        beam: LHC beam number to resolve the default sequence file, if
            ``sequence_file`` is not provided.
        sequence_file: Path to a MAD-X sequence file. Required if ``beam`` is not set.
        beam_energy: Beam energy in GeV used to set the particle reference.
        seq_name: Sequence name inside the environment; defaults to the file stem.
        rerun_madx: Force regeneration of the JSON cache by rerunning MAD-X.
        json_file: Optional path for the cached JSON representation.

    Returns:
        The loaded ``xt.Environment`` with particle reference set.

    Raises:
        ValueError: If neither ``beam`` nor ``sequence_file`` is provided.
        FileNotFoundError: If the sequence file is missing when regeneration is needed.
    """
    if sequence_file is None:
        raise ValueError("sequence_file must be provided.")

    if json_file is None:
        json_file = sequence_file.with_suffix(".json")
    if seq_name is None:
        seq_name = sequence_file.stem

    needs_regen = rerun_madx or not json_file.exists()
    if not needs_regen:
        needs_regen = sequence_file.stat().st_mtime > json_file.stat().st_mtime

    if needs_regen:
        if not sequence_file.exists():
            raise FileNotFoundError(f"Sequence file not found: {sequence_file}")
        env: xt.Environment = load_madx_lattice(file=sequence_file)
        env.to_json(json_file)
        logger.info(f"xsuite environment saved to {json_file}")
    else:
        logger.info(f"Loading existing xsuite environment from {json_file}")
        env = xt.Environment.from_json(json_file)  # type: ignore[attr-defined]

    # MAD-X converts sequence names to lowercase
    seq_name_lower = seq_name.lower()
    env[seq_name_lower].particle_ref = xt.Particles(
        mass=xp.PROTON_MASS_EV,
        energy0=beam_energy * 1e9,
    )
    return env


def _set_corrector_strengths(
    env: xt.Environment, corrector_table: tfs.TfsDataFrame, strict_set: bool = True
) -> None:
    """Apply corrector strengths from a TFS table to an xsuite environment.

    Args:
        env: Target xsuite environment.
        corrector_table: TFS table with ``ename``, ``hkick``, ``vkick`` and
            corresponding ``*_old`` columns.
        strict_set: If ``True``, assert that the environment matches the old values
            before applying the new strengths.

    Raises:
        AssertionError: If ``strict_set`` is ``True`` and current strengths differ.
    """
    logger.debug(f"Applying corrector strengths to {len(corrector_table)} elements")
    for _, row in corrector_table.iterrows():
        mag_name = row["ename"].lower()
        if strict_set:
            assert (
                np.isclose(env[mag_name].knl[0], -row["hkick_old"], atol=1e-10)  # ty:ignore[not-subscriptable]
                and np.isclose(env[mag_name].ksl[0], row["vkick_old"], atol=1e-10)  # ty:ignore[not-subscriptable]
            ), (
                f"Corrector {row['ename']} has different initial strengths in environment: "
                f"knl_env={env[mag_name].knl[0]}, expected={-row['hkick_old']}, "  # ty:ignore[not-subscriptable]
                f"ksl_env={env[mag_name].ksl[0]}, expected={row['vkick_old']}"  # ty:ignore[not-subscriptable]
            )
        knl_str: float = -row["hkick"] if abs(row["hkick"]) > 1e-10 else 0.0
        ksl_str: float = row["vkick"] if abs(row["vkick"]) > 1e-10 else 0.0
        env.set(mag_name, knl=[knl_str], ksl=[ksl_str])  # type: ignore[attr-defined]


def initialise_env(
    matched_tunes: dict[str, float],
    magnet_strengths: dict[str, float],
    corrector_table: tfs.TfsDataFrame,
    beam: int | None = None,
    sequence_file: Path | None = None,
    beam_energy: float = 6800,
    seq_name: str | None = None,
    json_file: Path | None = None,
    strict_set=True,
) -> xt.Environment:
    """Initialise an xsuite environment with tune knobs, magnet strengths, and correctors.

    Args:
        matched_tunes: Mapping of tune knob names to values (e.g., ``dqx_b1_op``).
        magnet_strengths: Mapping of element variable names to strengths, such as
            ``mqy.b5l2.b1.k1``.
        corrector_table: TFS table with corrector settings.
        beam: LHC beam number to resolve the default sequence file.
        sequence_file: Path to the MAD-X sequence file.
        beam_energy: Beam energy in GeV.
        seq_name: Sequence name inside the environment.
        json_file: Optional path for cached JSON environment.
        strict_set: If ``True``, validate existing corrector strengths before applying.

    Returns:
        The configured ``xt.Environment`` ready for tracking or optics analysis.

    Raises:
        ValueError: If tune knob names do not match the expected format.
    """
    base_env = create_xsuite_environment(
        beam=beam,
        sequence_file=sequence_file,
        beam_energy=beam_energy,
        seq_name=seq_name,
        rerun_madx=False,
        json_file=json_file,
    )

    for k, v in matched_tunes.items():
        knob = k[:3] + "." + k[4:]
        base_env.set(knob, v)  # type: ignore[attr-defined]
        import re

        if not re.match(r"dq[xy]\.b[12]_op", knob):
            raise ValueError(f"Unexpected tune knob name format: {knob}")

    for str_name, strength in magnet_strengths.items():
        magnet_name, var = str_name.rsplit(".", 1)
        logger.debug(f"Setting {magnet_name.lower()} {var} to {strength}")
        base_env.set(magnet_name.lower(), **{var: strength})  # type: ignore[attr-defined]

    _set_corrector_strengths(base_env, corrector_table, strict_set=strict_set)
    return base_env
