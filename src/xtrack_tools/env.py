from __future__ import annotations

import logging
from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np
import xpart as xp
import xtrack as xt
from packaging.version import Version

if TYPE_CHECKING:
    from pathlib import Path

    import tfs

logger = logging.getLogger(__name__)

# Number of kicks per thick element for the exact (drift-kick-drift) integrator.
# MAD-NG uses a single exact thick map; xsuite reproduces it by integrating with
# enough kicks. 64 reaches machine precision per element for LHC-strength quads
# (see tests/test_madng_element_precision.py); it is *not* a slicing count -- the
# thick element stays intact and is integrated internally.
_DEFAULT_NUM_MULTIPOLE_KICKS = 64

# Element types whose thick body is integrated with the exact drift-kick-drift
# model. Bends are configured separately (see below).
_EXACT_THICK_TYPES = ("Quadrupole", "Sextupole", "Octupole", "Multipole")


def _configure_line_models(
    line: xt.Line, num_multipole_kicks: int = _DEFAULT_NUM_MULTIPOLE_KICKS
) -> None:
    """Configure element models so xsuite reproduces MAD-NG's exact thick maps.

    MAD-NG integrates the *exact* Hamiltonian. xsuite's ``mat-kick-mat`` model,
    used previously for quadrupoles, uses the *expanded* (paraxial) Hamiltonian:
    its per-element error grows as the cube of the transverse angle and does not
    shrink with more kicks (it is a formulation error, not an integration error),
    so the two codes disagree by ~1e-9 relative per element. Over a full ring this
    accumulates into a ~5e-7 tune / ~1e-8 orbit difference, largest in the plane
    with the biggest angles (the crossing plane through the triplet).

    Using ``drift-kick-drift-exact`` with the ``yoshida4`` integrator and enough
    kicks matches MAD-NG to machine precision per element. Bends already match
    with ``bend-kick-bend`` (~3e-14); the ``expanded`` / ``drift-kick-drift``
    bend cores are the paraxial ones and must *not* be used.

    Args:
        line: The xsuite line to configure in place.
        num_multipole_kicks: Kicks used to integrate each thick quadrupole /
            multipole body. Higher is more accurate (and slower); the default
            reaches machine precision for LHC-strength elements.

    Raises:
        RuntimeError: If the installed xtrack is older than 0.105.1.
        ValueError: If a bend carries non-zero multipole components (it would not
            be reproduced by the pure-dipole ``bend-kick-bend`` configuration).
    """
    # This only works with xtrack version 0.105.1 or later, which makes the bend-kick-bend model stable even small bending angles.
    xtrack_version = Version(version("xtrack"))
    if xtrack_version < Version("0.105.1"):
        raise RuntimeError(
            f"xtrack version 0.105.1 or later is required for stable bend-kick-bend model, found {xtrack_version}"
        )

    logger.info(
        "Configuring xsuite line '%s' (length %.3f m) to match MAD-NG exact maps "
        "(bend core=bend-kick-bend, thick core=drift-kick-drift-exact, kicks=%d)",
        getattr(line, "name", "<unnamed>"),
        float(line.get_length()),
        num_multipole_kicks,
    )
    line.configure_drift_model(model="exact")
    line.configure_bend_model(
        core="bend-kick-bend",
        edge="full",
        integrator="uniform",
        num_multipole_kicks=1,
    )
    tt: xt.Table = line.get_table()
    for element_type in _EXACT_THICK_TYPES:
        rows = tt.rows[tt.element_type == element_type]
        if len(rows.name):
            line.set(
                rows,
                model="drift-kick-drift-exact",
                integrator="yoshida4",
                num_multipole_kicks=num_multipole_kicks,
            )

    # loop through all the elements, check that the bending magnets are only dipoles (no multipole components)
    for name, elm in line._element_dict.items():
        if isinstance(elm, xt.Bend) and (
            not np.isclose(elm.k1, 0.0, atol=1e-10)
            or not np.isclose(elm.k2, 0.0, atol=1e-10)
            or not np.all(elm.knl[1:] == 0.0)
            or not np.all(elm.ksl == 0.0)
        ):
            raise ValueError(
                f"Bend element '{name}' has non-zero multipole components: k1={elm.k1}, k2={elm.k2}"
            )


def create_xsuite_environment(
    sequence_file: Path | None = None,
    kinetic_energy: float = 6800,
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
        kinetic_energy: Beam kinetic energy in GeV used to set the particle reference.
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

    logger.info(
        "Preparing xsuite environment for sequence=%s seq_name=%s kinetic_energy=%s GeV cache=%s",
        sequence_file,
        seq_name,
        kinetic_energy,
        json_file,
    )
    logger.debug("Environment regeneration required: %s", needs_regen)

    if needs_regen:
        if not sequence_file.exists():
            raise FileNotFoundError(f"Sequence file not found: {sequence_file}")
        logger.info("Generating xsuite environment from MAD-X sequence %s", sequence_file)
        env: xt.Environment = xt.load(file=sequence_file, _rbend_correct_k0=True, format="madx")
        env.to_json(json_file)
        logger.info(f"xsuite environment saved to {json_file}")
    else:
        logger.info(f"Loading existing xsuite environment from {json_file}")
        env = xt.Environment.from_json(json_file)

    # MAD-X converts sequence names to lowercase
    seq_name_lower = seq_name.lower()
    env[seq_name_lower].particle_ref = xt.Particles(
        mass=xp.PROTON_MASS_EV,
        kinetic_energy0=kinetic_energy * 1e9,
    )
    logger.info(
        "Environment ready with line '%s' at energy %s GeV",
        seq_name_lower,
        env[seq_name_lower].particle_ref.energy0 / 1e9,
    )

    _configure_line_models(env[seq_name_lower])

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
            assert np.isclose(env[mag_name].knl[0], -row["hkick_old"], atol=1e-10) and np.isclose(
                env[mag_name].ksl[0], row["vkick_old"], atol=1e-10
            ), (
                f"Corrector {row['ename']} has different initial strengths in environment: "
                f"knl_env={env[mag_name].knl[0]}, expected={-row['hkick_old']}, "
                f"ksl_env={env[mag_name].ksl[0]}, expected={row['vkick_old']}"
            )
        knl_str: float = -row["hkick"] if abs(row["hkick"]) > 1e-10 else 0.0
        ksl_str: float = row["vkick"] if abs(row["vkick"]) > 1e-10 else 0.0
        env.set(mag_name, knl=[knl_str], ksl=[ksl_str])
    logger.info("Applied corrector strengths to %d elements", len(corrector_table))


def initialise_env(
    matched_tunes: dict[str, float],
    magnet_strengths: dict[str, float],
    corrector_table: tfs.TfsDataFrame,
    sequence_file: Path | None = None,
    kinetic_energy: float = 6800,
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
        kinetic_energy: Beam kinetic energy in GeV.
        seq_name: Sequence name inside the environment.
        json_file: Optional path for cached JSON environment.
        strict_set: If ``True``, validate existing corrector strengths before applying.

    Returns:
        The configured ``xt.Environment`` ready for tracking or optics analysis.

    Raises:
        ValueError: If tune knob names do not match the expected format.
    """
    logger.info(
        "Initialising environment with %d tune knobs, %d magnet strengths, and %d correctors",
        len(matched_tunes),
        len(magnet_strengths),
        len(corrector_table),
    )
    base_env = create_xsuite_environment(
        sequence_file=sequence_file,
        kinetic_energy=kinetic_energy,
        seq_name=seq_name,
        rerun_madx=False,
        json_file=json_file,
    )

    for k, v in matched_tunes.items():
        knob = k.lower()
        if "b1" in k.lower() or "b2" in k.lower():
            knob = k[:3] + "." + k[4:]
        old_value = base_env.get(knob)
        logger.info("Setting tune knob %s to %s from %s", knob, v, old_value)
        base_env.set(knob, v)

    _dknl_to_base = {"dk0l": "k0", "dk1l": "k1", "dk2l": "k2"}
    for str_name, strength in magnet_strengths.items():
        magnet_name, var = str_name.rsplit(".", 1)
        base_attr = _dknl_to_base.get(var)
        if base_attr is not None:
            element = base_env[magnet_name.lower()]
            length = float(getattr(element, "length", getattr(element, "l", 0.0)) or 0.0)
            if length == 0.0:
                raise ValueError(
                    f"Cannot apply integrated perturbation {str_name!r} to zero-length element"
                )
            delta = strength / length
            current = getattr(element, base_attr, 0.0) or 0.0
            logger.debug(
                "Applying integrated delta %s to %s.%s as per-length delta %s (was %s)",
                strength,
                magnet_name.lower(),
                base_attr,
                delta,
                current,
            )
            base_env.set(magnet_name.lower(), **{base_attr: current + delta})
        else:
            logger.debug(f"Setting {magnet_name.lower()} {var} to {strength}")
            base_env.set(magnet_name.lower(), **{var: strength})

    _set_corrector_strengths(base_env, corrector_table, strict_set=strict_set)
    logger.info("Environment initialisation complete")
    return base_env
