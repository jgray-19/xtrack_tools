from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import xtrack as xt

logger = logging.getLogger(__name__)


def resolve_element_name(line: xt.Line, element_name: str) -> str:
    """Return the line's canonical name for ``element_name`` (case-insensitive).

    Element names in the xsuite line are lower case, while callers commonly use
    the upper-case MAD-X convention (e.g. ``MKQA.6L4.B1``). Match on a
    case-insensitive basis and return the actual name stored in the line.
    """
    if element_name in line.element_names:
        return element_name
    lowered = element_name.lower()
    for name in line.element_names:
        if name.lower() == lowered:
            return name
    raise ValueError(f"Element '{element_name}' not found in the line.")


def get_element_s_centre(line: xt.Line, element_name: str, table: Any | None = None) -> float:
    """Return the longitudinal position of an element in a line."""
    element_name = resolve_element_name(line, element_name)

    line_table = line.get_table() if table is None else table
    return float(line_table["s_center", element_name])


def next_available_element_name(line: xt.Line, base_name: str) -> str:
    """Return an element name that is unused in both the line and its environment."""
    candidate = base_name
    suffix = 1
    while candidate in line.env.elements or candidate in line.element_names:
        candidate = f"{base_name}_{suffix}"
        suffix += 1
    return candidate


def make_element_thin(line: xt.Line, element_name: str, length_tol: float = 1e-12) -> str:
    """Collapse a thick element to a zero-length marker at its centre, in place.

    The thick body is preserved as an equal-length drift so the lattice length
    and downstream optics are unchanged, while a zero-length ``xt.Marker`` keeps
    the original element name and sits at the element centre. This pins the
    element's Twiss row to its centre, which is where point-like insertions (a
    single-turn kick, an AC dipole) actually act — reading optics from the
    original thick element's entry row would otherwise be off by half its length.

    The element's own transfer map is discarded (replaced by a drift), so this is
    intended for passive position markers such as tkickers and AC-dipole markers
    that carry no active strength in the optics. Elements already thin (within
    ``length_tol``) are left untouched.

    Args:
        line: Line to modify in place.
        element_name: Name of the element to thin (case-insensitive).
        length_tol: Elements with absolute length at or below this threshold are
            treated as already thin and returned unchanged.

    Returns:
        The canonical (line) name of the now-thin element.

    Raises:
        ValueError: If ``element_name`` is not present in the line.
    """
    import xtrack as xt

    element_name = resolve_element_name(line, element_name)
    length = float(getattr(line[element_name], "length", 0.0))
    if abs(length) <= length_tol:
        return element_name

    centre = get_element_s_centre(line, element_name)
    body_name = next_available_element_name(line, f"{element_name}__thinned_body")
    line.env.elements[body_name] = xt.Drift(length=length)
    line.replace(element_name, body_name)
    del line.env.element_dict[element_name]
    line.env.elements[element_name] = xt.Marker()
    line.insert(line.env.place(element_name, at=centre))
    logger.info(
        "Thinned element '%s' (length %.3f m) to a marker at its centre s=%.3f m",
        element_name, length, centre,
    )
    return element_name
