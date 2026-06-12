from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import xtrack as xt


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
