from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import xtrack as xt


def get_element_s_position(line: xt.Line, element_name: str, table: Any | None = None) -> float:
    """Return the longitudinal position of an element in a line."""
    if element_name not in line.element_names:
        raise ValueError(f"Element '{element_name}' not found in the line.")

    line_table = line.get_table() if table is None else table
    return float(line_table["s", element_name])
