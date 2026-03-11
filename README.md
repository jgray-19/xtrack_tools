# xtrack-tools

[![codecov](https://codecov.io/gh/jgray-19/xtrack_tools/graph/badge.svg?token=9QSX96FJQB)](https://codecov.io/gh/jgray-19/xtrack_tools)

Local tools for xtrack/xsuite workflows.

## Installation

```bash
python -m pip install -e .
```

## Usage

```python
import numpy as np
import xtrack_tools as xtt

# Convert action-angle values to coordinates at a chosen Twiss row.
coords = xtt.create_initial_conditions(
    action=1e-6,
    angle=0.5,
    twiss_data=twiss_df,
    kick_plane="x",
    starting_bpm="BPM.1",
)

# Vectorised inputs are supported as well.
coords = xtt.create_initial_conditions(
    action=np.array([1e-6, 2e-6]),
    angle=np.array([0.25, 0.75]),
    twiss_data=twiss_df,
    kick_plane="xy",
    starting_bpm="BPM.1",
)

# Track without an AC dipole. When use_diagonal_kicks=False, each
# action-angle pair produces two particles: one kicked in x and one in y.
tracked_line = xtt.run_tracking_without_ac_dipole(
    line=line,
    tws=tws,
    flattop_turns=128,
    action_list=[1e-6, 2e-6],
    angle_list=[0.25, 0.75],
    use_diagonal_kicks=False,
)

# Process turn-by-turn data after an AC-dipole ramp. The first kept turn
# is renumbered to 1 and exactly flattop_turns are returned.
tracking_df = xtt.process_tracking_data(
    monitored_line=tracked_line,
    ramp_turns=1000,
    flattop_turns=128,
)
```

## API Notes

- `create_initial_conditions` uses explicit `kick_plane` values: `"x"`, `"y"`, or `"xy"`.
- `run_tracking` succeeds only if all tracked particles survive.
- `run_tracking_without_ac_dipole(..., use_diagonal_kicks=False)` doubles the particle count by generating separate horizontal and vertical kicks for each action-angle pair.
- `process_tracking_data` removes the first `ramp_turns`, renumbers the remaining turns from `1`, and keeps exactly `flattop_turns`.

## Development

```bash
python -m pip install -e ".[dev]"
pre-commit install
pytest
```

## Documentation

```bash
python -m pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html
```
