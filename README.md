# xtrack-tools

[![codecov](https://codecov.io/gh/jgray-19/xtrack_tools/graph/badge.svg?token=9QSX96FJQB)](https://codecov.io/gh/jgray-19/xtrack_tools)

Local tools for xtrack/xsuite workflows.

## Installation

```bash
python -m pip install -e .
```

## Usage

```python
import xtrack_tools as xtt

# Example: run a tracking job (requires a configured xsuite line)
# xtt.run_tracking(line, particles, nturns=1000)
```

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
