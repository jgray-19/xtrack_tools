# xtrack-tools

[![Coverage](https://github.com/OWNER/xtrack-tools/actions/workflows/coverage.yml/badge.svg)](https://github.com/OWNER/xtrack-tools/actions/workflows/coverage.yml)

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
