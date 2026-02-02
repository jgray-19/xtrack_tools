from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "xtrack-tools"
author = "xtrack-tools contributors"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "alabaster"

autodoc_mock_imports = [
    "aba_optimiser",
    "numpy",
    "pandas",
    "tfs",
    "xobjects",
    "xpart",
    "xtrack",
]
