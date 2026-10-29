"""Test setup; scientific functions always come from the installed Pyrcel.

LAGRANGIAN_TEST_LIGHT_IMPORTS=1 skips Pyrcel's unrelated parcel-model package
initializer on hosts where its optional AOT compiler registration is unavailable.
It does not replace constants, thermodynamics, SciPy solvers, or netCDF readers.
"""
import importlib.util
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MPLBACKEND', 'Agg')

if os.environ.get('LAGRANGIAN_TEST_LIGHT_IMPORTS') == '1':
    spec = importlib.util.find_spec('pyrcel')
    if spec is None:
        raise ImportError('Install Pyrcel before running the scientific tests')
    package = importlib.util.module_from_spec(spec)
    sys.modules['pyrcel'] = package
    import microphysics
