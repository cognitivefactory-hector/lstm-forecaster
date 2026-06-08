"""Scaffold smoke test (M0).

Exists so `pytest` collects at least one test and CI goes green before the
real crown-jewel tests (test_no_leakage / test_windowing / test_metrics) land
in M1+. Replace/augment, do not delete the test suite, as those arrive.
"""

import sys


def test_python_is_312_plus():
    # The project pins Python 3.12 (pyproject requires-python). Guard against a
    # stray older interpreter that could mask version-specific behavior.
    assert sys.version_info >= (3, 12)
