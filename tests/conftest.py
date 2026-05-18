"""
conftest.py
===========
Pytest configuration for the Monte Carlo Dosimetry test suite.
"""
import sys
import os

# Ensure src/ is on the Python path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


def pytest_configure(config):
    """Register custom marks to avoid PytestUnknownMarkWarning."""
    config.addinivalue_line("markers", "slow: mark test as slow (skipped by default with -m 'not slow')")
    config.addinivalue_line("markers", "anyio: mark async tests")


# anyio backend configuration for async API tests
pytest_plugins = ('anyio',)
