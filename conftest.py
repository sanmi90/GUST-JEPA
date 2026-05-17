"""Pytest configuration for vortex-jepa.

Adds the ``slow`` marker for integration tests that should be skipped in
the default suite. Run slow tests explicitly with::

    pytest --runslow

This is the canonical pytest opt-in pattern; see the pytest docs on
"How to use markers".
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked ``slow``.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run (skipped by default)")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
