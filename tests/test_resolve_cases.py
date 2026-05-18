"""Tests for ``src.training.train_jepa.resolve_cases``.

Covers the three flag-mutex paths (--cases, --cases-from, --all-train)
introduced for Session 7's full-partition launch.
"""

from __future__ import annotations

import argparse

import pytest

from src.training.train_jepa import resolve_cases


def _ns(**kwargs) -> argparse.Namespace:
    base = dict(cases=None, cases_from=None, all_train=False)
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_resolve_cases_no_flag_returns_none() -> None:
    """Default: no flag -> None (downstream interprets as 'full train split')."""
    assert resolve_cases(_ns()) is None


def test_resolve_cases_all_train_returns_none() -> None:
    """--all-train is the explicit version of 'no flag': returns None."""
    assert resolve_cases(_ns(all_train=True)) is None


def test_resolve_cases_explicit_list_returns_list() -> None:
    """--cases passes the list through verbatim."""
    assert resolve_cases(_ns(cases=["A", "B"])) == ["A", "B"]


def test_resolve_cases_rejects_two_flags() -> None:
    """--cases AND --cases-from is an error."""
    with pytest.raises(SystemExit, match="pass at most one"):
        resolve_cases(_ns(cases=["A"], cases_from="foo.yaml"))


def test_resolve_cases_rejects_all_train_plus_other() -> None:
    """--all-train AND --cases is an error."""
    with pytest.raises(SystemExit, match="pass at most one"):
        resolve_cases(_ns(cases=["A"], all_train=True))
    with pytest.raises(SystemExit, match="pass at most one"):
        resolve_cases(_ns(cases_from="foo.yaml", all_train=True))


def test_resolve_cases_legacy_namespace_without_all_train_attr() -> None:
    """If a caller forgets the all_train attribute, ``getattr`` defaults to False."""
    ns = argparse.Namespace(cases=["A"], cases_from=None)
    assert resolve_cases(ns) == ["A"]
