"""
infotheory
==========

Information-theoretic / causal analysis toolkit for the vortex-jepa project.

Public API
----------
estimators     : KSG mutual information, Kozachenko-Leonenko entropy, histogram MI,
                 surrogate nulls, robustness sweeps.
observability  : aIND-style information-theoretic observability O = I(T;S)/H(T)
                 and the residual-information check.
surd           : Synergistic-Unique-Redundant Decomposition (discrete + reference
                 adapter + staged stand-in).
io_vortex      : data-loading bridge to the vortex-jepa cache/artifacts, plus a
                 synthetic design for end-to-end smoke tests.

Everything is leaf-level numpy/scipy/sklearn; no torch, no project imports, so it
unit-tests in isolation. See scripts/run_causal_analysis.py for the orchestration.
"""
from . import estimators, observability, surd, io_vortex  # noqa: F401

__all__ = ["estimators", "observability", "surd", "io_vortex"]
__version__ = "0.1.0"
