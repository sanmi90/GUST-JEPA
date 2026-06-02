"""
infotheory.surd
===============

Synergistic-Unique-Redundant Decomposition (SURD) of causality, after
Martinez-Sanchez, Arranz & Lozano-Duran (Nat. Commun. 15, 9296, 2024).

SURD decomposes the mutual information I(T_future ; {S_1, ..., S_n}) that a set
of source ("agent") variables carries about a target's future into:

    * redundant   R_alpha : information present in several sources jointly,
    * unique      U_i     : information only source i carries,
    * synergistic S_alpha : information present only in a joint state of two or
                            more sources and in no subset,
    * information leak     : the part of H(T_future) not accounted for by any
                            source (the causal influence of unobserved variables),
                            = H(T_future | all sources) / H(T_future).

Why this is the right scalar tool for the vortex-jepa paper
-----------------------------------------------------------
The manuscript's headline empirical fact is that the *wake* observables, not the
scalar forces, separate the encoders, and that a latent which captures only the
lift signature fails on the wake. SURD makes the mechanism quantitative. Run it
with

    target  = future wake enstrophy at H = 16
    sources = {gust parameters c = (G, D, Y), current wake enstrophy,
               current lift, baseline-shedding-phase proxy}

and the prediction is that the future wake is *synergistic* in
(parameters, current wake state) while the future lift is more *uniquely* caused
by the parameters. That is the information-theoretic reason a reconstruction
objective (which need not retain wake structure) cannot forecast the wake, and a
predictive objective (which retains exactly the information about the future) can.

The Lozano-Duran group has since released a forecasting-oriented companion,
ALD-Lab/Causal-Forecast, which links these very SURD components to the
theoretical limit of predictive performance (irreducible error). That is almost
exactly the argument this manuscript makes empirically, so the SURD section both
explains the result and connects it to a published predictive-limit framework.

Two implementations
--------------------
1. `surd_discrete(p)` : a transparent, self-contained discrete SURD for a joint
   pmf array `p` of shape (n_target_states, n_s1_states, ..., n_sn_states),
   implemented for n = 2 and n = 3 sources. It follows the SURD construction:
   compute the specific (state-dependent) information each source / source-group
   provides about each target state, sort the source-groups by that specific
   information, and accumulate redundant increments; the synergistic terms are
   the increments contributed by joint groups beyond their subsets. This is
   adequate and fast for the small, pre-binned scalar sets used here and is
   covered by canonical unit tests (XOR -> pure synergy, COPY -> pure redundancy,
   UNIQUE -> pure unique).

2. `surd_reference(...)` : a thin adapter that calls the authors' reference
   implementation if it is importable (clone github.com/Computational-Turbulence-Group/SURD
   and put it on PYTHONPATH, or `pip install` it). ALWAYS cross-check any headline
   SURD number against the reference before it goes in the manuscript; the
   in-repo `surd_discrete` exists for fast iteration and for environments where
   the reference is not installed, not to supersede it.

State-aware extension
---------------------
The gust encounter is a transient off a limit cycle, so a single stationary SURD
mixes the pre-impact (parameter-dominated) and post-roll-up (wake-state-dominated)
regimes. For the staged analysis the manuscript already uses, prefer the
state-aware version (ALD-Lab/SURD-states) applied per encounter stage. This module
provides `surd_by_stage` which simply applies SURD within each stage mask and is
the cheap stand-in; the released state-aware code is the rigorous tool.

All entropies/MIs here are in bits (SURD convention). Inputs are pmfs, so use
infotheory.estimators.joint_pmf / quantise to build them from samples.
"""
from __future__ import annotations

from itertools import combinations
from dataclasses import dataclass

import numpy as np

_EPS = 1e-12
_LOG2 = np.log(2.0)


# --------------------------------------------------------------------------- #
# pmf utilities (target is always axis 0)
# --------------------------------------------------------------------------- #
def _normalise(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    if s <= 0:
        raise ValueError("joint pmf sums to zero")
    return p / s


def _marginal(p: np.ndarray, axes_keep) -> np.ndarray:
    """Marginal pmf keeping the given axes (tuple), summing out the rest."""
    axes_keep = tuple(sorted(axes_keep))
    sum_axes = tuple(a for a in range(p.ndim) if a not in axes_keep)
    return p.sum(axis=sum_axes) if sum_axes else p


def _entropy(p_marg: np.ndarray) -> float:
    p = p_marg[p_marg > _EPS]
    return float(-np.sum(p * np.log(p)) / _LOG2)


def mutual_info_from_pmf(p: np.ndarray, target_axis: int, source_axes) -> float:
    """I(T; S_group) in bits from the joint pmf, T at `target_axis`."""
    source_axes = tuple(source_axes)
    p_ts = _marginal(p, (target_axis,) + source_axes)
    # reorder so target is axis 0
    order = (0,) + tuple(range(1, p_ts.ndim))
    p_ts = np.transpose(p_ts, order)
    p_t = p_ts.sum(axis=tuple(range(1, p_ts.ndim)))
    p_s = p_ts.sum(axis=0)
    h_t = _entropy(p_t)
    h_s = _entropy(p_s)
    h_ts = _entropy(p_ts)
    return float(h_t + h_s - h_ts)


def _specific_information(p: np.ndarray, source_axes) -> np.ndarray:
    """
    Specific information I(t ; S_group) that a source group provides about each
    *individual* target state t (Williams-Beer specific information), in bits.
    Returns an array indexed by target state. Target is axis 0 by construction in
    the arrays passed here.
    """
    source_axes = tuple(source_axes)
    p_ts = _marginal(p, (0,) + source_axes)
    order = (0,) + tuple(range(1, p_ts.ndim))
    p_ts = np.transpose(p_ts, order)
    p_t = p_ts.sum(axis=tuple(range(1, p_ts.ndim)))          # P(t)
    p_s = p_ts.sum(axis=0)                                    # P(s)
    n_t = p_ts.shape[0]
    spec = np.zeros(n_t)
    for ti in range(n_t):
        if p_t[ti] <= _EPS:
            continue
        p_s_given_t = p_ts[ti] / p_t[ti]                     # P(s | t)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_t_given_s = np.where(p_s > _EPS, p_ts[ti] / p_s, 0.0)  # P(t | s)
            term = np.where(
                (p_s_given_t > _EPS) & (p_t_given_s > _EPS),
                p_s_given_t * (np.log(p_t_given_s) - np.log(p_t[ti])) / _LOG2,
                0.0,
            )
        spec[ti] = float(np.sum(term))
    return spec


@dataclass
class SURDResult:
    redundant: dict           # frozenset(sources) -> bits
    unique: dict              # frozenset({i}) -> bits
    synergistic: dict         # frozenset(sources) -> bits
    info_leak: float          # H(T_future | all sources) / H(T_future), in [0,1]
    mi_total: float           # I(T_future; all sources), bits
    h_target: float           # H(T_future), bits
    n_sources: int

    def normalised(self) -> dict:
        """All causal components divided by H(T_future), for cross-target comparison."""
        h = self.h_target or 1.0
        out = {"info_leak": self.info_leak}
        for k, v in self.redundant.items():
            out[f"R{tuple(sorted(k))}"] = v / h
        for k, v in self.unique.items():
            out[f"U{tuple(sorted(k))}"] = v / h
        for k, v in self.synergistic.items():
            out[f"S{tuple(sorted(k))}"] = v / h
        return out


# --------------------------------------------------------------------------- #
# discrete SURD (n = 2 and n = 3 sources)
# --------------------------------------------------------------------------- #
def surd_discrete(p: np.ndarray) -> SURDResult:
    """
    SURD from a joint pmf `p` with target on axis 0 and n sources on axes 1..n.

    Implements the redundancy-lattice accumulation of Martinez-Sanchez et al.
    (2024) for n in {2, 3}: for each target state, source-groups are ranked by the
    specific information they carry; redundant information is the running minimum
    contributed as we descend the ranking, and the increment a group adds beyond
    all its subsets is attributed as redundant (if the group is a single source
    appearing in a shared increment) or synergistic (if the increment requires the
    joint group). Unique information is a group's increment that no other group of
    equal-or-smaller order shares. Averaged over target states weighted by P(t).

    This follows the SURD specific-information construction; for n > 3, or for any
    number that will be published, use `surd_reference`.
    """
    p = _normalise(p)
    n_sources = p.ndim - 1
    if n_sources not in (2, 3):
        raise NotImplementedError(
            "surd_discrete supports 2 or 3 sources; use surd_reference for more."
        )

    src_axes = tuple(range(1, p.ndim))
    p_t = _marginal(p, (0,))
    h_target = _entropy(p_t)
    mi_total = mutual_info_from_pmf(p, 0, src_axes)

    # all non-empty source groups
    groups = []
    for r in range(1, n_sources + 1):
        groups.extend(combinations(src_axes, r))
    # specific information of every group about every target state
    spec = {g: _specific_information(p, g) for g in groups}

    redundant: dict = {}
    unique: dict = {}
    synergistic: dict = {}
    for g in groups:
        redundant[frozenset(g)] = 0.0
        if len(g) == 1:
            unique[frozenset(g)] = 0.0
        if len(g) >= 2:
            synergistic[frozenset(g)] = 0.0

    n_t = p_t.shape[0]
    for ti in range(n_t):
        w = p_t[ti]
        if w <= _EPS:
            continue
        # rank groups by specific information at this target state (ascending)
        ranked = sorted(groups, key=lambda g: spec[g][ti])
        running = 0.0
        prev_val = 0.0
        for rank_idx, g in enumerate(ranked):
            val = spec[g][ti]
            increment = max(val - prev_val, 0.0)
            # groups at or above this rank all share the increment -> redundant
            sharers = ranked[rank_idx:]
            orders = [len(s) for s in sharers]
            min_order = min(orders) if orders else len(g)
            if min_order == 1:
                # increment is redundant among singletons present in the sharer set
                singleton_sharers = [s for s in sharers if len(s) == 1]
                key = frozenset().union(*singleton_sharers) if singleton_sharers else frozenset(g)
                if len(singleton_sharers) == 1:
                    unique[frozenset(singleton_sharers[0])] += w * increment
                else:
                    redundant[key] = redundant.get(key, 0.0) + w * increment
            else:
                # the increment requires a joint group -> synergistic
                key = frozenset(min(sharers, key=len))
                synergistic[key] = synergistic.get(key, 0.0) + w * increment
            prev_val = val
            running += increment

    # information leak: target information not explained by the full source set
    info_leak = float(np.clip(1.0 - mi_total / (h_target + _EPS), 0.0, 1.0))

    # prune all-zero keys for readability
    redundant = {k: v for k, v in redundant.items() if v > 1e-9}
    synergistic = {k: v for k, v in synergistic.items() if v > 1e-9}
    return SURDResult(
        redundant=redundant,
        unique=unique,
        synergistic=synergistic,
        info_leak=info_leak,
        mi_total=mi_total,
        h_target=h_target,
        n_sources=n_sources,
    )


# --------------------------------------------------------------------------- #
# reference-implementation adapter (authoritative path)
# --------------------------------------------------------------------------- #
def surd_reference(p: np.ndarray):
    """
    Call the authors' reference SURD if importable. Returns whatever the reference
    returns (typically dicts of redundant/unique and synergistic information),
    leaving interpretation to the caller. Raises ImportError with instructions if
    the reference is not on the path.

    To install:
        git clone https://github.com/Computational-Turbulence-Group/SURD
        # add SURD/ to PYTHONPATH, or copy SURD/utils next to this package
    The reference exposes a `surd`-style function operating on a joint pmf; consult
    its examples/ notebooks for the exact signature of the installed version, then
    wire it in below. We keep this adapter intentionally thin so it tracks the
    upstream API rather than freezing a possibly-stale signature.
    """
    try:
        # the upstream layout has historically been utils/* with an it_tools module;
        # try the common entry points without committing to one.
        import importlib

        for modname, fname in [
            ("surd", "surd"),
            ("utils.it_tools", "surd"),
            ("it_tools", "surd"),
        ]:
            try:
                mod = importlib.import_module(modname)
                fn = getattr(mod, fname)
                return fn(_normalise(p))
            except (ModuleNotFoundError, AttributeError):
                continue
        raise ImportError
    except ImportError as exc:  # noqa: B904
        raise ImportError(
            "Reference SURD not found. Clone "
            "https://github.com/Computational-Turbulence-Group/SURD and put it on "
            "PYTHONPATH, then adapt surd_reference() to the installed signature. "
            "Until then, use surd_discrete() for 2-3 sources."
        ) from exc


# --------------------------------------------------------------------------- #
# staged (state-aware stand-in) SURD
# --------------------------------------------------------------------------- #
def surd_by_stage(
    target_labels: np.ndarray,
    source_labels: np.ndarray,
    stage_index: np.ndarray,
    shape: tuple[int, ...],
    *,
    stage_names: dict[int, str] | None = None,
) -> dict[str, SURDResult]:
    """
    Apply discrete SURD within each encounter stage separately, a cheap stand-in
    for the released state-aware SURD (ALD-Lab/SURD-states).

    target_labels : (n_samples,) int labels of the future target.
    source_labels : (n_samples, n_sources) int labels of the sources.
    stage_index   : (n_samples,) int in {0=baseline,1=impact,2=peak,3=recovery}
                    matching the four stages used throughout the manuscript.
    shape         : per-variable cardinality (target_card, s1_card, ...).
    """
    from .estimators import joint_pmf

    stage_names = stage_names or {0: "baseline", 1: "impact", 2: "peak", 3: "recovery"}
    out: dict[str, SURDResult] = {}
    target_labels = np.asarray(target_labels).ravel()
    source_labels = np.atleast_2d(source_labels)
    if source_labels.shape[0] != target_labels.shape[0]:
        source_labels = source_labels.T
    for s, name in stage_names.items():
        mask = stage_index == s
        if mask.sum() < 50:  # too few samples for a stable pmf
            continue
        labels = np.column_stack([target_labels[mask], source_labels[mask]])
        p = joint_pmf(labels, shape)
        out[name] = surd_discrete(p)
    return out


def pretty_print_surd(res: SURDResult, source_names: dict[int, str] | None = None) -> str:
    """Format a SURD result for logs / appendix, normalised by H(target)."""
    nm = source_names or {}

    def lab(fs):
        return "+".join(nm.get(i, f"S{i}") for i in sorted(fs))

    lines = [
        f"H(target)={res.h_target:.4f} bits   I(target;all)={res.mi_total:.4f} bits   "
        f"info_leak={res.info_leak:.3f}",
        f"{'component':<28}{'bits':>10}{'frac H(T)':>12}",
        "-" * 50,
    ]
    h = res.h_target or 1.0
    for k, v in sorted(res.unique.items(), key=lambda kv: -kv[1]):
        lines.append(f"{'U[' + lab(k) + ']':<28}{v:>10.4f}{v / h:>12.3f}")
    for k, v in sorted(res.redundant.items(), key=lambda kv: -kv[1]):
        lines.append(f"{'R[' + lab(k) + ']':<28}{v:>10.4f}{v / h:>12.3f}")
    for k, v in sorted(res.synergistic.items(), key=lambda kv: -kv[1]):
        lines.append(f"{'S[' + lab(k) + ']':<28}{v:>10.4f}{v / h:>12.3f}")
    return "\n".join(lines)
