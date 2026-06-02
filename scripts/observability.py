"""
infotheory.observability
=========================

Information-theoretic observability and an aIND-style informative/residual split,
specialised to the vortex-jepa question: which reduced state (predictive JEPA,
reconstructive AE, or POD) carries the most information about a *future* physical
observable, and is that information recoverable from sparse wall pressure.

This is the principled version of two empirical results in the manuscript:

  * Section 4.7 ("the predictive state is the most observable from the wall")
    currently reports a kernel-ridge R^2 of pressure -> latent. Here we replace
    (or accompany) that with the information-theoretic observability of
    Lozano-Duran & Arranz (PRR 2022), used throughout Arranz & Lozano-Duran
    (JFM 2024, eq. 3.11):

        O_{S -> T} = I(T; S) / H(T),

    the fraction of the target's information that the source resolves. O = 1
    means T is a deterministic function of S (perfectly observable); O = 0 means
    S carries nothing about T. Reporting O for each latent family, with
    T = future wake enstrophy and S = {latent} or {wall-pressure taps}, turns the
    R^2 statement into an estimator-independent information statement.

  * The reconstruction-vs-prediction thesis. aIND (Arranz & Lozano-Duran 2024)
    splits a source field into an informative part (all the information about the
    target's future) and a residual part (none). A reconstruction objective must
    carry both; a predictive objective is free to keep only the informative part.
    We do not re-implement the full field-level aIND optimisation here (that is a
    companion-paper-scale effort and the authors release code at
    github.com/Computational-Turbulence-Group/aIND). Instead we provide the
    scalar / latent-level quantities that test the same hypothesis cheaply:
      - the observability O of each family's latent toward each future observable,
      - the residual-information check I(residual; target) used by aIND to certify
        that the discarded part is truly non-informative,
      - an "informative energy fraction" E_I for a latent coordinate, the share of
        its variance that is informative about the target.

All MI/entropy calls go through infotheory.estimators so the uncertainty
triangulation (kNN-vs-hist agreement, surrogate nulls) is uniform.

Nothing in this module fits a neural network or touches torch. It consumes the
scalar design matrices produced by infotheory.io_vortex.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .estimators import (
    entropy_knn,
    mutual_information_knn,
    surrogate_null_mi,
    estimator_robustness_sweep,
    quantise,
    entropy_hist,
)


# --------------------------------------------------------------------------- #
# target entropy (the denominator of observability)
# --------------------------------------------------------------------------- #
def target_entropy(t: np.ndarray, *, n_bins: int = 16, method: str = "hist") -> float:
    """
    Entropy H(T) of the target used as the observability denominator.

    For a continuous target the absolute differential entropy is scale dependent,
    so observability O = I/H is only meaningful if numerator and denominator use
    a consistent discretisation. We therefore default to a histogram H(T) with the
    same binning the histogram MI uses; pass method='knn' only if you also compute
    the numerator with a matched continuous convention and understand the caveat.
    """
    if method == "hist":
        labels = quantise(t, n_bins, strategy="quantile")
        counts = np.bincount(np.asarray(labels).ravel())
        return float(entropy_hist(counts, bias_correct=True))
    elif method == "knn":
        return float(entropy_knn(t, standardise=False))
    raise ValueError(f"unknown method {method!r}")


@dataclass
class ObservabilityResult:
    source_name: str
    target_name: str
    mi: float                # I(T; S), nats
    mi_debiased: float       # excess over surrogate null
    h_target: float          # H(T), nats
    observability: float     # O = mi_debiased / h_target, clipped to [0, 1]
    p_value: float
    z: float
    robustness_spread: float
    n_samples: int
    extra: dict = field(default_factory=dict)


def observability(
    source: np.ndarray,
    target: np.ndarray,
    *,
    source_name: str = "S",
    target_name: str = "T",
    k: int = 4,
    n_bins: int = 16,
    n_surrogate: int = 200,
    random_state: int | None = 0,
) -> ObservabilityResult:
    """
    Compute O_{S -> T} = I(T; S) / H(T) with surrogate null and robustness sweep.

    `source` may be multivariate (e.g. a whole latent vector, or K pressure taps);
    `target` is typically a scalar future observable (e.g. wake enstrophy at H=16).
    The debiased MI is used in the numerator so that a finite-sample MI floor does
    not inflate observability. O is clipped to [0, 1] for reporting (a debiased MI
    can be marginally negative under independence).
    """
    null = surrogate_null_mi(
        target, source,  # I(T; S): target first so multivariate source is the second arg
        estimator=mutual_information_knn,
        n_surrogate=n_surrogate,
        k=k,
        random_state=random_state,
    )
    h_t = target_entropy(target, n_bins=n_bins, method="hist")
    rob = estimator_robustness_sweep(target, source, random_state=random_state)
    o = float(np.clip(null["mi_debiased"] / h_t, 0.0, 1.0)) if h_t > 0 else 0.0
    return ObservabilityResult(
        source_name=source_name,
        target_name=target_name,
        mi=null["mi"],
        mi_debiased=null["mi_debiased"],
        h_target=h_t,
        observability=o,
        p_value=null["p_value"],
        z=null["z"],
        robustness_spread=rob["spread"],
        n_samples=int(np.asarray(target).shape[0]),
    )


def observability_table(
    families: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    *,
    k: int = 4,
    n_bins: int = 16,
    n_surrogate: int = 200,
    random_state: int | None = 0,
) -> list[ObservabilityResult]:
    """
    Cross every latent family (or sensor set) against every target observable.

    `families`: name -> source array (n_samples, d_family). Typical entries:
        'JEPA_d64', 'JEPA_d32', 'Fukami_d64', 'POD_d64', and the sensor variants
        'pressure_K8', 'pressure_K2'.
    `targets` : name -> target array (n_samples,). Typical entries:
        'wake_enstrophy_future', 'CL_future', 'circ_neg_future'.

    The headline prediction the manuscript should be able to state: for
    target='wake_enstrophy_future', O is highest for the predictive family and for
    the pressure sets routed through it, and the reconstructive/POD families are
    markedly lower, mirroring the forecast and recoverability results but on
    estimator-independent information-theoretic ground.
    """
    rows: list[ObservabilityResult] = []
    for tname, tvec in targets.items():
        for fname, fmat in families.items():
            rows.append(
                observability(
                    fmat,
                    tvec,
                    source_name=fname,
                    target_name=tname,
                    k=k,
                    n_bins=n_bins,
                    n_surrogate=n_surrogate,
                    random_state=random_state,
                )
            )
    return rows


# --------------------------------------------------------------------------- #
# aIND residual check and informative-energy fraction (scalar / latent level)
# --------------------------------------------------------------------------- #
def residual_information_check(
    source_coord: np.ndarray,
    informative_estimate: np.ndarray,
    target: np.ndarray,
    *,
    k: int = 4,
    n_surrogate: int = 200,
    random_state: int | None = 0,
) -> dict:
    """
    aIND certification that the residual carries no target information.

    Given a source coordinate phi (e.g. one latent component), an estimate of its
    informative part phi_I (e.g. the part predictable from the target via a
    monotone map, or the projection retained by the predictive model), define the
    residual phi_R = phi - phi_I and verify I(phi_R; target) ~ 0 against a
    surrogate null (Arranz & Lozano-Duran 2024, condition I(Phi_R; Psi_+) = 0).

    Returns the residual MI, its debiased value and p-value, plus the
    informative-energy fraction E_I = var(phi_I)/var(phi) (their eq. 2.8/2.11).
    """
    phi = np.asarray(source_coord, dtype=float).ravel()
    phi_i = np.asarray(informative_estimate, dtype=float).ravel()
    phi_r = phi - phi_i
    null = surrogate_null_mi(
        target, phi_r, n_surrogate=n_surrogate, k=k, random_state=random_state
    )
    var_phi = float(np.var(phi)) or 1.0
    e_i = float(np.var(phi_i) / var_phi)
    return {
        "residual_mi": null["mi"],
        "residual_mi_debiased": null["mi_debiased"],
        "residual_p_value": null["p_value"],
        "informative_energy_fraction": float(np.clip(e_i, 0.0, 1.0)),
        "note": "aIND wants residual_mi_debiased ~ 0 (large p) and E_I close to 1 "
        "for a coordinate that is mostly informative about the target.",
    }


def pretty_print_observability(rows: list[ObservabilityResult]) -> str:
    """Format an observability table as a fixed-width string for logs / appendix."""
    hdr = f"{'target':<24}{'source':<16}{'O':>7}{'I(T;S)':>9}{'I_deb':>9}{'H(T)':>8}{'p':>9}{'spread':>9}"
    lines = [hdr, "-" * len(hdr)]
    for r in sorted(rows, key=lambda x: (x.target_name, -x.observability)):
        lines.append(
            f"{r.target_name:<24}{r.source_name:<16}{r.observability:>7.3f}"
            f"{r.mi:>9.3f}{r.mi_debiased:>9.3f}{r.h_target:>8.3f}"
            f"{r.p_value:>9.3g}{r.robustness_spread:>9.3f}"
        )
    return "\n".join(lines)
