"""Session 31 Track F: assemble the ROM eval (Q1/Q2/Q3) into paper-ready form.

Single source of truth: every paper-bound quantity lands in
``outputs/session31/numbers.json`` keyed by a stable name, carrying
``{value, n, ci_low, ci_high}`` plus descriptor fields
``(model, predictor, window, target, metric)``. The manuscript sees the numbers
only through ``\\providecommand`` macros rendered from that file, so a numeric
literal never appears in a ``.tex`` source. This mirrors the Session 28 provenance
pattern (``scripts/session28/eval_all.py`` + ``emit_macros.py``).

This module holds the PURE, unit-tested pieces:

* case-clustered bootstrap over Test B cases (:func:`case_clustered_bootstrap`,
  :func:`random_row_bootstrap`, :func:`percentile_ci`) with sum-based aggregators
  for aggregated VRMSE (num/den), aggregated R^2 (SS_res/SS_tot) and mean-of-row
  metrics (SSIM);
* the headline-cell builder that walks the q1/q2/q3 payloads
  (:func:`build_headline_records`);
* merge + validation (no duplicate names/macros, required ``value`` key)
  (:func:`merge_and_validate`) and macro emission (:func:`emit_macros`).

The heavy per-row recompute (probe / pressure-estimator / decode-floor) that feeds
the bootstrap lives in ``scripts/session31/bootstrap_cis.py``; the CLI that stitches
everything together is ``scripts/session31/assemble_report.py``.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

# --------------------------------------------------------------------------- keys
CANONICAL_MODELS = (
    "jepa_nowake",
    "jepa_wake",
    "ae_nowake",
    "ae_wake",
    "supervised_only",
    "regAE",
)

# Alphabetic-only short names (LaTeX command names must be alphabetic).
MODEL_SHORT = {
    "jepa_nowake": "JepaNoWake",
    "jepa_wake": "JepaWake",
    "ae_nowake": "AeNoWake",
    "ae_wake": "AeWake",
    "supervised_only": "SupOnly",
    "regAE": "RegAE",
}
MODEL_LABEL = {
    "jepa_nowake": "JEPA",
    "jepa_wake": "JEPA+wake",
    "ae_nowake": "AE",
    "ae_wake": "AE+wake",
    "supervised_only": "supervised",
    "regAE": "regAE",
}

TARGETS = ("C_L", "C_D", "wake_enstrophy", "circulation_pos", "circulation_neg")
TARGET_SHORT = {
    "C_L": "CL",
    "C_D": "CD",
    "wake_enstrophy": "WakeEns",
    "circulation_pos": "CircPos",
    "circulation_neg": "CircNeg",
}
TARGET_LABEL = {
    "C_L": r"$C_L$",
    "C_D": r"$C_D$",
    "wake_enstrophy": r"$\Omega_w$",
    "circulation_pos": r"$\Gamma^{+}$",
    "circulation_neg": r"$\Gamma^{-}$",
}

# Horizon int -> alphabetic macro token.
HORIZON_WORD = {1: "HiOne", 8: "HiEight", 16: "HiSixteen"}
HEADLINE_HORIZONS = (1, 8, 16)

# The primary matched predictor for the headline forecast cells.
PRIMARY_PREDICTOR = "resunet_matched"


# =========================================================================== bootstrap
def _case_row_index(case_ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return ``(unique_cases, [row_index_array_per_case])`` preserving first-seen order."""
    case_ids = np.asarray(case_ids)
    uniq, first_idx = np.unique(case_ids, return_index=True)
    order = np.argsort(first_idx)
    uniq = uniq[order]
    rows = [np.where(case_ids == c)[0] for c in uniq]
    return uniq, rows


def case_clustered_bootstrap(
    case_ids: np.ndarray,
    contribs: Mapping[str, np.ndarray],
    aggregate: Callable[[Mapping[str, np.ndarray]], float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Case-clustered (block) bootstrap of an aggregate metric over Test B cases.

    Resamples the unique ``case_ids`` with replacement; a case drawn ``k`` times
    contributes its rows ``k`` times, so the whole error-correlation structure
    within a case is preserved (the honest ROM sampling unit is the case, not the
    frame). ``aggregate`` maps a dict of gathered per-row contribution arrays to a
    scalar; use the provided :func:`agg_vrmse` / :func:`agg_r2` / :func:`agg_mean`.

    Args:
        case_ids: ``(N,)`` per-row case identifier.
        contribs: named ``(N,)`` per-row contribution arrays (e.g. ``num``/``den``).
        aggregate: reducer over the gathered contribution dict.
        n_boot: number of bootstrap resamples.
        seed: RNG seed.

    Returns:
        ``(n_boot,)`` array of aggregate values.
    """
    case_ids = np.asarray(case_ids)
    contribs = {k: np.asarray(v) for k, v in contribs.items()}
    n = case_ids.shape[0]
    for k, v in contribs.items():
        if v.shape[0] != n:
            raise ValueError(f"contrib {k!r} has {v.shape[0]} rows, expected {n}")
    uniq, rows = _case_row_index(case_ids)
    c = len(uniq)
    rng = np.random.default_rng(seed)
    out = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        pick = rng.integers(0, c, size=c)
        idx = np.concatenate([rows[j] for j in pick])
        gathered = {k: v[idx] for k, v in contribs.items()}
        out[b] = float(aggregate(gathered))
    return out


def random_row_bootstrap(
    contribs: Mapping[str, np.ndarray],
    aggregate: Callable[[Mapping[str, np.ndarray]], float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Ordinary i.i.d. row bootstrap (resamples rows, ignores case structure).

    Present mainly as the naive comparator: on case-structured data with an
    outlier case it UNDER-estimates the CI width relative to
    :func:`case_clustered_bootstrap` (see the tests).
    """
    contribs = {k: np.asarray(v) for k, v in contribs.items()}
    n = next(iter(contribs.values())).shape[0]
    rng = np.random.default_rng(seed)
    out = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        gathered = {k: v[idx] for k, v in contribs.items()}
        out[b] = float(aggregate(gathered))
    return out


def percentile_ci(samples: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    """Percentile CI ``(p_lo, p_hi)`` from bootstrap samples (NaNs dropped)."""
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(s, lo)), float(np.percentile(s, hi)))


# --- sum-based aggregators (the metric recomputed from per-row contributions) ---
def agg_vrmse(c: Mapping[str, np.ndarray]) -> float:
    """Aggregated VRMSE ``sqrt(sum(num) / sum(den))`` from per-row num/den."""
    den = float(np.asarray(c["den"], dtype=np.float64).sum())
    if den <= 0.0:
        return float("nan")
    num = float(np.asarray(c["num"], dtype=np.float64).sum())
    return float(np.sqrt(num / den))


def agg_r2(c: Mapping[str, np.ndarray]) -> float:
    """Aggregated R^2 ``1 - sum(ss_res) / sum(ss_tot)`` from per-row SS contributions."""
    sst = float(np.asarray(c["ss_tot"], dtype=np.float64).sum())
    if sst <= 0.0:
        return float("nan")
    sse = float(np.asarray(c["ss_res"], dtype=np.float64).sum())
    return float(1.0 - sse / sst)


def agg_mean(c: Mapping[str, np.ndarray]) -> float:
    """Mean of a per-row metric (e.g. per-frame SSIM)."""
    v = np.asarray(c["val"], dtype=np.float64)
    return float(v.mean()) if v.size else float("nan")


def vrmse_contribs(pred: np.ndarray, true: np.ndarray) -> dict[str, np.ndarray]:
    """Per-row (num, den) for the aggregated VRMSE, mean_true fixed at the full-set mean.

    ``num_i = sum_cells (pred_i - true_i)^2`` and
    ``den_i = sum_cells (true_i - mean_true)^2`` with ``mean_true`` the single global
    mean over ALL rows and cells (so ``sqrt(sum num / sum den)`` equals
    ``aggregated_vrmse``). This makes the bootstrap a clean ratio-of-sums over the
    per-case blocks.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs true {true.shape}")
    n = pred.shape[0]
    mean_true = float(true.mean())
    num = ((pred - true) ** 2).reshape(n, -1).sum(axis=1)
    den = ((true - mean_true) ** 2).reshape(n, -1).sum(axis=1)
    return {"num": num, "den": den}


def three_curve_field_contribs(
    model_pred: np.ndarray,
    floor_pred: np.ndarray,
    persist_pred: np.ndarray,
    true: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-row (num, den) contribs for the Q2 three-curve field VRMSE.

    ``model``, ``floor`` and ``persistence`` all score against the SAME ``true``
    field at ``t + h`` (the DNS omega at the target frame), so they share the
    aggregated-VRMSE denominator (variance about the single global mean of
    ``true``); only the numerator differs. Returns ``{which: {num, den}}`` for
    ``which in ('model', 'floor', 'persistence')`` so each curve can be
    case-clustered-bootstrapped independently while ``agg_vrmse`` reproduces
    ``aggregated_vrmse`` per curve.
    """
    return {
        "model": vrmse_contribs(model_pred, true),
        "floor": vrmse_contribs(floor_pred, true),
        "persistence": vrmse_contribs(persist_pred, true),
    }


def r2_contribs(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    """Per-row (ss_res, ss_tot) for aggregated R^2, ybar fixed at the full-set mean."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    ybar = float(y_true.mean())
    ss_res = (y_true - y_pred) ** 2
    ss_tot = (y_true - ybar) ** 2
    return {"ss_res": ss_res, "ss_tot": ss_tot}


def bootstrap_ci(
    case_ids: np.ndarray,
    contribs: Mapping[str, np.ndarray],
    aggregate: Callable[[Mapping[str, np.ndarray]], float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Point estimate (full sample) + case-clustered percentile CI, as a record fragment."""
    point = float(aggregate({k: np.asarray(v) for k, v in contribs.items()}))
    samples = case_clustered_bootstrap(case_ids, contribs, aggregate, n_boot=n_boot, seed=seed)
    lo, hi = percentile_ci(samples)
    n = int(next(iter(contribs.values())).shape[0])
    return {"value": point, "ci_low": lo, "ci_high": hi, "n": n}


# =========================================================================== records
def _rec(value, macro, fmt="%.2f", **meta) -> dict:
    rec = {"value": float(value), "macro": macro, "fmt": fmt}
    rec.update({k: v for k, v in meta.items() if v is not None})
    return rec


def name_and_macro(kind: str, model: str, **parts) -> tuple[str, str]:
    """Stable (number-name, macro-name) for a headline cell.

    Number names are dotted and descriptive; macro names are the alphabetic
    CamelCase LaTeX command derived from the same fields.
    """
    ms = MODEL_SHORT[model]
    if kind == "repr_floor_vrmse":
        return f"q1.floor.vrmse.{model}", f"ReprFloorVrmse{ms}"
    if kind == "repr_floor_ssim":
        return f"q1.floor.ssim.{model}", f"ReprFloorSsim{ms}"
    if kind == "repr_lin":
        t = parts["target"]
        return f"q1.probe.linear.{t}.{model}", f"ReprLin{TARGET_SHORT[t]}{ms}"
    if kind == "repr_mlp":
        t = parts["target"]
        return f"q1.probe.mlp.{t}.{model}", f"ReprMlp{TARGET_SHORT[t]}{ms}"
    if kind == "fore_field":
        which, h = parts["which"], parts["h"]
        cap = {"model": "", "floor": "Floor", "persistence": "Pers"}[which]
        return (
            f"q2.field_vrmse.{which}.h{h}.{model}",
            f"ForeFieldVrmse{cap}{ms}{HORIZON_WORD[h]}",
        )
    if kind == "fore_gap":
        h = parts["h"]
        return f"q2.field_gap.h{h}.{model}", f"ForeFieldGap{ms}{HORIZON_WORD[h]}"
    if kind == "fore_clos":
        t, h = parts["target"], parts["h"]
        return (
            f"q2.closure.mlp.{t}.h{h}.{model}",
            f"ForeClos{TARGET_SHORT[t]}{ms}{HORIZON_WORD[h]}",
        )
    if kind == "fore_merit":
        h = parts["h"]
        return f"q2.merit.mean_obs.h{h}.{model}", f"ForeMerit{ms}{HORIZON_WORD[h]}"
    if kind == "pres_obs":
        t = parts["target"]
        return f"q3.pressure_obs.linear.{t}.{model}", f"PresObs{TARGET_SHORT[t]}{ms}"
    if kind == "pres_obs_mean":
        return f"q3.pressure_obs.mean_linear.{model}", f"PresObsMean{ms}"
    if kind == "pres_field_vrmse":
        return f"q3.pressure_field.vrmse.{model}", f"PresFieldVrmse{ms}"
    if kind == "pres_latent_r2":
        return f"q3.pressure_latent.r2.{model}", f"PresLatentRtwo{ms}"
    raise ValueError(f"unknown kind {kind!r}")


def build_headline_records(
    q1: dict,
    q2: dict,
    q3: dict,
    *,
    window: str = "windowed",
    predictor: str = PRIMARY_PREDICTOR,
    models: Sequence[str] = CANONICAL_MODELS,
    horizons: Sequence[int] = HEADLINE_HORIZONS,
) -> dict[str, dict]:
    """Merge the headline cells from the three ROM payloads into value-only records.

    Bootstrap CIs are attached later (:func:`apply_cis`). The chosen ``window`` scope
    (``windowed`` = Track 0.B impact+relaxation frames, comparable to Q2/Q3) applies to
    the Q1 decode-floor and probes; Q2/Q3 are already window-scoped.
    """
    recs: dict[str, dict] = {}

    def put(name, macro, value, fmt="%.2f", **meta):
        if name in recs:
            raise ValueError(f"duplicate headline name {name!r}")
        recs[name] = _rec(value, macro, fmt, **meta)

    for m in models:
        # ---- Q1: decode floor + representation probes ----
        floor = q1["models"][m]["decode_floor"][window]
        n, mc = name_and_macro("repr_floor_vrmse", m)
        put(
            n,
            mc,
            floor["vrmse"],
            "%.3f",
            n=floor["n"],
            model=m,
            window=window,
            metric="decode_floor_vrmse",
            source="q1",
        )
        n, mc = name_and_macro("repr_floor_ssim", m)
        put(
            n,
            mc,
            floor["ssim"],
            "%.3f",
            n=floor["n"],
            model=m,
            window=window,
            metric="decode_floor_ssim",
            source="q1",
        )
        probes = q1["models"][m]["probes"][window]
        for t in TARGETS:
            n, mc = name_and_macro("repr_lin", m, target=t)
            put(
                n,
                mc,
                probes[t]["linear_r2"],
                n=probes[t]["n_eval"],
                model=m,
                window=window,
                target=t,
                metric="probe_linear_r2",
                source="q1",
            )
            n, mc = name_and_macro("repr_mlp", m, target=t)
            put(
                n,
                mc,
                probes[t]["mlp_r2"],
                n=probes[t]["n_eval"],
                model=m,
                window=window,
                target=t,
                metric="probe_mlp_r2",
                source="q1",
            )

        # ---- Q2: field VRMSE (model/floor/persistence), gap, closure, merit ----
        pred = q2["models"][m]["predictors"][predictor]
        fv = pred["field_vrmse"]
        hz = list(fv["horizons"])
        for h in horizons:
            i = hz.index(h)
            for which in ("model", "floor", "persistence"):
                n, mc = name_and_macro("fore_field", m, which=which, h=h)
                put(
                    n,
                    mc,
                    fv[which][i],
                    "%.3f",
                    n=fv["n"][i],
                    model=m,
                    predictor=predictor,
                    window="windowed",
                    horizon=h,
                    metric=f"field_vrmse_{which}",
                    source="q2",
                )
            n, mc = name_and_macro("fore_gap", m, h=h)
            put(
                n,
                mc,
                fv["model"][i] - fv["floor"][i],
                "%.3f",
                n=fv["n"][i],
                model=m,
                predictor=predictor,
                window="windowed",
                horizon=h,
                metric="field_vrmse_model_minus_floor",
                source="q2",
            )
        oc = pred["observable_closure"]
        for h in horizons:
            i = list(oc["C_L"]["horizons"]).index(h)
            per = []
            for t in TARGETS:
                r2 = oc[t]["model_mlp_r2"][i]
                per.append(r2)
                n, mc = name_and_macro("fore_clos", m, target=t, h=h)
                put(
                    n,
                    mc,
                    r2,
                    n=oc[t]["n"][i],
                    model=m,
                    predictor=predictor,
                    window="windowed",
                    horizon=h,
                    target=t,
                    metric="closure_mlp_r2",
                    source="q2",
                )
            n, mc = name_and_macro("fore_merit", m, h=h)
            put(
                n,
                mc,
                float(np.mean(per)),
                n=oc["C_L"]["n"][i],
                model=m,
                predictor=predictor,
                window="windowed",
                horizon=h,
                metric="closure_mlp_r2_mean_obs",
                source="q2",
            )

        # ---- Q3: pressure -> obs / field ----
        po = q3["models"][m]["pressure_to_obs"]
        per_obs = po["per_observable"]
        lin = []
        for t in TARGETS:
            r2 = per_obs[t]["pressure_linear_r2"]
            lin.append(r2)
            n, mc = name_and_macro("pres_obs", m, target=t)
            put(
                n,
                mc,
                r2,
                n=per_obs[t]["n"],
                model=m,
                window="windowed",
                target=t,
                metric="pressure_obs_linear_r2",
                source="q3",
            )
        n, mc = name_and_macro("pres_obs_mean", m)
        put(
            n,
            mc,
            float(np.mean(lin)),
            n=per_obs["C_L"]["n"],
            model=m,
            window="windowed",
            metric="pressure_obs_linear_r2_mean",
            source="q3",
        )
        n, mc = name_and_macro("pres_latent_r2", m)
        put(
            n,
            mc,
            po["latent_r2"],
            model=m,
            window="windowed",
            metric="pressure_latent_r2",
            source="q3",
        )
        pf = q3["models"][m].get("pressure_to_field")
        if pf is not None:
            n, mc = name_and_macro("pres_field_vrmse", m)
            put(
                n,
                mc,
                pf["pressure_vrmse"],
                "%.3f",
                n=pf["n"],
                model=m,
                window="windowed",
                metric="pressure_field_vrmse",
                source="q3",
            )
    return recs


def apply_cis(records: dict[str, dict], ci_dump: Mapping[str, Mapping]) -> dict[str, dict]:
    """Overlay ``{ci_low, ci_high, n}`` from a CI dump onto value records by name.

    Records whose name is absent from ``ci_dump`` keep ``ci_low``/``ci_high`` unset
    (they render as a value macro with no CI companion). If the CI dump carries a
    recomputed point estimate that disagrees with the payload value by more than
    ``1e-6`` the payload value is kept and a ``note`` records the recompute delta.
    """
    out = {name: dict(rec) for name, rec in records.items()}
    for name, ci in ci_dump.items():
        if name not in out:
            continue
        rec = out[name]
        if "ci_low" in ci and ci["ci_low"] is not None:
            rec["ci_low"] = float(ci["ci_low"])
        if "ci_high" in ci and ci["ci_high"] is not None:
            rec["ci_high"] = float(ci["ci_high"])
        if ci.get("n") is not None:
            rec["n"] = int(ci["n"])
        rp = ci.get("recompute_value")
        if rp is not None and abs(float(rp) - float(rec["value"])) > 1e-6:
            rec["note"] = f"recompute={float(rp):.4f}"
    return out


# =========================================================================== merge / emit
ALLOWED_KEYS = {
    "value",
    "macro",
    "fmt",
    "ci_low",
    "ci_high",
    "n",
    "model",
    "predictor",
    "window",
    "target",
    "metric",
    "horizon",
    "source",
    "note",
}


def validate_record(name: str, rec: Mapping) -> list[str]:
    """Return a list of validation errors for a single number record."""
    errs: list[str] = []
    if "value" not in rec:
        errs.append(f"{name}: missing required 'value'")
    unknown = set(rec) - ALLOWED_KEYS
    if unknown:
        errs.append(f"{name}: unknown keys {sorted(unknown)}")
    macro = rec.get("macro")
    if macro is not None and not str(macro).isalpha():
        errs.append(f"{name}: macro {macro!r} must be alphabetic")
    return errs


def merge_and_validate(parts: Iterable[Mapping]) -> dict[str, dict]:
    """Accrete ``{"numbers": {...}}`` parts, refusing duplicate names and macros.

    Raises:
        ValueError: on any validation error (missing value, unknown key, non-alpha
            macro, duplicate number name, duplicate macro name).
    """
    numbers: dict[str, dict] = {}
    macros_seen: dict[str, str] = {}
    errors: list[str] = []
    for part in parts:
        for name, rec in part.get("numbers", {}).items():
            errors.extend(validate_record(name, rec))
            if name in numbers:
                errors.append(f"duplicate number name {name!r}")
                continue
            macro = rec.get("macro")
            if macro:
                if macro in macros_seen:
                    errors.append(f"duplicate macro {macro!r} ({macros_seen[macro]} vs {name})")
                else:
                    macros_seen[macro] = name
            numbers[name] = dict(rec)
    if errors:
        raise ValueError("numbers validation failed:\n  " + "\n  ".join(errors))
    return numbers


def _fmt_value(value, fmt: str) -> str:
    if fmt == "%s":
        return str(value)
    return fmt % value


def emit_macros(numbers: Mapping[str, Mapping], *, header: Sequence[str] = ()) -> str:
    """Render ``\\providecommand`` lines: one per macro, plus ``lo``/``hi`` CI companions."""
    lines = list(header)
    for name in sorted(numbers):
        rec = numbers[name]
        macro = rec.get("macro")
        if not macro:
            continue
        fmt = rec.get("fmt", "%.2f")
        lines.append(f"\\providecommand{{\\{macro}}}{{{_fmt_value(rec['value'], fmt)}}}")
        for suffix, key in (("lo", "ci_low"), ("hi", "ci_high")):
            if rec.get(key) is not None:
                lines.append(
                    f"\\providecommand{{\\{macro}{suffix}}}{{{_fmt_value(rec[key], fmt)}}}"
                )
    return "\n".join(lines) + "\n"
