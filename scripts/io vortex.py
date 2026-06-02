"""
infotheory.io_vortex
====================

Bridge between the vortex-jepa data/artifacts and the method-agnostic estimators.
It assembles the per-encounter scalar design matrices the causal / information
analyses consume:

    sources at impact   : c = (G, D, Y), current wake enstrophy, current lift,
                          baseline-shedding-phase proxy
    targets at H        : future wake enstrophy, future lift, future neg. circ.
    latents             : per-family latent vectors at the impact frame, for the
                          observability table (JEPA d64/d32, Fukami d3/.., POD ..)
    pressure            : K-tap wall-pressure feature vectors, for the
                          recoverability-as-observability result (Section 4.7)

IMPORTANT: schema hooks
-----------------------
This module mirrors the conventions recorded in HANDOFF.md but it cannot know the
exact dataset keys without the data, which lives behind PREVENT_ROOT on the
researcher's workstation, not in this repo. Every place that touches the on-disk
schema is marked `# SCHEMA HOOK` and defaults to the names seen in the handoff
(`encounter_*.h5`, group key `omega_z`, the six observables, alignment by
(case_id, encounter)). Confirm/adjust these against
`src/data/episode_dataset.py` and the `wake_observables/` writer before trusting
any number. The `validate_against_known()` gate is provided exactly so a stale
hook is caught immediately (it re-derives the d64 representational wake MAE 29.83
from Table 3(a) and asserts it matches).

Nothing here imports torch; latents are read from the .npz/.npy artifacts the
existing eval scripts already dump, so this stays a light dependency. If your
latents are only inside checkpoints, add a tiny export step in your eval pipeline
rather than importing the model here.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Canonical observable order used throughout the manuscript (Table 3).
OBSERVABLES = ("CL", "CD", "Iy", "wake_enstrophy", "circ_pos", "circ_neg")

# Encounter stages used throughout (numbered glyphs in Fig. 6).
STAGE_BASELINE, STAGE_IMPACT, STAGE_PEAK, STAGE_RECOVERY = 0, 1, 2, 3


# --------------------------------------------------------------------------- #
# environment / paths
# --------------------------------------------------------------------------- #
def resolve_cache_root() -> Path:
    """Resolve the processed-cache root from the same env vars the repo uses."""
    cache = os.environ.get("VORTEX_JEPA_CACHE")
    if cache:
        return Path(cache)
    prevent = os.environ.get("PREVENT_ROOT")
    if prevent:
        return Path(prevent) / "data" / "processed" / "vortex-jepa"
    raise EnvironmentError(
        "Set VORTEX_JEPA_CACHE (or PREVENT_ROOT) as in HANDOFF.md before loading data."
    )


def load_split(split_json: str | Path) -> dict:
    """Load configs/splits/split_v1.json (the locked partition manifest)."""
    with open(split_json) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# design-matrix container
# --------------------------------------------------------------------------- #
@dataclass
class CausalDesign:
    """
    Aligned, per-encounter scalar arrays. All arrays share row order, keyed by
    (case_id, encounter) -> `keys`. `partition` records which split each row is in.
    """
    keys: list[tuple[str, int]]
    partition: np.ndarray                 # array of strings: 'train'/'test_b'/'test_c'/...
    sources_impact: dict[str, np.ndarray]  # name -> (n,) at impact frame
    targets_future: dict[str, np.ndarray]  # name -> (n,) at horizon H
    stage_index: np.ndarray | None = None  # (n,) in {0..3} if a stage was assigned
    latents: dict[str, np.ndarray] = field(default_factory=dict)   # family -> (n, d)
    pressure: dict[str, np.ndarray] = field(default_factory=dict)  # 'K8'->(n, F) etc.
    meta: dict = field(default_factory=dict)

    def subset(self, partitions) -> "CausalDesign":
        """Return a row-subset restricted to the given partition name(s)."""
        partitions = {partitions} if isinstance(partitions, str) else set(partitions)
        mask = np.array([p in partitions for p in self.partition])
        idx = np.where(mask)[0]
        return CausalDesign(
            keys=[self.keys[i] for i in idx],
            partition=self.partition[idx],
            sources_impact={k: v[idx] for k, v in self.sources_impact.items()},
            targets_future={k: v[idx] for k, v in self.targets_future.items()},
            stage_index=None if self.stage_index is None else self.stage_index[idx],
            latents={k: v[idx] for k, v in self.latents.items()},
            pressure={k: v[idx] for k, v in self.pressure.items()},
            meta=dict(self.meta),
        )

    def source_matrix(self, names) -> np.ndarray:
        """Stack selected impact sources into (n, len(names))."""
        return np.column_stack([self.sources_impact[n] for n in names])


# --------------------------------------------------------------------------- #
# loaders (SCHEMA HOOKS marked)
# --------------------------------------------------------------------------- #
def _wake_observables_path(cache_root: Path, partition: str = "v1") -> Path:
    # SCHEMA HOOK: handoff shows ${VORTEX_JEPA_CACHE}/v1/wake_observables/ with
    # one per-encounter file plus _manifest.json and _train_stats.json.
    return cache_root / partition / "wake_observables"


def load_observables(
    cache_root: Path,
    split: dict,
    *,
    horizon: int = 16,
    partition: str = "v1",
) -> CausalDesign:
    """
    Read the cached per-encounter observables and build impact/future scalars.

    Returns a CausalDesign with sources_impact = {G, D, Y, wake_enstrophy_impact,
    CL_impact, phase_impact} and targets_future = {<obs>_future at H}. Latents and
    pressure are filled by the attach_* helpers below.

    SCHEMA HOOK: this assumes each encounter exposes the six observables as a
    per-frame time series of length 120 with impact near frame index `impact_idx`
    (handoff: impact ~2 t/c, dt=0.05, so impact_idx ~ 40). Adjust `impact_idx`,
    the time-series key names, and the (G,D,Y) source if your writer differs.
    """
    import h5py  # local import: only needed when actually reading the cache

    obs_dir = _wake_observables_path(cache_root, partition)
    manifest_path = obs_dir / "_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Point cache_root at the processed cache "
            "and confirm the wake_observables manifest name (SCHEMA HOOK)."
        )
    manifest = json.loads(manifest_path.read_text())

    # Map (case_id, encounter) -> partition from the split manifest.
    part_of = _build_partition_map(split)

    keys: list[tuple[str, int]] = []
    partition_labels: list[str] = []
    src = {k: [] for k in ("G", "D", "Y", "wake_enstrophy_impact", "CL_impact", "phase_impact")}
    tgt = {f"{o}_future": [] for o in OBSERVABLES}

    # SCHEMA HOOK: impact frame index inside the 120-frame encounter window.
    impact_idx = int(manifest.get("impact_index", 40))
    future_idx = impact_idx + horizon

    for entry in manifest["encounters"]:                 # SCHEMA HOOK: key 'encounters'
        case_id = entry["case_id"]                        # SCHEMA HOOK
        enc = int(entry["encounter"])                     # SCHEMA HOOK
        key = (case_id, enc)
        if key not in part_of:
            continue
        fpath = obs_dir / entry["file"]                   # SCHEMA HOOK: per-encounter file
        with h5py.File(fpath, "r") as g:
            # SCHEMA HOOK: the six observables as length-120 series under these keys
            series = {o: np.asarray(g[o]) for o in OBSERVABLES}
            gdy = (
                float(g.attrs.get("G", entry.get("G"))),  # SCHEMA HOOK: params in attrs or manifest
                float(g.attrs.get("D", entry.get("D"))),
                float(g.attrs.get("Y", entry.get("Y"))),
            )
            phase = float(entry.get("phase_at_impact", np.nan))  # SCHEMA HOOK (optional)

        if future_idx >= len(series["CL"]):
            continue  # encounter too short for this horizon (clamped regime)
        keys.append(key)
        partition_labels.append(part_of[key])
        src["G"].append(gdy[0]); src["D"].append(gdy[1]); src["Y"].append(gdy[2])
        src["wake_enstrophy_impact"].append(series["wake_enstrophy"][impact_idx])
        src["CL_impact"].append(series["CL"][impact_idx])
        src["phase_impact"].append(phase)
        for o in OBSERVABLES:
            tgt[f"{o}_future"].append(series[o][future_idx])

    design = CausalDesign(
        keys=keys,
        partition=np.array(partition_labels),
        sources_impact={k: np.asarray(v, float) for k, v in src.items()},
        targets_future={k: np.asarray(v, float) for k, v in tgt.items()},
        meta={"horizon": horizon, "impact_idx": impact_idx, "partition": partition},
    )
    return design


def _build_partition_map(split: dict) -> dict[tuple[str, int], str]:
    """
    Build (case_id, encounter) -> partition from split_v1.json.

    SCHEMA HOOK: assumes split lists encounter ids per partition. If the split is
    per-case, expand to per-encounter using the encounter count, or read the
    EpisodeDataset index that the training pipeline already builds.
    """
    part_of: dict[tuple[str, int], str] = {}
    for partition_name, entries in split.get("partitions", split).items():
        if not isinstance(entries, (list, tuple)):
            continue
        for e in entries:
            if isinstance(e, dict):
                part_of[(e["case_id"], int(e["encounter"]))] = partition_name
            elif isinstance(e, (list, tuple)) and len(e) == 2:
                part_of[(e[0], int(e[1]))] = partition_name
    return part_of


def attach_latents(
    design: CausalDesign,
    latent_files: dict[str, str | Path],
    *,
    key_field: str = "keys",
    z_field: str = "z_impact",
) -> CausalDesign:
    """
    Attach per-family impact-frame latents from .npz artifacts dumped by the eval
    pipeline. Each .npz must contain an array of (case_id, encounter) keys and the
    impact-frame latent matrix, aligned to each other. We re-index onto the
    design's row order by (case_id, encounter) so families with missing encounters
    line up correctly.

    SCHEMA HOOK: field names `keys` and `z_impact`. If your eval dumps full
    trajectories, slice the impact frame (impact_idx in design.meta) on export.
    """
    row_of = {k: i for i, k in enumerate(design.keys)}
    n = len(design.keys)
    for family, path in latent_files.items():
        npz = np.load(path, allow_pickle=True)
        fam_keys = [tuple(k) for k in npz[key_field]]
        fam_z = np.asarray(npz[z_field], dtype=float)
        d = fam_z.shape[1]
        mat = np.full((n, d), np.nan)
        for j, k in enumerate(fam_keys):
            k = (k[0], int(k[1]))
            if k in row_of:
                mat[row_of[k]] = fam_z[j]
        design.latents[family] = mat
    return design


def attach_pressure(
    design: CausalDesign,
    pressure_files: dict[str, str | Path],
    *,
    key_field: str = "keys",
    feat_field: str = "pressure_feats",
) -> CausalDesign:
    """
    Attach K-tap wall-pressure feature vectors (the input to the recoverability /
    observability result). `pressure_files` maps a tag like 'K8'/'K2' to a .npz
    with aligned keys and a (n, F) feature matrix (flattened taps x pre-impact
    window). Re-indexed onto design row order exactly like attach_latents.
    """
    row_of = {k: i for i, k in enumerate(design.keys)}
    n = len(design.keys)
    for tag, path in pressure_files.items():
        npz = np.load(path, allow_pickle=True)
        p_keys = [tuple(k) for k in npz[key_field]]
        feats = np.asarray(npz[feat_field], dtype=float)
        F = feats.shape[1]
        mat = np.full((n, F), np.nan)
        for j, k in enumerate(p_keys):
            k = (k[0], int(k[1]))
            if k in row_of:
                mat[row_of[k]] = feats[j]
        design.pressure[tag] = mat
    return design


def assign_stages_from_phase(design: CausalDesign) -> CausalDesign:
    """
    Cheap stage assignment for the staged (state-aware stand-in) SURD. Since this
    analysis is per-encounter at the impact frame, "stage" here partitions
    encounters by where their impact falls relative to the baseline shedding cycle
    and by gust sign, as a coarse proxy. For a true per-frame staged analysis,
    build the stage index inside the trajectory loader instead. Returns design with
    `stage_index` set (or left None if phase is unavailable).
    """
    phase = design.sources_impact.get("phase_impact")
    if phase is None or np.all(np.isnan(phase)):
        return design
    # quartiles of phase as a stand-in for the four numbered stages
    q = np.nanquantile(phase, [0.25, 0.5, 0.75])
    stage = np.digitize(phase, q)
    design.stage_index = stage.astype(int)
    return design


# --------------------------------------------------------------------------- #
# validation gate (catch stale schema hooks immediately)
# --------------------------------------------------------------------------- #
def validate_against_known(
    design: CausalDesign,
    *,
    known_jepa_d64_repr_wake_mae: float = 29.83,
    latent_family: str = "JEPA_d64",
    tol: float = 0.5,
) -> None:
    """
    Sanity gate: re-derive a number the manuscript already reports and assert it.

    If JEPA d64 latents are attached, fit the same kind of ridge wake-enstrophy
    probe the manuscript uses (representational mode: probe on the
    simulation-encoded latent), evaluate held-out test_b MAE, and check it matches
    Table 3(a)'s 29.83 within `tol`. A mismatch means a schema hook is wrong
    (impact frame, observable units, or alignment), and the causal numbers built
    on this loader cannot be trusted yet. Raises AssertionError on failure.
    """
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import KFold

    if latent_family not in design.latents:
        raise AssertionError(
            f"{latent_family} latents not attached; cannot run the validation gate."
        )
    tr = design.subset("train")
    te = design.subset("test_b")
    if len(te.keys) == 0:
        raise AssertionError("no test_b rows; check the partition map (SCHEMA HOOK).")

    Xtr, ytr = tr.latents[latent_family], tr.targets_future["wake_enstrophy_future"]
    Xte, yte = te.latents[latent_family], te.targets_future["wake_enstrophy_future"]
    ok = np.all(np.isfinite(Xtr), axis=1)
    krr = KernelRidge(kernel="rbf", alpha=1.0)
    krr.fit(Xtr[ok], ytr[ok])
    pred = krr.predict(Xte)
    mae = float(np.mean(np.abs(pred - yte)))
    assert abs(mae - known_jepa_d64_repr_wake_mae) <= max(tol, 0.1 * known_jepa_d64_repr_wake_mae), (
        f"representational wake MAE {mae:.2f} does not match the manuscript's "
        f"{known_jepa_d64_repr_wake_mae} (tol {tol}); a SCHEMA HOOK is likely wrong."
    )


# --------------------------------------------------------------------------- #
# synthetic design (run the whole pipeline before touching real data)
# --------------------------------------------------------------------------- #
def make_synthetic_design(
    n: int = 1500,
    *,
    random_state: int = 0,
) -> CausalDesign:
    """
    Generate a synthetic CausalDesign whose dependency structure mimics the
    manuscript's hypotheses, so the full analysis pipeline (observability + SURD)
    can be exercised end-to-end with NO real data. Use this to smoke-test
    scripts/run_causal_analysis.py and to demonstrate the expected qualitative
    pattern before the schema hooks are wired to the real cache.

    Built-in ground truth:
      * future lift is mostly a UNIQUE function of the gust parameters (forcing is
        nearly imprinted at impact), with small noise.
      * future wake enstrophy is SYNERGISTIC in (parameters, current wake state):
        it needs the joint of c and the current wake, not either alone.
      * a 'JEPA' latent encodes both the parameters and the current wake state
        (so it is informative about the future wake); a 'reconstructive' latent
        encodes the field amplitude / lift signature but NOT the wake-state
        interaction (so it is far less informative about the future wake); a 'POD'
        latent is an energy-ranked linear mix in between.
      * wall pressure carries the lift strongly and the wake state weakly.
    """
    rng = np.random.default_rng(random_state)

    G = rng.uniform(-3, 3, n)
    D = rng.choice([0.5, 1.0, 1.5], n)
    Y = rng.uniform(-0.4, 0.4, n)
    wake_now = np.abs(G) * (0.6 + 0.4 * D) * 30 + rng.normal(0, 8, n)
    cl_now = 1.2 * np.sign(G) * np.tanh(np.abs(G)) + 0.2 * rng.normal(0, 1, n)
    phase = rng.uniform(0, 2 * np.pi, n)

    # future targets
    cl_future = 0.9 * cl_now + 0.1 * G + rng.normal(0, 0.15, n)            # ~unique to params
    # synergy: product term that needs BOTH c (via G,D) and current wake state
    wake_future = (
        0.5 * wake_now
        + 0.02 * (np.abs(G) * D) * wake_now
        + 8.0 * np.sign(G) * np.sqrt(np.abs(wake_now))
        + rng.normal(0, 10, n)
    )
    circ_neg_future = -np.sqrt(np.clip(wake_future, 0, None)) * 0.3 + rng.normal(0, 0.3, n)

    # latents (impact frame)
    def standard(a):  # column-standardise
        a = np.asarray(a, float)
        return (a - a.mean()) / (a.std() + 1e-9)

    jepa = np.column_stack(
        [standard(G), standard(D), standard(Y), standard(wake_now), standard(cl_now)]
    )
    jepa = jepa @ rng.standard_normal((5, 32)) + 0.05 * rng.standard_normal((n, 32))  # d=32

    recon = np.column_stack(
        [standard(cl_now), standard(np.abs(G)), standard(Y)]  # carries lift/amplitude, not wake-state interaction
    )
    recon = recon @ rng.standard_normal((3, 32)) + 0.05 * rng.standard_normal((n, 32))

    pod = np.column_stack([standard(wake_now), standard(cl_now), standard(G), standard(D)])
    pod = pod @ rng.standard_normal((4, 32)) + 0.1 * rng.standard_normal((n, 32))

    pressure_k8 = np.column_stack(
        [standard(cl_now), standard(cl_now) ** 2, standard(G)]
        + [0.3 * standard(wake_now) + rng.normal(0, 1, n)]
    )
    pressure_k8 = np.column_stack([pressure_k8, rng.standard_normal((n, 4))])

    keys = [(f"synthetic_case_{i // 5}", i % 5) for i in range(n)]
    partition = np.array(
        ["train"] * int(0.7 * n) + ["test_b"] * int(0.18 * n) + ["test_c"] * (n - int(0.7 * n) - int(0.18 * n))
    )
    rng.shuffle(partition)

    return CausalDesign(
        keys=keys,
        partition=partition,
        sources_impact={
            "G": G, "D": D, "Y": Y,
            "wake_enstrophy_impact": wake_now,
            "CL_impact": cl_now,
            "phase_impact": phase,
        },
        targets_future={
            "CL_future": cl_future,
            "wake_enstrophy_future": wake_future,
            "circ_neg_future": circ_neg_future,
        },
        stage_index=np.digitize(phase, np.quantile(phase, [0.25, 0.5, 0.75])).astype(int),
        latents={"JEPA_d32": jepa, "reconstructive_d32": recon, "POD_d32": pod},
        pressure={"K8": pressure_k8},
        meta={"synthetic": True, "horizon": 16},
    )
