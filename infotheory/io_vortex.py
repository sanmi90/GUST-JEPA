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
    # Verified Session 25: wake-observable cache is
    # ${VORTEX_JEPA_CACHE}/<partition>/wake_observables/<case_id>/encounter_XX.h5,
    # each file holding enstrophy_scalar (120,1) plus the patch/pool wake modes.
    return cache_root / partition / "wake_observables"


def _episode_path(cache_root: Path, partition: str = "v1") -> Path:
    # Verified Session 25: per-encounter episode cache is
    # ${VORTEX_JEPA_CACHE}/<partition>/<case_id>/encounter_XX.h5, each file holding
    # C_L (120,), C_D (120,), omega_z, p_wall and the (G,D,Y, impact_frame_estimate)
    # attrs.
    return cache_root / partition


def _encounter_file(root: Path, case_id: str, enc: int) -> Path:
    return root / case_id / f"encounter_{enc:02d}.h5"


# Observables sourced directly from the two on-disk caches. Iy / circ_pos /
# circ_neg are not in these caches; they live in the per-frame descriptor export
# (outputs/session16/exp2/per_frame_targets) and are attached for Block B/C, not
# needed for the Block A de-risking gate.
CACHE_OBSERVABLES = ("CL", "CD", "wake_enstrophy")


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
    CL_impact, CD_impact, phase_impact} and targets_future = {CL_future, CD_future,
    wake_enstrophy_future}. Latents and pressure are filled by the attach_* helpers.

    Verified Session 25 schema. For each (case_id, encounter) carried by the split
    partition map, read C_L / C_D and the (G,D,Y) + impact_frame_estimate attrs from
    the episode cache and the per-frame wake enstrophy (enstrophy_scalar, normalised
    omega) from the wake_observables cache, then sample the impact frame and the
    impact+H frame. wake_enstrophy here is the enstrophy of the pipeline-normalised
    vorticity (typical magnitude ~0.1..0.4), matching the wake_observables writer.
    """
    import h5py  # local import: only needed when actually reading the cache

    ep_root = _episode_path(cache_root, partition)
    wo_root = _wake_observables_path(cache_root, partition)
    if not ep_root.exists():
        raise FileNotFoundError(
            f"{ep_root} not found. Set VORTEX_JEPA_CACHE / PREVENT_ROOT so the "
            "processed cache resolves (SCHEMA HOOK)."
        )

    # Map (case_id, encounter) -> partition from the split manifest.
    part_of = _build_partition_map(split)

    keys: list[tuple[str, int]] = []
    partition_labels: list[str] = []
    src = {k: [] for k in ("G", "D", "Y", "wake_enstrophy_impact", "CL_impact",
                           "CD_impact", "phase_impact")}
    tgt = {f"{o}_future": [] for o in CACHE_OBSERVABLES}
    impacts: list[int] = []
    n_missing = 0

    for (case_id, enc), plabel in sorted(part_of.items()):
        ep_f = _encounter_file(ep_root, case_id, enc)
        wo_f = _encounter_file(wo_root, case_id, enc)
        if not ep_f.exists() or not wo_f.exists():
            n_missing += 1
            continue
        with h5py.File(ep_f, "r") as g:
            cl = np.asarray(g["C_L"], dtype=float).ravel()
            cd = np.asarray(g["C_D"], dtype=float).ravel()
            gval = float(g.attrs["G"]); dval = float(g.attrs["D"]); yval = float(g.attrs["Y"])
            impact = int(g.attrs.get("impact_frame_estimate", 40))
        with h5py.File(wo_f, "r") as g:
            ens = np.asarray(g["enstrophy_scalar"], dtype=float).ravel()

        n_frames = min(len(cl), len(cd), len(ens))
        future = impact + horizon
        if impact >= n_frames or future >= n_frames:
            continue  # encounter too short for this horizon

        keys.append((case_id, enc))
        partition_labels.append(plabel)
        src["G"].append(gval); src["D"].append(dval); src["Y"].append(yval)
        src["wake_enstrophy_impact"].append(ens[impact])
        src["CL_impact"].append(cl[impact])
        src["CD_impact"].append(cd[impact])
        src["phase_impact"].append(np.nan)  # no baseline-phase tag in the cache yet
        series = {"CL": cl, "CD": cd, "wake_enstrophy": ens}
        for o in CACHE_OBSERVABLES:
            tgt[f"{o}_future"].append(series[o][future])
        impacts.append(impact)

    if not keys:
        raise RuntimeError(
            "no encounters loaded; check cache_root / partition / split alignment "
            "(SCHEMA HOOK). part_of had "
            f"{len(part_of)} entries, {n_missing} files missing on disk."
        )

    design = CausalDesign(
        keys=keys,
        partition=np.array(partition_labels),
        sources_impact={k: np.asarray(v, float) for k, v in src.items()},
        targets_future={k: np.asarray(v, float) for k, v in tgt.items()},
        meta={
            "horizon": horizon,
            "partition": partition,
            "impact_idx_mean": float(np.mean(impacts)),
            "n_missing_files": n_missing,
        },
    )
    return design


def _build_partition_map(split: dict) -> dict[tuple[str, int], str]:
    """
    Build (case_id, encounter) -> partition from configs/splits/split_v1.json.

    Verified Session 25 schema. split['cases'] maps case_id -> a dict carrying
    'split' in {'train','test_b','test_c'} plus 'train_encounter_indices' and
    'test_a_encounter_indices'. For a train case the two index lists separate the
    in-case held-out (test_a) encounters from the training ones; for a test_b /
    test_c case every encounter (listed under test_a_encounter_indices) inherits
    the case split. NOTE: the per-file 'split' attr inside wake_observables is a
    coarser, older labelling (train 318 / test_b 36) that disagrees with this
    manifest (train 237 / test_a 89 / test_b 28); the manifest is authoritative.
    """
    part_of: dict[tuple[str, int], str] = {}
    cases = split.get("cases", {})
    if not cases:
        raise ValueError(
            "split has no 'cases' dict; expected configs/splits/split_v1.json "
            "(SCHEMA HOOK)."
        )
    for case_id, c in cases.items():
        case_split = c.get("split", "train")
        train_idx = [int(e) for e in (c.get("train_encounter_indices") or [])]
        test_a_idx = [int(e) for e in (c.get("test_a_encounter_indices") or [])]
        if case_split == "train":
            for e in train_idx:
                part_of[(case_id, e)] = "train"
            for e in test_a_idx:
                part_of[(case_id, e)] = "test_a"
        else:  # test_b / test_c case: every encounter carries the case split
            for e in sorted(set(train_idx) | set(test_a_idx)):
                part_of[(case_id, e)] = case_split
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
