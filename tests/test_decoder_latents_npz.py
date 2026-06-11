"""CPU-only tests for the --latents-npz mode of ``scripts/session9_train_decoder.py``.

The mode trains the SAME decoder recipe post-hoc on frozen latents from ANY
family (Fukami AE, POD) instead of a JEPA encoder (Session 28 T9 fairness;
referee M2/C5 decode fairness). Covered here without running real training:

- CLI: --latents-npz parses; argparse mutex vs --jepa-checkpoint /
  --encoder-run; SystemExit on --input-mode omega_direct conflict.
- LatentStore: tolerates BOTH key conventions, singular (case_id,
  encounter_index; session14 style) and plural (case_ids,
  encounter_indices; session18/28 style); clear errors on missing keys,
  missing rows, duplicate rows, and out-of-range frame slices.
- (case, encounter, frame) -> z alignment: distinct constant z per
  encounter plus a per-frame ramp, with the dataset's encounter order
  scrambled relative to the NPZ row order, so any positional or frame
  misalignment is detectable.
- evaluate_split with a latent store and enc=None (recording stub decoder).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "session9_train_decoder.py"

N_FRAMES = 120
D = 4
H, W = 8, 4


def _load_script_module():
    spec = importlib.util.spec_from_file_location("session9_train_decoder", SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_script_module()


def _parse(mod, argv):
    old = sys.argv
    sys.argv = ["session9_train_decoder.py"] + argv
    try:
        return mod.parse_args()
    finally:
        sys.argv = old


def _z_full(n_rows: int) -> np.ndarray:
    """z_full where dim 0 is the frame index and dim 1 a per-row constant.

    z[i, t, 0] = t exposes frame misalignment; z[i, t, 1] = 100 * (i + 1)
    exposes row (encounter) misalignment.
    """
    z = np.zeros((n_rows, N_FRAMES, D), dtype=np.float32)
    for i in range(n_rows):
        z[i, :, 0] = np.arange(N_FRAMES, dtype=np.float32)
        z[i, :, 1] = 100.0 * (i + 1)
    return z


ROWS = [("caseA", 0), ("caseA", 1), ("caseB", 0)]


def _write_npz(path: Path, style: str, rows=ROWS) -> None:
    """Write a tiny synthetic {split}.npz in the requested key style."""
    case_ids = np.array([c for c, _ in rows])
    enc_idx = np.array([k for _, k in rows], dtype=np.int32)
    keys = {
        "plural": {"case_ids": case_ids, "encounter_indices": enc_idx},
        "singular": {"case_id": case_ids, "encounter_index": enc_idx},
    }[style]
    np.savez(path, z_full=_z_full(len(rows)), **keys)


def _write_encounter_h5(path: Path) -> None:
    """omega_z[t] is a constant-t field so the frame index is readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    omega = np.broadcast_to(
        np.arange(N_FRAMES, dtype=np.float32)[:, None, None], (N_FRAMES, H, W)
    ).copy()
    with h5py.File(path, "w") as f:
        f.create_dataset("omega_z", data=omega)


def _make_encs(tmp_path: Path, rows=ROWS) -> list[dict]:
    encs = []
    for cid, k in rows:
        p = tmp_path / "cache" / cid / f"encounter_{k:02d}.h5"
        _write_encounter_h5(p)
        encs.append({"case_id": cid, "k": int(k), "path": str(p)})
    return encs


# ---------------------------------------------------------------------------
# CLI: parsing and conflicts
# ---------------------------------------------------------------------------


def test_latents_npz_arg_parses(mod) -> None:
    args = _parse(mod, ["--latents-npz", "/tmp/latents", "--output-dir", "/tmp/out"])
    assert args.latents_npz == "/tmp/latents"
    assert args.jepa_checkpoint is None
    assert args.encoder_run is None
    assert args.input_mode == "latent"


def test_latents_npz_mutex_with_jepa_checkpoint(mod) -> None:
    with pytest.raises(SystemExit):
        _parse(
            mod,
            [
                "--latents-npz",
                "/tmp/latents",
                "--jepa-checkpoint",
                "/tmp/dummy.pt",
                "--output-dir",
                "/tmp/out",
            ],
        )


def test_latents_npz_mutex_with_encoder_run(mod) -> None:
    with pytest.raises(SystemExit):
        _parse(
            mod,
            [
                "--latents-npz",
                "/tmp/latents",
                "--encoder-run",
                "/tmp/run",
                "--output-dir",
                "/tmp/out",
            ],
        )


def test_latents_npz_conflicts_with_omega_direct(mod) -> None:
    args = _parse(
        mod,
        [
            "--latents-npz",
            "/tmp/latents",
            "--input-mode",
            "omega_direct",
            "--decoder-type",
            "lapfilm",
            "--output-dir",
            "/tmp/out",
        ],
    )
    with pytest.raises(SystemExit, match="omega_direct"):
        mod.check_latents_npz_conflicts(args)


def test_no_conflict_for_compatible_invocations(mod) -> None:
    args = _parse(mod, ["--latents-npz", "/tmp/latents", "--output-dir", "/tmp/out"])
    mod.check_latents_npz_conflicts(args)  # must not raise
    args = _parse(
        mod,
        [
            "--input-mode",
            "omega_direct",
            "--decoder-type",
            "lapfilm",
            "--output-dir",
            "/tmp/out",
        ],
    )
    mod.check_latents_npz_conflicts(args)  # latents_npz unset: must not raise


# ---------------------------------------------------------------------------
# LatentStore: key conventions and error paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("style", ["plural", "singular"])
def test_latent_store_tolerates_both_key_conventions(mod, tmp_path, style) -> None:
    npz = tmp_path / "train.npz"
    _write_npz(npz, style)
    store = mod.LatentStore(npz)
    assert store.d == D
    assert store.n_frames == N_FRAMES
    assert store.row == {("caseA", 0): 0, ("caseA", 1): 1, ("caseB", 0): 2}


def test_latent_store_decodes_bytes_case_ids(mod, tmp_path) -> None:
    npz = tmp_path / "train.npz"
    np.savez(
        npz,
        z_full=_z_full(2),
        case_ids=np.array([b"caseA", b"caseB"]),
        encounter_indices=np.array([0, 0], dtype=np.int64),
    )
    store = mod.LatentStore(npz)
    assert store.row == {("caseA", 0): 0, ("caseB", 0): 1}


def test_latent_store_error_paths(mod, tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="latents-npz"):
        mod.LatentStore(tmp_path / "nope.npz")

    no_z = tmp_path / "no_z.npz"
    np.savez(no_z, case_ids=np.array(["caseA"]), encounter_indices=np.array([0]))
    with pytest.raises(KeyError, match="z_full"):
        mod.LatentStore(no_z)

    no_keys = tmp_path / "no_keys.npz"
    np.savez(no_keys, z_full=_z_full(1))
    with pytest.raises(KeyError, match="case_id"):
        mod.LatentStore(no_keys)

    dup = tmp_path / "dup.npz"
    np.savez(
        dup,
        z_full=_z_full(2),
        case_ids=np.array(["caseA", "caseA"]),
        encounter_indices=np.array([0, 0]),
    )
    with pytest.raises(ValueError, match="duplicate"):
        mod.LatentStore(dup)

    ok = tmp_path / "ok.npz"
    _write_npz(ok, "plural")
    store = mod.LatentStore(ok)
    with pytest.raises(KeyError, match="caseZ"):
        store.row_index("caseZ", 0)
    with pytest.raises(ValueError, match="out of range"):
        store.frames("caseA", 0, N_FRAMES - 5, 10)


# ---------------------------------------------------------------------------
# (case, encounter, frame) -> z alignment
# ---------------------------------------------------------------------------


def test_dataset_aligns_z_by_key_and_frame(mod, tmp_path) -> None:
    """Lookup must be keyed on (case_id, encounter_index), never positional.

    The dataset's encounter list is deliberately scrambled relative to the
    NPZ row order, z carries a distinct constant per encounter row plus a
    per-frame ramp, and omega_z carries the frame index, so any row or
    frame misalignment changes the asserted values.
    """
    npz = tmp_path / "train.npz"
    _write_npz(npz, "plural")  # rows: (caseA,0)=0, (caseA,1)=1, (caseB,0)=2
    store = mod.LatentStore(npz)
    encs = _make_encs(tmp_path)
    scrambled = [encs[2], encs[0], encs[1]]  # (caseB,0), (caseA,0), (caseA,1)
    expected_row = {("caseB", 0): 2, ("caseA", 0): 0, ("caseA", 1): 1}

    ds = mod.EncounterFrameDataset(scrambled, T=32, seed=0, latent_store=store)
    assert len(ds) == 3
    for _ in range(5):  # several draws so the random start varies
        for idx, e in enumerate(scrambled):
            x, z = ds[idx]
            assert x.shape == (32, H, W)
            assert z.shape == (32, D)
            # Frame alignment: z dim 0 carries the frame index, as does omega.
            assert torch.equal(z[:, 0], x[:, 0, 0])
            # Row alignment: z dim 1 carries the NPZ-row constant.
            row = expected_row[(e["case_id"], e["k"])]
            assert torch.all(z[:, 1] == 100.0 * (row + 1))


def test_dataset_without_store_returns_plain_tensor(mod, tmp_path) -> None:
    """Existing modes stay byte-identical: no store -> bare (T, H, W) tensor."""
    encs = _make_encs(tmp_path, rows=[("caseA", 0)])
    ds = mod.EncounterFrameDataset(encs, T=32, seed=0)
    out = ds[0]
    assert isinstance(out, torch.Tensor)
    assert out.shape == (32, H, W)


def test_collate_with_latents_shapes(mod) -> None:
    batch = [(torch.zeros(32, H, W), torch.zeros(32, D)) for _ in range(3)]
    x, z = mod.collate_with_latents(batch)
    assert x.shape == (3, 32, H, W)
    assert z.shape == (3, 32, D)


def test_check_store_coverage(mod, tmp_path) -> None:
    npz = tmp_path / "train.npz"
    _write_npz(npz, "plural")
    store = mod.LatentStore(npz)
    covered = [{"case_id": "caseA", "k": 0}, {"case_id": "caseB", "k": 0}]
    mod.check_store_coverage(store, covered, "train")  # must not raise
    with pytest.raises(SystemExit, match="missing"):
        mod.check_store_coverage(store, covered + [{"case_id": "caseZ", "k": 7}], "train")


# ---------------------------------------------------------------------------
# evaluate_split with a latent store (fake decoder, enc=None)
# ---------------------------------------------------------------------------


class _RecordingZeroDecoder(nn.Module):
    """Stub decoder: records the z it receives, returns zeros (B, T, 1, H, W)."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[torch.Tensor] = []

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.seen.append(z.detach().clone())
        b, t, _ = z.shape
        return torch.zeros(b, t, 1, H, W)


def test_evaluate_split_pulls_z_from_store(mod, tmp_path) -> None:
    npz = tmp_path / "test_b.npz"
    _write_npz(npz, "singular")
    store = mod.LatentStore(npz)
    encs = _make_encs(tmp_path, rows=[("caseA", 1), ("caseA", 0)])  # scrambled order
    dec = _RecordingZeroDecoder()

    result = mod.evaluate_split(None, dec, encs, torch.device("cpu"), latent_store=store)

    assert result["n_encounters"] == 2
    assert "mse_mean" in result and "ssim_mean" in result
    # First eval encounter is (caseA, 1) = NPZ row 1; full 120-frame trajectory.
    assert len(dec.seen) == 2
    z0 = dec.seen[0].squeeze(0)
    assert z0.shape == (N_FRAMES, D)
    assert torch.equal(z0[:, 0], torch.arange(N_FRAMES, dtype=torch.float32))
    assert torch.all(z0[:, 1] == 200.0)  # row 1 constant
    z1 = dec.seen[1].squeeze(0)
    assert torch.all(z1[:, 1] == 100.0)  # row 0 constant
