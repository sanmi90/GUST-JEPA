# Spatio-temporal JEPA encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Project commit rule:** CLAUDE.md says commit/push only when the user asks. The commit steps below are the intended units of work, but actually running `git commit` is gated on user approval. Stage and show the diff, then commit when told.

**Goal:** Add a causal spatio-temporal encoder (3D-conv tubelet stem) to the JEPA so each latent integrates a short causal motion window, and measure whether it beats the per-frame encoder on the full six-metric suite at d=64 and d=16.

**Architecture:** A new `SpatioTemporalCNNViTEncoder` replaces only the 2D conv stem of `HybridCNNViTEncoder` with a causal 3D-conv stem; the per-frame ViT + `[CLS]` readout + BatchNorm projection are reused unchanged, so the `(B, T, 1, H, W) -> (B, T, d)` contract is preserved and the module drops into the existing training loop, predictor, and eval. Temporal mixing is causal (left zero-pad, temporal stride 1) and GroupNorm is applied per-frame so the normalization statistics never mix across time.

**Tech Stack:** PyTorch 2.x, the existing `src/models/encoder.py` primitives (`_ViTBlock`, `_sin_cos_2d_pos_embed`), `src/training/train_jepa.py`, `scripts/session18/encode_baseline_latents.py`, and the session29 eval scripts. RTX 6000 Blackwell only via `require_rtx6000`.

**Reference baseline (on disk):** per-frame `jepa_tf_noc_d64_s{0,1,2,42}` and `jepa_tf_noc_d16_s{0,1,42}` at `outputs/runs/session28/<tag>/encoder/checkpoint_iter020000.pt`; measured d16 reversal: regAE-matched +0.78[0.69,0.85] vs JEPA-own +0.30[0.07,0.45] at h=1.

**Spec:** `docs/superpowers/specs/2026-06-18-spatiotemporal-jepa-design.md`

---

## File structure

- `src/models/encoder.py` — add `_CausalConv3dBlock` and `SpatioTemporalCNNViTEncoder` (one new responsibility: temporal stem). No change to existing classes.
- `tests/test_encoder.py` — add ST shape + causality + batchnorm + param-bound tests.
- `src/training/train_jepa.py` — extend the `--encoder` choice list, add `--temporal-kernel`, add the construction branch.
- `scripts/session18/encode_baseline_latents.py` — add the `st_hybrid` branch in `_load_jepa_encoder`.
- `scripts/session29/st_band.sh` — new launcher (train -> extract -> roll -> decode), queued behind d32.
- `scripts/session29/st_compare.py` — new six-metric comparison report (ST vs per-frame).

---

### Task 1: `SpatioTemporalCNNViTEncoder` with causal 3D stem

**Files:**
- Modify: `src/models/encoder.py` (add new classes after `HybridCNNViTEncoder`, before `CNNOnlyEncoder` at line 238)
- Test: `tests/test_encoder.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_encoder.py`:

```python
from src.models.encoder import SpatioTemporalCNNViTEncoder


def test_st_encoder_shape_contract() -> None:
    """Input (2, 8, 1, 192, 96) -> output (2, 8, 32) at default config."""
    torch.manual_seed(0)
    enc = SpatioTemporalCNNViTEncoder()
    x = torch.randn(2, 8, 1, 192, 96)
    z = enc(x)
    assert z.shape == (2, 8, 32)


def test_st_encoder_is_causal() -> None:
    """Perturbing a future frame must not change earlier latents.

    With temporal kernel 3 in the stem + 2 downsamples, the causal receptive
    field is 1 + (3-1)*3 = 7 frames, so z_t depends only on frames in
    [t-6, t]. Perturbing frame 7 must leave z_0..z_6 unchanged. Run in eval()
    so BatchNorm uses fixed running stats and the only temporal coupling under
    test is the 3D stem.
    """
    torch.manual_seed(0)
    enc = SpatioTemporalCNNViTEncoder(latent_dim=16).eval()
    x = torch.randn(1, 10, 1, 192, 96)
    with torch.no_grad():
        z = enc(x)
        x2 = x.clone()
        x2[:, 7] = torch.randn(1, 1, 192, 96)
        z2 = enc(x2)
    assert torch.allclose(z[:, :7], z2[:, :7], atol=1e-5)
    assert not torch.allclose(z[:, 7], z2[:, 7], atol=1e-5)


def test_st_encoder_projection_is_batchnorm_by_default() -> None:
    """The default projection_norm selects nn.BatchNorm1d at proj[-1]."""
    enc = SpatioTemporalCNNViTEncoder()
    assert isinstance(enc.proj[-1], nn.BatchNorm1d)


def test_st_encoder_param_count_bound() -> None:
    """ST encoder stays within ~1.5x the per-frame encoder's parameter count."""
    from src.models.encoder import HybridCNNViTEncoder
    base = sum(p.numel() for p in HybridCNNViTEncoder().parameters())
    st = sum(p.numel() for p in SpatioTemporalCNNViTEncoder().parameters())
    assert st < 1.5 * base, f"ST encoder {st} params exceeds 1.5x base {base}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `timeout 120 python -m pytest tests/test_encoder.py -k st_encoder -q`
Expected: FAIL with `ImportError: cannot import name 'SpatioTemporalCNNViTEncoder'`

- [ ] **Step 3: Implement the encoder**

In `src/models/encoder.py`, add `import torch.nn.functional as F` to the imports, then insert after `HybridCNNViTEncoder` (line 236):

```python
class _CausalConv3dBlock(nn.Module):
    """Causal Conv3d -> per-frame GroupNorm -> GELU.

    The time axis is left-padded by ``t_kernel - 1`` with zeros and uses
    temporal stride 1, so output frame ``t`` depends only on input frames
    ``<= t``. GroupNorm is applied per frame (reshape to ``(B*T, C, H, W)``)
    so the normalization statistics never mix across time, which would
    otherwise leak the future into the past and break causality.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        t_kernel: int,
        s_kernel: int = 3,
        spatial_stride: int = 1,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        self._t_pad = t_kernel - 1
        s_pad = s_kernel // 2
        self.conv = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=(t_kernel, s_kernel, s_kernel),
            stride=(1, spatial_stride, spatial_stride),
            padding=(0, s_pad, s_pad),
            bias=True,
        )
        self.norm = nn.GroupNorm(n_groups, out_ch)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T, H, W). Causal left-pad on the time axis only.
        x = F.pad(x, (0, 0, 0, 0, self._t_pad, 0))
        x = self.conv(x)
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.act(self.norm(x))
        return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


class SpatioTemporalCNNViTEncoder(nn.Module):
    """Causal 3D-conv tubelet stem + per-frame ViT, emitting per-frame latents.

    Same ``(B, T, 1, H, W) -> (B, T, d)`` contract as
    ``HybridCNNViTEncoder``, but the stem mixes a causal window of frames so
    each ``z_t`` integrates frames ``<= t`` instead of a single snapshot. The
    temporal receptive field is ``1 + (t_kernel - 1) * 3`` frames (the stem and
    the two spatial-downsampling convs are temporal; the residual blocks are
    spatial-only at temporal kernel 1). The ViT, ``[CLS]`` readout, and
    BatchNorm projection are identical to ``HybridCNNViTEncoder``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        cnn_channels: tuple[int, int, int] = (64, 128, 256),
        vit_depth: int = 6,
        vit_hidden: int = 256,
        vit_heads: int = 8,
        vit_mlp_ratio: float = 4.0,
        latent_dim: int = 32,
        dropout: float = 0.0,
        projection_norm: str = "batchnorm",
        temporal_kernel: int = 3,
    ) -> None:
        super().__init__()
        if projection_norm not in ("batchnorm", "layernorm"):
            raise ValueError(
                f"projection_norm must be 'batchnorm' or 'layernorm', got {projection_norm!r}"
            )
        if temporal_kernel < 1:
            raise ValueError(f"temporal_kernel must be >= 1, got {temporal_kernel}")
        self.projection_norm = projection_norm
        self.temporal_kernel = temporal_kernel
        c1, c2, c3 = cnn_channels
        tk = temporal_kernel

        # Causal 3D stem: temporal in stem + the two downsamples (RF = 1+(tk-1)*3);
        # residual blocks are spatial-only (t_kernel=1).
        self.stem = _CausalConv3dBlock(in_channels, c1, t_kernel=tk, s_kernel=7, spatial_stride=2)
        self.block1 = nn.Sequential(
            _CausalConv3dBlock(c1, c1, t_kernel=1),
            _CausalConv3dBlock(c1, c1, t_kernel=1),
        )
        self.down1 = _CausalConv3dBlock(c1, c2, t_kernel=tk, spatial_stride=2)
        self.block2 = nn.Sequential(
            _CausalConv3dBlock(c2, c2, t_kernel=1),
            _CausalConv3dBlock(c2, c2, t_kernel=1),
        )
        self.down2 = _CausalConv3dBlock(c2, c3, t_kernel=tk, spatial_stride=2)
        self.block3 = nn.Sequential(
            _CausalConv3dBlock(c3, c3, t_kernel=1),
            _CausalConv3dBlock(c3, c3, t_kernel=1),
        )

        h_feat, w_feat = 192 // 8, 96 // 8
        self._num_spatial_tokens = h_feat * w_feat

        self.token_proj: nn.Module = (
            nn.Identity() if c3 == vit_hidden else nn.Linear(c3, vit_hidden)
        )
        pos_embed = _sin_cos_2d_pos_embed(h_feat, w_feat, vit_hidden)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, vit_hidden))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.vit = nn.ModuleList(
            [_ViTBlock(vit_hidden, vit_heads, vit_mlp_ratio, dropout) for _ in range(vit_depth)]
        )
        self.norm = nn.LayerNorm(vit_hidden)
        proj_norm: nn.Module = (
            nn.BatchNorm1d(latent_dim)
            if projection_norm == "batchnorm"
            else nn.LayerNorm(latent_dim)
        )
        self.proj = nn.Sequential(nn.Linear(vit_hidden, latent_dim), proj_norm)

    @property
    def num_spatial_tokens(self) -> int:
        return self._num_spatial_tokens

    def forward(self, x: Tensor) -> Tensor:
        """Encode a sub-trajectory into causal-window-aware per-frame latents.

        Args:
            x: ``(B, T, C, H, W)`` with ``C = 1``, ``H = 192``, ``W = 96``.

        Returns:
            ``z`` of shape ``(B, T, latent_dim)``.
        """
        if x.dim() != 5:
            raise ValueError(f"x must be (B, T, C, H, W), got {tuple(x.shape)}")
        b, t = x.shape[0], x.shape[1]
        h = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        h = self.stem(h)
        h = self.block1(h)
        h = self.down1(h)
        h = self.block2(h)
        h = self.down2(h)
        h = self.block3(h)  # (B, c3, T, 24, 12)

        c3, hf, wf = h.shape[1], h.shape[3], h.shape[4]
        h = h.permute(0, 2, 1, 3, 4).reshape(b * t, c3, hf, wf)
        h = h.flatten(2).transpose(1, 2)  # (B*T, 288, c3)
        h = self.token_proj(h)
        h = h + self.pos_embed
        cls = self.cls_token.expand(b * t, -1, -1)
        h = torch.cat([cls, h], dim=1)
        for block in self.vit:
            h = block(h)
        h = self.norm(h)
        z = self.proj(h[:, 0, :])
        return z.view(b, t, -1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `timeout 180 python -m pytest tests/test_encoder.py -k st_encoder -q`
Expected: 4 passed. If `test_st_encoder_is_causal` fails, the most likely cause is a normalization that mixes across time (a plain 3D GroupNorm instead of the per-frame reshape) or a non-causal pad; do not weaken the test, fix the stem.

- [ ] **Step 5: Style check**

Run: `black --line-length 100 src/models/encoder.py tests/test_encoder.py && flake8 --max-line-length=100 src/models/encoder.py`
Expected: no output (clean).

- [ ] **Step 6: Commit** (gated on user approval)

```bash
git add src/models/encoder.py tests/test_encoder.py
git commit -m "feat(encoder): causal spatio-temporal 3D-conv stem encoder"
```

---

### Task 2: Wire `st_hybrid` into `train_jepa.py`

**Files:**
- Modify: `src/training/train_jepa.py:42` (import), `:174-183` (choices), `:725-731` (construction), and add `--temporal-kernel` near the `--encoder` arg.

- [ ] **Step 1: Add the import**

At `src/training/train_jepa.py:42`, change:

```python
from src.models.encoder import CNNOnlyEncoder, HybridCNNViTEncoder
```
to:
```python
from src.models.encoder import (
    CNNOnlyEncoder,
    HybridCNNViTEncoder,
    SpatioTemporalCNNViTEncoder,
)
```

- [ ] **Step 2: Extend the `--encoder` choices and add `--temporal-kernel`**

In the `--encoder` argparse block (line ~174), change `choices=["hybrid", "cnn_only"]` to `choices=["hybrid", "cnn_only", "st_hybrid"]` and append to the help string: ``" 'st_hybrid' adds a causal 3D-conv tubelet stem (spatio-temporal encoder, variant A of the 2026-06-18 spec)."``

Immediately after that block, add:

```python
    p.add_argument(
        "--temporal-kernel",
        type=int,
        default=3,
        help=(
            "Temporal conv kernel for --encoder st_hybrid. Causal receptive "
            "field is 1 + (k-1)*3 frames (k=3 -> 7 frames ~ 0.35 t/c). Ignored "
            "for other encoders."
        ),
    )
```

- [ ] **Step 3: Add the construction branch**

In the encoder construction (line ~725), change:

```python
    if args.encoder == "cnn_only":
        encoder: nn.Module = CNNOnlyEncoder(
            latent_dim=args.d, projection_norm=args.projection_norm
        )
    else:
        encoder = HybridCNNViTEncoder(latent_dim=args.d, projection_norm=args.projection_norm)
```
to:
```python
    if args.encoder == "cnn_only":
        encoder: nn.Module = CNNOnlyEncoder(
            latent_dim=args.d, projection_norm=args.projection_norm
        )
    elif args.encoder == "st_hybrid":
        encoder = SpatioTemporalCNNViTEncoder(
            latent_dim=args.d,
            projection_norm=args.projection_norm,
            temporal_kernel=args.temporal_kernel,
        )
    else:
        encoder = HybridCNNViTEncoder(latent_dim=args.d, projection_norm=args.projection_norm)
```

- [ ] **Step 4: Verify argparse parses and the branch constructs**

Run:
```bash
timeout 60 python -c "
import sys; sys.argv=['x','--encoder','st_hybrid','--d','16','--temporal-kernel','3']
from src.training.train_jepa import parse_args
a=parse_args(); print('encoder=',a.encoder,'tk=',a.temporal_kernel,'d=',a.d)
from src.models.encoder import SpatioTemporalCNNViTEncoder
import torch
e=SpatioTemporalCNNViTEncoder(latent_dim=a.d, temporal_kernel=a.temporal_kernel)
print('z', e(torch.randn(1,4,1,192,96)).shape)
"
```
Expected: `encoder= st_hybrid tk= 3 d= 16` then `z torch.Size([1, 4, 16])`.

- [ ] **Step 5: Commit** (gated on user approval)

```bash
git add src/training/train_jepa.py
git commit -m "feat(train_jepa): --encoder st_hybrid + --temporal-kernel"
```

---

### Task 3: `st_hybrid` branch in the latent extractor

**Files:**
- Modify: `scripts/session18/encode_baseline_latents.py:205-216`

- [ ] **Step 1: Add the import and branch**

In `_load_jepa_encoder`, change line 205:
```python
    from src.models.encoder import CNNOnlyEncoder, HybridCNNViTEncoder
```
to:
```python
    from src.models.encoder import (
        CNNOnlyEncoder,
        HybridCNNViTEncoder,
        SpatioTemporalCNNViTEncoder,
    )
```
and change the construction (lines 212-216):
```python
    encoder_kind = str(targs.get("encoder", "hybrid"))
    if encoder_kind == "cnn_only":
        encoder = CNNOnlyEncoder(latent_dim=d, projection_norm=proj_norm).to(device)
    else:
        encoder = HybridCNNViTEncoder(latent_dim=d, projection_norm=proj_norm).to(device)
```
to:
```python
    encoder_kind = str(targs.get("encoder", "hybrid"))
    if encoder_kind == "cnn_only":
        encoder = CNNOnlyEncoder(latent_dim=d, projection_norm=proj_norm).to(device)
    elif encoder_kind == "st_hybrid":
        tk = int(targs.get("temporal_kernel", 3))
        encoder = SpatioTemporalCNNViTEncoder(
            latent_dim=d, projection_norm=proj_norm, temporal_kernel=tk
        ).to(device)
    else:
        encoder = HybridCNNViTEncoder(latent_dim=d, projection_norm=proj_norm).to(device)
```

- [ ] **Step 2: Style check**

Run: `flake8 --max-line-length=100 scripts/session18/encode_baseline_latents.py`
Expected: clean.

- [ ] **Step 3: Commit** (gated on user approval)

```bash
git add scripts/session18/encode_baseline_latents.py
git commit -m "feat(encode_latents): reconstruct st_hybrid encoder from checkpoint"
```

(No standalone unit test here; Task 4 exercises this path end-to-end on a real checkpoint, which is the meaningful verification.)

---

### Task 4: End-to-end smoke (train -> extract -> roll-own)

This proves the three code changes compose: a tiny ST run produces a checkpoint whose `args` round-trips through the extractor, and the extracted latents roll through the co-trained predictor. Uses ONE RTX 6000; only run when a card is free (d32 band may still hold both).

**Files:** none (verification only).

- [ ] **Step 1: 200-iter ST smoke train (1 card, CPU-capped)**

Run:
```bash
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
OUT=outputs/runs/session29/st_smoke
taskset -c 0-15 python -m src.training.train_jepa --encoder st_hybrid --temporal-kernel 3 \
  --d 16 --predictor-cond-dim 0 --partition v2p1 \
  --split configs/splits/split_v2p1.json \
  --omega-pipeline-manifest outputs/data_pipeline/v2p1/manifest.json \
  --max-iters 200 --checkpoint-every 200 --diagnostic-every 200 --log-every 50 \
  --num-workers 3 --wandb-mode offline --gpu 0 --tag-suffix st_smoke \
  --output-dir "$OUT" 2>&1 | tail -20
```
Expected: runs on an RTX 6000 (no `NoRTX6000Error`), prints loss lines, writes `$OUT/encoder/checkpoint_iter000200.pt`. If both cards are busy with d32, defer this step.

- [ ] **Step 2: Confirm the checkpoint carries the ST args**

Run:
```bash
python -c "
import torch
b=torch.load('outputs/runs/session29/st_smoke/encoder/checkpoint_iter000200.pt',map_location='cpu',weights_only=False)
a=b['args']; print('encoder=',a['encoder'],'tk=',a.get('temporal_kernel'),'d=',a['d'])
assert a['encoder']=='st_hybrid' and a.get('temporal_kernel')==3
print('OK args round-trip')
"
```
Expected: `encoder= st_hybrid tk= 3 d= 16` then `OK args round-trip`.

- [ ] **Step 3: Extract latents via the st_hybrid branch**

Run:
```bash
taskset -c 0-15 python scripts/session18/encode_baseline_latents.py --baseline jepa --d 16 \
  --checkpoint outputs/runs/session29/st_smoke/encoder/checkpoint_iter000200.pt \
  --partition v2p1 --split configs/splits/split_v2p1.json \
  --pipeline-manifest outputs/data_pipeline/v2p1/manifest.json \
  --splits test_b --gpu 0 --output-dir outputs/session29/latents/st_smoke 2>&1 | tail -5
python -c "
import numpy as np; d=np.load('outputs/session29/latents/st_smoke/test_b.npz')
print('z_full', d['z_full'].shape); assert d['z_full'].shape[2]==16; print('OK extract')
"
```
Expected: `z_full (N, 120, 16)` then `OK extract` (strict state-dict load succeeded, proving the reconstructed ST encoder matches the checkpoint).

- [ ] **Step 4: Roll the co-trained predictor**

Run:
```bash
taskset -c 0-15 python scripts/session29/roll_own_predictor.py \
  outputs/runs/session29/st_smoke/encoder/checkpoint_iter000200.pt \
  outputs/session29/latents/st_smoke/test_b.npz \
  outputs/session29/rollouts/st_smoke --device cuda:0 2>&1 | tail -5
ls -1 outputs/session29/rollouts/st_smoke/test_b.npz
```
Expected: prints `[roll-own] predictor d=16 ...` and writes the rollout npz. This confirms the full ST pipeline composes. (Metrics on a 200-iter smoke are meaningless; this step only checks plumbing.)

- [ ] **Step 5: No commit** (smoke artifacts are gitignored under `outputs/`). Delete the smoke dir if desired: `rm -rf outputs/runs/session29/st_smoke outputs/session29/latents/st_smoke outputs/session29/rollouts/st_smoke`.

---

### Task 5: Full ST band launcher (queued behind d32)

**Files:**
- Create: `scripts/session29/st_band.sh`

- [ ] **Step 1: Write the launcher**

Create `scripts/session29/st_band.sh`:

```bash
#!/usr/bin/env bash
# Spatio-temporal JEPA band (variant A): train --encoder st_hybrid at d in
# {64,16}, seeds {0,1,2,42}, 20k iters (convergence-matched to the per-frame
# band), then extract latents -> JEPA-own rollouts -> T9 decoders. Both RTX 6000
# cards, 2-packed, idempotent, CPU-capped so >=64 cores stay free for asolera.
# Queue this AFTER the d32 band frees the cards. RTX 6000 only.
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE
SPLIT="configs/splits/split_v2p1.json"; MAN="outputs/data_pipeline/v2p1/manifest.json"
RUN=outputs/runs/session29/st; LAT=outputs/session29/latents/st; ROLL=outputs/session29/lowd_rollouts
DEC=outputs/runs/session29
SEEDS_D64="0 1 2 42"; SEEDS_D16="0 1 42"
dev(){ [ "$1" -eq 0 ] && echo cuda:0 || echo cuda:1; }

st_enc(){ local d=$1 s=$2 g=$3; local out=$RUN/st_d${d}_s${s}
  [ -f "$out/encoder/checkpoint_iter020000.pt" ] && { echo "[st] skip enc d$d s$s"; return 0; }
  mkdir -p "$out"; echo "[st] enc d$d s$s gpu$g $(date -Iseconds)"
  nice -n 10 python -m src.training.train_jepa --encoder st_hybrid --temporal-kernel 3 \
    --d "$d" --predictor-cond-dim 0 --partition v2p1 --split "$SPLIT" \
    --omega-pipeline-manifest "$MAN" --lambda-sigreg 0.01 --lambda-lift 0.01 \
    --wake-observable-type patch_signed_spectrum --lambda-wake 1.0 \
    --max-iters 20000 --checkpoint-every 10000 --diagnostic-every 2000 --log-every 200 \
    --num-workers 3 --wandb-mode offline --seed "$s" --gpu "$g" \
    --tag-suffix s29_st_d${d}_s${s} --output-dir "$out" >"$out/train.log" 2>&1; }
st_lat(){ local d=$1 s=$2 g=$3; local o=$LAT/st_d${d}_s${s}
  [ -f "$o/test_b.npz" ] && return 0; mkdir -p "$o"; echo "[st] extract d$d s$s"
  taskset -c 0-31 python scripts/session18/encode_baseline_latents.py --baseline jepa --d "$d" \
    --checkpoint "$RUN/st_d${d}_s${s}/encoder/checkpoint_iter020000.pt" --partition v2p1 \
    --split "$SPLIT" --pipeline-manifest "$MAN" --splits train test_a test_b test_c \
    --gpu "$g" --output-dir "$o" >"$o/encode.log" 2>&1 || true; }
st_roll(){ local d=$1 s=$2 g=$3; local r=$ROLL/st_own_d${d}_s${s}
  [ -f "$r/test_b.npz" ] && return 0
  local ck=$RUN/st_d${d}_s${s}/encoder/checkpoint_iter020000.pt
  [ -f "$ck" ] || { echo "[st] no enc d$d s$s"; return 0; }; mkdir -p "$r"; echo "[st] roll d$d s$s"
  nice -n 10 python scripts/session29/roll_own_predictor.py "$ck" "$LAT/st_d${d}_s${s}/test_b.npz" \
    "$r" --device "$(dev $g)" >"$r.own.log" 2>&1 || true; }
st_dec(){ local d=$1 s=$2 g=$3; local tag=dec_st_d${d}_s${s}
  [ -f "$DEC/$tag/decoder_iter030000.pt" ] && { echo "[st] skip dec d$d s$s"; return 0; }
  echo "[st] decoder d$d s$s"
  taskset -c 0-31 bash scripts/session29/dec_posthoc_launch.sh "$LAT/st_d${d}_s${s}" "$tag" "$g"; }

echo "[st] PHASE A encoders, 2-packed at $(date -Iseconds)"
st_enc 64 0 0 & st_enc 64 1 1 & wait
st_enc 64 2 0 & st_enc 64 42 1 & wait
st_enc 16 0 0 & st_enc 16 1 1 & wait
st_enc 16 42 0 & wait
echo "[st] PHASE B extract latents at $(date -Iseconds)"
for s in $SEEDS_D64; do st_lat 64 $s 0; done
for s in $SEEDS_D16; do st_lat 16 $s 0; done
echo "[st] PHASE C JEPA-own rollouts, 2-packed at $(date -Iseconds)"
st_roll 64 0 0 & st_roll 64 1 1 & wait
st_roll 64 2 0 & st_roll 64 42 1 & wait
st_roll 16 0 0 & st_roll 16 1 1 & wait
st_roll 16 42 0 & wait
echo "[st] PHASE D decoders, 2-packed at $(date -Iseconds)"
st_dec 64 0 0 & st_dec 64 1 1 & wait
st_dec 64 2 0 & st_dec 64 42 1 & wait
st_dec 16 0 0 & st_dec 16 1 1 & wait
st_dec 16 42 0 & wait
echo "[st] COMPLETE at $(date -Iseconds)"
```

- [ ] **Step 2: Lint the script**

Run: `bash -n scripts/session29/st_band.sh && echo "syntax OK"`
Expected: `syntax OK`.

- [ ] **Step 3: Commit** (gated on user approval)

```bash
git add scripts/session29/st_band.sh
git commit -m "feat(session29): spatio-temporal JEPA band launcher"
```

- [ ] **Step 4: Launch (only once the d32 band has freed both cards)**

Run (background, CPU-capped):
```bash
taskset -c 0-31 bash scripts/session29/st_band.sh > outputs/runs/session29/st_band.log 2>&1 &
echo "launched pid $!"
```
Verify the first encoder is on an RTX 6000: `grep -m1 gpu_name outputs/runs/session29/st/st_d64_s0/train.log`. Watch progress via the per-run `train.log` iter lines. Do NOT launch while d32 is still on both cards (oversubscription); confirm with `nvidia-smi` first.

---

### Task 6: Six-metric comparison report

**Files:**
- Create: `scripts/session29/st_compare.py`

This reuses the existing, d-parameterized eval machinery and adds an ST family alongside the per-frame JEPA. The forecast band is already covered by `m_seed_forecast_band.py`; this script adds the ST tags and prints all six metrics side by side.

- [ ] **Step 1: Write the comparison driver**

Create `scripts/session29/st_compare.py`:

```python
"""Six-metric ST-vs-per-frame comparison at a given d. Usage: st_compare.py [D].

Reads ST rollouts/latents/decoders produced by st_band.sh and the per-frame
jepa_tf_noc artifacts, then prints: (1) forecast band (wake-enstrophy R^2 vs
horizon), (2) SSIM test_a/b/c, (3) drift Mahalanobis, (4) impact (G,D,Y) probe
R^2, (5) participation ratio + near-null-dim count, (6) instantaneous wake R^2.
Each metric is computed by the existing session29 modules; this is the
aggregator only.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session29"))
import m_lowd_forecast as MF  # noqa: E402

D = int(sys.argv[1]) if len(sys.argv) > 1 else 64
ROLL = MF.ROLL
LAT = REPO / "outputs" / "session29" / "latents" / "st"
DEC = REPO / "outputs" / "runs" / "session29"
HS = (1, 2, 4, 8, 12, 16)
SEEDS = {64: [0, 1, 2, 42], 16: [0, 1, 42]}[D]


def part_ratio(z):  # z: (n, d)
    s = np.linalg.svd(z - z.mean(0), compute_uv=False)
    s2 = s ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def ssim_of(tag):
    f = DEC / tag / "decoder_summary.json"
    if not f.exists():
        return None
    j = json.loads(f.read_text())
    return {k: j.get(f"ssim_{k}", j.get(k)) for k in ("test_a", "test_b", "test_c")}


def forecast_band(st_lat_fn, roll_fn):
    otr, itr, _ = MF.load_obs("train")
    otb, itb, _ = MF.load_obs("test_b")
    curves = []
    for s in SEEDS:
        lat, rk = st_lat_fn(s), roll_fn(s)
        if not (ROLL / rk / "test_b.npz").exists() or not (lat / "train.npz").exists():
            continue
        gs = MF.fit_probe(str(lat.relative_to(MF.LAT)) if False else lat, "wake_enstrophy", otr, itr)
        c = MF.forecast_curve(rk, lat, "wake_enstrophy", gs, otb, itb)
        if c is not None:
            curves.append(c)
    if not curves:
        return None
    return {h: np.array([c[h] for c in curves]) for h in HS}


def main():
    print(f"\n=== ST vs per-frame, d={D} ===")
    # Forecast band: ST tags vs the existing per-frame band (m_seed_forecast_band covers per-frame).
    band = forecast_band(lambda s: LAT / f"st_d{D}_s{s}", lambda s: f"st_own_d{D}_s{s}")
    print("forecast (wake R^2 mean[min,max]):  h=" + " ".join(f"{h}" for h in HS))
    if band:
        print("  ST-own : " + "  ".join(
            f"{band[h].mean():+.2f}[{band[h].min():+.2f},{band[h].max():+.2f}]" for h in HS))
    else:
        print("  ST-own : (rollouts not present yet)")
    print("  (per-frame band: run `python scripts/session29/m_seed_forecast_band.py", D, "`)")
    # SSIM
    print("SSIM (test_a/test_b/test_c):")
    for s in SEEDS:
        v = ssim_of(f"dec_st_d{D}_s{s}")
        print(f"  ST s{s}: {v}")
    # PR + null-dim, instantaneous wake R^2: per seed from train latents
    print("participation ratio / near-null dims (<1% of max var):")
    for s in SEEDS:
        f = LAT / f"st_d{D}_s{s}" / "test_b.npz"
        if not f.exists():
            print(f"  ST s{s}: (latents absent)"); continue
        z = np.load(f)["z_full"].reshape(-1, D)
        v = z.var(0)
        nnull = int((v < 0.01 * v.max()).sum())
        print(f"  ST s{s}: PR={part_ratio(z):.1f}/{D}  null={nnull}/{D}")
    print("\n(drift Mahalanobis: `python scripts/session29/m2_drift_nowake.py` on the ST tags;")
    print(" impact (G,D,Y) probe: reuse the impact-frame KRR probe on st_d{D}_s* latents.)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity-run on the smoke latents (shape-only)**

Run: `python scripts/session29/st_compare.py 16` (after Task 4, before the full band)
Expected: prints the section headers and "(rollouts not present yet)"/"(latents absent)" placeholders without crashing. This verifies the aggregator's plumbing; real numbers come after Task 5 completes.

- [ ] **Step 3: Commit** (gated on user approval)

```bash
git add scripts/session29/st_compare.py
git commit -m "feat(session29): ST-vs-per-frame six-metric comparison report"
```

- [ ] **Step 4: After the band completes, produce the comparison and decide the A->B gate**

Run:
```bash
python scripts/session29/m_seed_forecast_band.py 64   # per-frame baseline band, d64
python scripts/session29/st_compare.py 64
python scripts/session29/m_seed_forecast_band.py 16
python scripts/session29/st_compare.py 16
```
Decision: escalate to variant B (full V-JEPA) only if the ST forecast band beats per-frame at either d by a margin outside the seed bands, or materially improves SSIM or drift. Otherwise report A as a clean null. Either way, report the full table to the user honestly (no softening), per the project writing rule.

---

## Self-review

**Spec coverage:** the one architectural change (Task 1), causal-only + per-frame-norm safety (Task 1 step 3 + causality test), the `--encoder st_hybrid` wiring (Task 2), latent extraction (Task 3), d=64 and d=16 seeds {0,1,2,42}/{0,1,42} at 20k convergence-matched (Task 5), and all six metrics (Task 6) are each covered. The A->B gate is in Task 6 step 4. No spec requirement is unmapped.

**Placeholder scan:** no TBD/TODO/"handle edge cases"; every code step shows full code; every command shows expected output. The two cross-references to existing tools (`m_seed_forecast_band.py`, `m2_drift_nowake.py`) point at on-disk, verified scripts, not undefined work.

**Type consistency:** `SpatioTemporalCNNViTEncoder(latent_dim=, projection_norm=, temporal_kernel=)` is used identically in Tasks 1, 2, 3, 4. The checkpoint key `args["temporal_kernel"]` written by argparse (Task 2) is read with that exact name in Task 3 and asserted in Task 4 step 2. The latent npz schema (`z_full (n,120,d)`) matches the existing extractor output verified on `jepa_tf_noc_d64_s0`.
