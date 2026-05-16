# SESSION3_ENCODER_PREDICTOR.md

Session 3 plan for the vortex-jepa project.

Last updated: 2026-05-16.

## Session goal

Build the encoder and the predictor modules, with unit tests that pass before any
training loop or JEPA wrapper imports them. The JEPA wrapper, scheduled sampling,
diagnostics, and the training entrypoint are deferred to Session 4.

|Module                   |Purpose                                                      |
|-------------------------|-------------------------------------------------------------|
|`src/models/encoder.py`  |Hybrid CNN + 6-layer ViT, [CLS] readout, BatchNorm projection|
|`src/models/predictor.py`|6-layer AR transformer, AdaLN-Zero, RoPE, causal mask        |

Pass criteria: all unit tests in `tests/test_encoder.py` and `tests/test_predictor.py`
pass on a clean checkout (`pytest tests/ -v` returns 0 with no non-third-party
warnings), AND the three Session 2 test files remain green.

## Why these two together

The encoder and predictor share no code, but they share a single shape contract
(`(B, T, d=32)` between them) and three Session 2 dependencies (AdaLN, RoPE,
indirectly SIGReg via the projection sizing). Building them in the same session
keeps the contract testable end-to-end without yet committing to a training loop.
The JEPA wrapper, the loss aggregation, scheduled sampling, and diagnostics are
Session 4’s job and are not in scope here.

## arXiv MCP plugin

The arXiv MCP plugin is enabled in this session. If a specification below is
ambiguous or contradicts your prior reading, consult the primary source directly
rather than guessing. Recommended primary sources:

|Reference                                       |arXiv ID  |Used for                                                |
|------------------------------------------------|----------|--------------------------------------------------------|
|LeWM: Maes, Le Lidec, Scieur, LeCun, Balestriero|2603.19312|Encoder projection (Section 3.1), predictor AdaLN-Zero  |
|DiT: Peebles, Xie                               |2212.09748|AdaLN-Zero block structure, figure 3, official code     |
|RoFormer: Su, Lu, Pan, Murtadha, Wen, Liu       |2104.09864|RoPE applied to Q and K only (Section 3.4)              |
|LeJEPA: Balestriero, LeCun                      |2511.08544|BatchNorm vs LayerNorm projection debate (see D17 below)|
|V-JEPA 2 / V-JEPA 2-AC: Assran et al.           |2506.09985|Block-causal attention precedent, deferred to Session 4 |

LeWM is the more modern and closer-to-our-case template. Where LeWM and LeJEPA
disagree, default to LeWM. This is recorded as D17 in HANDOFF.md.

If after consulting the source there is still genuine ambiguity, record the
decision and rationale as a brief note in HANDOFF.md under “Decision history”
with a new identifier (D18, D19, …) before proceeding.

## Files to create

```
src/models/encoder.py
src/models/predictor.py
tests/test_encoder.py
tests/test_predictor.py
```

No changes to `src/data/`, no changes to the three Session 2 primitives
(`src/models/sigreg.py`, `src/models/adaln.py`, `src/models/rope.py`), no changes
to their unit tests. If a Session 3 change forces a Session 2 primitive to evolve,
stop, update the primitive’s unit tests first, confirm they pass, then continue.

## Locked decisions baked into Session 3

The following decisions are locked in CLAUDE.md and HANDOFF.md. Do not revisit
without explicit user approval.

1. **Encoder input shape**: `(B, T, 1, 192, 96)`, single channel (omega_z from
   the per-encounter cache). Native cache resolution, no upsampling, no SDF
   auxiliary channel in the default run.
1. **Encoder output shape**: `(B, T, d)` with `d = 32`. Per-frame latent.
1. **Encoder pooling**: [CLS] token followed by 1-layer MLP projection with
   `nn.BatchNorm1d` (NOT LayerNorm). Per LeWM Section 3.1, see D17.
1. **CNN stem stages**: 3 downsampling stages, channels 64 -> 128 -> 256,
   ending at a 24 x 12 feature map (288 spatial tokens at 256 channels).
1. **ViT depth**: 6 layers, 8 heads, hidden 256, MLP ratio 4, dropout 0.0 in
   the encoder, 2D sinusoidal positional embeddings on the 288 spatial tokens.
1. **Predictor input shape**: `(B, T, d)` plus `(B, cond_dim)` per episode,
   with `cond_dim = 3` for `c = (G, D, Y)`. Phase variable phi_t is NOT included
   in the default. See D16. The API call inside the predictor takes
   `(B, T, cond_dim)` with c broadcast across t, so a future switch to
   `cond_dim = 4` is one line.
1. **Predictor architecture**: 6 layers, 16 heads, hidden 384, MLP ratio 4,
   dropout 0.1. Two AdaLN-Zero modules per block (one before attention, one
   before MLP), each producing one `(shift, scale, gate)` triple. See “DiT
   block structure” below. RoPE on Q and K, applied to the temporal axis only.
   Causal mask (lower-triangular).
1. **Predictor output shape**: `(B, T, d)`, same dimension as the encoder
   latent. The final projector is a 1-layer MLP with BatchNorm matching the
   encoder projector (per LeWM Section 3.1).
1. **No conditioning of the encoder on c**. The encoder is unconditional by
   design (D6). The c-in-encoder ablation is a deliberate negative-result run
   handled by a separate config, not by this module.

## Module 1: `src/models/encoder.py`

### Class signature

```python
class HybridCNNViTEncoder(nn.Module):
    """Hybrid CNN stem followed by a small ViT, with [CLS] readout and a
    BatchNorm-projected MLP head to the latent dimension d.

    Reference architecture for the projection-with-BatchNorm rationale:
        Maes et al., "LeWorldModel: Stable End-to-End Joint-Embedding
        Predictive Architecture from Pixels", arXiv:2603.19312, Section 3.1.
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
    ) -> None:
        ...

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, C, H, W) with C=1, H=192, W=96
        Returns z: (B, T, latent_dim).
        """
        ...

    @property
    def num_spatial_tokens(self) -> int:
        """288 for the default 3-stage stem on (192, 96) input."""
        ...
```

### Internal structure

CNN stem with three downsampling stages, taking `(B*T, 1, 192, 96)` to
`(B*T, 256, 24, 12) = 288 spatial tokens of dim 256`. Concrete layout:

|Stage  |Op                                    |Channels|Stride|Output (H x W)|
|-------|--------------------------------------|--------|------|--------------|
|Stem   |Conv 7x7, stride 2, GELU, GroupNorm(8)|64      |2     |96 x 48       |
|Block 1|2 x (Conv 3x3, GroupNorm(8), GELU)    |64      |1     |96 x 48       |
|Down 1 |Conv 3x3, stride 2, GroupNorm(8), GELU|128     |2     |48 x 24       |
|Block 2|2 x (Conv 3x3, GroupNorm(8), GELU)    |128     |1     |48 x 24       |
|Down 2 |Conv 3x3, stride 2, GroupNorm(8), GELU|256     |2     |24 x 12       |
|Block 3|2 x (Conv 3x3, GroupNorm(8), GELU)    |256     |1     |24 x 12       |

The 24 x 12 = 288 spatial tokens of dim 256 are flattened to `(B*T, 288, 256)`.

GroupNorm is preferred over BatchNorm in the stem because the effective batch
size at training time is small (B episodes x T = 32 frames x 1 channel per slot).
BatchNorm in the projection head is fine because the projection sees
`(B*T, latent_dim)` which is a larger effective batch.

ViT over the 288 tokens. Prepend a learnable [CLS] token (one per frame), add
2D sinusoidal positional embeddings (computed once, registered as a non-trainable
buffer), apply 6 standard transformer blocks (pre-norm LayerNorm, MHA with 8
heads, GELU MLP at ratio 4, dropout 0.0). After the final block, take the
[CLS] token of each frame as the per-frame summary, shaped `(B*T, vit_hidden)`.

Projection head: 1-layer MLP with BatchNorm.

```
proj = nn.Sequential(
    nn.Linear(vit_hidden, latent_dim),
    nn.BatchNorm1d(latent_dim),
)
```

The BatchNorm in the projection head is the LeWM-specific choice that interacts
correctly with SIGReg downstream (D17). Do not replace with LayerNorm in the
default run.

Reshape back to `(B, T, latent_dim)` before returning.

### Numerical notes (mandatory)

- The encoder runs under bf16 autocast in production. Build it as a plain
  fp32 module; the autocast wrapper at the training loop handles the cast.
- The 2D sinusoidal positional embedding is computed at init time from the
  feature-map dimensions (24, 12) and `vit_hidden = 256`. Standard sin/cos
  encoding: pair the 256 channels into 64 quadruples of (sin_x, cos_x, sin_y,
  cos_y), with separate frequency bases for x and y. Register as
  `self.pos_embed: Tensor = nn.Buffer(...)` so it moves with `.to(device)` but
  is not optimized. Shape: `(1, 288, vit_hidden)`.
- The [CLS] token is `nn.Parameter(torch.zeros(1, 1, vit_hidden))`, broadcast
  to `(B*T, 1, vit_hidden)` at forward time. Initialize with
  `nn.init.trunc_normal_(self.cls_token, std=0.02)` matching ViT convention.
- Reshape between (B, T, C, H, W) and (B*T, C, H, W) is the cleanest way to
  pass frames through the CNN. Use `x.flatten(0, 1)` and `x.view(B, T, ...)`
  symmetrically; do not use `permute` followed by `contiguous()`.
- Verify the parameter count is in the 8M to 12M range. The CNN stem is
  ~2.5M, the ViT is ~5M at hidden 256 depth 6 heads 8, the projection is
  ~16k.

### Unit tests for the encoder

`tests/test_encoder.py`:

```python
def test_encoder_shape_contract():
    """Input (2, 8, 1, 192, 96) -> output (2, 8, 32). Float tolerance n/a;
    this is purely a shape check on a fresh module under torch.manual_seed(0)."""

def test_encoder_num_spatial_tokens():
    """encoder.num_spatial_tokens == 288 for the default stem on (192, 96)."""

def test_encoder_projection_is_batchnorm():
    """The final layer of encoder.proj is an instance of nn.BatchNorm1d, NOT
    LayerNorm. This is the LeWM-specific constraint, see D17 in HANDOFF.md.
    Use isinstance(module, nn.BatchNorm1d) on the right submodule."""

def test_encoder_parameter_count_in_range():
    """8e6 < sum(p.numel() for p in encoder.parameters()) < 12e6 for the
    default config."""

def test_encoder_gradient_flows():
    """Backward pass on a scalar loss produces non-zero gradients on the
    input. Deterministic via torch.manual_seed(0)."""

def test_encoder_bf16_autocast_roundtrip():
    """Under torch.autocast(device_type='cuda', dtype=torch.bfloat16) (or 'cpu'
    if no CUDA is available), the forward pass runs end-to-end on a small
    batch (B=2, T=4) without dtype errors and returns a tensor whose dtype is
    bfloat16 or float32 (autocast policy is layer-dependent). Skip the test if
    autocast is not supported on the available device."""

def test_encoder_deterministic_with_fixed_seed():
    """Two encoder instances built with the same torch.manual_seed(0) produce
    identical outputs on the same input. Confirms init reproducibility."""
```

Use `torch.manual_seed(0)` at the top of every test. Default batch size 2,
sub-trajectory length 8 (not the production 32) to keep tests fast.

## Module 2: `src/models/predictor.py`

### Class signature

```python
class AutoregressivePredictor(nn.Module):
    """Autoregressive transformer over latent trajectories with AdaLN-Zero
    conditioning and RoPE temporal positions.

    Predictor architecture and AdaLN-Zero placement:
        Maes et al., arXiv:2603.19312, Section 3.1.
    AdaLN-Zero block structure (two AdaLN modules per block, one before
    attention and one before the MLP, with shift/scale/gate triples each):
        Peebles, Xie, "Scalable Diffusion Models with Transformers",
        arXiv:2212.09748, figure 3, plus the official code at
        https://github.com/facebookresearch/DiT/blob/main/models.py.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        cond_dim: int = 3,
        hidden_dim: int = 384,
        depth: int = 6,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 32,
    ) -> None:
        ...

    def forward(self, z: Tensor, cond: Tensor) -> Tensor:
        """Teacher-forced forward pass.

        Args:
            z: (B, T, latent_dim). Encoder latents over the sub-trajectory.
            cond: (B, cond_dim). Static episode descriptor c = (G, D, Y).
                Broadcast internally to (B, T, cond_dim).

        Returns:
            z_hat: (B, T, latent_dim). Per-position next-step prediction.
                z_hat[:, t, :] is the prediction of z[:, t+1, :] from z[:, :t+1, :].
                The last position z_hat[:, T-1, :] is the prediction of the
                (would-be) (T+1)-th frame, which is the natural rollout step.
        """
        ...

    def rollout(self, z_init: Tensor, cond: Tensor, steps: int) -> Tensor:
        """Open-loop autoregressive rollout.

        Args:
            z_init: (B, T_init, latent_dim). Seed latents (typically T_init = 1
                for full rollout from a single starting frame, or T_init > 1 to
                warm-start with several ground-truth frames).
            cond: (B, cond_dim). Static episode descriptor.
            steps: number of additional frames to predict beyond T_init.

        Returns:
            z_full: (B, T_init + steps, latent_dim). The seed plus the rolled-out
                predictions, concatenated along the time axis. z_full[:, :T_init, :]
                equals z_init exactly.
        """
        ...
```

### Internal structure

Embedding. A `nn.Linear(latent_dim, hidden_dim)` lifts each frame’s z to the
predictor hidden dim 384. No positional embedding added here; RoPE handles
positions inside attention.

Conditioning preprocessor. A 2-layer MLP from `cond_dim` to `hidden_dim` produces
the conditioning vector that feeds the AdaLN modules:

```
self.cond_mlp = nn.Sequential(
    nn.Linear(cond_dim, hidden_dim),
    nn.SiLU(),
    nn.Linear(hidden_dim, hidden_dim),
)
```

This is the conditioning stream. Inside the predictor it is broadcast across
time: `c_seq = self.cond_mlp(cond).unsqueeze(1).expand(-1, T, -1)` of shape
`(B, T, hidden_dim)`. Each `AdaLN` call inside a block receives `c_seq` as its
cond input and returns `(shift, scale, gate)` of shape `(B, T, hidden_dim)`.

Block. Each predictor block contains exactly TWO AdaLN modules from Session 2,
one before attention and one before the MLP, following the DiT figure 3 layout:

```python
class PredictorBlock(nn.Module):
    def __init__(self, hidden_dim, heads, mlp_ratio, dropout, max_seq_len):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.adaln1 = AdaLN(hidden_dim=hidden_dim, cond_dim=hidden_dim)
        self.attn = CausalSelfAttentionWithRoPE(
            hidden_dim=hidden_dim, heads=heads, dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.adaln2 = AdaLN(hidden_dim=hidden_dim, cond_dim=hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, c_seq):
        shift1, scale1, gate1 = self.adaln1(c_seq)
        h = self.norm1(x) * (1 + scale1) + shift1
        x = x + gate1 * self.attn(h)
        shift2, scale2, gate2 = self.adaln2(c_seq)
        h = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.mlp(h)
        return x
```

Note `elementwise_affine=False` on the LayerNorms. The AdaLN scale and shift
do the affine work; a learnable affine inside LayerNorm would compose with
AdaLN and the zero-init identity property would no longer hold. This matches
the DiT convention.

The attention sub-module applies RoPE inside, to query and key tensors only,
not values. Use the Session 2 `apply_rope` function with a RoPE cache built
at init for `max_seq_len = 32` and the per-head dimension
`head_dim = hidden_dim // heads = 384 // 16 = 24`. RoPE requires even
head_dim; 24 is even, fine. Causal mask is a standard lower-triangular mask
materialised on demand.

```python
class CausalSelfAttentionWithRoPE(nn.Module):
    def __init__(self, hidden_dim, heads, dropout, max_seq_len):
        super().__init__()
        assert hidden_dim % heads == 0
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout
        cos, sin = build_rope_cache(max_seq_len, self.head_dim)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
        k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True,
        )
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.proj(out)
```

Final projector. Following LeWM Section 3.1, the predictor output passes
through a projector with the same architecture as the encoder projector
(1-layer MLP with BatchNorm). This is so the predicted and target embeddings
live in the same space:

```python
self.out_proj = nn.Sequential(
    nn.Linear(hidden_dim, latent_dim),
    nn.BatchNorm1d(latent_dim),
)
```

Apply by flattening time into the batch dimension for the BatchNorm pass,
then reshaping back to `(B, T, latent_dim)`.

### Numerical notes (mandatory)

- All AdaLN modules are zero-initialised by construction (Session 2 already
  enforces this in `AdaLN.__init__`). Verify in the test below that the
  predictor returns the input (after the input embedding) at initialisation.
- The input embedding `nn.Linear(latent_dim, hidden_dim)` is NOT
  zero-initialised. The identity-at-init property is residual identity inside
  the stack, not end-to-end identity from input to output.
- RoPE cache is built once at init in `CausalSelfAttentionWithRoPE` and
  registered as a buffer. Slicing it to `[:T]` at forward time handles
  variable T up to `max_seq_len = 32`. If a future use case needs T > 32,
  rebuild the cache.
- The `is_causal=True` flag in `F.scaled_dot_product_attention` is the right
  way to apply the causal mask; PyTorch will use the efficient kernel.
  Do not pass both `is_causal=True` and an explicit `attn_mask`.
- Parameter count target: ~14M to ~18M. The dominant cost is the 6 attention
  blocks at hidden 384.

### Conditioning broadcast

Inside `forward`, the cond preprocessor produces `c = self.cond_mlp(cond)` of
shape `(B, hidden_dim)`. Broadcast to `(B, T, hidden_dim)` once before the
block loop: `c_seq = c.unsqueeze(1).expand(-1, T, -1)`. Pass `c_seq` to each
block’s two AdaLN modules. With static cond (cond_dim = 3, no phi_t), c_seq is
constant across t. The block still works correctly; the AdaLN modules just
return constant-in-t triples. When phi_t is later added (cond_dim = 4 in a
future ablation), the broadcast is replaced with a time-varying input and no
other code changes.

### Unit tests for the predictor

`tests/test_predictor.py`:

```python
def test_predictor_shape_contract():
    """Input z: (2, 8, 32), cond: (2, 3). Output: (2, 8, 32). Deterministic
    via torch.manual_seed(0)."""

def test_predictor_identity_at_init():
    """At initialization, with all AdaLN modules returning (0, 0, 0), the
    residual stream is unchanged by the block stack. Concretely:
        x_after_embedding = predictor.embed(z)
        x_after_blocks    = predictor.blocks(x_after_embedding, c_seq)
        assert torch.allclose(x_after_blocks, x_after_embedding, atol=1e-6)
    This test reaches inside the predictor; use a helper or expose
    .blocks as a public attribute for testability."""

def test_predictor_all_adaln_modules_zero_init():
    """Count the AdaLN instances inside predictor (should be 2 per block x 6
    blocks = 12), and verify each one's final linear layer has zero weight
    and zero bias."""

def test_predictor_causal_mask():
    """Perturb z[:, t, :] for some fixed t in {0, 3, 6} and check that the
    output at positions strictly less than t is UNCHANGED to within float
    tolerance. Positions >= t are allowed to change. This verifies the
    causal mask."""

def test_predictor_rope_q_k_only_not_v():
    """Register a forward hook on the first attention layer that captures q,
    k, v before and after the apply_rope call. Assert that q and k are
    rotated (changed) and that v is NOT (unchanged). One clean way to do
    this is to add a debug flag to CausalSelfAttentionWithRoPE that stores
    pre-RoPE and post-RoPE q, k, v on self for inspection during the test
    only; otherwise factor apply_rope into a helper that the test calls
    directly with the same RoPE cache."""

def test_predictor_rollout_shape_and_seed_match():
    """rollout(z_init, cond, steps=4) with z_init of shape (2, 1, 32)
    returns shape (2, 5, 32), and z_full[:, 0, :] == z_init[:, 0, :]
    exactly (no perturbation of the seed)."""

def test_predictor_rollout_matches_teacher_when_groundtruth_fed():
    """If we manually feed ground-truth z into the predictor via the
    teacher-forced forward (not the autoregressive rollout), and we
    separately call rollout(z_init=z[:, :1], cond, steps=T-1), the
    AUTOREGRESSIVE prediction at step t will diverge from the
    teacher-forced prediction at step t for t > 0 (because the rollout
    feeds its own previous output). They should agree at step 0 only.
    Test that step 0 agrees within float tolerance and step T-1 does NOT
    (assert NOT allclose, with non-zero tolerance to avoid flaky tests on
    the AdaLN-Zero initial-identity case; use a non-zero seed and a few
    training steps of random gradient noise to break the identity)."""

def test_predictor_parameter_count_in_range():
    """1.4e7 < sum(p.numel() for p in predictor.parameters()) < 1.8e7
    for the default config."""

def test_predictor_gradient_flows():
    """Backward pass on a scalar loss produces non-zero gradients on every
    trainable parameter. Deterministic via torch.manual_seed(0)."""
```

The “rollout matches teacher when ground-truth fed” test is subtle. At init,
the predictor IS identity-on-residual, so teacher-forced forward and
autoregressive rollout will agree at every step on the trivial fixed point.
The test needs to break that identity (one optimizer step on random data) so
the divergence at later steps actually appears. Document this in the test.

## DiT block structure recap (for reference)

The DiT-style block, from Peebles and Xie figure 3 and the official Facebook
code at `https://github.com/facebookresearch/DiT/blob/main/models.py`:

```
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp =
    self.adaLN_modulation(c).chunk(6, dim=1)
x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
```

`adaLN_modulation` is a single Linear that outputs `6 * hidden_dim`, then chunks
into 6 triples. Our Session 2 `AdaLN` module returns one `(shift, scale, gate)`
triple per call, so we instantiate it TWICE per block instead of producing all
six values in one shot. Functionally equivalent; the two-instance factoring
makes the identity-at-init test trivial because we just check 12 modules instead
of 6.

## Coding conventions (recap from CLAUDE.md and SESSION2)

- Python 3.10+, PyTorch 2.x.
- One module per file. No catch-all `utils.py`.
- Type hints everywhere. Google-style docstrings on every public class and
  function. Cite the paper and arXiv ID in each module’s top-level docstring.
- Imports: `torch`, `torch.nn`, `torch.nn.functional`, `math`, Python stdlib,
  and the three Session 2 primitives (`from src.models.adaln import AdaLN`,
  `from src.models.rope import build_rope_cache, apply_rope`). No new
  third-party dependencies. SIGReg is NOT imported here; the JEPA wrapper in
  Session 4 will compose encoder + predictor + sigreg.
- `ruff check src/models/ tests/` and `black --check --line-length 100 src/models/ tests/` must pass before commit.
- `torch.manual_seed(0)` at the top of each test. Default test batch B=2,
  test sub-trajectory T=8 (not the production T=32, to keep tests under one
  second each).

## Out of scope for Session 3

- `src/models/jepa.py` (the wrapper composing encoder + predictor + sigreg).
  Session 4.
- `src/training/scheduled_sampling.py`. Session 4; the design decision
  between V-JEPA 2-AC-faithful (two-loss sum, fixed coefficients) and
  Bengio-style scheduled sampling (probabilistic teacher/student mixing) is
  the open D18. Recommended default for Session 4 is the V-JEPA 2-AC two-loss
  sum (simpler, more faithful to the published recipe), but this is a
  Session-4 decision, not a Session-3 one.
- `src/training/diagnostics.py` (participation ratio, probe R^2, variance
  histogram). Session 4.
- `src/training/train_jepa.py`. Session 5.
- The decoder. Trained separately on a frozen encoder, never part of JEPA
  loss. Out of scope until after the first full training run.
- The four baselines (POD, Fukami AE, Solera-Rico beta-VAE, PLDM). Parallel
  work, not part of the core JEPA path.
- The c-in-encoder negative-result ablation. Handled by a separate config
  later; the default encoder is unconditional by D6 and stays that way.

## Expected duration

Three to four hours if tests are written before implementations (TDD style).
The encoder is the bigger module by lines of code, but its tests are simpler
(shape checks, isinstance check on the projection, parameter count bracket).
The predictor is smaller but has subtler tests (causal mask, RoPE-on-q-k-not-v,
identity-at-init via AdaLN-Zero, rollout consistency).

Suggested order:

1. Write `tests/test_encoder.py` end to end, with all tests failing.
1. Implement `src/models/encoder.py` until all encoder tests pass.
1. Write `tests/test_predictor.py` end to end, with all tests failing.
1. Implement `src/models/predictor.py` until all predictor tests pass.
1. Run the full suite: `pytest tests/ -v`. All Session 2 tests must remain
   green; all new tests must pass.

## If something is unclear

The arXiv MCP plugin is enabled. Recommended consultation order if you have
doubts:

1. **DiT AdaLN-Zero block specifics**: arXiv:2212.09748 figure 3, and the
   official code at `facebookresearch/DiT/models.py`. The 6-value chunk into
   `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)` is the
   canonical structure; our two-instance-per-block factoring is mathematically
   equivalent.
1. **LeWM predictor wiring**: arXiv:2603.19312 Section 3.1 and Appendix D
   “Implementation details, Predictor Architecture”. LeWM’s defaults are
   6 layers, 16 heads, hidden 384, dropout 0.1, which match ours. LeWM uses
   history length N=3 with frame-skip 5; we deviate to full causal history
   within L=32 (recorded in the architectural spec Section 3.2). The
   deviation is locked, do not revisit.
1. **LeWM encoder projection (BatchNorm vs LayerNorm)**: arXiv:2603.19312
   Section 3.1 is emphatic that BatchNorm is required because the final ViT
   LayerNorm prevents anti-collapse optimization. The LeJEPA official
   reference repo at `github.com/galilai-group/lejepa` says “no clear
   difference observed” at large scale. Our regime (low-intrinsic-dim,
   small dataset) is closer to LeWM’s setting, so we follow LeWM. See D17.
1. **RoPE on Q and K only**: arXiv:2104.09864 Section 3.4. RoPE is applied
   only to query and key, not to value. This is the defining property that
   makes the dot product depend only on the relative offset.
1. **V-JEPA 2-AC scheduled sampling**: arXiv:2506.09985 Section 6 and
   appendices. Their actual recipe is teacher-forcing over T=15 positions
   PLUS two-step rollout, summed with fixed coefficients. NOT Bengio-style
   probabilistic mixing. This is relevant to Session 4, not Session 3.

If after consulting the source there is genuine ambiguity (for instance,
multiple equally valid interpretations of the rollout API), record the
decision and rationale as a new D18/D19/… entry in HANDOFF.md before
proceeding with the code.

## After Session 3 lands

Carlos triggers Session 4 with one message. Session 4 builds the JEPA wrapper,
the scheduled-sampling strategy (D18 to be decided), and the diagnostics
module, then runs a 200-iteration smoke test on a 5-case subset. Session 2
and Session 3 unit tests must remain green throughout Session 4.

## Decision references

- D6 (CLAUDE.md, HANDOFF.md): encoder is unconditional, c enters only the
  predictor.
- D13 (HANDOFF.md): SIGReg follows LeWM Appendix A without the N multiplier.
- D14 (HANDOFF.md): partition v1 absorbed two more run3 cases on 2026-05-16,
  total now 45 cases / 222 encounters. Affects the data loader, not Session 3.
- D16 (HANDOFF.md): default predictor cond_dim = 3, no phi_t. Phi_t is a
  contained ablation if forecast horizon disappoints.
- D17 (HANDOFF.md): BatchNorm at encoder projection per LeWM precedent, with
  the caveat that LeJEPA at scale shows no difference; if SIGReg partially
  collapses (H4), swap to LayerNorm at the projection as a first diagnostic
  before invoking the VICReg fallback.
- D18 (pending Session 4): scheduled sampling recipe, V-JEPA 2-AC-faithful vs
  Bengio-style. Not a Session 3 concern.
