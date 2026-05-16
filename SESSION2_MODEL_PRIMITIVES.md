# SESSION2_MODEL_PRIMITIVES.md

Session 2 plan for the vortex-jepa project.

Last updated: 2026-05-15.

## Session goal

Implement three model primitives with unit tests that pass before any larger module
imports them. No encoder, no predictor, no training loop yet.

|Module                |Purpose                                |
|----------------------|---------------------------------------|
|`src/models/sigreg.py`|SIGReg anti-collapse regularizer       |
|`src/models/adaln.py` |AdaLN-Zero conditioning block          |
|`src/models/rope.py`  |Rotary position embeddings, 1D temporal|

Pass criteria: all 14 unit tests pass on a clean checkout
(`pytest tests/ -v` returns 0 with no non-third-party warnings).

## Why these three first

The encoder and predictor (Session 3) depend on all three primitives. If SIGReg has a
numerical bug, the entire JEPA loss is silently wrong and the participation-ratio
diagnostic gives misleading signal. If AdaLN-Zero is not actually zero-initialized,
the predictor does not start as identity-on-residual and the AdaLN gradient signal is
misshapen from step one. RoPE is the smallest, but a wrong rotation matrix corrupts
the attention pattern subtly and is almost impossible to debug post hoc. Get them
tested in isolation.

## Reference papers

You have the arXiv MCP plugin available during this session. If anything in the
specifications below is ambiguous, consult the primary sources directly rather than
guessing.

|Reference                                                                                                 |arXiv ID        |Used for                                                       |
|----------------------------------------------------------------------------------------------------------|----------------|---------------------------------------------------------------|
|LeJEPA: Balestriero, LeCun, “Provable and Scalable Self-Supervised Learning Without the Heuristics”       |2511.08544      |SIGReg derivation, Cramer-Wold guarantee, Epps-Pulley statistic|
|LeWM: Maes, Le Lidec, Scieur, LeCun, Balestriero, “LeWorldModel”                                          |2603.19312      |Applied SIGReg setup (M, knot range, weighting)                |
|Epps and Pulley (1983), “A test for normality based on the empirical characteristic function”             |Biometrika 70(3)|The univariate normality test underlying SIGReg                |
|DiT: Peebles, Xie, “Scalable Diffusion Models with Transformers”                                          |2212.09748      |AdaLN-Zero exact specification and initialization              |
|RoFormer: Su, Lu, Pan, Murtadha, Wen, Liu, “RoFormer: Enhanced Transformer with Rotary Position Embedding”|2104.09864      |RoPE derivation and the rotation property                      |

Recommended consultation sequence if unsure:

1. arXiv 2511.08544 sections 2 to 3 for SIGReg and its proof sketch.
1. arXiv 2603.19312 algorithm 1 and appendix A for the applied LeWM defaults
   (M = 1024 was the LeWM choice; we use M = 256 because d = 32, see CLAUDE.md).
1. arXiv 2212.09748 figure 3 for the AdaLN-Zero diagram, including the zero-init.
1. arXiv 2104.09864 sections 3.3 to 3.4 for RoPE’s rotation matrix in 2D pairs.

## Files to create

```
src/models/__init__.py
src/models/sigreg.py
src/models/adaln.py
src/models/rope.py
tests/__init__.py
tests/test_sigreg.py
tests/test_adaln_zero.py
tests/test_rope.py
```

## Module 1: `src/models/sigreg.py`

Implement the Sketched Isotropic Gaussian Regularizer of Balestriero and LeCun
(arXiv:2511.08544, 2025), as used in LeWM (arXiv:2603.19312, 2026).

### Class signature

```python
class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer.

    References:
        Balestriero, LeCun. "LeJEPA: Provable and Scalable Self-Supervised
        Learning Without the Heuristics." arXiv:2511.08544, 2025.
    """

    def __init__(
        self,
        dim: int,
        num_projections: int = 256,
        num_knots: int = 17,
        knot_range: tuple[float, float] = (0.2, 4.0),
        resample_each_step: bool = True,
        weight_lambda: float = 1.0,
    ) -> None:
        ...

    def forward(self, z: Tensor) -> Tensor:    # z: (B, dim) -> scalar loss
        ...
```

### Algorithm

1. Sample M unit-norm directions `u^(m)` uniformly on `S^{dim-1}`. If
   `resample_each_step` is True, draw fresh directions every forward call; otherwise
   cache them as a non-trainable buffer at init.
1. Project: `h^(m) = z @ u^(m)`, resulting shape `(B, M)`.
1. For each projection, compute the Epps-Pulley test statistic against N(0, 1).
1. Return the mean over the M projections.

The Epps-Pulley statistic with Gaussian weighting `w(t) = exp(-t^2 / (2 * weight_lambda^2))`
is

```
T^(m) = integral over t of  w(t) * |phi_N(t; h^(m)) - phi_0(t)|^2  dt
```

where `phi_N(t; h) = (1 / N) sum_n exp(i t h_n)` is the empirical characteristic
function and `phi_0(t) = exp(-t^2 / 2)` is the standard Gaussian characteristic
function.

Implement the integral as a trapezoidal quadrature with `num_knots` knots uniformly
spaced in `knot_range = (0.2, 4.0)`. Endpoint weights are halved per trapezoidal rule.

### Numerical notes (mandatory)

- Compute the regularizer body in fp32 even if the surrounding model runs in bf16.
  The complex exponentials and their squared magnitude differences are not stable at
  bf16. Cast the input to fp32 at the top of forward and return an fp32 scalar.
  PyTorch’s autograd will handle the dtype promotion correctly when used in a mixed
  precision training loop.
- Use `torch.complex64` for the empirical characteristic function. Real and imaginary
  parts go through autograd independently.
- Trapezoidal quadrature: spacing is `(knot_range[1] - knot_range[0]) / (num_knots - 1)`.
  Endpoint weights are 0.5, interior weights are 1.0, multiplied by the spacing.
- Direction sampling: draw `u ~ N(0, I_dim)` then normalize to unit length. Avoid
  computing on the CPU and transferring; sample on `z.device`.

### Unit tests for SIGReg

`tests/test_sigreg.py`:

```python
def test_sigreg_low_on_isotropic_gaussian():
    """A batch from N(0, I_32) should give SIGReg below 0.1."""

def test_sigreg_high_on_heavy_tailed():
    """A batch from Student-t with df=2 should give SIGReg above 5.0."""

def test_sigreg_high_on_uniform():
    """A batch from Uniform(-1, 1) should give SIGReg above 1.0."""

def test_sigreg_invariant_to_projection_count():
    """Mean SIGReg with M=64, 256, 1024 should agree to within 20 percent on
    isotropic Gaussian samples (a robustness check for the M ablation later)."""

def test_sigreg_gradient_flows():
    """Backward pass produces non-zero gradients on z. Use a random seed so
    the test is deterministic."""

def test_sigreg_dtype_promotion():
    """Input in bf16 (with autocast) produces fp32 output and the backward
    pass runs without overflow. The gradient on the bf16 input is bf16."""
```

Use `torch.manual_seed(0)` at the top of every test. Batch size 4096. Standard
`pytest` fixtures, no other framework.

Suggested reference implementation strategy: write a 30-line numpy version of the
quadrature first, verify it gives the expected magnitudes on Gaussian and Student-t
samples, then translate to PyTorch and verify against the numpy reference on the same
seeded inputs.

## Module 2: `src/models/adaln.py`

AdaLN-Zero from Peebles and Xie (DiT, arXiv:2212.09748, 2023). Mandatory: the final
linear layer that produces `(shift, scale, gate)` must be zero-initialized so the
block starts as identity-on-residual.

### Class signature

```python
class AdaLN(nn.Module):
    """Adaptive LayerNorm conditioning with zero-init (AdaLN-Zero).

    Returns the (shift, scale, gate) tuple used inside a DiT-style transformer
    block. The block consumer applies them as:

        x_in = layer_norm(x) * (1 + scale) + shift
        x_out = x + gate * sublayer(x_in)

    Reference:
        Peebles, Xie. "Scalable Diffusion Models with Transformers."
        arXiv:2212.09748, 2023, figure 3.
    """

    def __init__(self, hidden_dim: int, cond_dim: int) -> None:
        ...

    def forward(self, cond: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """cond: (B, cond_dim) or (B, T, cond_dim).
        Returns (shift, scale, gate), each shaped (B, [T,] hidden_dim).
        """
        ...
```

### Internal structure

```
cond -> SiLU -> Linear(cond_dim, 3 * hidden_dim) -> split into (shift, scale, gate)
```

The Linear weights AND biases are zero-initialized. This guarantees `shift = scale = gate = 0` at init, which makes the gated residual equal to the input.

A typical use inside a transformer block (the consumer, not part of this module):

```python
shift, scale, gate = adaln(cond)
x = x + gate * mlp(layer_norm(x) * (1 + scale) + shift)
```

### Unit tests for AdaLN

`tests/test_adaln_zero.py`:

```python
def test_adaln_zero_output_at_init():
    """At initialization, shift, scale, and gate are all exactly zero."""

def test_adaln_block_is_identity_at_init():
    """A small toy transformer block using AdaLN at init satisfies
    forward(x, cond) == x within float tolerance."""

def test_adaln_gradient_nonzero_after_step():
    """After one optimizer step with a non-trivial loss, the AdaLN linear
    weights become non-zero."""

def test_adaln_conditioning_broadcasts_over_time():
    """When cond has shape (B, T, cond_dim), shift/scale/gate have shape
    (B, T, hidden_dim) with no broadcast surprises."""
```

## Module 3: `src/models/rope.py`

Rotary Position Embeddings (Su et al., arXiv:2104.09864, 2021) applied along the
temporal axis only. For our use case the sequence length is the sub-trajectory length
L = 32, so the maximum position is 32. The encoder transformer in Session 3 uses 2D
sinusoidal embeddings for the spatial tokens; RoPE here is only for the time axis in
the predictor.

### Function signatures

```python
def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Return (cos_cache, sin_cache), each of shape (seq_len, head_dim // 2).
    head_dim must be even.
    """

def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to query or key tensor.

    Args:
        x: shape (B, n_heads, seq_len, head_dim)
        cos: shape (seq_len, head_dim // 2)
        sin: shape (seq_len, head_dim // 2)

    Returns:
        Tensor of same shape as x, with each adjacent pair of dims rotated.
    """
```

### Algorithm

Pair adjacent dimensions of the head: `(x_0, x_1), (x_2, x_3), ...`. For position `t`
and pair index `i`, the angle is `t / base^(2i / head_dim)`. Apply the rotation:

```
x'_{2i}   = x_{2i}   * cos(theta_{t,i}) - x_{2i+1} * sin(theta_{t,i})
x'_{2i+1} = x_{2i}   * sin(theta_{t,i}) + x_{2i+1} * cos(theta_{t,i})
```

The cache stores `cos(theta_{t,i})` and `sin(theta_{t,i})` for all `(t, i)`.

### Unit tests for RoPE

`tests/test_rope.py`:

```python
def test_rope_identity_at_position_zero():
    """apply_rope(x, cos[0:1], sin[0:1]) returns x unchanged at position 0."""

def test_rope_preserves_dot_product_relative_to_offset():
    """For random q and k, dot(rope(q, t1), rope(k, t2)) depends only on
    (t1 - t2), not on t1 and t2 individually. Verify by computing the dot
    product for (t1, t2) = (3, 5) and (10, 12) and asserting equality
    within float tolerance."""

def test_rope_cache_shapes():
    """build_rope_cache returns tensors of shape (seq_len, head_dim // 2)."""

def test_rope_cache_dtypes():
    """build_rope_cache respects the requested dtype (fp32, bf16) and device."""

def test_rope_rejects_odd_head_dim():
    """build_rope_cache raises ValueError for odd head_dim."""
```

## Coding conventions (recap from CLAUDE.md)

- Python 3.10+ (matches the project’s `.venv`), PyTorch 2.x.
- One module per file. No catch-all `utils.py`.
- Type hints everywhere. Google-style docstrings on every public class and function.
- Cite the paper and arXiv ID in each module’s top-level docstring.
- No imports beyond `torch`, `torch.nn`, `math`, and Python stdlib in the
  implementation modules. scipy may appear in the tests only (for
  `scipy.stats.normaltest` as an optional cross-check, not as a required dependency
  of the test).
- `ruff check src/models/ tests/` and `black --check --line-length 100 src/models/ tests/` must pass before commit.
- All random sources seeded with `torch.manual_seed(0)` at the top of each test.

## Out of scope for Session 2

- Encoder, predictor, JEPA wrapper (Session 3).
- Hydra configs (until training loop exists, configs are not exercised).
- W&B integration (Session 4, when the training loop lands).
- The 4-stage encoder ablation (72 tokens). Recorded in HANDOFF D11 as a deferred
  option.
- VICReg fallback module. It is invoked only when SIGReg fails the auto-fallback
  criterion at iteration 20k of training, which is Session 4 at the earliest. Add
  `src/models/vicreg.py` then.

## Expected duration

Two to three hours if the unit tests are written before the implementations (TDD
style). SIGReg is the bulk of the time, mostly because the Epps-Pulley quadrature
should be checked against a small numpy reference implementation before being
translated to PyTorch.

## If something is unclear

The arXiv MCP plugin is enabled in this session. If a specification below is
ambiguous or contradicts your prior reading, consult the primary source directly:

- SIGReg numerical details: arXiv:2511.08544, especially the appendix on the
  Epps-Pulley quadrature, and arXiv:2603.19312 algorithm 1 for the LeWM applied
  hyperparameters.
- AdaLN-Zero diagram and initialization: arXiv:2212.09748 figure 3.
- RoPE rotation matrix: arXiv:2104.09864 section 3.4.

Prefer consulting the source over guessing. If after consulting the source there is
still genuine ambiguity (for example, multiple equally valid quadrature schemes for
the Epps-Pulley integral), record the decision and rationale as a brief note in
HANDOFF.md under “Decision history” with a new identifier (D12, D13, …) before
proceeding.

## After Session 2 lands

Carlos triggers Session 3 (encoder, predictor, JEPA wrapper) with one message. The
unit tests from Session 2 must remain green throughout Session 3 because the
primitives are imported as-is, not modified. If a Session 3 change forces a Session 2
primitive to evolve, that is a flag to stop, update the unit tests first, and confirm
they pass before continuing.
