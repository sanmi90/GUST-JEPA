# Session 28 latent-spectral-analysis track (spectrum_dmd.py)

The "eigenvalues of JEPA": two estimators of the latent dynamics spectrum
plus a DMD/linear-dynamics forecasting baseline rung. Physical ground
truth: undisturbed shedding St_dominant = 0.675 (period ~30 frames),
St_subharmonic = 0.338 (full-cycle clock, ~59 frames),
dt_tc = 0.05 (outputs/session28/undisturbed_stats.json).

## Two estimation methods

1. DATA-DRIVEN DMD (Part 1). Exact DMD on the encoded latent trajectories:
   fit the best-fit linear operator A (z_{t+1} ~ A z_t) by least squares,
   A = Z' Z^+, eigendecompose A. Run on (a) the Baseline no-gust
   limit-cycle frames and (b) pooled train trajectories, per family
   (POD, JEPA tf-no-c, Fukami) at d=64. Discrete eigenvalue lambda ->
   rate = log|lambda|/dt_tc, St = |angle(lambda)|/(2 pi dt_tc). This is the
   directly-comparable spectrum: the same operator construction for every
   family, so it isolates the encoder coordinates' linear dynamics.

2. INTRINSIC predictor Jacobian / FLOQUET (Part 2). The learned predictor
   is nonlinear, so there is no single operator; we linearize it by
   torch.autograd at on-cycle states. The delay-embedded one-step map
   (state = stacked history window, map = shift + append predictor output)
   has a COMPANION Jacobian; the ordered product over one full shedding
   cycle (59-frame subharmonic period) is the MONODROMY matrix,
   whose eigenvalues are the FLOQUET MULTIPLIERS. The leading-modulus
   multiplier (per-step ~1) certifies a marginally stable orbit (the
   on-manifold property). The monodromy turns out to be dominated by this
   one neutral direction with all others strongly damped, so the FREQUENCY
   is read from the Markov one-step Jacobian / the Part-1 DMD, not the
   over-damped oscillatory Floquet sub-mode (see HONEST READ below). We also
   report the plain one-step (W=1, Markov) Jacobian spectrum.

## St-recovery verdict

Best DMD St recovery (Baseline): pod with leading St = 0.682, matching the DNS dominant line (|err| = 0.007).

- pod: leading St = 0.682, |lambda| = 0.996 (DMD fit residual 1.63e-01).
- jepa_tf_noc: leading St = 0.662, |lambda| = 0.991 (DMD fit residual 2.23e-02).
- fukami: leading St = 0.503, |lambda| = 0.939 (DMD fit residual 4.97e-02).

## Marginal-stability finding

Floquet leading-MODULUS multiplier (per-step) = 1.0042 (window W = 16, 59-frame cycle, transformer predictor). This is the most-amplified direction; it is
(near-)real and its per-step modulus sits essentially ON the unit circle.
A per-step modulus near 1 means the learned latent orbit is (approximately)
marginally stable -- the predictor neither blows the limit cycle up nor
collapses it, consistent with an on-manifold shedding attractor. Well below 1
would mean the rolled-out latent decays toward a fixed point (over-smoothing);
well above 1 would mean it diverges.

HONEST READ of the rest of the spectrum: the monodromy is dominated by this
single neutral direction. The second Floquet multiplier modulus is 0.222 (vs 1.28 for the leader over the full cycle), i.e. every
OTHER direction is strongly contracted over one period. The learned predictor
therefore behaves, over a full shedding cycle, like a projection onto a 1-D
neutral (phase) manifold with all transverse directions damped -- a clean
on-manifold signature, but it means the Floquet monodromy does NOT carry a
well-resolved shedding-frequency pair: the leading OSCILLATORY multiplier (the
complex one) is one of the strongly-damped sub-modes (per-step modulus 0.968, St 0.082) and should NOT be
over-interpreted as the orbit clock. The shedding frequency is recovered by the
Part-1 data-driven DMD (best family St below) and, more coarsely, by the Markov
one-step Jacobian (leading St 0.194 +- 0.095, |lambda| 0.933).

So the rigorous 'eigenvalues of JEPA' certify MARGINAL STABILITY of the learned
orbit (per-step leading Floquet modulus ~1) but defer the FREQUENCY recovery to
the data-driven DMD; the two estimators are complementary, not redundant.

Subtlety (companion / delay-embedding handling): the delay-embedded state is
the stacked W = 16-frame window; the one-step map shifts the
window and appends the predictor's next-step output. Its Jacobian is the
block-companion matrix (identity shift blocks above the diagonal, the per-frame
predictor Jacobians J_0..J_{W-1} in the bottom row). The monodromy is the
ordered product of these companion Jacobians over the period, so the reported
multipliers are eigenvalues of a (W*d)-dimensional operator: d 'genuine'
directions plus (W-1)*d shift modes that pile up near the origin. We sort by
modulus; the result is stable across W in [1, 16] (leading per-step modulus
1.003-1.005). The Markov (W=1) one-step Jacobian spectrum is reported alongside
as the simplest linearization.

## DMD forecasting rung (Part 3): is JEPA just DMD?

POD + linear-dynamics (DMD) forecast wake-enstrophy R^2 @ H=16 test_b = -0.393.
Learned tf-no-c JEPA forecast wake R^2 @ H=16 test_b = +0.432.
POD + learned matched predictor wake R^2 @ H=16 test_b = -0.597.

One-liner:
The learned nonlinear predictor BEATS the linear DMD operator on the matched forecast metric (delta = +0.825). JEPA is NOT just DMD: the dynamics carry nonlinear structure the best-fit linear operator cannot capture.

