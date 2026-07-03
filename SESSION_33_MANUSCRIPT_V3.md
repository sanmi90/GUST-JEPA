# SESSION 33: Manuscript v3 revision guide and drop-in draft prose

**EXECUTION STATUS (2026-07-03, after the Session 33 overnight campaign).** Every [RUN] and
[RE-RUN] item in Section 11 is EXECUTED and committed; the numbers are FROZEN (299 values,
14 report anchors PASS, paper/macros_v3.tex, HANDOFF D243). Verdicts and prose flags live
in HANDOFF D240-D243; the load-bearing ones: Gate T2 STRONG (K1xW30 = K8xW1 static
recovery) but Gate T2b FAIL (the frozen filter needs its 8 taps -- harness verified
bit-exact at K8, so the trade is a property of the static delay-coordinate reconstruction,
not of the frozen-tuned filter; D-T2's main-text figure becomes the honest filter-vs-K
panel); the DMD dichotomy softens (regAE/POD also keep the shedding clock; only the
Fukami-AE families lose it); Y is now READABLE (0.53 KRR test_b, was -0.03 in v2.1); the
single-seed merit tie does NOT replicate across seeds (supervised_only collapses on 1/3
seeds; the predictive objective supplies seed-robust forecastability -- STRENGTHENS the
division of labour). Remaining: Phase 5 figures (drafts of the two Track T figures are in
outputs/session33/figures/) and the Phase 6 LaTeX restructure (L0-L9).

**What this is.** The standing v3 planning document, merged 2026-07-02 (Session 33) from two
inputs: the v3 revision guide (section-by-section transformation of the v2.1 manuscript onto
the executed Session 32 results) and the delay-coordinate estimability addendum (Takens
layer: theory, drop-in prose, Tables T1-T3, Track T). Every number below is from
`outputs/session32/` as reported in the Session 32 report; every v2.1 physics result that
must be recomputed on v2.2 is flagged **[RE-RUN]**; every new Track T quantity is flagged
**[RUN]**. Prose blocks marked **DRAFT** are written to JFM house style (British spelling,
no em-dashes, physics before method, honest hedging) and are meant to be pasted into the
LaTeX with numbers wired through `numbers.json` -> macros.

**The one-sentence change.** The paper stops asking "which reduced state reads the wake" and
asks "which reduced coefficient state can a sequential wall-pressure estimator track through
the encounter", answers it with a division of labour (supervision supplies readability and a
protected latent geometry; the multi-step predictive objective supplies rollout stability),
and demonstrates it with a leakage-free filter that extends the usable gust envelope beyond
single-frame recovery.

**The one-sentence addition (delay-coordinate layer).** No single pressure frame fixes the
coefficient state (the wall is instantaneously blind to the wake circulation), but the
encounter is low-dimensional, so by the delay-embedding theorem a few wall-pressure taps read
over a short delay window reconstruct the state up to diffeomorphism; the sequential filter
is the noise-robust form of that reconstruction, the sensor count and delay window trade
against each other, and the envelope boundary is where the strengthening gust raises the
effective dimension beyond what a fixed sensing budget can embed.

---

## 0. The reframing, and why the results force it

Three facts from Session 32 drive the restructure.

1. **Attribution is C2.** On the pooled tier the objective-free supervised encoder
(`supervised_only`) matches the full predictive latent (`jepa_wake`) on every observable's
readability (all five within |Delta| < 0.05; wake-enstrophy Delta = +0.026 with a CI
including zero) and ties it on matched-predictor merit (0.637 vs 0.639). The predictive
objective does not add instantaneous readability or fitted-predictor merit on this tier.
This is not a weakness to hide; it is a cleaner mechanistic result than v2.1 claimed, and it
relocates the predictive objective's value precisely.

2. **The predictive objective's value is dynamical and long-horizon.** The multi-step rollout
objective (H_roll = 8 vs 1, isolated cleanly) lifts C_L forecast closure 0.429 -> 0.691 and
reduces on-manifold drift 0.731 -> 0.618 at horizon 16, with the merit-versus-horizon curves
tied at one step (~0.54 both) and separating by sixteen steps (single-step collapses to
0.046, multi-step holds 0.312). Supervision alone (`supervised_only`) is the worst field
forecaster (VRMSE 0.992) despite its readability tie. So the predictive/multi-step objective
buys forward usability, exactly the on-manifold story, now measured against the right
controls.

3. **The mechanism is supervision-induced anisotropy.** Rolling each frozen latent forward
and projecting the departure onto the bottom-quartile (near-null) eigenvectors of the
training covariance: `regAE` (anti-collapse, no supervision) is near-isotropic (condition
number 11) and leaks (near-null fraction 0.100); `jepa_wake` and `supervised_only` (both
supervised) are anisotropic (613, 1041) and stay in distribution (0.010, 0.029). The v2.1
mechanism credited the anti-collapse regulariser; the pooled result shows anti-collapse alone
is not enough, supervision is what pins variance into a low-rank subspace the rollout
respects. LuMamba reports the same isotropy-versus-structure behaviour for SIGReg/LeJEPA in
EEG, giving a cross-domain corroboration.

Two facts set the new headline and the honesty band.

4. **The envelope is the new central result.** On all 450 encounters the filter tracks lift
for every |G| >= 1 (analysis R^2 0.71 to 0.90) and extends the usable range beyond static
single-frame recovery, which becomes worse than the mean by |G| ~ 3 (R^2 -1.22 at |G|=3,
-0.33 at |G|=4) while the filter still tracks (0.69, 0.90). Open-loop forecast never works,
so the correction step is essential. This is the working realisation of the v2.1 Section 8
speculative pathway (static lead-time pressure -> C_L was R^2 = 0.35).

5. **The honesty band is estimability and calibration.** The pooled state's
wall-recoverability margin over a raw high-dimensional field latent is real but modest
(Delta R^2 = +0.120, CI [0.096, 0.145], below the pre-registered 0.2 bar), and the filter is
under-dispersed at the strongest and widest gusts (innovations not white, |lag-1| ~ 0.88;
mean NIS rises 0.9 -> 19.4 with |G|). The defensible strong claims are the physics-recovery
ordering and the envelope; the strong-form estimability gate and a calibrated filter are not
claimed.

The paper is therefore still a representation-design paper, but the design criterion is
estimability-and-forward-usability, and the filter is the demonstration that makes the
criterion operational and yields the envelope. The joint-embedding predictive architecture
remains the instrument, not the subject. The delay-embedding theorem (Section 5.4 below) is
the theoretical thread that ties the estimation contribution together, not a fourth
contribution.

---

## 1. Title

v2.1: "Designing wake-supervised latent states for predictive modelling of vortex-gust
airfoil interactions."

**RESOLVED (D238, 2026-07-02): T1** -- "Wake-supervised coefficient states for wall-pressure
estimation of extreme vortex-gust airfoil encounters". T1 puts the physics object (the
state), the sensing modality (wall pressure), and the flow up front, which is the lineage
pattern (compare Fukami and Taira titles).

Key words (JFM controlled list): keep "vortex shedding, low-dimensional models, machine
learning"; consider swapping one for a sensing/estimation term if the list allows.

---

## 2. Abstract (DRAFT, drop-in)

> Extreme vortex-gust encounters of a post-stall airfoil are wake-reorganisation events: the
transient load is the surface signature of leading-edge-vortex roll-up and shedding, and the
release parameters alone do not fix the wake once the gust has passed. A reduced-order model
of such an encounter is useful in deployment only if its state can be recovered from the
sensors that are actually available, the wall pressure, and advanced by a forward model
without leaving the region on which that recovery was calibrated. Using direct numerical
simulations of a NACA 0012 airfoil at alpha = 14 degrees and Re = 5000, perturbed by Taylor
vortices spanning gust strength, core diameter and wall-normal offset, we ask which reduced
coefficient state a sequential wall-pressure estimator can track through the encounter. At
fixed latent dimension and under one shared probe and predictor, we compare a wake-supervised
predictive state, a lift-augmented reconstructive autoencoder in the Fukami-Taira lineage,
and a proper orthogonal decomposition basis; matched controls separate the training objective
from the observable supervision and the anti-collapse geometry. We find a division of labour.
Observable supervision on the wake supplies both the instantaneous readability of the wake
and an anisotropic latent geometry whose principal directions the forward rollout respects,
so a supervised state stays in distribution under prediction where an anti-collapse
regulariser applied to a bare reconstruction does not. The multi-step predictive objective
supplies long-horizon rollout stability rather than readability, which a matched single-step
objective and an objective-free supervised encoder both lack. On the resulting coefficient
state we build a leakage-free ensemble Kalman filter that senses eight span-averaged
wall-pressure taps and never observes the vorticity field: it tracks the lift through the
encounter (analysis R-squared of 0.71 to 0.90 for gust ratios of unity and above) and extends
the usable gust-intensity envelope beyond static single-frame recovery, which becomes worse
than predicting the mean near a gust ratio of three. Because the encounter is
low-dimensional, the state is recoverable from a few taps read over a short delay window
rather than from many simultaneous sensors, a spatial-for-temporal trade we quantify and read
through the delay-embedding theorem, which frames the envelope boundary as the gust strength
at which the interaction becomes three-dimensional and its rising effective dimension outruns
a fixed sensing budget. The predictive coefficient state carries the most wall-recoverable
physics of the three families, although the reconstructive state is the most recoverable in
raw variance, a distinction that separates energy from information at the wall. We are
explicit about the limits: the state's recoverability advantage over a raw high-dimensional
field latent is real but modest, and the filter is under-dispersed at the strongest and
widest gusts, where a calibrated process and observation noise model, rather than covariance
inflation, is the outstanding requirement.

**Note.** The v2.1 abstract's flagship number (R^2 = 0.79 wake readability) is not the
headline any more; readability is stated as supervision-supplied, and the numbers that lead
are the filter's C_L tracking and the envelope. This is deliberate and matches the honest
attribution. The delay-embedding sentence sits after the filter sentence, per the addendum.

---

## 3. Introduction (DRAFT of the changed parts)

Keep v2.1 paragraphs 1-4 (the wake-reorganisation opening, the ROM ladder from POD/DMD
through observable-augmented autoencoders, the diffeomorphism-freedom argument, the JEPA
description) essentially intact; they are excellent and correctly framed. Make four
insertions and one rewrite.

**Insertion A, after the ROM-ladder paragraph (the "forward-state usability" rung).** Add the
estimation motivation, which is the paper's new reason to exist and is already on record in
the lineage:

> The rung beyond forward usability is deployment: at flight time the vorticity field is
never measured, only the surface pressure, so a reduced state earns its place only if it can
be recovered sequentially from sparse wall sensing and advanced without leaving the region on
which that recovery was fitted. This is the pathway the extreme-aerodynamic control
literature has named but not closed. Fukami, Nakao and Taira (2024) end their gust-mitigation
study by noting that estimating the low-dimensional latent from pressure sensors is what a
real-time controller would require, and Fukami and Taira (2025) frame state estimation as the
manifold's purpose; the disturbed-aerodynamics estimation front has meanwhile been opened on
the Bayesian side with sparse-pressure flow reconstruction and uncertainty quantification
(Mousavi and Eldredge 2025; Eldredge and Mousavi 2025). We take the representation question
that sits underneath all of this: holding the estimator fixed, which learned reduced state
can a sequential wall-pressure filter actually track through an extreme encounter, and which
training ingredient supplies which of the properties such a state must have.

**Insertion B, in the JEPA paragraph.** Note that pooled predictive latents evaluated by
frozen linear probing are now an established practice, which supports both the latent form
and the protocol:

> Pooled predictive latents evaluated by frozen linear probing have since been used for
procedural video, where a masked latent-prediction objective reorganises a pooled feature
stream so that a linear probe recovers structure the raw features do not expose (Tristram et
al. 2026); the same anti-collapse regulariser we adopt has been carried to biosignal time
series, where it is reported to drive embeddings toward isotropy so that a structuring
objective is needed alongside it for useful geometry (Broustail et al. 2026). Both
observations recur below in physical form.

**Insertion C** (companion-paper division of labour): keep the v2.1 sentence on Solera-Rico
et al. (under review), unchanged in spirit.

**Rewrite of the three contributions.** Replace the v2.1 contribution list with:

> We make three contributions. First, we pose the vortex-gust encounter as a predictive-state
estimation problem and fix its endpoints in advance: the primary endpoint is the sequential
recovery and forecast of the transient load and the wake enstrophy from sparse wall pressure
across the impact and relaxation windows, with a model-free parameter floor and a
single-frame recovery baseline that bound what the state must beat.
>
> Second, we separate the ingredients of an estimable, forward-usable state with matched
controls, and reach a division of labour. Observable supervision on the wake supplies both
the instantaneous readability of the wake and an anisotropic latent geometry, a low-rank
subspace into which the encoded variance is concentrated and which the forward rollout
respects; an anti-collapse regulariser applied to a bare reconstruction produces instead a
near-isotropic latent with no protected subspace, and its rollout leaks into low-variance
directions. The multi-step predictive objective supplies long-horizon rollout stability,
which neither a matched single-step objective nor an objective-free supervised encoder
provides. We establish these on a pooled coefficient latent at matched dimension, with the
reconstructive and linear recipes as references.
>
> Third, we demonstrate the state in a leakage-free sequential filter that senses only wall
pressure, and we read the demonstration through delay-coordinate embedding. Because the
encounter is low-dimensional, a few taps over a short window place the state, the wake
circulation the instantaneous pressure cannot see is recovered through the dynamics, and the
tap count trades against the window length; the filter tracks the load and extends the usable
gust-intensity range beyond single-frame recovery, and the delay-embedding view identifies
the envelope boundary as the point at which the strengthening gust raises the effective
dimension beyond what a fixed sensing budget can embed. We characterise, rather than hide,
where it degrades: the recoverability margin over a raw field latent is modest, and the
filter is under-dispersed at the strongest gusts.

(The third contribution is the addendum's extended form, which supersedes the earlier draft;
the honesty clause from the earlier draft is retained as the closing sentence.)

**Keep** the v2.1 closing framing paragraph ("The question this paper addresses is therefore
not which encoder reconstructs the flow best...") but retarget its final clause to
estimability and forward usability rather than single-instant readability.

---

## 4. Section 2, flow configuration and data (change map)

- **Split.** Replace the v2.1 counts with v2.2: 102 cases, 450 encounters (84 train / 10
  test_b / 8 test_c / 100 validation), with a **symmetric |G| = 4 Test C** (four periodic
  plus four run4). This is a genuine improvement over v2.1's four-case one-sided boundary and
  should be stated as such.

- **New data-provenance note (DRAFT), one short paragraph in Section 2.2 or an appendix:**

> The extrapolation boundary at |G| = 4 is sampled symmetrically by combining the periodic
archive cases with four additionally released cases. We verified the additional cases against
the archive by a physical consistency check on the load rather than by trusting the file
metadata: the two signs of the |G| = 4 gust produce near-mirror-image lift transients, one a
positive-first excursion peaking near +8.3 and the other troughing near -7.6, as the sign of
the gust ratio sets the sign of the first excursion (Section 2.1), which confirms that the
two halves of the symmetric boundary set are physically consistent and that the
sign-asymmetry analyses below are sound.

  (The solver sign convention s = -G stays as the appendix data note it already is in v2.1.)

- **Table 1** (DNS author-fill checklist): unchanged, still owned by the simulation
  collaborators.

- **Normalisation constants** used by the metrics: `train_std = 3.5396`, SSIM data range
  L = 8.487 on v2.2. State these where the normalisation pipeline is described (they anchor
  the VRMSE and SSIM definitions).

- **Observables** (five: C_L, C_D, E_w, Gamma+, Gamma-): unchanged definitions. The wake
  enstrophy remains the pre-registered primary wake endpoint; C_L becomes co-primary because
  it is what the filter tracks and what the envelope is stated in (D238, resolving D-primary).

---

## 5. Section 3, model, loss, and the estimator

### 5.1 The latent state and the loss (rewrite + new anatomy paragraph)

Keep the encoder/predictor description, but present the latent honestly as a **pooled
coefficient state** (class-token projection to R^d), which is v2.1's own choice, and headline
**d = 32** (Session 32's operating dimension), with the dimension plateau reported as
robustness **[RE-RUN the {16, 32, 64} pooled plateau on v2.2]**. Motivate d = 32 by
estimability: a 32-dimensional state admits an ensemble (and effectively exact) filter
cheaply and is orders of magnitude cheaper to advance than the field, which is the operative
reason for order reduction here.

**New loss-anatomy paragraph (DRAFT), replacing the apologetic "auxiliary heads with small
weights" framing:**

> The encoder, predictor and auxiliary heads are trained jointly, and it is worth being
explicit about what the combined objective is, because its structure predicts the attribution
results below. Writing the terms with their roles, the loss couples a transition term over
trajectories, the multi-step latent rollout, with a chart regulariser that fixes the gauge
freedom a decoder-free latent would otherwise retain, and an emission term that reads the
observables from each instantaneous embedding,
>
> L = sum_k || z_hat(a+k) - sg(z(a+k)) ||^2  +  lambda_S SIGReg({z})  +  lambda_l
|| g_l(z_t) - C_L(t) ||^2  +  (lambda_w / 80) || g_w(z_t) - s_w(t) ||^2 ,
>
> with a stop-gradient on the prediction targets. This is a jointly trained state-space
realisation: a transition map trained on sequences, an emission map trained per frame, and an
anti-collapse regulariser in place of a decoder. A dynamics equation is a statement about
trajectories and an observation equation is a statement about single instants, so the mixed
timescales are the structure of an observer, not an inconsistency, and they mirror the
predict-correct filter of Section 3.3. The anatomy also predicts the attribution: with the
targets detached, the only terms that shape the per-frame embedding geometry are the chart
regulariser and the emission heads, so instantaneous readability is supervision-supplied by
construction, and the transition term can express itself only in what a rollout does, which
is exactly where Section 4.2 finds it.

Then, the anti-collapse choice (SIGReg, the characteristic-function/Epps-Pulley regulariser)
keeps its v2.1 Appendix A treatment; pin the weight and cite the from-pixels line plus
LuMamba as an independent adoption. **lambda_S = 0.02, RESOLVED** (commit dda57b7; the v2.1
text and the reorganisation notes disagreed between 0.01 and 0.02; 0.02 is the value used in
every Session 31/32 canonical and pooled run).

**Heads held identical across families** (current-frame emission, the one shape every family
can share): keep this as the controlled-comparison spine. Note explicitly that v2.1's tuned
reconstructive baseline used future-lift heads while JEPA used current-frame heads; v3 pins
current-frame heads for all families and says so, which removes an accidental asymmetry.

### 5.2 Matched-predictor protocol (keep, update numbers)

Unchanged in principle: one shared predictor family fit on each frozen latent, one shared
frozen probe family (Ridge and small MLP), latent dimension matched. Report at d = 32 pooled;
note the matched predictor is the transformer (the ResUNet of Session 31 is a spatial-tier
object and does not apply to a vector state).

### 5.3 The predict-correct estimator (NEW section, DRAFT)

> At deployment the vorticity field is unavailable and the wall pressure is not, by itself,
enough to fix the state: a single pressure frame recovers little of the wake through any of
the latents (Section 4.3). The estimator must therefore be sequential, accumulating pressure
innovations through the learned dynamics, which is the setting of data assimilation. We use
an ensemble Kalman filter (Evensen 1994, 2003) in the latent space. An ensemble of N = 64
members is advanced one step at a time by the frozen predictor, the forecast step, and
corrected against K = 8 span-averaged wall-pressure taps through a frozen observation
operator h : z -> p_K, the analysis step; a deterministic square-root variant is available
for the strongly non-Gaussian innovations at impact. The observation operator maps the latent
to pressure only. The physical observables the paper cares about, the lift and the wake
enstrophy, are read from the state by the same frozen probes used throughout and can never
enter the innovation; this leakage guarantee is enforced by construction and by a unit test,
so the reported analysis closure of the load and the wake is a genuine estimate, not a
quantity the filter was given.
>
> The taps are placed by a target-blind, then a model-conditioned, criterion; the
model-conditioned placement (which selects, per family, the taps that best expose that
family's own latent) improves the analysis lift closure over a shared target-blind array by
36 per cent in root-mean-square error at impact, and the placements are strongly
family-specific, sharing at most a leading-edge tap. The filter is initialised, field-free,
from a windowed pressure-to-latent regressor, and its ensemble spread from that regressor's
residual covariance, so no encoded truth is used at inference. Process noise is calibrated
from the predictor's one-step residuals; a multiplicative inflation was selected on the
validation split alone and then frozen, and it selected no inflation (the filter is
under-dispersed rather than over-dispersed, mean normalised innovation squared near 5 against
K = 8, so added inflation would hurt). All tuning is frozen before any test encounter is
seen.

Baselines, all read through the same probes: open-loop rollout (no correction), a
sliding-window pressure-only regression (no dynamics), persistence, and a model-free
parameter floor (the gust parameters given, diagnostic only, the model never sees them).

**Delay-embedding insert (DRAFT, from the addendum; placed after the "estimator must be
sequential" sentence above):**

> The reason a single pressure frame is not enough, and a window is, is a statement about the
dynamics rather than about the number of sensors. The wall pressure is a low-dimensional
observation of a low-dimensional state: the encounter lies on an attractor of effective
dimension of order a few, a limit cycle with a gust excursion and a relaxation, into which
the encoded latent concentrates its variance (Section 4.6). By the delay-embedding theorem
(Takens 1981; Sauer, Yorke and Casdagli 1991), a generic observation of such a system, read
over a window of delays, reconstructs its state up to a diffeomorphism, and with K
simultaneous taps over m delays the reconstruction is generic once the product mK exceeds
twice the attractor dimension. Because that dimension is small, a handful of taps over a
modest window suffices, and the delay window can substitute for sensor count, a trade we
quantify in Section 4.3. A coordinate the instantaneous pressure misses, the wake
circulation, is still reconstructed by the window provided it is dynamically coupled to what
the pressure sees, which for the coupled vortex dynamics it is; instantaneous blindness is
therefore recoverable through time, and the sequential filter is the noise-robust, recursive
form of the delay-coordinate reconstruction, integrating the pressure history through the
learned dynamics rather than inverting a fixed delay map. This also fixes where the
reconstruction must fail: as the gust strengthens and the interaction becomes
three-dimensional (Section 2.1), the effective dimension grows and a fixed sensing budget
eventually ceases to embed the state, which is the reading of the operating-envelope boundary
we return to in Section 4.5.

### 5.4 The delay-embedding thread (theory, stated for this problem; from the addendum)

The reduced state z lives on the encounter attractor, which is low-dimensional: the
undisturbed flow is a limit cycle, the gust adds an excursion and a relaxation, and the
encoded latent concentrates its variance into a handful of directions (participation ratio
near two at d = 64 in v2.1; six components span ninety per cent), so the attractor's
effective dimension d_eff is of order a few, not the ambient thirty-two. The wall pressure is
a low-dimensional observation of this state: Session 32 shows it reads the force and the wake
enstrophy but is instantaneously blind to the wake-circulation direction (the near-null
direction of the encoded covariance, O2).

The delay-embedding theorem (Takens 1981; Sauer, Yorke and Casdagli 1991) states that a
generic observation function of a dynamical system, sampled over a window of delays,
reconstructs the state on the attractor up to a diffeomorphism. In the multivariate form
relevant here, K simultaneous observables read over m delays give a delay-coordinate map into
R^(mK) that is generically an embedding once

    mK > 2 d_eff        (sufficient, generic; box-counting dimension, prevalence sense).

Two consequences organise the estimation results:

1. **Few sensors suffice, and delays substitute for sensors.** Because d_eff is small, the
   product mK need only clear a small bound, so a handful of taps read over a modest window
   carries enough information to place the state. Increasing the delay window m lets K fall
   and vice versa: a spatial-for-temporal trade. This is the "require fewer sensors" lever,
   and it is quantifiable (Track T).

2. **The hidden coordinate is recoverable through time.** A coordinate the instantaneous
   observation misses (the wake circulation) is still reconstructed by the delay map provided
   it is dynamically coupled to what the observation sees, which for the coupled vortex
   dynamics it is. Instantaneous blindness is therefore not un-recoverability; it is the
   statement that the coordinate must be read through the dynamics, which is what a
   sequential filter does.

The filter is the recursive, noise-robust form of this reconstruction: sequential Bayesian
filtering on a Markov latent integrates the pressure history through the learned dynamics
rather than inverting a fixed delay map, so it needs no explicit delay-coordinate observation
operator, the sequentiality already carries the delays. The static delay-coordinate map is
then the characterisation that quantifies how many sensors and delays are needed and explains
why the filter behaves as it does.

The same reading bounds the failure. As the gust strengthens the interaction becomes
three-dimensional (Section 2.1: the out-of-plane enstrophy fraction rises about threefold by
|G| = 4), so d_eff grows with |G|, and a fixed (K, m) eventually fails to clear the bound and
ceases to embed the state. That is the delay-embedding reading of the operating-envelope
boundary: the filter diverges at strong gusts not because the dynamics are unlearnable but
because a fixed sensing budget stops embedding an attractor whose dimension has grown.

Read through delay coordinates the encounter also has a familiar structure, an intermittently
forced quasi-periodic system (Brunton et al. 2017): the baseline shedding is the
delay-coordinate dynamics and the vortex impact is a forcing event whose amplitude is the
gust ratio, which is consistent with the envelope being organised by |G|. This is the
sensing-side counterpart of the multi-time-delay latent predictor introduced for controlled
wakes in our companion study (Solera-Rico et al. 2025): there the latent is advanced from a
window of its own past, here it is estimated from a window of wall pressure, and both rest on
the same low-dimensional delay structure.

What this thread explains that we already have, at no new computational cost beyond stating
it: **O2 wall-blindness** (an instantaneous observability gap the delay map closes if the
circulation is dynamically coupled to the observed directions; turns a limitation into a
motivation for sequential estimation); **the envelope** (a single frame is not an embedding,
a delay window is; the filter succeeds precisely because it is a delay-coordinate estimator);
**the calibration boundary** (d_eff grows with |G| as the flow becomes three-dimensional, so
the fixed sensing budget stops embedding; the calibration limit and the |G| = 4
three-dimensional observability boundary are the same phenomenon seen from the filter and
from the field).

---

## 6. Section 4, results (full restructure)

New order: 4.1 what the state carries and who supplies it; 4.2 why it stays usable under
rollout; 4.3 what the wall can see (including 4.3.1 sensors traded for delays); 4.4 tracking
the encounter; 4.5 the operating envelope; 4.6 physics of the latent state. The
spatial-versus-pooled decodability trade goes to Discussion + appendix.

### 6.1 Section 4.1, what the coefficient state carries (DRAFT + table)

> The comparison is meaningful only if the wake is a discriminating endpoint, and it is: at
the readout horizon the gust parameters no longer fix the wake enstrophy (the model-free
floor is negative there), while they still fix the forces, so the wake is where
representations that captured the gust-vortex physics separate from those that captured only
its force signature. Table X reports the held-out closure of the pooled coefficient states at
matched dimension. The wake enstrophy is read cleanly by every state that carries the wake
head (R-squared of 0.75 to 0.79) and collapses for every state that does not (0.16 to 0.46),
so the instantaneous readability of the wake is supplied by the observable supervision, not
by any objective. The objective-free supervised encoder reads the wake at least as well as
the full predictive latent (0.792 against 0.766) and ties it on the matched-predictor
forecast merit (0.637 against 0.639), so on this tier we do not attribute the readability or
the fitted-predictor merit to the predictive objective over the reconstructive one. We state
the representational result as the supervision's, and turn to what the predictive objective
does add, which is dynamical.

Table X (pooled d = 32, test_b, v2.2; controlled matrix then references):

| state | objective | wake head | wake-enst. R^2 | merit (5-obs, h8) | field VRMSE (h8) | decode SSIM |
|---|---|---|---|---|---|---|
| jepa_wake | predictive (multi-step) | yes | 0.766 | 0.639 | 0.969 | 0.778 |
| supervised_only | none | yes | 0.792 | 0.637 | 0.992 | 0.774 |
| ae_wake | reconstruction | yes | 0.750 | 0.548 | 0.976 | 0.776 |
| jepa_nowake | predictive (multi-step) | no | 0.160 | 0.471 | 0.983 | 0.767 |
| ae_nowake | reconstruction | no | 0.456 | 0.432 | 0.976 | 0.773 |
| regAE | reconstruction | no | 0.333 | 0.197 | 0.949 | 0.782 |
| beta-VAE | reconstruction (KL) | yes | 0.728 | 0.356 | 1.007 | 0.724 |
| Fukami | reconstruction (lineage) | no | -0.094 (measured v2.2) | -0.20 | 1.016 | 0.708 |
| Fukami+wake | reconstruction (lineage) | yes | +0.432 (measured v2.2) | 0.203 | 1.026 | 0.705 |
| POD | linear | none | +0.186 (measured v2.2) | 0.29 | [RE-RUN] | 0.775 |

Gate P1 (readability attribution): PASS, supervised_only >= jepa_wake on all five
observables, |Delta| < 0.05, wake Delta = +0.026 with CI including zero. Gate P2 (merit
ordering): PASS, 0.639 >= 0.637 > 0.548 > 0.197. The POD and Fukami wake-readability cells
are ALREADY MEASURED on v2.2 (`outputs/session31/q1_reference.json`): POD +0.186,
fukami -0.094, fukami_wake +0.432. NOTE the POD sign flip vs v2.1 (-0.16): the prose follows
the measured value; "collapses relative to the wake-headed states" still holds (0.19 << 0.75)
but POD is not "negative" on v2.2. The remaining [RE-RUN] is POD field VRMSE at pooled d=32
if it is kept as a table column.

Note the instructive inversion worth one sentence: `jepa_nowake` has the highest
single-observable C_L forecast closure (0.691) yet the lowest wake readability (0.160),
because the forces are carried redundantly across the latent while the wake is not, which is
the coordinate-level content of the distributed-code result (Section 4.6).

### 6.2 Section 4.2, why the state stays usable under rollout (DRAFT + table)

> A state that reads the wake at a single instant need not survive a rollout, because a probe
attached to a drifted latent is queried outside its support. We ask where each rollout goes
relative to the training distribution by decomposing its departure onto the principal and the
near-null directions of the encoded covariance (Table Y). The reconstruction with only an
anti-collapse term is near-isotropic, its covariance condition number is near unity, and its
rollout leaks a tenth of its departure into the near-null directions; the supervised states,
predictive and objective-free alike, are strongly anisotropic, their variance concentrated in
a low-rank subspace, and their rollouts keep their departures in that subspace (a hundredth
and three-hundredths, against a quarter for an isotropic null). Supervision, not the
anti-collapse regulariser alone, is what builds the protected subspace: it pins the encoded
variance onto the observable-aligned directions, and the forward model, having only ever seen
states in that subspace, moves within it. This refines the mechanism of our earlier report,
which credited the anti-collapse term; the anti-collapse term alone drives the latent toward
isotropy, exactly as an independent adaptation of the same regulariser to biosignals reports
(Broustail et al. 2026), and it is the supervision that supplies the geometry the rollout
needs.

Table Y (pooled d = 32, test_b, rollout departure at H = 16):

| state | latent covariance condition number | near-null departure fraction (k = 8) |
|---|---|---|
| regAE | 11 (near-isotropic) | 0.100 |
| jepa_wake | 613 (anisotropic) | 0.010 |
| supervised_only | 1041 (anisotropic) | 0.029 |

Isotropic null baseline = k/d = 0.25. Gate P3: PASS under both a minimally-regularised and a
shrinkage covariance estimator, CIs excluding zero (jepa_wake Delta = +0.090 vs regAE, CI
[0.081, 0.099]; supervised_only Delta = +0.072, CI [0.062, 0.080]); as in v2.1, the robust
statement is the departure direction, not the ratio magnitude.

> What the predictive objective does add is long-horizon stability, and it is the multi-step
form of the objective that adds it. Training the same predictive latent with a single-step
rather than a multi-step rollout objective leaves the two tied one step out and separates
them under rollout: by sixteen steps the single-step latent's wake-forecast skill collapses
to near zero while the multi-step latent holds a third, its lift forecast closure is higher
by 0.26, and its rollout drift is lower (Table Z). The objective-free supervised state, which
has the readability and the protected subspace but no predictive training, is the worst field
forecaster of the supervised group. The predictive objective's contribution is therefore
forward, not instantaneous: it keeps the rolled state both in distribution and close to the
true trajectory, which is what a sequential estimator queries.

Table Z (H_roll ablation, pooled, test_b, h8):

| axis (lower VRMSE/drift better) | H_roll = 1 | H_roll = 8 | Delta(8 - 1) |
|---|---|---|---|
| observable merit (mean 5-obs) | 0.407 | 0.471 | +0.064 |
| C_L forecast closure | 0.429 | 0.691 | +0.262 |
| wake-enstrophy closure | 0.284 | 0.305 | +0.021 |
| field VRMSE | 1.006 | 0.983 | -0.024 |
| on-manifold drift (h16) | 0.731 | 0.618 | -0.113 |

Caveat to keep (JFM honesty): this isolates the objective's effect on the frozen
representation via matched downstream operators; native co-trained predictors would likely
show a larger gap, which we do not claim here.

### 6.3 Section 4.3, what the wall can see (DRAFT + table)

> At deployment the state must be read from the wall, so we ask how much of each latent a
sparse set of pressure taps recovers, over a window ending at the readout instant and, as a
stricter test, over a window that ends before the gust reaches the leading edge. Two findings
matter (Table W). First, raw-variance recoverability and physics recoverability disagree: the
reconstruction latent is the most recoverable in state variance (0.921) yet carries the least
wall-recoverable physics (observable R-squared 0.470), while the predictive latent carries
the most (0.637). The wall aligns with many of the reconstruction latent's directions, but
they are largely the low-variance ones that do not carry the observables; it aligns with
fewer of the predictive latent's directions, but those are the ones that carry the state.
Second, a direct visibility analysis of the predictive state, perturbing it along its
principal directions and measuring the signal that reaches the pressure head, shows the wall
sees the force and wake-enstrophy directions and is blind to the highest-variance
wake-circulation direction; this ranking is unchanged under a fixed-amplitude perturbation,
so it is a property of the learned dynamics and not of the normalisation. The wall, in short,
reads the force well and the wake circulation not at all, which is precisely why the state
must be estimated sequentially rather than inverted from one frame.

Table W (recovery at K = 8, test_b, v2.2, per-family taps and CV-selected estimator):

| state | recovery estimator | state recovery R^2 | observable recovery R^2 |
|---|---|---|---|
| predictive (JEPA) | LSTM | 0.707 | 0.637 |
| reconstructive (Fukami) | MLP | 0.921 | 0.470 |
| linear (POD) | MLP | 0.561 | 0.534 |

Gate O (estimability of the coefficient state over a raw field latent): **WEAK**. Report it
honestly: the pooled state's advantage over a flattened field latent is Delta R^2 = +0.120
(CI [0.096, 0.145]), below the pre-registered 0.2 bar, so we do not claim a strong-form
estimability gate; the defensible statement is the physics-versus-variance recoverability
ordering above, which is clean.

### 6.3.1 Section 4.3.1, sensors traded for delays (DRAFT + tables, from the addendum)

> The delay-embedding view makes a testable prediction: the state should be recoverable from
fewer taps if they are read over a longer window, and the wake circulation, invisible to the
instantaneous pressure, should become recoverable as the window lengthens. We test both.
Holding the tap count fixed and lengthening the pressure window (Table T1), the recovery of
the wall-blind wake circulation improves from its near-zero single-frame value and saturates
once the window is long enough to clear the embedding bound, while the force, already visible
at one instant, improves little; the delays recover the coordinate the sensors cannot.
Varying the tap count and the window together (Table T2) traces a spatial-for-temporal trade:
recovery is set to first order by the product of tap count and window length, so a short
window with many taps, a long window with few, and intermediate combinations reach the same
recovery, and the encounter is trackable from as few as [K_min] taps once a window of [W_min]
frames is used. The trade is favourable here because the encounter is low-dimensional; a
state that filled its ambient dimension would not admit it.

Table T1 (fixed K = 8, recovery R^2 vs window W; test_b, v2.2) [RUN]:

| observable | W = 1 | W = 4 | W = 8 | W = 16 | W = 30 |
|---|---|---|---|---|---|
| C_L | [.] | [.] | [.] | [.] | [.] |
| wake enstrophy | [.] | [.] | [.] | [.] | [.] |
| wake circulation (wall-blind) | [.] | [.] | [.] | [.] | [.] |
| coefficient state | [.] | [.] | [.] | [.] | [.] |

Table T2 (coefficient-state recovery R^2 over the (K, W) grid; test_b, v2.2) [RUN]:

| K \ W | 1 | 4 | 8 | 16 | 30 |
|---|---|---|---|---|---|
| 1 | [.] | [.] | [.] | [.] | [.] |
| 2 | [.] | [.] | [.] | [.] | [.] |
| 4 | [.] | [.] | [.] | [.] | [.] |
| 8 | [.] | [.] | [.] | [.] | [.] |

> The requirement tracks the theory. Estimating the encounter's effective dimension per gust
stratum (Table T3) and comparing the smallest window that reaches the recovery target against
the delay-embedding bound, the empirical requirement follows twice the effective dimension
divided by the tap count, and the effective dimension rises with gust strength as the
interaction becomes three-dimensional, so the window needed to embed the state grows with
|G|. This is the same growth that closes the operating envelope (Section 4.5): a sensing
budget fixed at deployment embeds the weak-gust attractor and fails to embed the strong-gust
one.

Table T3 (effective dimension vs |G|; and the embedding requirement) [RUN]:

| \|G\| | effective dimension d_eff | out-of-plane enstrophy fraction | window m at K = 8 to meet target | bound 2 d_eff / K |
|---|---|---|---|---|
| 1 | [.] | [.] | [.] | [.] |
| 2 | [.] | [.] | [.] | [.] |
| 3 | [.] | [.] | [.] | [.] |
| 4 | [.] | [.] | [.] | [.] |

### 6.4 Section 4.4, tracking the encounter from the wall (DRAFT, the first hero result)

> We now run the estimator. The filter is initialised from wall pressure alone before impact
and corrected at every frame through the encounter; it never sees the vorticity field. Figure
H shows, for representative held-out encounters, the true lift and wake enstrophy against the
open-loop rollout (no correction) and the filter analysis (with correction). The open-loop
rollout tracks the pre-impact baseline and then loses the encounter, as expected of a
forecast with no measurement; the pressure-only regression tracks the force while its window
is representative and goes stale through the transient; the filter tracks the load through
impact and into relaxation. Across the tracking band the analysis lift closure is 0.71 to
0.90, well above the static single-frame recovery and the open-loop forecast, so the
correction step is doing the work, and it is doing it from pressure that the visibility
analysis showed to be blind to the wake circulation, so the sequential dynamics are carrying
the unobserved part of the state.

Then the honest calibration paragraph, in the same subsection:

> The filter's point tracking is good; its uncertainty is not yet calibrated. The innovations
are temporally correlated rather than white, because the wall pressure is smooth in time, and
at the strongest and widest gusts the normalised innovation grows and the filter passes into
statistical divergence. Divergence here is a statement about the ensemble spread, not about
the tracking: the analysis still follows the load (Section 4.5). We read this as evidence
that the outstanding requirement is a genuine process and observation noise model rather than
the covariance inflation we deliberately did not apply, and we return to it in the
discussion.

### 6.5 Section 4.5, the gust-intensity operating envelope (DRAFT + table, the central new result)

> The value of a sequential estimator over a single-frame inverse is that it should extend
the range of encounters it can handle, and it does. Running the frozen filter, the open-loop
forecast and the static recovery on all encounters, stratified by gust strength (Table V),
the static single-frame recovery of the lift is good for weak and moderate gusts and then
fails, becoming worse than predicting the mean by a gust ratio of three, where the encounter
is most strongly reorganised; the filter, over the same encounters, still tracks the lift
(0.69 at a ratio of three, 0.90 at the four boundary). What degrades monotonically with gust
strength is not the tracking but the calibration: the divergence rate rises from zero to four
in five, and the mean normalised innovation from near unity to nearly twenty, crossing the
half-divergence mark near a ratio of three for the compact cores and near two for the widest.
The envelope is also sign-asymmetric, and the asymmetry is representation-general (the linear
basis reproduces it): the physically positive gusts, which produce the larger and longer-lived
leading-edge vortex, have the wider usable envelope. The limit at a ratio of three lies
inside the training range, so it is an observability and filtering limit of the mid-plane
pressure signal, not a gap in training coverage; the mid-plane representation reaches a
genuine three-dimensional observability boundary only at the four extrapolation, as the field
itself does.

Table V (all encounters, headline predictive state, frozen filter):

| \|G\| | filter C_L R^2 | filter divergence rate | static single-frame recovery C_L R^2 |
|---|---|---|---|
| <= 0.5 | ~ 0 (weak-signal) | 0.00 | +0.60 |
| 1 | +0.71 | 0.00 | +0.63 |
| 1.5 | +0.87 | 0.18 | +0.64 |
| 2 | +0.78 | 0.41 | +0.35 |
| 3 (in-distribution) | +0.69 | 0.75 | -1.22 |
| 4 (test_c boundary) | +0.90 | 0.82 | -0.33 |

The |G| <= 0.5 lift closure is near zero because the true lift is near-constant there (the
R^2 denominator vanishes), not because the filter fails; state this in a footnote as the
Session 32 report does.

**Framing sentence for the paper's contribution:** this table, not the single-instant
readability, is the paper's operational claim. It should be Figure/Table front-and-centre in
Section 4, and the abstract leads with it.

**Delay-embedding addition to the envelope discussion (DRAFT, from the addendum):**

> The envelope has a delay-embedding reading that unifies it with the wall-observability
result and the calibration limit. The filter tracks the load where single-frame recovery
fails because it is a delay-coordinate estimator and a single frame is not an embedding; it
loses calibration as the gust strengthens because the interaction becomes three-dimensional
and the effective dimension grows past what the fixed tap count and delay window can embed
(Section 4.3). The boundary at a gust ratio of three, which lies inside the training range,
is therefore not a gap in training coverage but the point at which the encounter's dimension
outruns a fixed sensing budget, the same three-dimensional observability boundary the
mid-plane representation reaches at the highest gust strength, seen now from the filter.

The reduced-budget filter result (T2b) is reported here or in 4.3.1: the encounter tracked
from [K_min] taps at |G| in {1, 1.5, 2} within CI of the K = 8 filter [RUN]; main-text
figure per D238 (resolving D-T2).

### 6.6 Section 4.6, physics of the latent state (port from v2.1, RE-RUN on v2.2 pooled)

Keep the three v2.1 physics results, recomputed on the pooled v2.2 latents, and keep the
Fukami-coherence framing:

- **Latent spectrum / DMD** [RE-RUN]: v2.1 found the predictive latent recovers the shedding
  Strouhal number (0.66 against a measured 0.68) on a marginally stable orbit while the
  reconstructive latent is damped and off-frequency (0.50). Recompute; expect the same
  qualitative ordering.
- **Parametric manifold atlas** [RE-RUN]: 3D PCA coloured by G, D, Y and phase, per family,
  with the parameter probes recomputed (v2.1 test_b: G 0.83, D 0.65, Y -0.03; the report's
  v2.2 recompute keeps Y as the marginal axis). This is the panel that answers the "does the
  manifold cohere with physical parameters" question in the Fukami and Taira (2023) and Tran
  et al. (2026) tradition.
- **Distributed code** [RE-RUN]: the wake is a collective code (v2.1 full-versus-best-
  coordinate gap 0.36 for the predictive latent against 0.05/0.04 for the baselines), which
  is the coordinate-level reason the forces are carried redundantly while the wake is not,
  and it explains the `jepa_nowake` inversion in Section 4.1. Add the minimum-dimension panel
  (smallest d at which each family reaches R^2 >= 0.5 on the wake) as the d = 32 defence.

---

## 7. Section 5, discussion (DRAFT of the changed parts)

**5.1 What a wall-estimable state must contain (rewrite around the division of labour).**

> The central result is a design rule for a reduced state that a wall-pressure estimator can
track. The state must carry the wake observables that govern the post-impact transient, must
be organised so a forward model can advance it without leaving its own support, and must be
recoverable from the surface pressure that is the only available measurement. These three
requirements are supplied by different ingredients, and separating them is the paper's
contribution. Observable supervision on the wake supplies the first, the instantaneous
readability, and, less obviously, much of the second: it concentrates the encoded variance
into a low-rank subspace aligned with the observables, and the forward rollout, trained and
living in that subspace, stays in distribution, where an anti-collapse regulariser applied to
a bare reconstruction produces an isotropic latent with no protected subspace and a rollout
that leaks. The multi-step predictive objective supplies the rest of the second requirement,
the long-horizon stability that a single-step objective and an objective-free supervised
encoder both lack. The third requirement, wall recoverability, is where energy and
information part company: the reconstruction latent is the most recoverable in variance but
the least in physics, while the predictive latent carries the most wall-recoverable physics,
so the coefficient state a controller would want is not the one a reconstruction objective
produces.

**5.2 The estimation demonstration and its limits (new).**

> The filter turns the design rule into a working estimator: from eight wall-pressure taps
and no field measurement it tracks the load through the encounter and extends the usable
gust-intensity range beyond single-frame recovery, realising the online-estimator pathway the
extreme-aerodynamic control literature has named. We are deliberate about its limits. Its
recoverability advantage over a raw high-dimensional field latent is real but modest, so the
case for the coefficient state rests on the physics-recovery ordering and the envelope, not
on a large recoverability margin. Its point tracking is good but its uncertainty is
under-dispersed at the strongest and widest gusts, where the innovations are temporally
correlated and the ensemble collapses; the visibility analysis suggests a mechanism, the
forward model has large gain along the data-unconstrained near-null directions it never
learned to damp, and those inject spread-consuming noise during assimilation, which points to
a calibrated process and observation noise model, rather than covariance inflation, as the
outstanding requirement. We report the filter as a feasibility demonstration with a
characterised soft spot, not as a deployed estimator.

**Delay-embedding addition to 5.2 (DRAFT, from the addendum):**

> The delay-embedding view also names the deployment knob and its limit. Because the
encounter is low-dimensional, the state can be tracked from few taps read over a short
window, which is the sensing budget a small vehicle could carry; the trade between tap count
and window length is explicit (Section 4.3), so a sensor-poor platform can compensate with a
longer window within the encounter's own timescale. The limit is the growth of the effective
dimension with gust strength: no fixed budget embeds the strongest, most three-dimensional
encounters, which is why the envelope closes and why the outstanding requirement is a noise
model calibrated to that growth rather than more sensors.

**5.3 Decodability versus estimability (the two-tier trade, condensed to a discussion point,
DRAFT).**

> A reduced state can be optimised for two different deployments, and they pull apart. A
spatially resolved latent decodes the vorticity field markedly better than the coefficient
state, but it is not recoverable from a sparse set of wall sensors, being of far higher
dimension than the measurement; the coefficient state decodes the field less sharply but is
the object a wall-pressure estimator can track. We therefore separate the two roles: the
coefficient state is the estimable, forward-usable object this paper is about, and the
field-resolved latent is a decodability instrument, not an estimation target. The trade is
quantified in Appendix C: pooling the encoder costs field-reconstruction similarity and
forecast fidelity across every family while gaining nothing in wall recoverability that the
coefficient state does not already have.

**5.4 Limitations and outlook (keep v2.1's structure, add two).** Keep the wall-normal-offset
resolution limit and the |G| = 4 three-dimensional observability boundary. Add: (i) the
attribution is C2, so the honest claim is that a wake-supervised coefficient state is
estimable and forward-usable, with supervision supplying readability and geometry and the
predictive objective supplying rollout stability, not that the predictive objective makes the
wake more readable; (ii) the filter's calibration is the priority open problem, and the
natural next step is a learned or physically motivated (Q, R) model and, beyond it, the
closed-loop use the estimator is built for and that we do not demonstrate here. Keep the
intermittently-forced-system sentence (Brunton et al. 2017) here, light, tied to the |G|
envelope, not to encounter phase (D238, resolving D-havok).

---

## 8. Section 6, conclusions (DRAFT)

> We asked which reduced-order state a sequential wall-pressure estimator can track through
an extreme vortex-gust encounter, and reached a design rule with a division of labour and a
working demonstration. On a pooled coefficient latent at matched dimension, observable
supervision on the wake supplies both the instantaneous readability of the wake and an
anisotropic latent geometry whose low-variance directions the forward rollout avoids, so a
supervised state stays in distribution under prediction where an anti-collapse regulariser on
a bare reconstruction, which is near-isotropic, does not; the multi-step predictive objective
supplies the long-horizon rollout stability that a single-step objective and an
objective-free supervised encoder both lack. On this state we built a leakage-free ensemble
Kalman filter that senses eight wall-pressure taps and never observes the vorticity field: it
tracks the lift through the encounter, from an analysis closure of 0.71 to 0.90 for gust
ratios of unity and above, and extends the usable gust-intensity range beyond single-frame
recovery, which becomes worse than the mean near a ratio of three. The recovery is a
delay-coordinate reconstruction: the encounter is low-dimensional, so a few wall taps read
over a short window place the state and recover through the dynamics the wake circulation the
instantaneous pressure cannot see, while the envelope closes where the strengthening gust
raises the effective dimension beyond what a fixed sensing budget can embed. The predictive
coefficient state carries the most wall-recoverable physics of the three families we
compared, though the reconstruction latent is the most recoverable in raw variance, a
separation of energy from information at the wall. Three facts bound the result: the
coefficient state's recoverability advantage over a raw field latent is modest, the filter is
under-dispersed at the strongest gusts and needs a calibrated noise model rather than
inflation, and the mid-plane representation reaches a three-dimensional observability
boundary at the highest gust strength. What the paper establishes is a wall-estimable,
forward-usable reduced state and the ingredients that produce it, together with a filter that
demonstrates the state in the deployment the extreme-aerodynamic control literature has been
pointing toward.

---

## 9. Figures (main text, ~10) and their sources

- **F1** encounter staging (keep, v2.1 Figure 1).
- **F2** parameter-space sampling of the v2.2 split (update; symmetric Test C).
- **F3** the coefficient state, the observable heads, and the predict-correct filter loop,
  one schematic (redraw of v2.1 Figure 3 plus the filter).
- **F4** what the state carries and who supplies it: wake-readability bars across the
  controlled matrix with the attribution annotation (supervision on/off), decode floor, merit
  (Table X visual).
- **F5** the mechanism: covariance anisotropy and near-null departure (Table Y) plus the
  H_roll merit-versus-horizon curve (Table Z).
- **F6** what the wall can see: recovery vs K per state, the visibility spectrum (Table W +
  O2), and the K x W spatial-for-temporal trade panel (Table T2) [RUN].
- **F7** the hero: assimilated encounter traces, truth vs open-loop vs pressure-only vs
  filter (Section 4.4).
- **F8** the operating envelope: filter vs static recovery vs divergence, stratified by |G|,
  with the sign-asymmetry panel (Table V), the reduced-budget filter overlay (T2b, main text
  per D238) and the effective-dimension/bound overlay (T3, main text per D238) [RUN].
- **F9** physics of the latent state: parametric atlas coloured by (G, D, Y, phase) with the
  DMD spectrum inset (Section 4.6) [RE-RUN].
- **F10** (optional) the decodability-vs-estimability trade (Appendix C pooling cost) if kept
  in main text; otherwise appendix.

Everything else (topology, decode galleries, paired tables, per-axis breakdowns, filter
diagnostics) to appendices, as v2.1 already does.

---

## 10. Citations to add

From the v3 guide:

- **Tristram, Gasperini, Killeen, Walch, Benz, Navab, Ghazaei 2026**, P-JEPA
  (arXiv:2606.23256): pooled predictive latents, frozen-encoder linear probing as the
  representation-quality diagnostic. Cite in Section 1 (insertion B) and Section 3 (protocol).
- **Broustail, Tegon, Ingolfsson, Li, Benini 2026**, LuMamba (arXiv:2603.19100):
  SIGReg/LeJEPA anti-collapse adapted to biosignals; the finding that the anti-collapse
  objective drives isotropy and needs a structuring objective. Cite in Section 1 (insertion
  B), Section 4.2 (mechanism, cross-domain corroboration), and Appendix A (anti-collapse).
- **Mousavi and Eldredge 2025** (JFM 1013 A41), sparse-pressure low-order reconstruction and
  UQ for disturbed aerodynamics; **Eldredge and Mousavi 2025** (arXiv:2502.20280), the
  Bayesian sensor-based estimation review. Cite in Section 1 (insertion A) and Section 5.2 to
  position the estimation contribution.
- **Evensen 1994** (JGR) and **Evensen 2003** (Ocean Dynamics), the ensemble Kalman filter,
  in Section 3.3.
- Keep all v2.1 lineage citations. The Fukami-Nakao-Taira 2024 and Fukami-Taira 2025 outlook
  sentences are now load-bearing (they are the paper's motivation), so quote/paraphrase them
  precisely in Section 1.

From the delay-embedding addendum:

- **Takens, F. 1981.** Detecting strange attractors in turbulence. In Dynamical Systems and
  Turbulence, LNM 898, 366-381. (Cite the theorem.)
- **Sauer, T., Yorke, J. A. and Casdagli, M. 1991.** Embedology. J. Stat. Phys. 65, 579-616.
  (Fractal-dimension / prevalence extension; the multivariate mK > 2 d_eff form.)
- **Brunton, S. L., Brunton, B. W., Proctor, J. L., Kaiser, E. and Kutz, J. N. 2017.** Chaos
  as an intermittently forced linear system. Nat. Commun. 8, 19. (The intermittently-forced
  delay-coordinate reading; HAVOK.)
- **Arbabi, H. and Mezic, I. 2017.** Koopman/Hankel from delays (verify already in lineage).
- **Bakarji, J., Champion, K., Kutz, J. N. and Brunton, S. L. 2023.** Governing equations
  from delay embeddings (verify already in lineage).
- **Solera-Rico et al. 2025** (companion, under review): the multi-time-delay latent
  predictor; cite as the sensing-side counterpart.
- Optional practical-knob citations: **Fraser and Swinney 1986** (delay by mutual
  information), **Kennel, Brown and Abarbanel 1992** (embedding dimension by false nearest
  neighbours), for the Appendix B cross-checks only.

---

## 11. What must be re-run (since all valid v2.1 analyses are re-run on v2.2)

Executed and usable as-is (Session 32, pooled v2.2): the controlled matrix (Table X core),
the mechanism (Table Y), the H_roll ablation (Table Z), the wall recovery (Table W), the
filter and its diagnostics, the operating envelope (Table V), Gate O, the O2 visibility
spectrum. Also already measured (Session 31, harvest only): the POD/Fukami wake-readability
reference cells (`outputs/session31/q1_reference.json`) and the Appendix C pooling cost
(Track P4, `outputs/session32/track_p_gates.json`).

**[RE-RUN] on pooled d = 32, v2.2**, to complete the paper:
1. POD and Fukami wake-readability cells (Table X reference column). STATUS: measured
   (POD +0.186 sign-flips vs v2.1; prose follows the measured value); remaining: POD field
   VRMSE if kept.
2. The dimension plateau {16, 32, 64} pooled (Section 3.1 robustness, and the minimum-d
   panel for the d = 32 defence). Training: jepa_pool at d = {4, 8, 16, 64};
   fukami_wake at d = {4, 8, 16}; POD by truncation.
3. The DMD/Strouhal spectrum per family (Section 4.6, F9 inset).
4. The parametric manifold atlas and the (G, D, Y) parameter probes (Section 4.6, F9).
5. The distributed-code gap and energy-information curve (Section 4.6).
6. The topology of the no-gust cycle (appendix, if kept per D223).
7. Paired per-encounter tests and the Holm family, regenerated for the new endpoint set
   {analysis C_L and E_w in the impact and relaxation windows} (Section 4.4/4.5, appendix).
8. Reference CIs and a 3-seed variance pass on the spine pair {jepa_wake, supervised_only}
   (appendix; the rest stay 1-seed with the community-standard justification).
9. Appendix C pooling-cost trade (P4), for the decodability-vs-estimability discussion.
   STATUS: harvest only.

**[RUN] Track T (delay-coordinate estimability; new, no training):**
10. **T1, delays recover the hidden coordinate.** Fix K = 8 target-blind taps. Vary W in
    {1, 2, 4, 8, 16, 30} frames at the cache cadence. Report recovery R^2 of each observable,
    the wake circulation in particular, and of the coefficient state (Table T1).
    Gate T1 (strong): wake-circulation recovery rises monotonically with W and the
    longest-window value exceeds the single-frame value with a case-clustered CI excluding
    zero. Weak: any positive trend, reported descriptively.
11. **T2, the spatial-for-temporal trade.** Grid (K, W) with K in {1, 2, 4, 8} (nested
    target-blind taps) and W in {1, 4, 8, 16, 30}. Report coefficient-state recovery and
    C_L, E_w recovery on each cell (Table T2), and run the filter at the smallest (K, W)
    that meets the recovery target to confirm it still tracks the envelope (T2b).
    Gate T2 (strong): a cell with K <= 2 and W >= 8 matches the (K = 8, W = 1) recovery
    within a case-clustered CI, and the reduced-budget filter tracks C_L for |G| in
    {1, 1.5, 2} within CI of the K = 8 filter. Weak: a monotone tradeoff surface of the
    expected sign. Selection discipline: the (K_min, W_min) pick is computed on train/val
    rows only; test_b report-only; test_c untouched.
12. **T3, dimension and the bound.** Estimate d_eff per |G| stratum: a correlation-dimension
    estimate of the encoded encounter trajectory, cross-checked against the
    participation-ratio / PCA estimate, with the out-of-plane enstrophy fraction (Section
    2.1) as the three-dimensional-onset proxy. Overlay the smallest window that meets the
    target (from T2) against the bound 2 d_eff / K, and the envelope divergence boundary
    (Section 4.5) against the |G| at which d_eff exceeds the embeddable dimension at the
    deployed budget (Table T3).
    Gate T3 (descriptive): consistency, not a fit.

Practical delay knobs (state briefly in Appendix B): the stride tau is set from the cache
cadence (cross-checked against the first mutual-information minimum, Fraser and Swinney
1986); the window W is chosen from the T2 tradeoff rather than from a false-nearest-
neighbours criterion (Kennel et al. 1992), which we report as a cross-check. Keep this light;
the paper's claim is the trade and its physical reading, not a delay-embedding parameter
study.

Not for this paper (S34+): the filter's (Q, R) calibration; the warning-horizon study
(Track D); any closed-loop demonstration.

---

## 12. Decisions (all RESOLVED 2026-07-02, D238)

- **D-T** Title: **T1** ("Wake-supervised coefficient states for wall-pressure estimation of
  extreme vortex-gust airfoil encounters").
- **D-dim** Headline latent dimension: **d = 32** with the {16, 32, 64} plateau as
  robustness.
- **D-tier** Spatial tier placement: **Discussion point + Appendix C**.
- **D-primary** Primary endpoint statement: **co-primary C_L and wake enstrophy**.
- **D-lambda** Anti-collapse weight: **lambda_S = 0.02** (commit dda57b7), recorded in
  Appendix A and numbers.json.
- **D-obs5** Physics-native 5-observable head ablation: **skipped**.
- **D-T1** Delay stride tau: **cache cadence** (dt_tc = 0.05), with the mutual-information
  cross-check reported in Appendix B.
- **D-T2** Reduced-budget filter (T2b): **main-text figure** (deployment-legible "tracked
  from K_min taps").
- **D-T3** Effective-dimension-vs-envelope overlay (T3): **main text** (theorem-to-data
  closure).
- **D-havok** Intermittently-forced framing (Brunton et al. 2017): **kept, one light
  discussion sentence tied to the |G| envelope, not to encounter phase** (phase evidence is
  mixed).
- **Tap policy (Track T)**: T1/T2 recovery grid uses **qDEIM target-blind nested prefixes**
  (matches the addendum's "target-blind" wording; avoids the model-conditioned-placement
  confound), with one bridge cell (K = 8, W = 30) on osp_per_model jepa_pool taps to
  reconcile with the Track O1 headline and the frozen filter. T2b filter stays
  **osp_per_model nested prefixes** (consistency with the frozen D220 filter; only K/taps
  change, never rho/members/mode).

---

## 13. A note on voice, for the LaTeX pass

Match the lineage: open each results subsection on the physical question, put the number in a
sentence that says what it means physically, and hedge where the data hedges. Keep the v2.1
sentence patterns that already do this ("The honest statement is therefore...", "we report
this as ... not ..."). Do not let the estimation framing turn the paper into a
machine-learning paper: the filter is the instrument that answers a fluid-mechanics question
about what a reduced state must contain, and the joint-embedding architecture remains a
probe, not the subject. The delay-embedding theorem is the theoretical thread that ties the
estimation contribution together, not a fourth contribution. British spelling throughout;
commas, colons or parentheses instead of dashes.

**JFM honesty caveats for the delay-embedding layer (state these where the claims are made):**

- **Autonomy.** Takens is for autonomous deterministic dynamics; within an encounter, after
  release, the flow relaxes autonomously and the theorem applies, but the impact itself is a
  forcing event (the intermittently-forced reading, Brunton et al. 2017). The filter is not a
  pure autonomous delay map, it carries learned dynamics that include the forced response, so
  it is not invalidated at impact. The honest framing: the delay-embedding argument is exact
  for the relaxation and approximate through the forced impact, and the filter's learned
  dynamics cover the gap.
- **Noise and finiteness.** Takens is exact for noise-free infinite data; real reconstruction
  degrades with noise and finite windows, which is why the ensemble filter (noise-robust,
  recursive) is the deployed tool and the static delay map is the characterisation. The
  under-dispersion at strong gusts is consistent with delay reconstruction becoming
  ill-conditioned as d_eff rises.
- **Effective dimension is an estimate.** Report d_eff as a bounded estimate (correlation
  dimension cross-checked against PCA), not a fitted constant, and use it for the qualitative
  bound and the |G| trend, not for a precise sensor count.
- **Genericity.** Wall pressure is a physical, non-generic observable, strongly coupled to
  the near-body vorticity; the delay-embedding guarantees are generic, so the operative
  evidence is the measured recovery (Track T), with Takens as the organising principle, not a
  proof that eight taps suffice.
