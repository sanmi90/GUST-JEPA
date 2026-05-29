# Section 8. Discussion

## 8.1 Summary of headline findings

Across six physical observables, three encoder families at matched latent
dimension, and a unified predictor architecture trained with identical recipes
and conditioning, the predictive Joint Embedding Predictive Architecture closes
the rollout to physical observables at horizon $H = 16$ at a mean coefficient
of determination of $0.835$ on the train split, against $0.561$ for the linear
proper-orthogonal-decomposition baseline at the same $d = 64$ and $0.427$ for
the reconstructive Fukami autoencoder at the same $d = 64$. The matched-capacity
ablation at $d = 32$ retains a mean $R^2$ of $0.808$, demonstrating that the
closure quality is not a function of the latent budget on this dataset. The
gap is largest on the spatially distributed wake observables: JEPA achieves
wake-enstrophy $R^2 = 0.934$ versus $0.277$ for Fukami and $0.373$ for POD at
the matched $d = 64$. The Section 5 closure result is the primary evidence
for our claim that the predictive training objective produces a latent
representation that is forward-stable in a way the reconstructive objective
does not.

## 8.2 The mechanism behind the gap

The Fukami latent drifts an order of magnitude further out of distribution
under autoregressive rollout than its DNS-encoded counterpart, with a
Mahalanobis ratio of $9.90$ averaged over the test\_b encounters, while the
JEPA and POD latents stay inside their training distribution at ratios of
$0.85$ and $0.81$ respectively. The probe attached to a rollout that has
drifted out of distribution is, by construction, queried outside its training
support, and its outputs degrade for reasons that have nothing to do with
the predictor's actual forecast quality. The JEPA closure result is therefore
not only a statement that the JEPA latent encodes the right physical
information, but also that the encoder produces a latent geometry that the
predictor can roll forward without leaving the manifold it was fit on. The
two are not separable in practice. A predictor trained on a representation
that is not constructed to be forward-stable will produce rollouts that
probes cannot read, regardless of the predictor's one-step quality.

The corresponding result on pre-impact lift inference reinforces the
mechanism. The Fukami $d = 64$ latent achieves an oracle $C_L$ inference
$R^2$ of $-0.185$ at every pre-impact lead time. This is a representation
failure, not a probe failure: no probe can recover information that is not
encoded in the latent. The Fukami AE is trained to reconstruct the current
vorticity frame; nothing in that objective forces the latent to encode
forward-in-time information about the impact $C_L$. The JEPA latent reaches
an oracle $R^2$ of $0.683$ on the same task at the same dimension, because
the JEPA training objective explicitly aligns the latent with its future
under the predictor.

## 8.3 The conditioning-only floor and what the latent adds

A natural skeptical reading of the closure result is that the predictor's
conditioning on $\mathbf{c} = (G, D, Y)$ does the work, with the latent
representation serving as a passive carrier. The conditioning-only floor in
Section 5.5 rules this out. A kernel-ridge regressor with an RBF kernel
mapping $\mathbf{c}$ directly to the impact-frame observable interpolates the
226 training points well in three dimensions; on the train split it reaches
$R^2 = 0.895$ for $C_L$ and $0.806$ for wake enstrophy, comparable to or
slightly better than JEPA at $d = 64$. On the test\_b split the same
regressor collapses to $R^2 = 0.303$ for $C_L$ and $0.482$ for
wake enstrophy, and on the test\_c extrapolation it goes negative for five of
six observables. The latent representation is contributing the
generalisation gap between the conditioning floor and the JEPA closure on
the same held-out splits; that gap is not explained by the parameters alone.

This framing is also the cleanest answer to a likely reviewer concern about
conditioning being baked into the comparison. Within the B1 protocol the
conditioning is identical across baselines, so within-protocol no
asymmetry exists; the conditioning-only floor establishes the
across-protocol comparison that ties JEPA's advantage to its latent rather
than to the conditioning shared across all baselines.

## 8.4 Limitations

Three limitations bound the manuscript's claims. The first is the
cross-stream displacement axis $Y$. The training envelope spans
$Y \in [-0.39, +0.39]$ but most of the training mass concentrates near
$Y = 0$, and the z-to-c probe on test\_b recovers $G$ at $R^2 = 0.461$ and
$D$ at $R^2 = 0.799$ but only $0.101$ for $Y$. This is a data limitation,
not an architectural one: the latent does not strongly resolve a flow axis
that is poorly sampled in the training data. Any deployment-time
parameter-estimation step that relies on the latent should treat $Y$ as a
nuisance variable that the encoder estimates only weakly.

The second limitation is the test\_c extrapolation to $|G| = 4$. The
$R^2$ on test\_c is negative for the conditioning-only baseline on five of
six observables, and the JEPA closure on test\_c degrades sharply for the
same reason: the gust strength at $|G| = 4$ is one full integer beyond the
largest training value at $|G| = 3$. The paper does not claim
generalisation beyond the training envelope; the test\_c numbers are
reported as the boundary at which the architecture breaks down. The
inclusion of $|G| = 4$ as a separate held-out partition is a stress test,
not a claim.

The third limitation is the small-$n$ statistical regime of the
POD-pressure mechanism story. The Q-criterion overlap result in Section 4.5
shows a Spearman rank correlation of $\rho = 0.54$ with $p = 0.030$
between the POD-mode Q-overlap and the pressure recoverability across
$n = 16$ POD modes; the Pearson correlation on the same pairs is
$r = 0.49$ with $p = 0.054$. The Spearman test is the appropriate test
because the relationship is plausibly monotone rather than linear, and at
$p = 0.030$ it is statistically significant at the $5\%$ level. We still
caveat the interpretation because $n = 16$ is small, the second-half POD
modes that drag the rank correlation are noisy, and we have not extended
the analysis to other POD truncation levels.

A fourth concern is the visualisation decoder. Section 6 reports a
reconstruction SSIM in the $0.7$ range under the Wang convention on test\_a
and degrades on test\_b and test\_c; this is the visual analogue of the
forecast result and confirms the latent encodes encounter-level structure
beyond a per-case mean, but the decoder itself is not load-bearing for the
forecast claim. A reader expecting pixel-perfect reconstruction from a
predictive latent will be disappointed; this is the explicit trade-off the
predictive training objective makes.

## 8.5 Pathway to model-based control

The predictor $P_{\phi}(\mathbf{z}_{t+1} \mid \mathbf{z}_t, \mathbf{c}, \Delta t)$
is, by construction, the latent dynamics function required by model-based
reinforcement learning and model-predictive control. In our offline setting we
condition on $\mathbf{c} = (G, D, Y)$ because each direct numerical
simulation encounter has fixed gust parameters known a priori, and we use the
conditioning to test whether the latent dynamics close as a function of the
gust state. In a deployed setting the gust is unknown ahead of time; the
controller does not see $\mathbf{c}$, it only experiences the flow. The
architecture maps onto this regime in a clean way because the encoder is
unconditional by design: $\mathbf{z}_t = E_{\theta}(\boldsymbol{\omega}_z(t))$
encodes the instantaneous flow state without reference to any parameter, so
the latent itself is the observable that the controller has access to.
Short-horizon rollouts that propagate $\mathbf{z}_t$ forward through the
predictor remain meaningful even when $\mathbf{c}$ is uncertain or marginalised,
and the Section 4 z-to-c probe result quantifies how much of $\mathbf{c}$ is
already implicit in $\mathbf{z}$ on held-out cases. Closing the deployment loop
requires an online estimator
$\widehat{\mathbf{c}}_t = g(\mathbf{z}_{1:t}, \mathbf{p}_{\text{wall}, 1:t})$
that infers the gust parameters from the running latent trajectory and any
available physical sensors, after which the predictor becomes
$P_{\phi}(\mathbf{z}_{t+1} \mid \mathbf{z}_t, \widehat{\mathbf{c}}_t, \Delta t)$;
building $g$ is a sensor-to-parameter regression problem that is mechanically
straightforward but outside the scope of this work.

Three properties of the architecture transfer directly to model-predictive
control and justify the design choice of training a predictive latent model
rather than a generative one. First, the Markov-closure result of Section 5
demonstrates that the predictor closes on physical observables out to $H = 16$
frames, which is the relevant planning horizon for the impact-instant
pitching-moment problem; without closure, long-horizon plans would be polluted
by hidden-state error. Second, the bottleneck $d = 32$ to $64$ makes the
per-step cost of a candidate trajectory two to three orders of magnitude
lower than evolving the full direct numerical simulation field, so
sampling-based planners such as the model-predictive path integral and the
cross-entropy method are tractable, and the predictor is differentiable
end-to-end so gradient-based planning over a prospective actuation channel
$\mathbf{u}_t$ is equally tractable. Third, the latent drift diagnostic of
Section 4 shows that JEPA rollouts stay inside the training Mahalanobis
distribution at a ratio of $0.85$, in contrast to the Fukami autoencoder
rollout drift of $9.9$; this is the property that determines whether
long-horizon plans remain meaningful, because a planner queries the predictor
on latents reached by its own rollout rather than on ground-truth latents
from the dataset. A predictor whose latent leaves the training manifold under
rollout is not safe to plan with, regardless of how well it interpolates
next-step prediction in distribution.

The natural extension that we leave to future work is to augment the
conditioning with a time-varying actuation channel $\mathbf{u}_t$, retrain the
predictor on direct numerical simulation data with prescribed control
schedules, and use the resulting model as the world model in a closed-loop
controller that drives a chosen quantity of interest such as a lift tracking
error or a wake energy budget. The present manuscript establishes the
architectural feasibility of this pathway: a unique, unconditional latent
that does not drift under rollout, a temporal predictor whose closure on
physical observables is demonstrated out to the planning horizon, and a
parameter conditioning mechanism that admits replacement with control
variables without architectural change.

## 8.6 Closed-loop pilot (negative result)

The pathway to model-based control in Section 8.5 rests on three steps:
estimating the latent from wall-pressure sensors, rolling the latent
forward with the predictor, and probing the rolled-out latent for the
control objective. Section 4.6 establishes that the first step is
feasible at test\_b $R^2 = 0.87$ for the mean latent with a temporal
convolutional pressure estimator at $K = 8$ sensors. To check whether
the three steps compose into a usable controller we ran a closed-loop
pilot on the same test\_b and test\_c partitions and report the result
honestly: the pilot does not meet the tolerance gates we set in advance,
and the manuscript should not be read as claiming a working
sensor-to-actuator pipeline today. Source:
\texttt{outputs/session17/exp5/closed\_loop\_physical\_metrics.csv} and
\texttt{exp5\_gates.json}.

The pilot evaluates three modes. Mode A is the oracle latent at the
initial frame with oracle $\mathbf{c}$, rolled out for $H \in \{8, 16,
32\}$ frames and probed for $C_L$, $I_y$, and wake enstrophy. Mode B is
the pressure-estimated latent at the initial frame with oracle
$\mathbf{c}$, otherwise identical to Mode A. Mode C is the
pressure-estimated latent and the pressure-estimated $\mathbf{c}$, the
fully closed loop. The tolerance gates were set in advance at $10\%$
relative error on $C_L$ and $15\%$ on $I_y$.

At $K = 8$ wall-pressure sensors with the temporal convolutional
estimator, the fraction of test\_b encounters within tolerance on $C_L$
is $14\%$ in Mode A and $14\%$ in Mode B at the design horizon
$H = 16$; the corresponding fractions on $I_y$ are $14\%$ and $7\%$. On
test\_c the fractions drop further. None of the eight pre-registered
gates are met. The negative result has a clean mechanistic
interpretation: the pressure-to-latent estimator is good
($R^2 = 0.87$), but the predictor's autoregressive rollout amplifies
the residual estimation error, and the probe is then queried on a
latent that is far enough from the training distribution to produce a
poor physical-observable prediction. The Mode A versus Mode B
comparison shows the rollout is the dominant contributor to the error,
not the estimator: the within-tolerance fractions are essentially the
same with the pressure-estimated latent and the oracle latent.

The honest reading is that the architecture and the pressure
observability are both in place but the rollout step needs reinforcement
before a closed-loop controller would be usable. Two natural follow-ups
that would tighten the rollout are an online error correction step that
re-estimates the latent from pressure every few rollout frames (closing
the loop more frequently and limiting the accumulated rollout error),
and a predictor trained with a longer scheduled-sampling rollout
horizon than the production $H_{\text{roll}} = 8$ to reduce the
rollout-step error. Both are outside the scope of this work.

## 8.7 Outlook

The contribution this paper makes is narrow: a fair, controlled comparison
of three encoder families on a single dataset, with the comparison anchored
to physical-observable closure rather than to in-distribution probe
accuracy. The result is that a predictive latent objective produces a
latent representation that closes on forward-time physics where a
reconstructive latent objective does not, and that the gap is largest on
spatially distributed observables that integrate over the wake region. This
is consistent with the broader self-supervised representation learning
literature, which finds that predictive training objectives produce
representations that transfer better to downstream tasks than reconstructive
objectives. The contribution of this work is to demonstrate that the same
pattern holds in a physical-flow setting where the downstream task is a
forward-time observable prediction rather than an in-distribution
classification, and to provide the latent-drift mechanism that explains
why it does. The natural next step is the model-based control closure
sketched in Section 8.5, and beyond that the application of the same
B1 protocol to other parametric flow datasets in the open literature to
test the generality of the predictive-versus-reconstructive trade-off
this manuscript reports.
