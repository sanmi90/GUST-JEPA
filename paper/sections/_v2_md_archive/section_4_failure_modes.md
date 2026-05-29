# Section 4. Failure modes of latent dynamics models on gust-vortex flow

Before reporting the full B1 protocol results in Section 5, we describe four
failure modes that any latent dynamics model on this data must avoid. Two of
them are mechanistic limits we have measured in the data and that bound the
manuscript's claims; two are properties of the AE family that the JEPA
architecture sidesteps. The numerical values in this section come from
\texttt{paper/HEADLINE\_NUMBERS.md}, which is anchored to on-disk artifacts
under \texttt{outputs/session18/exp\_b1\_test3/}.

## 4.1 Latent drift under autoregressive rollout

A predictor that achieves low one-step prediction error in distribution does
not necessarily produce a useful long-horizon rollout. The reason is that the
predictor's output at step $t + 1$ is fed back as the input at step $t + 2$,
so any small drift in the latent geometry accumulates and pushes the rollout
away from the training distribution. Beyond a few steps, the probe attached to
the predictor (Section 5) is being queried on latent states it has never seen
during fitting, and its output degrades not because the rollout is wrong but
because the probe is being asked to extrapolate.

We measure this drift by computing, at each rollout step, the Mahalanobis
distance of the rolled-out latent from the distribution of DNS-encoded latents
in the same encounter set. The reference Mahalanobis distance
$d_{\text{Mahal, DNS}}$ is itself non-trivial, because the DNS-encoded latents
are themselves spread out over the test\_b distribution. The ratio
$r = d_{\text{Mahal, Markov}} / d_{\text{Mahal, DNS}}$ is the drift index. A
ratio close to one means the rollout stays in distribution; a ratio of ten
means the rollout has gone an order of magnitude further out of distribution
than the natural spread of the DNS-encoded latents.

\input{sections/tables/latent_drift.tex}

Table~\ref{tab:latent_drift} reports the drift index for the four production
encoders at horizon $H = 16$, averaged over the 42 test\_b encounters. The
Fukami AE rollout latent reaches a Mahalanobis distance of $28.96$, against a
reference DNS distance of $2.92$, for a drift ratio of $9.90$. The JEPA
$d = 64$ and $d = 32$ rollouts stay within their training distribution, with
ratios of $0.85$ and $0.86$ respectively. The POD rollout has a ratio of
$0.81$.

The mechanism behind the Fukami drift is the AE training objective. The Fukami
encoder is trained jointly with a decoder via a pixel-space reconstruction
loss; there is no explicit pressure on the latent to remain in a region the
predictor will see during rollout. The encoder is free to place latent codes
wherever the decoder finds it convenient, and the resulting latent geometry is
not constructed to be predictable in time. The JEPA encoder is trained against
a predictive target in the same latent space (Section 3) and is regularised
against geometry collapse via SIGReg, so the latent distribution it produces
is by construction closer to one that supports stable autoregressive rollout.
The POD basis, being a fixed linear projection of the input, inherits a
latent distribution that matches the input distribution exactly and so
trivially remains in distribution under any predictor that does not extrapolate
the linear coefficients out of their training range.

The practical consequence of this drift is Section 5's headline closure
result: the Fukami latent, despite achieving competitive in-distribution
probe accuracy, fails by a factor of two to three on long-horizon physical
observable closure compared with JEPA. The failure is not a probe failure; it
is the predictor being queried by a probe whose support no longer covers the
rolled-out latent. Section 5.3 returns to this diagnostic when reading the
closure table.

## 4.2 Parametric conditioning is not enough out of distribution

The predictor in this work is conditioned on $\mathbf{c} = (G, D, Y)$. A
natural concern is that the rollout quality is dominated by this conditioning
rather than by the latent representation. We test this by fitting a kernel
ridge regressor with an RBF kernel from $\mathbf{c}$ directly to each physical
observable at the impact frame, with no latent input, and evaluating its
$R^2$ on the train, test\_b, and test\_c splits.

\input{sections/tables/conditioning_floor.tex}

The conditioning-only floor (Table~\ref{tab:conditioning_floor}) is competitive
with JEPA $d = 64$ on the train split: an RBF kernel interpolating 226 points in
a three-dimensional parameter space inevitably interpolates well. On the train
split, $R^2$ of the (G, D, Y)-only baseline reaches $0.895$ for $C_L$ and
$0.941$ for $C_D$. This is what reviewers should expect from a kernel ridge
regressor with a flexible kernel and a low-dimensional input; it is not a
representational property of the parameters.

The same regressor evaluated on test\_b drops to $R^2 = 0.303$ for $C_L$, and
the other observables fall to near zero or negative. On test\_c (G = +4
extrapolation) the floor is negative for five of six observables. The
conditioning-only baseline cannot generalise outside the train envelope; the
JEPA closure result on test\_b and test\_c (Section 5) is therefore not a
function of $\mathbf{c}$ alone, and the latent representation contributes a
real generalisation property that conditioning alone cannot replicate. The
manuscript's headline claim that JEPA encodes flow physics is the gap between
the conditioning floor and the JEPA closure on the same held-out splits, not
the JEPA closure itself in isolation.

## 4.3 The pre-impact information frontier

Forecasting the impact $C_L$ from pre-impact information is the controllable
task this paper is most squarely aimed at. There is a hard upper bound on how
early in the encounter the impact $C_L$ can be predicted, set by when the gust
signature first reaches the airfoil surface (and so the wall pressure) and the
mid-plane vorticity field that drives the encoder. We probe this frontier by
fitting a probe on the DNS-encoded latent at various lead times $\tau$
(frames before impact) and evaluating its test\_b $R^2$ for the impact $C_L$.
The DNS-encoded latent is an oracle for the encoder family: no rollout error,
no predictor extrapolation.

For JEPA $d = 64$, the oracle $R^2$ is $0.683$ at every lead time, since the
oracle has the DNS truth at hand. This is the information-theoretic ceiling
for any pipeline that builds on the JEPA latent: with $\tau$ frames of
pre-impact latent and the JEPA encoder fixed, no probe can do better than
$R^2 = 0.683$. The fact that the oracle is constant across $\tau$ means the
JEPA latent at any pre-impact frame already carries the same information about
the impact $C_L$ as the latent at any other pre-impact frame, which is
consistent with the encoder summarising the encounter-level physics rather
than locally varying state.

The Fukami $d = 64$ oracle $R^2$ for the same task is $-0.185$, negative at
every lead time. This is not a probe failure; an oracle probe cannot do
better than the latent it is given, and a negative $R^2$ means the latent
does not carry the information required for the prediction. The Fukami AE,
trained to reconstruct the vorticity field at the current frame, has no
training signal that would force its latent to encode pre-impact gust
information that is informative about the future $C_L$. The smaller Fukami
$d = 3$ shows a slightly more degenerate version of the same pattern. This
is the strongest single piece of evidence in the paper that the predictive
training objective is not interchangeable with the reconstructive one: the
two produce latents with materially different information content even
when the encoder architecture and capacity match.

POD $d = 64$ reaches an oracle $R^2$ of $-0.238$ on this task, also negative.
POD modes computed on the full DNS field are tuned for reconstructive variance
and inherit the AE limitation. POD at smaller $d$ does slightly better
(POD $d = 16$ reaches $R^2 = 0.431$); the smaller basis acts as an implicit
regulariser, but the underlying objective is still the wrong one for
forecasting.

## 4.4 Pressure observability of the gust parameters

The deployment-time corollary to Section 4.3 is that the gust parameters
$\mathbf{c} = (G, D, Y)$ are not directly observed in the real system; only
the pressure and partial flow fields are. We measure how much of $\mathbf{c}$
is implicit in the JEPA latent by training a ridge probe on
$\mathbf{z}_{1:K}$ at $K = 8$ pre-impact frames to recover $(G, D, Y)$ on
held-out splits.

For JEPA $d = 64$ at $K = 8$ pre-impact frames, the test\_b $R^2$ values are
$0.461$ for $G$ (the gust strength), $0.799$ for $D$ (the depth), and
$0.101$ for $Y$ (the cross-stream displacement). The $G$ and $D$ axes are
recoverable from the latent; the $Y$ axis is essentially not. On test\_c
($G = +4$ extrapolation), the recovery degrades sharply for all three axes,
which is the expected behaviour for any data-driven probe queried outside the
training envelope.

The failure mode this characterises is the cross-stream displacement $Y$
axis. The training envelope spans $Y \in \{-0.387, -0.10, +0.10, +0.20,
+0.387\}$ for the periodic cases plus a few values at $+0.20$ and $-0.20$
for the run3 extension cases. The total span is roughly $\pm 0.39$ chord
lengths, but most of the training mass is at $Y \approx 0$. A probe fit on
this distribution does not learn a sharp $Y$ axis, and the latent inherits
the same weakness: gust encounters at different $Y$ values within the
training envelope produce latent trajectories that are not strongly
separated by $Y$. This is a data limitation, not an architectural one, and
it bounds the manuscript's claims on $Y$-conditional behaviour. The
Section 8 discussion of deployment uses this number to set the scope for
what an online estimator $g(\mathbf{z}_{1:t})$ would be able to recover; G
and D are recoverable, Y is marginal.

## 4.5 Pressure-based state estimation

The deployment-time consequence of Sections 4.3 and 4.4 is that the
controller has access to wall pressure but not to the mid-plane vorticity
field that the encoder consumes. The relevant question for closed-loop use
is therefore how well a learned regressor can recover the JEPA latent from a
pre-impact window of wall pressure sensors, and on which axes the recovery
is clean versus weak.

We train a kernel-ridge regressor with an RBF kernel
($\alpha = 0.1$, $\gamma = 0.01$) on the test\_b split mapping a window of
$K = 8$ wall-pressure sensors over the 30 frames preceding impact to the
JEPA latent at impact, and report the per-POD-mode $R^2$ of the
reconstruction; the POD basis is computed on the same JEPA latent at
impact across test\_b. Source: \texttt{pod\_q\_overlap\_pressure.json}.

The per-mode recoverability spans $R^2 = -0.02$ (modes 11 and 12, in which
pressure carries no information about the corresponding latent direction) to
$R^2 = 0.84$ (mode 1) with median $R^2 \approx 0.50$ over the 16 modes
retained from the impact-frame latent. The non-uniform recoverability is
mechanistically explained by the spatial overlap of each POD mode with the
high-$Q$-criterion region of the mid-plane velocity at impact: modes whose
spatial support overlaps strongly with the vortex core at impact are
recovered well from wall pressure, while modes localised away from the
vortex core are not. The rank correlation between POD-mode Q-overlap (hard
indicator at the median $p_{99}(|Q|)$ threshold) and pressure
recoverability across the 16 modes is Spearman $\rho = 0.54$ with
$p = 0.030$. The Pearson correlation on the same pairs is $r = 0.49$ with
$p = 0.054$. The rank-based test is the appropriate test here because the
relationship between Q-overlap and recoverability is plausibly monotone
rather than linear, and that test is statistically significant at the 5\%
level.

The practical interpretation: a deployment-time state estimator that maps
wall pressure to the JEPA latent should expect to recover the half of the
latent space that aligns with vortex-core-overlapping POD modes well and the
remaining half poorly. The recovery is best on the modes that matter most
for the gust-vortex impact dynamics, which is the favourable case for the
closed-loop pathway sketched in Section 8.5.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.92\linewidth]{sections/figures/results/figS_pressure_to_z.png}
  \caption{Pressure-based state estimation diagnostic. Left:
  per-POD-mode $R^2$ of the kernel-ridge regression from a pre-impact
  wall-pressure window ($K = 8$ sensors, 30 frames) to the JEPA latent
  at impact; green bars indicate $R^2 > 0.30$ recovery, red bars are
  modes that pressure cannot decode. Right: scatter of the
  mode-by-mode recoverability against the spatial overlap of each POD
  mode with the high-$Q$ region at impact, with the linear-regression
  guide and the Spearman and Pearson rank statistics annotated.
  Spearman $\rho = 0.54$, $p = 0.030$ over $n = 16$ modes.}
  \label{fig:pressure_to_z}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.92\linewidth]{sections/figures/results/figS_pod_q_overlap.png}
  \caption{POD-mode Q-criterion overlap mechanism. Spatial support of each
  POD mode (left columns) compared with the high-$Q$ region of the
  mid-plane velocity at impact (right). Modes whose support overlaps
  with the vortex core are recoverable from wall pressure; modes localised
  away from the core are not.}
  \label{fig:pod_q_overlap}
\end{figure}

## 4.6 Nonlinear pressure-based state estimation

The kernel-ridge regressor of Section 4.5 is linear in its features after the
RBF feature map. A natural extension is to use a nonlinear sequence model
that can exploit temporal structure in the pre-impact pressure window. We
test three estimator families on the same pressure-to-latent task at sensor
counts $K \in \{2, 4, 8, 16\}$: a small temporal convolutional network with
200 hidden units, a multilayer perceptron baseline, and the kernel-ridge
RBF baseline of Section 4.5. Sources:
\texttt{outputs/session17/exp5/nonlinear\_estimator\_R2.csv} and
\texttt{pressure\_to\_c\_R2.csv}.

The temporal convolutional network reaches a test\_b $R^2$ on the mean latent
recovery of $0.87$ at $K = 8$ sensors, compared with $-0.12$ for the
linear KRR-RBF baseline on the same task. Per-axis, the temporal model
recovers gust strength $G$ at $R^2 = 0.93$ and depth $D$ at $R^2 = 0.84$
on test\_b at $K = 8$; the cross-stream displacement $Y$ remains the
weakest axis at $R^2 = 0.49$, consistent with the data-side concentration
noted in Section 4.4. The multilayer perceptron lands close behind the
temporal model at $R^2 = 0.90$ on the mean latent and $R^2 = 0.96$ on $G$,
$0.90$ on $D$, $0.50$ on $Y$. On test\_c (the $|G| = 4$ extrapolation),
the $G$ recoverability collapses to large negative values for all three
estimator families, with the kernel ridge baseline producing the most
extreme numerical instabilities; the depth axis $D$ remains recoverable at
$R^2 \approx 0.93$ for the temporal model.

The direct pressure-to-parameter regressor at the impact-frame
(parameters-by-c kernel ridge with the same sensor counts) recovers $G$ on
test\_b at $R^2 = 0.91$ at $K = 2$, dropping to $R^2 = 0.74$ at $K = 4$ and
$0.57$ at $K = 8$; the depth axis $D$ recovery is negative across all
$K$ on test\_b, and $Y$ recovery is also negative across all $K$. The
pressure-to-parameter task is harder than the pressure-to-latent task
because the parameters are coarse summaries that the wall pressure does
not separate cleanly, whereas the latent has a finer-grained spatial
structure that the temporal model can exploit. The deployment-time
implication is that a nonlinear estimator should be preferred over a
linear one, and that the pressure-to-latent route is preferable to the
pressure-to-parameter route as the first stage of a closed-loop
controller. Section 8.6 reports the full closed-loop pilot that chains
these regressors with the predictor and probe.

## 4.7 Optimal sensor placement and selection

Sections 4.5 and 4.6 report the pressure-based state estimation at a fixed
sensor budget of $K = 8$ uniformly spaced wall-pressure stations. In a
real deployment the choice of which stations to instrument is a design
question of its own, and the answer is not "uniform" if the sensors have a
target the estimator is trying to recover. We run a sensor-selection pilot
that compares four selection rules at $K \in \{2, 3, 4, 8, 16\}$: a
target-conditioned structural-information criterion (TCSI) inspired by the
epiplexity framework of Finzi and Wilson, a mutual-information greedy
forward selection on the pressure-to-latent mapping, an $L^1$-regularised
LASSO regression, and a $q$-DEIM placement based on the wall-pressure
empirical interpolation. Uniform-$K$ and random-$K$ ($50$ seeds) are the
controls. Source: \texttt{outputs/session14/tcsi\_pilot/}.

Figure~\ref{fig:sensor_selection} shows the wall-pressure tap positions
on the NACA 0012 surface chosen by each rule at $K = 16$, together with
the decision-gate $z$ recovery $R^2$ on test\_b. The TCSI rule clusters
sensors in two bands: a tight leading-edge cluster covering taps $0$ to
$15$ where the gust signature arrives first, and a wider suction-side
cluster between taps $30$ and $60$ that captures the post-impact shedding
phase. The $q$-DEIM placement chooses taps in similar regions but with a
slightly different leading-edge distribution. The MI-greedy and LASSO
rules choose taps that are less spatially clustered and include
trailing-edge positions that the TCSI criterion does not select; on the
performance side, MI-greedy is consistently $0.05$ to $0.10$ behind TCSI
at $K = 2$ and $K = 4$, while LASSO is competitive at $K = 4$ ($C_L$
$R^2 = 0.95$) but degrades at larger $K$.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{sections/figures/results/figS_sensor_selection.png}
  \caption{Optimal sensor placement on the NACA 0012 surface. Each column
  corresponds to a selection rule at $K = 16$ sensors; the airfoil is
  shown in chord-normalised coordinates with the chosen wall-pressure
  taps marked. TCSI (target-conditioned structural information,
  leftmost) clusters at the leading edge and along the suction surface;
  $q$-DEIM (next) chooses similar but slightly distinct positions; MI
  greedy and LASSO produce less clustered selections that include
  trailing-edge stations. The decision-gate $z$ recovery $R^2$ on test\_b
  is shown below each placement.}
  \label{fig:sensor_selection}
\end{figure}

At $K = 8$, the TCSI sensor placement gives the temporal convolutional
pressure-to-latent estimator a test\_b $z$ recovery of $R^2 = 0.82$ and
a $C_L$ recovery (linear ridge from the latent) of $R^2 = 0.82$. At
$K = 4$, the TCSI placement reaches $R^2 = 0.79$ on $z$ and $R^2 = 0.92$
on $C_L$. At $K = 2$ the $z$ recovery drops to $R^2 = 0.70$ but the
$C_L$ recovery actually peaks at $R^2 = 0.93$, because two
well-chosen taps near the leading edge suffice to capture the bulk lift
response. The deployment-time implication is that four to eight
target-conditioned taps are sufficient for the $C_L$-tracking task
relevant to a pitching-moment controller; uniform placement would
require sixteen or more taps to reach the same performance.

The result also bounds the closed-loop pilot of Section 8.6: that
pilot used uniform-$K = 8$ taps and found the tolerance gates failed
predominantly at the rollout step rather than at the latent estimation
step. Substituting the TCSI placement at the same $K$ raises the
estimator $R^2$ from $0.87$ (Section 4.6, uniform) to $0.89$ (this
section, TCSI), a small improvement that does not change the closed-loop
conclusion because the bottleneck is the rollout, not the estimator.

## 4.8 Why the AE family fails on this dataset

The four failure modes above together account for the AE-family closure
deficit in Section 5. The Fukami AE is trained to reconstruct each frame's
vorticity field, and this objective produces a latent that is the right
representation for that task: it captures the current-frame visual content
of the flow at high fidelity, and the visualisation decoder in Section 6
confirms this fidelity is preserved by all three architectures at matched
$d$. But the latent is not constructed to remain in distribution under
autoregressive rollout (Section 4.1), it does not carry forward-in-time
information about quantities the encoder was not asked to predict
(Section 4.3), and it is not designed to be probed for gust parameters that
are not part of the reconstruction loss (Section 4.4). Each of these is a
consequence of the reconstructive training objective being mismatched to
the downstream forecast task.

The JEPA architecture replaces the pixel-space reconstruction with a
latent-space prediction loss, which forces the encoder to construct a
latent that is by construction stable under rollout and informative about
the predictable future. The remainder of the paper quantifies this
qualitative claim against the four failure modes above and against the POD
linear baseline at matched dimension.
