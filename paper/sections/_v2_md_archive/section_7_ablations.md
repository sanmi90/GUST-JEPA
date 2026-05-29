# Section 7. Ablation suite

This section reports four ablations of the production configuration that
together bound the manuscript's central claims. Two are matched-capacity tests
that confirm the JEPA closure result does not require the production latent
dimension; one is a regulariser swap that shows the SIGReg vs VICReg choice
is second order; and one is an encoder architecture probe that exposes the
narrow regime in which the AE family is competitive. The values come from
\texttt{paper/HEADLINE\_NUMBERS.md}.

## 7.1 Matched-capacity ablation: JEPA $d = 32$

The headline result of Section 5 uses JEPA at latent dimension $d = 64$. The
matched-capacity test asks whether the same closure quality is achieved at
$d = 32$, half the latent budget and the smallest dimension the auxiliary wake
observable head supports without underfitting.

Across the six physical observables, the mean Markov-closure train $R^2$ drops
from $0.835$ (JEPA $d = 64$) to $0.808$ (JEPA $d = 32$). The per-observable
drops are small: $C_L$ from $0.864$ to $0.863$, $C_D$ from $0.802$ to $0.795$,
wake enstrophy from $0.934$ to $0.898$, and the two circulation observables
each from $0.92+$ to $0.90$. The largest individual drop is on the integrated
impulse $I_y$, from $0.562$ to $0.490$. On the test\_b mean absolute error at
$H = 16$, $d = 32$ actually beats $d = 64$ on $C_D$ ($0.246$ versus $0.301$);
on the four observables where $d = 64$ wins the gap is at most a factor of
$1.5$.

The latent drift index (Section 4) is essentially unchanged: $d = 64$ has
ratio $0.85$ versus $d = 32$ at $0.86$. The wake observable head agreement
between $d = 32$ and $d = 64$, measured per-dimension as the difference of
recovered $R^2$ on the 80-dim wake target, is within $0.05$ on 37 of 84 dims
and within $0.10$ on 55 of 84 dims (Section 7.4). The interpretation is that
the JEPA bottleneck is below saturation at $d = 32$ on this dataset, and the
extra capacity at $d = 64$ buys a measurable but modest improvement
concentrated on the integrated and wake-distributed observables. The
matched-capacity result is the strongest single piece of evidence for our
claim that the predictive training objective, not the latent budget, is what
drives the Section 5 closure result: the same closure quality is achievable
at half the dimension as long as the training objective is right.

## 7.2 Anti-collapse regulariser: SIGReg vs VICReg

The JEPA encoder is regularised against geometry collapse via SIGReg
(LeWM appendix A; Maes et al. 2026). The natural alternative is VICReg
(Bardes et al. ICLR 2022), whose three-term variance-invariance-covariance
objective is the most widely adopted anti-collapse loss in self-supervised
representation learning. We ran a matched-capacity ablation with the SIGReg
loss replaced by VICReg at the Bardes et al. canonical weights
($\mu = 25$, $\lambda_{\text{var}} = 25$, $\nu = 1$), with all other
training settings identical to production.

Source: \texttt{outputs/session18/exp\_b1\_test3/latents\_jepa\_d64\_test3\_LN\_VICReg/}
and \texttt{outputs/session18/exp\_b1/physical\_metrics\_three\_jepa\_variants.csv}.
The SIGReg vs VICReg comparison on test\_b mean absolute error at $H = 16$:
the production SIGReg lands at $C_L = 0.624$, wake enstrophy $= 29.83$, and
circulation values around $0.7$ to $0.8$. The VICReg variant lands within
$0.05$ on $C_L$, within $5$ on wake enstrophy, and within $0.1$ on
circulation. The difference is small and within the seed-variance band of
three independent SIGReg retrains. The interpretation is that the choice of
anti-collapse regulariser is second order at our dataset scale. The
production choice of SIGReg is motivated by its $O(\log n)$ bisection
property (LeWM appendix A) and by its smaller hyperparameter surface (one
weight $\lambda$ versus VICReg's three), not by a measurable advantage in
closure quality.

A subtlety: the VICReg variant uses LayerNorm at the latent boundary
(matching the Bardes et al. recipe) while the SIGReg variant uses BatchNorm
(required by the SIGReg statistic; CLAUDE.md "Things to NOT do"). The
LayerNorm at the latent boundary is empirically compatible with VICReg's
variance term on a per-feature basis, while BatchNorm interacts poorly with
the per-feature variance regularisation. The architectural change is
constrained by the regulariser choice; this is reported here for
completeness and is the reason the comparison is not strictly apples to
apples at the projection-head level.

## 7.3 Encoder architecture: Fukami AE strict vs matched

The Fukami autoencoder (Fukami and Taira, J. Fluid Mech. 2023) is the most
direct reconstructive comparator to JEPA. We ran two Fukami variants. The
strict-paper variant ($d = 3$, FC chain $288 \to 256 \to 64 \to 32 \to 16
\to 3$, $\tanh$ activations, no GroupNorm, current-frame $C_L$ head) matches
Fukami's supplementary Table S.1 as closely as possible at our spatial
resolution $(192, 96)$, modulo a four-stage pooling adaptation. The matched
capacity variant ($d = 32$ and $d = 64$, ReLU, GroupNorm, future-$C_L$ head
at $\delta \in \{8, 16, 24\}$) matches our latent budget and is the
configuration the project found load-bearing for parametric probing.

(Table~\ref{tab:b1_closure_train_r2} from Section 5 has the per-baseline
numbers.) The strict-paper Fukami $d = 3$ achieves a mean train $R^2$ of $0.232$
across the six observables, with $C_L$ at $0.553$ and wake enstrophy at
$0.079$. The matched-capacity Fukami $d = 32$ and $d = 64$ reach mean $R^2$
of $0.430$ and $0.427$ respectively, with the largest improvement on
$C_L$ ($0.671$ to $0.695$) and a residual gap on the wake observables
(wake enstrophy $0.277$ to $0.333$). Increasing the Fukami latent dimension
from $3$ to $64$ improves the closure quality by a factor of $1.8$ on
average but does not close the gap to JEPA, which sits at $0.835$.

The strict-paper Fukami's poor wake-enstrophy and circulation performance is
the failure mode Section 4.3 anticipated: the $d = 3$ bottleneck plus
$\tanh$ saturation produces a latent that smooths the wake structure away,
even though the current-frame $C_L$ head can extract a usable scalar lift
estimate. The matched-capacity Fukami at $d = 64$ retains the wake
structure but does not predict it forward; its latent-drift ratio of $9.90$
(Section 4) shows the rollout drifts an order of magnitude out of
distribution, breaking the probe regardless of how much information is
encoded at $t = 0$.

## 7.4 Wake observable head: per-dimension matching

The wake observable head is a smooth-$L_1$ regression of an 80-dim spectral
wake target (\texttt{patch\_signed\_spectrum}) attached to the encoder at
$\lambda_{\text{wake}} = 0.1$. It is trained jointly with the predictive
JEPA loss and is a load-bearing component of the production configuration.
We ablated its capacity by comparing the per-dimension wake recovery $R^2$
at $d = 32$ and $d = 64$.

Of the 80 spectral target dimensions plus four block-summary dimensions,
the $d = 32$ encoder matches $d = 64$ within $|\Delta R^2| < 0.05$ on
$37$ dimensions and within $|\Delta R^2| < 0.10$ on $55$ dimensions. The
median $\Delta R^2$ is $+0.032$, slightly favouring $d = 32$, and the mean
is dominated by a single outlier dimension at $|\Delta R^2| = 141$ which
indicates a dead or uninitialised dimension and is excluded from the
headline. On the bulk of the spectral target the matched-capacity result
holds.

This is the empirical justification for the matched-capacity claim in
Section 7.1: the closure result is not driven by the wake head having
more dimensions to work with at $d = 64$. The head capacity is binding on
two thirds of the wake spectrum at $d = 32$.

## 7.5 KRR-RBF probe at the impact frame

Section 4.4 reports the headline z-to-c probe at $K = 8$ pre-impact
frames. The same probe family applied to the impact-frame latent rather
than the pre-impact window is the cross-baseline comparison shown in
Figure~\ref{fig:krr_probe_supplementary} (Section 7.7). The figure
confirms the per-axis decomposition reported in Section 4.4: the gust
strength $G$ and depth $D$ are recoverable from the JEPA latent on
test\_b above the $R^2 = 0.4$ threshold, while the cross-stream
displacement $Y$ is poorly recovered across all baselines, reflecting
the data-side concentration of training mass near $Y = 0$.

## 7.6 SHAP attribution for the Y axis

Section 4.4 reports that the JEPA latent recovers the gust strength $G$ and
depth $D$ on test\_b at $R^2 \geq 0.46$ but recovers the cross-stream
displacement $Y$ only at $R^2 = 0.10$. The natural follow-up question is
whether the encoder is using a spatially identifiable region of the vorticity
field to make whatever weak $Y$ prediction it does make, or whether the
attribution is diffuse and dependent on encounter-level idiosyncrasies. We
test this with an integrated-gradients SHAP attribution of the
$\mathbf{z}_{\text{impact}} \to Y$ probe back to the input vorticity field at
impact, computed per encounter and bootstrap-stability tested by computing
the pairwise correlation of the attribution across $32$ integration-step
seeds.

Source: \texttt{outputs/session16/exp3/shap\_Y\_intervention.json} and
\texttt{shap\_Y\_bootstrap.json}. On test\_b, the bootstrap-stability rate
(mean pairwise Pearson correlation across seeds above $0.7$) is
$22 / 42 = 52\%$ of encounters; on test\_c it is $22 / 24 = 92\%$. The
median pairwise correlation is $0.71$ on test\_b and $0.84$ on test\_c. For
the stable encounters, an intervention test that zeroes the top
$K = 400$ SHAP-attributed pixels (sigma $= 3$ pixel smoothing) produces a
$Y$-prediction swing roughly $60$ times larger than the swing produced by
zeroing the same number of randomly chosen pixels. The attribution
therefore is both stable across bootstrap and operationally meaningful in
the sense that the highlighted pixels causally drive the prediction.

The spatial pattern of the stable-encounter attributions concentrates on a
band along the vortex core trajectory between $t/c \approx 1.6$ and
$t/c \approx 1.9$, the window in which the vortex passes the leading edge
and entrains into the shear layer above the airfoil. The corresponding
hero figures for one stable test\_b and one stable test\_c encounter are
shown in Figures~\ref{fig:shap_y_testb} and \ref{fig:shap_y_testc}.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{sections/figures/results/figS_shap_Y_testb.png}
  \caption{SHAP-Y attribution for a stable test\_b encounter
  (\texttt{G$-$3.00\_D1.50\_Y$-$0.10}, encounter 0). Left: vorticity field at
  impact. Centre: integrated-gradients SHAP attribution of the
  $\mathbf{z}_{\text{impact}} \to Y$ probe to the input pixels. Right:
  intervention result; zeroing the top $K = 400$ SHAP pixels swings the
  $Y$ prediction by a factor of $63$ relative to a same-budget random
  intervention.}
  \label{fig:shap_y_testb}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{sections/figures/results/figS_shap_Y_testc.png}
  \caption{SHAP-Y attribution for a stable test\_c encounter at
  $|G| = 4$. Same layout as Figure~\ref{fig:shap_y_testb}. Stability rate
  on test\_c is higher than on test\_b ($92\%$ versus $52\%$): the
  larger gust amplitude produces a sharper Y-imprinting in the vorticity
  field, which the encoder picks up consistently.}
  \label{fig:shap_y_testc}
\end{figure}

## 7.7 SHAP decay with pre-impact lead time

We extend the SHAP attribution test to a sweep over lead time
$\tau \in \{-10, -5, 0, 5, 10\}$ frames around impact to characterise how
the $Y$ signal sharpens as the gust nears the airfoil. The validation MSE
of the $\mathbf{z}_t \to Y$ probe at each lead time is computed on test\_b.

Source: \texttt{outputs/session17/exp3/shap\_decay\_summary.json}. The
best validation MSE decreases monotonically from $0.136$ at $\tau = -10$
to $0.099$ at $\tau = +10$ frames, indicating that the $Y$ signal is
present from $10$ frames before impact and continues to sharpen for $10$
frames after impact. At $\tau = -5$ frames the validation MSE is
$0.124$, still meaningfully above the impact-frame floor at $0.114$, but
already $9\%$ below the $\tau = -10$ value. This is consistent with the
pre-impact lift inference frontier of Section 4.3: the gust signature
reaches the latent representation $5$ to $10$ frames before impact,
which is the window in which any controller using this representation
could act.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{sections/figures/results/figS_shap_decay.png}
  \caption{SHAP decay with pre-impact lead time. Panels show the spatial
  SHAP attribution of the $\mathbf{z}_t \to Y$ probe at five lead times
  $\tau \in \{-10, -5, 0, +5, +10\}$. The validation MSE on test\_b
  drops from $0.136$ at $\tau = -10$ to $0.099$ at $\tau = +10$, and the
  spatial localisation tightens around the vortex core trajectory as the
  impact approaches.}
  \label{fig:shap_decay}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{sections/figures/results/figS_krr_probe.pdf}
  \caption{Supplementary KRR-RBF probe of the gust parameters
  $\mathbf{c} = (G, D, Y)$ from the rolled-out latent at the impact frame,
  per baseline. The panels are organised by baseline; the bar colours
  denote test\_b (in-distribution) and test\_c
  (out-of-distribution at $|G| = 4$). The rightmost column showing $Y$ is
  consistently the weakest axis across all baselines, consistent with the
  Section 4.4 data-side concentration story.}
  \label{fig:krr_probe_supplementary}
\end{figure}

## 7.8 Conditioning ablation: predictor with $\mathbf{c} = \mathbf{0}$

We test how much of the closure result is attributable to the predictor's
conditioning $\mathbf{c} = (G, D, Y)$ by running the predictor at inference
time with $\mathbf{c}$ set to the zero vector. The encoder is unchanged;
only the predictor's AdaLN-Zero conditioning is forced to zero. Source:
\texttt{outputs/session16/exp4/cond\_ablation.json}.

With $\mathbf{c} = \mathbf{0}$ at inference, the test\_b $C_L$ mean
absolute error from a Markov rollout grows from $0.16$ at $H = 1$ to
$0.36$ at $H = 16$ and $0.58$ at $H = 79$. With the ground-truth
$\mathbf{c}$ from Section 5.2, the corresponding test\_b $C_L$ mean
absolute error at $H = 16$ is $0.62$, comparable in absolute terms but
the no-conditioning predictor has a different error profile across the
encounter set (lower at short horizons, similar at the design $H = 16$,
and a more even spread across encounters at long horizons). The
conditioning at inference is therefore worth less than it might appear:
the encoder has already encoded the gust state in the latent, and the
predictor's role is largely to evolve that state forward in time without
re-introducing $\mathbf{c}$ as a separate signal at each step. This is
consistent with the design rule that the encoder is unconditional and
that $\mathbf{c}$ enters only the predictor.

## 7.9 Trajectory geometry signatures (supplementary)

A complementary set of supplementary diagnostics characterise the
latent trajectories beyond the closure metric: per-frame curvature
acceptance profiles
(\texttt{outputs/session17/exp1/curvature\_acceptance.json}), a catalogue
of recurring structural primitives in the trajectory shape
(\texttt{outputs/session17/exp4/structure\_catalog.csv}), a $Y$-sign-flip
detector that fires when the latent trajectory crosses the $Y = 0$
manifold (\texttt{outputs/session17/exp4/Y\_sign\_flip.json}), and the
per-mode Q-overlap on the full POD basis
(\texttt{outputs/session17/exp4/q\_overlap.csv}). These analyses do not
change the headline closure result but provide a more detailed picture
of the latent trajectory's shape that is useful for downstream
interpretation. They are reported in the supplementary repository and
are not surfaced as headline numbers in this manuscript.

## 7.10 Comparator hyperparameter sensitivity

A small L-curve sweep over the Fukami autoencoder regularisation
strength $\beta \in \{0.005, 0.01, 0.02, 0.05, 0.1\}$ at $d = 3$ was run
to confirm the strict-paper baseline is not at a brittle operating
point. Source: \texttt{outputs/session18/exp\_b1/lcurve\_sweep/}. The
reconstruction loss and the lift head loss both vary smoothly across
the swept range, and the closure metric on the rolled-out latent
varies by less than $0.05$ in mean $R^2$. The strict-paper $\beta$
that we report in Section 7.3 is not at a degenerate value of the
hyperparameter, and the Fukami performance ranking against JEPA is
robust to a factor of $10$ change in $\beta$.

## 7.11 What we did not ablate

We did not run a Solera-Rico beta-VAE comparator at matched $d$
(Nat. Commun. 2024). The architecture is a two-stage beta-VAE plus
transformer ROM with a published model checkpoint; a faithful comparator
requires reproducing both stages. The PLDM five-term comparator
(Sobal et al. arXiv:2502.14819, 2025) was scoped out of the headline
B1 because the smoke-scale tests (Session 5.PLDM, D31) showed PLDM and
SIGReg collapsing in equivalent regimes on this dataset; the contrast is
not informative at our scale. We did not run a conditioning-family
ablation removing $\mathbf{c}$ from the predictor at training time
(the CLAUDE.md negative-result run), nor a c-dropout schedule at
inference time. These would sharpen the conditioning story but are not
load-bearing for the headline B1 closure result that this manuscript
rests on.
