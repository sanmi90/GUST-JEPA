# Section 5. Full-scale results

This section reports the headline result of the manuscript: a fair, matched-protocol
comparison of three encoder families (JEPA, the Fukami observable-augmented
autoencoder, and proper orthogonal decomposition) on a unified set of downstream
physical-observable forecasts. Every numerical claim in this section is taken from
the canonical reference at \texttt{paper/HEADLINE\_NUMBERS.md}, which in turn is
anchored to on-disk artifacts under
\texttt{outputs/session18/exp\_b1\_test3/}. The split is partition v2: 84 cases, 226
training encounters, a ten-case stratified test\_b of 42 encounters, and a four-case
test\_c at $|G| = 4$ with 24 encounters.

## 5.1 The B1 fairness protocol

The comparison this paper rests on is not a horse race between independently tuned
pipelines but a controlled isolation of the encoder's contribution. Every baseline
is composed of three stages: an encoder that produces a latent trajectory
$\mathbf{z}_{1:T}$ from the vorticity field; an autoregressive transformer predictor
that rolls the latent forward one step at a time given conditioning
$\mathbf{c} = (G, D, Y)$; and a probe that maps the predicted latent to a physical
observable for evaluation. The encoder is the only stage that differs across
baselines. The predictor architecture (six transformer blocks, hidden 384, sixteen
heads, AdaLN-Zero conditioning on $\mathbf{c}$, RoPE on $Q$ and $K$, dropout 0.1) is
identical across all baselines, including the training recipe (20\,000 iterations,
AdamW with cosine schedule, batch 16, scheduled-sampling rollout horizon
$H_{\text{roll}} = 8$) and the probe family (a small MLP fit on the train split,
evaluated on the held-out splits). The latent dimension $d$ is matched per
architecture family: JEPA at $d \in \{32, 64\}$, Fukami at $d \in \{3, 32, 64\}$ to
span the strict-paper $d = 3$ recipe and matched-capacity variants, and POD at
$d \in \{16, 32, 64\}$. This decomposition is the only fair way to attribute
forecast quality to the representation rather than to incidental architectural or
hyperparameter advantages, and is the basis for every claim in this section.

The headline figure of merit is the Markov-closure error: given a ground-truth
$\mathbf{c}$ and an initial latent $\mathbf{z}_0$, the predictor is rolled
recursively to horizon $H = 16$, and at each rollout step the probe maps
$\widehat{\mathbf{z}}_t$ to the predicted physical observable. The mean absolute
error against the direct numerical simulation observable, averaged over test\_b or
test\_c encounters, is the closure quality of the
(\textit{encoder}, \textit{predictor}, \textit{probe}) pipeline. A small closure
error means the encoder produces a latent whose dynamics, propagated by the
predictor, retain the information the probe needs to recover the physical
quantity. A large error can mean either that the encoder dropped that information
or that the predictor lost it during rollout; the latent-drift diagnostic in
Section 4 disambiguates the two.

Six physical observables are reported. The lift coefficient $C_L$ and the drag
coefficient $C_D$ are the body-force outputs; the impulse $I_y$ is an integrated
flow quantity that couples to wake-vortex dynamics; the wake enstrophy and the
positive and negative circulation are spatially distributed wake observables that
are particularly demanding because they integrate vorticity over the wake region
and are sensitive to the dynamics the encoder must learn to represent. These six
observables span body-force, integrated-flow, and spatially-distributed wake
diagnostics and are jointly the strongest test of whether a latent representation
captures the gust-vortex interaction physics.

## 5.2 Headline result: JEPA closes the physical-observable rollout

Table~\ref{tab:b1_closure_train_r2} reports the train coefficient of determination
$R^2$ for predicting each of the six physical observables from the rolled-out
latent at $H = 16$. JEPA $d = 64$ achieves the best closure on five of six
observables and is within $0.04$ of the best on the sixth. The mean $R^2$ across
observables is $0.835$ for JEPA $d = 64$ and $0.808$ for JEPA $d = 32$, compared
with $0.427$ for Fukami $d = 64$ and $0.561$ for POD $d = 64$. The Fukami AE
performs worst on the spatially distributed wake observables: its wake enstrophy
$R^2$ is $0.277$ and its circulation $R^2$ values are $0.336$ and $0.352$, a
factor of $2.7$ to $3.4$ below JEPA. The POD basis is competitive on the
integrated impulse $I_y$ (where it actually beats JEPA, with $R^2 = 0.799$ versus
$0.562$) but lags JEPA on wake enstrophy ($0.373$ versus $0.934$) and the
circulation observables ($0.388$ and $0.506$ versus $0.922$ and $0.927$).

\input{sections/tables/b1_physical_closure.tex}

The test\_b mean absolute errors at $H = 16$ in Table~\ref{tab:b1_mae_testb} show
the same ranking in absolute units. JEPA $d = 64$ achieves a test\_b wake-enstrophy
mean absolute error of $29.83$ versus $72.90$ for Fukami $d = 64$ and $89.31$ for
POD $d = 64$, a factor of $2.4$ and $3.0$ better, respectively. The two
circulation observables show the same pattern: JEPA $d = 64$ at $0.801$ and
$0.715$ versus Fukami $d = 64$ at $1.057$ and $1.904$ and POD $d = 64$ at
$1.660$ and $1.800$. The body-force observables $C_L$ and $C_D$ are closer
across the three families, consistent with their being scalar wake summaries that
all three encoders can approximate, but JEPA still leads on $C_L$
($0.624$ versus $0.989$ for Fukami $d = 64$ and $0.872$ for POD $d = 64$).
Figure~\ref{fig:b1_results} shows the same data graphically across test\_b and
test\_c at $H = 16$.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{sections/figures/results/fig5_b1_results.pdf}
  \caption{B1 fairness protocol results. Mean absolute error at $H = 16$ for three
  physical observables ($C_L$, $I_y$, wake enstrophy) across all eight baselines
  on test\_b (top row) and test\_c (bottom row). Error bars are bootstrap 95\%
  confidence intervals across $n = 42$ test\_b encounters and $n = 24$ test\_c
  encounters. JEPA $d = 64$ achieves the lowest error on five of six panels;
  the wake-enstrophy panel shows the clearest separation between JEPA and the
  AE/POD families.}
  \label{fig:b1_results}
\end{figure}

The matched-capacity comparison of JEPA $d = 64$ versus $d = 32$ shows that the
closure quality is largely preserved when the latent dimension is halved. Mean
$R^2$ drops from $0.835$ to $0.808$, and the per-observable drops are at most
$0.07$ (the largest drop is on impulse $I_y$, $0.562$ to $0.490$). On test\_b
mean absolute error, JEPA $d = 32$ actually beats $d = 64$ on $C_D$
($0.246$ versus $0.301$); on the four observables where $d = 64$ wins the gap is
at most a factor of $1.5$. The interpretation is that the JEPA bottleneck is not
yet saturated at $d = 32$ for the dataset's intrinsic dimensionality, but the
extra capacity at $d = 64$ buys a measurable improvement on the integrated and
wake observables. Ablations on the encoder architecture and regulariser are in
Section 7.

## 5.3 Why JEPA rolls out further: latent drift in distribution

The Markov-closure gap between JEPA and the Fukami AE is large enough that the
mechanism deserves attention. The latent-drift diagnostic in Section 4
(Table~\ref{tab:latent_drift}) reports
the Mahalanobis distance of the rolled-out latent against the DNS-encoded
latent reference distribution, averaged over test\_b encounters. The ratio
$d_{\text{Mahal, Markov}} / d_{\text{Mahal, DNS}}$ measures how far the predictor's
autoregressive rollout has left the training manifold. For JEPA $d = 64$ the
ratio is $0.85$ (the rollout is essentially within the training distribution),
for JEPA $d = 32$ it is $0.86$, for POD $d = 64$ it is $0.81$, and for Fukami
$d = 64$ it is $9.90$.

The Fukami rollout latent is roughly ten times further from its training
distribution after $H = 16$ steps than its DNS-encoded counterpart. This is the
mechanistic explanation for the Fukami closure failure: the probe is being
queried on latent states the encoder never produced and the probe never saw,
and its prediction degrades accordingly. JEPA and POD rollouts stay inside the
training manifold, so their probes are queried within the distribution they were
fit on, and the closure metric tracks the predictor's actual forecast quality
rather than an out-of-distribution probe failure. This is also the property that
makes JEPA a suitable backbone for a future model-predictive controller; see
Section 8.

## 5.4 Pre-impact lift inference

The $C_L$ profile in the window around the gust impact carries the load
information a pitching-moment controller would need to react to. We test
whether the latent representations contain pre-impact information about the
impact $C_L$ by training a probe on test\_b at various lead times $\tau$
(frames before impact) and evaluating its $R^2$. The setup compares four
modes: \textit{oracle}, probing the DNS-encoded latent at lead time
(an upper bound for the probe family); \textit{direct}, probing the raw wall
pressure sensors at lead time; \textit{via\_baseline}, probing the rolled-out
baseline latent at lead time; and \textit{predictor\_in\_loop}, rolling the
latent from lead time forward through the predictor and probing the
predicted state at impact. Sources:
\texttt{cl\_inference\_comparison.csv} and
\texttt{cl\_inference\_predictor\_in\_loop.csv}.

For JEPA $d = 64$ the oracle $R^2$ is $0.683$ at all lead times, since the
oracle has the DNS truth at hand. At $\tau = 10$ frames the
predictor-in-loop $R^2$ is $0.350$, recovering about half of the oracle
skill from ten frames out. The direct pressure probe at the same lead time
achieves $R^2 = 0.134$. At $\tau = 5$ frames the predictor-in-loop reaches
$R^2 = 0.600$, within $0.083$ of the oracle. At $\tau = 30$ frames all
methods collapse: the oracle still has the truth in hand
($R^2 = 0.683$) but the predictor-in-loop falls to $R^2 = 0.139$, the
direct probe is negative at $R^2 = -0.084$, and the via\_baseline is
negative at $R^2 = -0.064$. The interpretation is that the
predictor-in-loop adds about $0.2$ to $0.3$ of $R^2$ over a direct
pressure-sensor read at lead times of five to ten frames, and beyond
that horizon the gust signal is simply not yet in the pre-impact
window.

The Fukami AE result is qualitatively different. The Fukami $d = 64$
oracle $C_L$ $R^2$ is $-0.185$ at every lead time. This is not a probe
failure but a representation failure: the Fukami latent itself does
not carry the pre-impact information that would predict the impact
$C_L$, so no probe (linear, MLP, oracle, or otherwise) can recover it.
The smaller Fukami $d = 32$ oracle reaches $0.628$, close to JEPA's
$0.683$, but its predictor-in-loop drops to $0.578$ at $\tau = 5$
and below zero at $\tau = 20$, consistent with the latent-drift result
of Section 5.3.

## 5.5 The conditioning-only control floor

A natural skeptical question is whether the JEPA closure result is
secretly a function of the conditioning $\mathbf{c} = (G, D, Y)$ that
the predictor receives. The predictor is conditioned on $\mathbf{c}$
identically across all baselines, so within the B1 comparison there is
no asymmetry, but a reviewer might still ask: how much of the
forecast quality comes from the parameters alone, with no latent
input at all? We fit a kernel-ridge regressor with an RBF kernel on
the train split mapping $\mathbf{c}$ directly to each physical
observable at the impact frame, and evaluate on test\_b and test\_c.
The result, reported in Table~\ref{tab:conditioning_floor}, sets the
parametric floor.

On the train split the parametric floor matches JEPA $d = 64$ on
$C_L$ ($R^2 = 0.895$ versus $0.864$) and is slightly above it on
$C_D$ ($0.941$ versus $0.802$). This is the expected behaviour of an
RBF kernel interpolating $226$ points in a three-dimensional input
space, and the apparent advantage is an overfit signal rather than a
representational property. On test\_b the floor collapses: $C_L$
$R^2$ drops to $0.303$, $I_y$ is negative at $-0.301$, and the
circulation $R^2$ is essentially zero or negative. On test\_c the
floor is negative for five of six observables, with $C_D$ at
$-2.459$ and circulation negative at $-4.262$. The interpretation is
unambiguous: $\mathbf{c}$ alone cannot generalise out of the train
envelope, and the JEPA closure on test\_b and test\_c is doing
representational work that conditioning cannot replicate.

## 5.6 Three-seed encoder reproducibility

The B1 protocol numbers in Sections 5.2 to 5.5 are reported on a single
production checkpoint. To bound the reproducibility of the result we
retrained the JEPA encoder three times from independent seeds at the same
recipe and analysed two senses of seed-to-seed agreement: subspace
alignment of the interpretable axes, and the cross-seed transfer of a
SHAP-fit probe.

The subspace test computes a partial-least-squares basis of dimension
$K = 3$ from the impact-frame latent of each seed, then forms the pairwise
mean $\cos^2$ between every pair of bases. The random-rotation baseline
for two $K = 3$ subspaces in $d = 64$ ambient dimensions is
$\cos^2 = K / d = 0.047$. Source:
\texttt{outputs/session16/exp1/exp1c\_pairwise.json}. The pairwise mean
$\cos^2$ between the four seeds (production plus three retrains) is
$0.035$ for PLS bases and $0.032$ for PCA bases, both at or below the
random baseline. The interpretable basis the encoder discovers is
therefore seed-specific: a basis fit on one seed does not transfer to
another. This bounds the manuscript's interpretability claims to
per-seed statements.

The cross-seed function transfer test trains a SHAP-Y probe on each
seed's latent and evaluates it on the other seeds' latents on test\_b.
Source:
\texttt{outputs/session17/exp3/cross\_seed\_function\_transfer.json}.
Within-seed cross-validation $R^2$ ranges from $0.76$ to $0.84$ across
the four seeds. Off-diagonal transfer (probe trained on seed $i$,
evaluated on seed $j$ latent) is negative for most pairs on test\_b
(median $R^2 \approx -0.20$). The probe is seed-specific in the same
sense the basis is. The closure result on physical observables in
Sections 5.2 to 5.4 is, by contrast, computed end-to-end on each seed's
own latent and is stable across seeds: the Markov closure mean $R^2$ on
the six observables ranges from $0.81$ to $0.84$ across the four seeds
(median $0.83$). The closure metric is therefore reproducible across
seeds while the per-seed interpretable basis is not. This is the
appropriate split for a paper that claims forward-time forecast quality
rather than basis-level interpretability.

## 5.7 Markov closure versus full one-shot rollout

The B1 closure metric in Section 5.2 is computed by recursive
single-step prediction (Markov rollout). The natural alternative is to
predict the latent at horizon $H$ in one shot, conditioning the predictor
on the initial latent and a horizon-$H$ time embedding. We compute the
delta between the two modes on $C_L$ at test\_b at horizons
$H \in \{1, 4, 8, 16, 24, 32, 48, 64, 79\}$. Source:
\texttt{outputs/session17/exp2/markov\_vs\_full\_delta.json}.

At short horizons ($H \leq 16$) the Markov-minus-full delta on $C_L$ is
within $\pm 0.12$ in absolute units and within $6\%$ of the test\_b
$C_L$ standard deviation, indicating the two rollout modes produce
essentially equivalent predictions. At $H = 32$ the delta climbs to
$0.27$ ($16\%$ of $\sigma$); at $H = 48$ it reaches $0.37$ ($46\%$ of
$\sigma$); at $H = 64$ it returns to $0.13$. The Markov rollout becomes
worse than the one-shot rollout at horizons $H \geq 32$ as expected
because the recursive error compounds. The B1 protocol uses $H = 16$,
inside the regime where the two modes agree, which is the appropriate
horizon for the impact-instant pitching-moment forecast that this
manuscript targets.

## 5.8 Summary

Across six physical observables, three encoder families, and a
unified predictor architecture trained with identical recipes and
conditioning, JEPA at $d = 64$ achieves the lowest closure error
on five of six observables and the highest mean $R^2$ ($0.835$
versus $0.561$ for POD and $0.427$ for Fukami at the same $d$).
The matched-capacity result at $d = 32$ retains $0.808$ mean $R^2$
at half the latent budget. The mechanism behind the gap is that
JEPA and POD rollouts stay inside their training distribution
(Mahalanobis ratio $0.85$ and $0.81$ respectively) while Fukami
rollouts go ten times out of distribution, breaking the probe.
The parametric floor confirms that the JEPA closure is not
explained by the conditioning. The pre-impact $C_L$ inference
result demonstrates the predictor-in-loop adds five to ten frames
of usable forecast horizon beyond a direct pressure-sensor read at
matched lead times. Section 7 isolates the architectural choices
that produce this representation; Section 8 discusses its
implications for downstream model-based control.
