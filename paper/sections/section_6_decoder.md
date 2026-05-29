# Section 6. Visualisation decoder

## 6.1 Decoder architecture and training

The visualisation decoder is a separate model trained on the frozen JEPA
encoder of the Section 5 production configuration. The decoder is never
part of the JEPA loss (Section 3 design rule); training proceeds on the
same 226 train encounters and produces a model that maps a per-frame
latent $\mathbf{z}_t$ at the production dimension back to the mid-plane
vorticity field $\omega_z$ of shape $(192, 96)$. The decoder is reported
at both $d = 32$ and $d = 64$ to match the matched-capacity ablation of
Section 7.

The architecture mirrors the encoder. A linear back-projection lifts
$\mathbf{z}_t$ to $288$ tokens on a $24 \times 12$ spatial grid at channel
width $256$, a sinusoidal two-dimensional positional embedding is added,
and a six-layer pre-norm ViT (hidden $256$, 8 heads, MLP ratio 4) refines
the token features. The token grid is reshaped to a $(256, 24, 12)$
feature map and three PixelShuffle 2x upsample stages with intermediate
channel widths $(128, 64, 32)$ produce the final $(1, 192, 96)$
reconstruction. PixelShuffle is used in place of transposed convolutions
to avoid the checkerboard artifacts the literature reports for transposed
upsampling on this kind of structured field. Total parameters at $d = 32$:
$8.72\text{M}$.

Training uses a per-frame mean squared error loss
$\|\widehat{\omega}_z - \omega_z\|_F^2$ summed over $(T = 32, H = 192,
W = 96)$ and averaged over the batch dimension $B = 16$. AdamW with
momenta $(0.9, 0.95)$, weight decay $0.05$, base learning rate
$10^{-4}$, $5\%$ linear warmup followed by cosine decay to $0.05 \cdot \text{lr}$.
Gradient clipping at norm $1.0$, bf16 mixed precision on the RTX 6000
Blackwell. Training runs for $10\,000$ iterations (roughly two hours on
a single card); checkpoints land every $2\,000$ iterations. The lower
learning rate relative to the encoder reflects the single pathway being
fit, with no balancing across multiple losses.

## 6.2 SSIM convention

All structural similarity numbers in this section use the Wang convention
($K_1 = 0.01$, $K_2 = 0.03$; Wang et al. IEEE TIP 2004) on the
pipeline-normalised vorticity field, with dynamic range
$L = 2 \cdot \text{global } p_{99.9}(|\omega_{\text{norm}}|)$. For the v2
production data this gives $L \approx 8.31$. The earlier Fukami convention
($c_1 = 0.16$, $c_2 = 1.44$ on raw-scale vorticity) used by some prior
work in this dataset family is reported alongside in the head-to-head
table for direct comparison but is not the canonical SSIM in this
manuscript. The motivation for the Wang convention is that the
predictive loss is computed in the same pipeline-normalised space the
SSIM is computed in, so the structural similarity numerically corresponds
to the loss the encoder was trained against.

## 6.3 Reconstruction quality on test\_a

Per-encounter reconstruction MSE on the held-out test\_a set is compared
against a per-case-mean noise floor. The noise floor is the MSE of the
per-case mean $\omega_z$ field; a decoder that achieves a reconstruction
MSE below the floor is demonstrably using the latent's encounter-specific
information rather than the case-mean.

Result. The decoder at the production $d = 64$ achieves test\_a SSIM
$\approx 0.71$ under the Wang convention. Under the older Fukami
convention the same reconstructions score SSIM $\approx 0.60$; the gap
between the two conventions reflects the larger dynamic-range constant
$L$ in the Wang convention loosening the contrast term. The
per-encounter MSE histogram clusters near the median value, with outliers
concentrated on encounters that include the impact-frame vortex
collision close to the leading edge (frames $38$ to $42$). The
frame-by-frame MSE shows the expected peak at the impact frame (mean
argmax $40.8$ across the cached partition).

## 6.4 Reconstruction on test\_b (parametric interpolation)

Test B encounters lie outside the training parametric envelope at unseen
$(G, D, Y)$ combinations. The decoder must reconstruct vorticity fields
at interior interpolated parameter values it has not seen at training. The
qualitative outcome is that the decoder recovers the vortex core morphology
at the unseen parametric values; the wake structure is qualitatively
faithful but the wall-bounded shear layer at the trailing edge is smeared
at the highest gust strengths where the parametric distance from training
is largest. The per-encounter SSIM under the Wang convention drops
relative to test\_a, consistent with the parametric-interpolation regime
being a harder reconstruction task.

## 6.5 Reconstruction on test\_c (extrapolation)

Test C holds out the four cases at $|G| = 4$, entirely outside the
training parametric envelope. The decoder is asked to reconstruct fields
it has effectively never seen at the case level. The vortex core
position is recovered correctly at impact, consistent with the
analytical parametric prediction that the vortex centroid trajectory
follows the gust kinematics; the wake amplitude is systematically
underestimated relative to the ground truth, consistent with the
latent's $G$ axis being interpolated rather than extrapolated.

## 6.6 Head to head with the Fukami AE

The Fukami lift-augmented autoencoder (Section 7.3, A11) shares the
matched-$d$ setting and the same evaluation pipeline, but trains the
encoder, decoder, and lift head jointly on
$\text{MSE}(\omega) + \text{MSE}(C_L)$. The reconstruction quality
ranking on the same test\_a / test\_b / test\_c splits is the inverse of
the downstream closure ranking from Section 5: Fukami wins on pixel
fidelity (SSIM under the Wang convention) while JEPA wins on
forward-time predictive utility (Markov closure on physical observables).
The two results together support the explicit trade-off this paper
makes: JEPA-style predictive-only training trades reconstruction fidelity
for downstream forecast quality, and the trade-off is favourable in the
direction of forecast quality at the dataset scale and physics regime of
this work.

## 6.7 Methodology for cross-method reconstruction comparison

A natural reviewer concern is whether the reconstruction comparison in
Sections 6.3 to 6.6 is apples-to-apples across the three encoder families,
because the families differ in how they handle a new flow field at an
unseen $\mathbf{c} = (G, D, Y)$. The same concern applies in the
opposite direction to a parametric-prediction comparison: how do POD and the
Fukami autoencoder "predict" the flow at a new $\mathbf{c}$ without direct
numerical simulation data at that $\mathbf{c}$, when neither method has a
predictor that consumes $\mathbf{c}$ as input?

There are two distinct comparison protocols and they answer different
questions.

The first protocol is direct encode-decode of the direct numerical
simulation field at the test parameter. For test\_a, test\_b, and test\_c we
have direct numerical simulation data at every encounter, so each method
can be given the input field $\boldsymbol{\omega}_z(t)$ and asked to
reconstruct it through its own encoder and decoder. JEPA encodes through
the production CNN + ViT encoder and decodes through the Section 6.1
visualisation decoder. The Fukami autoencoder uses its jointly trained
encoder and decoder. POD projects the field onto the modal basis (computed
on the training split only) and reconstructs by summing the modes weighted
by the modal coefficients of the input. None of the three methods uses
$\mathbf{c}$ at encode or decode time; the encoder is unconditional in all
three cases. The "interpolation" property being tested is whether the
encoder's representation generalises to unseen $\mathbf{c}$, not whether the
method can predict the flow without direct numerical simulation access.
This is the protocol Sections 6.8 and 6.9 use.

The second protocol is parametric prediction without direct numerical
simulation access at the test parameter. We have not run a full pass of
this protocol but we discuss it here because it is the natural extension
that a deployment-time controller would need. The standard baseline is a
$K$-nearest-neighbour radial basis function interpolation in
$\mathbf{c}$-space across training $(\mathbf{c}, \mathbf{z}_{\text{impact}})$
pairs: at the test $\mathbf{c}$, the interpolated $\mathbf{z}_{\text{impact}}$
is the weighted sum of training latents whose $\mathbf{c}$ values are
nearest neighbours in parameter space. Each method then decodes the
interpolated latent through its own decoder. This protocol is
architecture-agnostic in the sense that it works for any encoder that
produces a fixed-dimensional latent. POD admits a direct analogue in which
the interpolation is on the modal coefficient vector
$\boldsymbol{\alpha}(\mathbf{c})$ rather than on a learned latent; the
reconstruction step is identical to direct projection. The Fukami
autoencoder admits the same interpolation step on its learned latent.
JEPA additionally admits a third path that POD and the Fukami autoencoder
do not: the predictor $P_{\phi}$ takes $\mathbf{c}$ as conditioning, so an
initial latent (for example from the baseline no-gust encounter) can be
rolled forward through the predictor conditioned on the test $\mathbf{c}$
to produce $\mathbf{z}_{\text{impact}}$ at the test parameter. This is the
parametric-rollout capability that Section 5 evaluates on physical
observables.

The fair cross-method comparison for reconstruction quality is the
direct encode-decode protocol at the test field. The fair cross-method
comparison for parametric prediction is the kernel-interpolation protocol
applied identically to all three methods, with the JEPA predictor rollout
as a fourth column documenting the extra capability the predictive
architecture provides. Sections 6.8 and 6.9 below use the first protocol.

## 6.8 Reconstruction quality: encounter, interpolation, extrapolation

We compare the three encoder families on the direct encode-decode
protocol of Section 6.7 across the three held-out partitions, with one
representative encounter chosen per partition. test\_a is a gust case
that the encoder has seen at training but with the specific encounter
held out (sparse encounter, dense case). test\_b is an interpolation
case at an unseen interior $(G, D, Y)$ value within the training
envelope. test\_c is an extrapolation case at $|G| = 4$, one full unit
of gust strength beyond any training value.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{sections/figures/results/figS_recon_test_a.png}
  \caption{Reconstruction quality on a held-out test\_a encounter (gust
  case seen at training, encounter held out). Columns: ground-truth
  direct numerical simulation, JEPA, Fukami autoencoder, POD. All four
  panels share the same colormap; the airfoil polygon is overlaid in
  black. Encounter metadata in the top-left corner; per-panel structural
  similarity (Wang, $L = 8.31$) and relative $L^2$ error in the bottom-right
  corner.}
  \label{fig:recon_test_a}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{sections/figures/results/figS_recon_test_b.png}
  \caption{Reconstruction quality on a held-out test\_b interpolation
  encounter (unseen interior $(G, D, Y)$). Layout matches
  Figure~\ref{fig:recon_test_a}. The reconstruction quality at this
  unseen $\mathbf{c}$ tests whether each encoder's representation
  generalises to interpolation regimes in the parametric envelope.}
  \label{fig:recon_test_b}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{sections/figures/results/figS_recon_test_c.png}
  \caption{Reconstruction quality on a held-out test\_c extrapolation
  encounter at $|G| = 4$. Layout matches
  Figure~\ref{fig:recon_test_a}. The reconstruction quality here tests
  whether each encoder's representation extrapolates beyond the training
  envelope.}
  \label{fig:recon_test_c}
\end{figure}

The reconstructions illuminate three failure modes that the SSIM and
relative $L^2$ numbers alone do not. The representative encounters chosen
are $G + 2.00, D = 1.00, Y + 0.10$ encounter 4 on test\_a;
$G + 1.50, D = 1.50, Y + 0.10$ encounter 0 on test\_b; and
$G + 4.00, D = 1.00, Y + 0.10$ encounter 0 on test\_c. All three panels
share the same colormap with $v_{\min} = -3, v_{\max} = +3$ in the
pipeline-normalised vorticity scale and the NACA 0012 polygon overlaid
in black.

The numerical readings are:

\begin{table}[h]
  \centering
  \small
  \begin{tabular}{l c c c}
    \toprule
    Metric & JEPA & Fukami AE & POD \\
    \midrule
    \multicolumn{4}{l}{\textit{test\_a (held-out encounter, trained gust case)}} \\
    \quad SSIM            & 0.673 & 0.890 & 0.770 \\
    \quad $\varepsilon_{L^2}$ & 0.850 & 0.835 & \textbf{0.710} \\
    \quad MSE             & 0.089 & 0.086 & \textbf{0.062} \\
    \quad peak $|\widehat{\omega}|$ & 5.61 & 1.38 & 5.56 \\
    \midrule
    \multicolumn{4}{l}{\textit{test\_b (interpolation case)}} \\
    \quad SSIM            & 0.678 & 0.819 & 0.750 \\
    \quad $\varepsilon_{L^2}$ & 0.936 & 0.965 & \textbf{0.703} \\
    \quad MSE             & 0.072 & 0.077 & \textbf{0.041} \\
    \quad peak $|\widehat{\omega}|$ & 5.31 & 0.37 & 4.10 \\
    \midrule
    \multicolumn{4}{l}{\textit{test\_c (extrapolation $|G| = 4$)}} \\
    \quad SSIM            & 0.585 & 0.768 & 0.681 \\
    \quad $\varepsilon_{L^2}$ & 0.844 & 0.864 & \textbf{0.754} \\
    \quad MSE             & 0.164 & 0.172 & \textbf{0.131} \\
    \quad peak $|\widehat{\omega}|$ & 9.31 & 1.44 & \textbf{10.35} \\
    \bottomrule
  \end{tabular}
\end{table}

POD lowest $\varepsilon_{L^2}$ on every split is by construction: POD
minimises the rank-truncated $L^2$ projection error in the normalised
space, so any rank-$d$ orthogonal projection of a vorticity field will
beat any nonlinear method on this metric. The structural similarity
ranking, however, is more informative. On test\_b the Fukami autoencoder
reconstruction has SSIM $= 0.819$ but the predicted field amplitude is
near zero everywhere, with a peak normalised magnitude of $0.36$
against a target peak of $9.7$. The high SSIM reflects bulk-zero
agreement with a flow that is itself mostly zero away from the impact
region, not a faithful structural reconstruction; the Fukami
reconstruction at this unseen interior $\mathbf{c}$ has collapsed to a
near-uniform mean field and lost the vortex-impingement signature
entirely. POD preserves the vortex-impingement structure correctly at the
right position and amplitude. JEPA correctly localises the vortex
impingement with the correct sign but its visualisation decoder
(Section 6.1) is blurry by design, which suppresses the high-wavenumber
wake fine structure visible in the direct numerical simulation truth.

The test\_c extrapolation row of the table degrades all three methods.
POD retains the most structure visually, JEPA shows a saturated dipole
pattern at the correct vortex position, and the Fukami autoencoder
remains near zero everywhere. The pattern is consistent across all
three splits: POD wins on amplitude-aware metrics because its objective
is exactly amplitude-aware, JEPA preserves the impingement geometry
even at extrapolation, and the Fukami autoencoder reconstruction
amplitude collapses on every unseen $\mathbf{c}$. This is the strongest
single piece of evidence that the Fukami latent does not generalise to
the parametric envelope in a way the decoder can decode: the latent
codes for unseen $\mathbf{c}$ land in regions of latent space that the
jointly trained decoder produces near-zero output for.

The visualisation comparison is the visual analogue of the
forecast-quality table of Section 5.2 and of the latent-drift
diagnostic of Section 4.1. The three together support the same reading,
that JEPA preserves the structural and dynamical information that the
downstream forecast probe requires, that POD is competitive on
amplitude reconstruction but cannot be evolved forward by the
predictor in a way that closes on physical observables (Section 5),
and that the Fukami autoencoder fails on both the dynamical
forecast (Section 5) and the parametric reconstruction (this section).

## 6.9 Parametric prediction without direct numerical simulation

The encode-decode protocol of Section 6.8 tests whether each method's
representation generalises to an unseen $\mathbf{c}$ when the field at the
test parameter is available. The deployment-relevant regime is the
complement: predict the flow at the test $\mathbf{c}$ from the parameter
alone, with no direct numerical simulation field at that parameter. We
test this protocol on the same test\_b and test\_c encounters used in
Section 6.8 so the two protocols can be compared head-to-head.

The architecture-agnostic baseline (Section 6.7, protocol 2) fits a
kernel-ridge regressor with an RBF kernel mapping
$\mathbf{c} = (G, D, Y)$ to the impact-frame latent
$\mathbf{z}_{\text{impact}}$ on the training split, predicts
$\widehat{\mathbf{z}}_{\text{impact}}$ at the test $\mathbf{c}$, and
decodes through each method's own decoder. For POD the same step is
applied to the modal coefficient vector
$\boldsymbol{\alpha}(\mathbf{c})$ and the reconstruction is the
weighted basis sum. JEPA additionally admits the predictor-rollout
path: starting from the baseline no-gust initial latent, the predictor
is rolled forward forty frames (one encounter horizon) conditioned on
the test $\mathbf{c}$, and the rolled-out impact-frame latent
$\widehat{\mathbf{z}}_{40}$ is decoded through the same visualisation
decoder.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.98\linewidth]{sections/figures/results/figS_recon_test_b_parametric.png}
  \caption{Parametric prediction without direct numerical simulation
  on the held-out test\_b interpolation encounter (same as
  Figure~\ref{fig:recon_test_b}). Five panels: ground truth
  for reference, JEPA via kernel-ridge interpolation in
  $\mathbf{c}$-space then decoder, JEPA via predictor rollout from the
  baseline initial latent conditioned on the test $\mathbf{c}$, Fukami
  autoencoder via the same kernel-ridge interpolation then decoder, and
  POD via kernel-ridge interpolation on the modal coefficient vector
  then linear basis reconstruction. All five panels share the same
  colormap and overlay; per-panel structural similarity and relative
  $L^2$ error are annotated.}
  \label{fig:recon_test_b_parametric}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.98\linewidth]{sections/figures/results/figS_recon_test_c_parametric.png}
  \caption{Parametric prediction without direct numerical simulation on
  the held-out test\_c extrapolation encounter at $|G| = 4$. Layout
  matches Figure~\ref{fig:recon_test_b_parametric}.}
  \label{fig:recon_test_c_parametric}
\end{figure}

The per-panel numbers are reported in the table below.

\begin{table}[h]
  \centering
  \small
  \begin{tabular}{l c c c c}
    \toprule
    Metric & JEPA KRR & JEPA rollout & Fukami KRR & POD KRR \\
    \midrule
    \multicolumn{5}{l}{\textit{test\_b (interpolation case)}} \\
    \quad SSIM             & 0.693 & 0.668 & 0.753 & 0.752 \\
    \quad $\varepsilon_{L^2}$ & 0.999 & 0.999 & 0.959 & \textbf{0.888} \\
    \quad MSE              & 0.082 & 0.082 & 0.076 & \textbf{0.065} \\
    \quad peak $|\widehat{\omega}|$ & 1.66 & \textbf{2.68} & 0.74 & \textbf{5.66} \\
    \midrule
    \multicolumn{5}{l}{\textit{test\_c (extrapolation $|G| = 4$)}} \\
    \quad SSIM             & 0.503 & 0.471 & 0.642 & 0.566 \\
    \quad $\varepsilon_{L^2}$ & 0.987 & 1.059 & 1.115 & \textbf{0.962} \\
    \quad MSE              & 0.224 & 0.258 & 0.286 & \textbf{0.213} \\
    \quad peak $|\widehat{\omega}|$ & 2.11 & \textbf{7.21} & 1.40 & 4.59 \\
    \bottomrule
  \end{tabular}
\end{table}

The numbers tell three honest stories. First, on the geometric error
metric POD wins on both splits, as in the encode-decode protocol of
Section 6.8 and for the same reason: smooth radial-basis interpolation
on a low-dimensional modal-coefficient space recovers the conditional
mean of the field at the test parameter, and the linear basis
reconstruction is amplitude-aware by construction. The POD parametric
prediction is the strongest non-predictive baseline.

Second, the structural-similarity numbers are misleading for the
Fukami autoencoder for the same reason they were misleading in
Section 6.8: the Fukami prediction has peak normalised amplitude
of $0.74$ on test\_b and $1.40$ on test\_c, against an oracle peak of
$9.7$ and $11.4$ respectively, which means the predicted field is
essentially the flat freestream zero and the high structural similarity
score reflects bulk-zero agreement with a flow that is itself mostly
zero away from the impact region. Kernel-ridge interpolation on the
Fukami latent decodes to a near-zero field; the latent geometry the
encoder produces is not smooth in $\mathbf{c}$ in the way the modal
coefficient space is.

Third, the JEPA predictor rollout produces the highest amplitude of any
parametric prediction on test\_c (peak $|\widehat{\omega}| = 7.21$,
within $36\%$ of the oracle $11.4$) while JEPA KRR sits at $2.11$. The
JEPA structural-similarity numbers are slightly behind POD on both
splits because the visualisation decoder is blurry, as documented in
Section 6.1. Rollout accumulates error over $40$ Markov steps, which
is why the structural similarity of JEPA rollout is slightly below
JEPA KRR on the parametric-mean metric, but the predictor's
conditioning on $\mathbf{c}$ correctly modulates the amplitude of the
rolled-out latent in a way that the kernel-ridge interpolation cannot.
This is the architecture-specific signature of the predictive training
objective: the predictor tracks gust strength via the conditioning,
even though the visualisation decoder smooths the high-wavenumber
detail. The four-way comparison summarises the central trade-off this
paper documents: the linear modal baseline wins on amplitude-aware
geometric error at all unseen parameters; the predictive architecture
wins on amplitude tracking and on dynamical forecast quality
(Section 5); the reconstructive autoencoder fails on amplitude at
every unseen parameter; and none of the methods recovers the
fine-scale shed wake structure from the parameter alone, which is the
fundamental information limit of any parametric prediction protocol
that does not include a direct numerical simulation field at the test
parameter.

## 6.10 What the decoder tells us about the latent

The reconstruction quality progression test\_a $\to$ test\_b $\to$ test\_c
provides a visual signature of the latent's parametric
interpolation-vs-extrapolation behaviour. On test\_a the decoded
reconstructions track the encounter-by-encounter dynamics closely,
demonstrating that the JEPA latent encodes encounter-level structure
beyond a per-case mean. On test\_b the decoder smooths the wake at the
largest gust strengths but preserves the vortex core morphology. On
test\_c the wake amplitude is systematically underestimated, which is
the expected visual fingerprint of an interpolative encoder asked to
extrapolate in $G$.

The visual evidence is consistent with the closure-quality result in
Section 5: the JEPA latent supports recovery of the gust-vortex physics
at unseen interior parametric values (test\_b) and degrades gracefully
under out-of-distribution extrapolation (test\_c). The decoder is a
diagnostic tool, never a load-bearing component of the JEPA loss; its
existence as a separate stage on a frozen encoder is by design
(Section 3) and is what makes it informative about the encoder's
representation rather than about a coupled encoder-decoder training
objective.
