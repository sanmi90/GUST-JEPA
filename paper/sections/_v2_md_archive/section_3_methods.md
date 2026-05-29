# Section 3. Methods

## 3.1 Data and partition

The dataset is a direct numerical simulation of the incompressible Navier-Stokes
equations around a NACA 0012 airfoil at chord Reynolds number $\text{Re} = 5000$
and angle of attack $\alpha = 14^\circ$ (deeply post-stall). The flow is perturbed
by a Taylor vortex parameterised by three scalars: the gust strength $G$ (signed
circulation, with the training envelope covering $|G| \in \{0.5, 1.0, 1.5, 2.0, 3.0\}$
and $G = 0$ as a baseline reference, plus $|G| = 4$ as an out-of-distribution
extrapolation), the vortex core diameter $D \in \{0.5, 1.0, 1.5\}$ chords, and the
wall-normal offset $Y/c \in [-0.4, +0.4]$ chords. Each direct numerical simulation
case is one long run partitioned into encounters of 120 cache frames at
$\Delta t = 0.05 t/c$; one encounter is one gust release at $t = 0$.

The partition is split\_v2, locked at \texttt{configs/splits/split\_v2.json},
sha256-anchored to the inventory at \texttt{data\_manifest/raw\_cases\_inventory.yaml}.
It contains 84 cases organised into a four-way train + val + test\_b + test\_c
split, with a ten-case stratified test\_b designed to span the five gust strength
levels, the three diameter levels, and the two source groups (periodic and run3).
Test C is the four cases at $|G| = 4$, reserved for the out-of-distribution
extrapolation reading. The Baseline (no-gust) case is included in train with the
flag \texttt{is\_calibration\_reference: true} so calibration tooling can still
identify it, but it is otherwise treated as a regular training case.

The per-encounter cache stores the mid-plane spanwise vorticity
$\omega_z(192, 96)$ (single channel, the second component of $\nabla \times \mathbf{u}$
at the mid-span $z$-station 16), the span-averaged wall pressure
$p_{\text{wall}}(192)$ at the airfoil surface, and the lift and drag coefficients
$C_L, C_D$ as per-frame scalars. The partition counts at horizon $H = 16$ used
throughout the paper: 226 training encounters, 28 validation encounters, 42
test\_b encounters, and 24 test\_c encounters. The encoder is unconditional; the
predictor sees $\mathbf{c} = (G, D, Y)$.

## 3.2 Omega preprocessing

The vorticity field is normalised through a frozen three-stage pipeline at
\texttt{outputs/data\_pipeline/v1/manifest.json}. First, a spatial mask of 140
cells (the inside-solid cells plus one adjacent cell, removing the leading-edge
finite-difference artifact) is applied. Second, a per-encounter $p_{99.99}$ clip
caps spikes at 282 thresholds in $[52, 178]$ over the 60 cases that participated
in the original calibration. Third, the field is scale-normalised by the train
standard deviation $\sigma_{\text{train}} = 3.5526$ with divisor
$3 \sigma = 10.658$, with no mean shift to preserve the vorticity antisymmetry.
All training and probe losses are computed in this $3\sigma$-normalised space;
unnormalised values are recovered only for figure rendering and physical-units
metrics.

## 3.3 Architecture

The encoder is a hybrid CNN + ViT mapping $\omega_z(192, 96)$ to a latent
$\mathbf{z} \in \mathbb{R}^d$ at $d \in \{32, 64\}$. Three CNN downsampling stages
(approximately 3 million parameters) produce a $24 \times 12 \times 256$ feature
map (288 spatial tokens). A six-layer ViT (approximately 7 million parameters,
hidden 256, 8 heads) processes the tokens; the [CLS] token output is projected
through a single-layer MLP with BatchNorm to $\mathbf{z}$. BatchNorm at the
latent boundary is required by the SIGReg regulariser (LeWM appendix A) and is
the project's "things to NOT do" deviation from a LayerNorm alternative. The
encoder is unconditional by design; it does not see $\mathbf{c}$.

The predictor is a six-layer autoregressive transformer (hidden 384, 16 heads,
dropout 0.1) with rotary position embeddings on queries and keys only, causal
mask, and AdaLN-Zero conditioning on $\mathbf{c} = (G, D, Y)$ via a two-layer
MLP. The output head matches the encoder's projection space. In the B1
fairness protocol (Section 5), the predictor architecture and training recipe
are held identical across the three baseline encoder families (JEPA, Fukami AE,
POD) so that closure quality is attributable to the encoder alone.

Two auxiliary observable heads are attached to the encoder. The lift head
maps $\mathbf{z}_t$ to $C_L(t)$ (at $\delta = 0$) through a
$\text{Linear}(d, 64) \to \text{GELU} \to \text{Linear}(64, 1)$ MLP with loss
weight $\eta = 0.01$. The wake head maps $\mathbf{z}_t$ to an 80-dimensional
spectral wake target \texttt{patch\_signed\_spectrum} through a Linear layer,
trained with a smooth-$L_1$ loss at weight $\lambda_{\text{wake}} = 0.1$. The
heads are trained jointly with the encoder and predictor; their parameters
share the predictor's learning-rate group.

Architecture diagram in Figure~\ref{fig:jepa_architecture}; the
contrast between the predictive JEPA training objective and a reconstructive
autoencoder objective in Figure~\ref{fig:predictive_vs_reconstructive}; the
predictor internal detail in Figure~\ref{fig:predictor_detail}; the B1 fairness
diagram in Figure~\ref{fig:b1_fairness}.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{sections/figures/tikz/fig1_jepa_architecture.pdf}
  \caption{JEPA architecture. The encoder is unconditional and outputs a latent
  $\mathbf{z}_t$ from the mid-plane vorticity field. The autoregressive
  transformer predictor takes the latent sequence and the gust conditioning
  $\mathbf{c} = (G, D, Y)$ and produces $\widehat{\mathbf{z}}_{t+\Delta t}$ in
  the same space. Two auxiliary observable heads are attached to the encoder
  output. The visualisation decoder is trained on the frozen encoder and is
  never part of the JEPA loss.}
  \label{fig:jepa_architecture}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{sections/figures/tikz/fig2_predictive_vs_reconstructive.pdf}
  \caption{Predictive versus reconstructive training. The Fukami autoencoder
  (left) takes the vorticity field at time $t$ through an encoder to a latent
  $\mathbf{z}_t$, a decoder to a reconstructed field, and a pixel-space mean
  squared error loss. JEPA (right) takes the vorticity field at time $t$
  through the same encoder to $\mathbf{z}_t$ and the field at $t + \Delta t$
  through the same encoder to a target latent $\mathbf{z}_{t + \Delta t}$;
  the predictor produces $\widehat{\mathbf{z}}_{t + \Delta t}$ from
  $\mathbf{z}_t$ and the conditioning, and the loss is in latent space.
  JEPA never sees a pixel-space loss; the visualisation decoder of Section 6
  is trained separately on the frozen encoder.}
  \label{fig:predictive_vs_reconstructive}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{sections/figures/tikz/fig3_predictor_detail.pdf}
  \caption{Predictor internal detail. The six-layer autoregressive transformer
  applies multi-head attention with rotary position embeddings on the queries
  and keys, a feed-forward network, and AdaLN-Zero modulation conditioned on
  $\mathbf{c} = (G, D, Y)$ at every block. The causal mask is shown as a
  lower-triangular grid; the hyperparameters of the production configuration
  are summarised to the right.}
  \label{fig:predictor_detail}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{sections/figures/tikz/fig4_b1_fairness.pdf}
  \caption{B1 fairness protocol. Each baseline encoder produces its own latent
  stream which is then fed into an identical autoregressive transformer
  predictor trained with the identical recipe and identical conditioning
  $\mathbf{c} = (G, D, Y)$. Closure quality on physical observables is
  attributable to the encoder alone.}
  \label{fig:b1_fairness}
\end{figure}

## 3.4 Loss

The JEPA loss is

$$
\mathcal{L} = \mathcal{L}_{\text{pred}} + 0.5 \cdot \mathcal{L}_{\text{roll}}
            + \lambda_{\text{SIGReg}} \cdot \mathcal{L}_{\text{SIGReg}}(\mathbf{z})
            + \eta \cdot \mathcal{L}_{C_L}
            + \lambda_{\text{wake}} \cdot \mathcal{L}_{\text{wake}}
$$

where $\mathcal{L}_{\text{pred}}$ is the teacher-forced one-step MSE in latent
space, $\mathcal{L}_{\text{roll}}$ is the V-JEPA 2-AC scheduled-sampling rollout
loss with $H_{\text{roll}} = 8$, $\mathcal{L}_{\text{SIGReg}}$ is the LeJEPA
characteristic-function regulariser with $M = 256$ projections and 17 Epps-Pulley
knots in $[0.2, 4]$, and $\mathcal{L}_{C_L}$ and $\mathcal{L}_{\text{wake}}$ are
the two auxiliary observable losses. The default weights are
$\lambda_{\text{SIGReg}} = 0.1$, $\eta = 0.01$, and $\lambda_{\text{wake}} = 0.1$.

The auto-fallback rule from CLAUDE.md applies: if at iteration 20\,000 the
participation ratio $\text{PR}(\mathbf{z})$ falls below $0.3 d$ and the linear
probe $R^2$ for $\mathbf{c}$ falls below $0.7$, the SIGReg term is replaced by
VICReg with weights $\mu = 25, \nu = 1$. The B1 production checkpoints
reported in Section 5 did not trigger the fallback.

## 3.5 Optimization and schedule

AdamW with $\beta = (0.9, 0.95)$, weight decay $0.05$. Two parameter groups
with separate learning rates: encoder LR $1.5 \times 10^{-4}$, predictor LR
$5 \times 10^{-4}$. Linear warmup over $5\%$ of training, then cosine decay to
$5\%$ of peak. Gradient clipping at $1.0$. bf16 mixed precision on the
RTX 6000 Blackwell GPU (sm\_120). Batch size $B = 16$. Sub-trajectory length
$L = 32$ (=$1.6 t/c$) with a $70\%$ impact-aware and $30\%$ uniform mixture for
the sampler. The production runs train 20\,000 iterations on the v2 train
partition; the predictor in the B1 fairness protocol also trains 20\,000
iterations on the frozen latents from each encoder.

## 3.6 Evaluation protocol

The headline metric is the Markov-closure mean absolute error at horizon
$H = 16$ on the test\_b and test\_c splits. The predictor is rolled
recursively from the encoder's initial latent, and at each step the
auxiliary probe maps the rolled-out latent to the physical observable. The
mean absolute error is reported as median plus bootstrap 95\% confidence
interval over $n = 42$ test\_b encounters and $n = 24$ test\_c encounters,
following the D130 reporting protocol (bootstrap $n = 2000$, three-seed
encoder variance, five-fold probe cross-validation).

The encoder-level diagnostic block is computed every 250 iterations on a
held-out batch and includes the participation ratio
$\text{PR}(\mathbf{z}) = (\sum_i s_i)^2 / \sum_i s_i^2$ over the singular
values of the $(N \times d)$ latent batch, the linear probe $R^2$ for the
gust parameters, and the per-dimension variance histogram. The visualisation
decoder is evaluated under the Wang SSIM convention with $K_1 = 0.01$ and
$K_2 = 0.03$ on the pipeline-normalised vorticity, with dynamic range
$L = 2 \cdot \text{global } p_{99.9}(|\omega_{\text{norm}}|) \approx 8.31$
for v2 production (Section 6).

## 3.7 Hardware and reproducibility

All training runs report \texttt{gpu\_name} from
\texttt{torch.cuda.get\_device\_name(device.index)}; the production
JEPA $d = 64$ and $d = 32$ checkpoints, the three Fukami AE checkpoints,
and the three POD bases all run on RTX 6000 Blackwell cards
(\texttt{sm\_120}). One exceptional set of B1 predictor retrains was
launched on an NVIDIA L40S (\texttt{sm\_89}) under the
\texttt{VORTEX\_JEPA\_ALLOW\_NON\_RTX6000=1} bypass and is recorded as
such in the W\&B run metadata; the predictor architecture is identical
across cards. All random seeds (Python random, NumPy, torch CPU, torch CUDA)
are seeded from the run's nominal seed and logged. Every reported number
in this manuscript traces back through the split sha256 in W\&B to the
locked partition manifest at \texttt{configs/splits/split\_v2.json}.
