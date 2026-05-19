# Section 3: Methods

LaTeX-friendly markdown. Approximate target length: 4 pages. Citations
are placeholder `\cite{key}` tokens whose `key` matches HANDOFF.md
"Key references"; the final BibTeX will resolve them.

## 3.1 Data

Direct numerical simulation (DNS) of the incompressible Navier-Stokes
equations around a NACA 0012 airfoil at chord-Reynolds number Re=5000,
angle of attack alpha=14 degrees (deeply post-stall). The flow is
perturbed by a Taylor vortex parameterised by three scalars: gust
strength G (signed circulation, range [-3, +3] in the train partition,
extended to |G|=4 in the held-out Test C), vortex core diameter D in
[0.5, 1.5] chords, and wall-normal offset Y/c in [-0.4, +0.4].

The partition v1.2 is locked at `configs/splits/split_v1.json` with
sha256 `a721dc92f6e278ee054bb952933c14ba20a58137f79f3a19fc6ad71b70a007dd`
(decision D35 in HANDOFF). The partition contains 51 cases, organised
as 41 train cases (138 train sub-trajectory encounters across periodic
and run3 source groups), 6 Test B cases at unseen interior (G, D, Y)
values (28 encounters), and 4 Test C cases at the extrapolation
boundary |G|=4 (24 encounters). Test A is the held-out subset of train
cases (56 encounters within the 41 train cases) reserved for
in-distribution validation. The encoder is unconditional; the predictor
sees c=(G, D, Y).

Each case is one DNS run partitioned into encounters of 120 cache
frames at dt=0.05 t/c (D34). The cache stores the mid-plane spanwise
vorticity omega_z (192x96), span-averaged wall pressure (192,), and
two force coefficients (lift CL and drag CD, both per-frame scalars).
Sub-trajectories of length L=32 (=1.6 t/c) are sampled with a
70%/30% impact-aware/uniform mixture (configs/splits/split_v1.json,
"subtrajectory_sampling").

## 3.2 Architecture

Encoder. A hybrid CNN + ViT mapping omega_z(192, 96) to a latent
z in R^d with d=32 (D2). Three CNN downsampling stages (3M params)
produce a 24x12x256 feature map (288 spatial tokens). A 6-layer ViT
(7M params, hidden 256, 8 heads) processes the tokens; the [CLS]
token output is projected via a one-layer MLP with BatchNorm to z
(D17; LeWM Appendix A; \cite{lewm}). The encoder is unconditional by
design (it does NOT see c, per D6).

Predictor. A 6-layer autoregressive transformer (~14M params, hidden
384, 16 heads, dropout 0.1) with rotary position embeddings on
queries and keys only (RoFormer; \cite{rope}), causal mask, and
AdaLN-Zero conditioning on c=(G, D, Y) via a 2-layer MLP. The output
head is BatchNorm-projected to match the encoder's projection space
(D17). The Session 6 F-NC variant introduces a `cond_dim=0` mode that
swaps AdaLN-Zero for a plain pre-norm transformer when there is no
conditioning to inject; the production runs in Session 7 use the
default cond_dim=3 path.

Observable head (Session 6 F-OBS, Session 7 R1 and R3). A two-layer
MLP with hidden width 64 and GELU activation that maps z_t to
CL(t + Delta) for Delta in (8, 16, 24) frames (=0.4, 0.8, 1.2 t/c at
dt=0.05). Total ~2k params. Loss weight eta=0.01 (D37). The head is
trained jointly with the encoder + predictor; its parameters share
the predictor's learning-rate group in AdamW.

## 3.3 Loss compositions

This paper compares two latent-dynamics objectives that share the
encoder + predictor architecture and differ only in the anti-collapse
+ rollout machinery.

Two-term SIGReg JEPA (D5, D21). The loss is

L_JEPA = L_pred + 0.5 * L_roll + lambda * L_SIGReg(z.flatten(0, 1))

where L_pred is the teacher-forced one-step MSE in latent space,
L_roll is the open-loop H_roll=8 step rollout MSE with V-JEPA-2-AC
scheduled sampling (\cite{vjepa2}), and L_SIGReg is the LeJEPA SIGReg
characteristic-function regulariser (M=256 projections, 17 Epps-Pulley
knots in [0.2, 4]) with lambda=0.1 (\cite{lejepa}).

Five-term PLDM (D8 -> D30 -> D32). The loss is

L_PLDM = L_sim + lambda_var * L_var + lambda_cov * L_cov
                + lambda_time_sim * L_time_sim + lambda_idm * L_idm

with all four lambda set to 1.0 (the unit-weight starting point per
\cite{pldm} Appendix D.1.1). L_sim is the multi-step rollout MSE from
a single seed frame to H=8 frames ahead, L_var and L_cov are the
VICReg-derived variance and decorrelation terms applied to the rolled-
out latents \cite{vicreg}, L_time_sim is the temporal-similarity
prediction term, and L_idm is the inverse-dynamics-model loss
(L_sim + 4 collapse-prevention terms).

Observable-augmented variants (R1 = PLDM + OBS; R3 = SIGReg + OBS).
The corresponding loss is `L_base + eta * L_obs` where L_obs is the
mean-squared error of CL(t + Delta) prediction across the (B, T, 3)
output tensor and eta=0.01.

## 3.4 Optimization and training schedule

AdamW with betas (0.9, 0.95), weight decay 0.05. Two parameter groups
with separate learning rates: encoder LR 1.5e-4, predictor LR 5e-4
(including IDM-MLP for PLDM and the observable head when present).
Linear warmup over 5% of training iterations, then cosine decay to
5% of peak. Gradient clipping at 1.0. bf16 mixed-precision autocast
on the RTX 6000 Blackwell GPU (sm_120). Batch size B=16. Production
runs use 20k iterations on the 138-encounter train partition (~145
samples per epoch at B=16, so each iteration cycles through one full
pass per 10 iters, and 20k iters covers roughly 2000 epochs of the
sub-trajectory sampler's mixture).

## 3.5 Diagnostic suite

A bookkeeping diagnostic block is computed every 250 to 500 iterations
on a held-out Test B batch:

- Participation ratio `PR(z) = (sum s_i)^2 / sum s_i^2` over the
  singular values s_i of the (N x d) latent batch. PR ranges from 1
  (rank-1 collapse) to d (isotropic).
- Linear probe r^2 on z -> c via closed-form least squares with a
  75/25 fit/evaluation split.
- Per-dimension variance histogram for dimensional-collapse detection.

The Session 6 D39 evaluation extends this with the static-vs-dynamic
decomposition: `z_dyn = z - mean_per_case(z)`. Metrics computed on
both z and z_dyn distinguish "encoder encodes case identity" (high
r^2(z -> c), low r^2(z_dyn -> c)) from "encoder encodes dynamics within
each case" (high r^2(z_dyn -> phase), high r^2(z_dyn -> CL_future)).
Session 7 carries the same decomposition forward to Test B and Test C.

## 3.6 Hardware and reproducibility

Single workstation with two RTX 6000 Blackwell cards (sm_120, 96 GB
each, D40). Training entrypoints select between the two cards via
`--gpu {0,1}`; the per-run W&B `run_config["gpu_name"]` records the
device. PyTorch built with the cu128 wheel that ships sm_120 kernels.
All random sources seeded (Python random, NumPy, torch CPU and CUDA),
seed logged with the run. Inventory and split sha256 logged in W&B
`run_config` so any reported number can be traced back to a frozen
partition. W&B mode is `offline` during this session; `wandb sync`
will upload the run histories before paper submission.

## Open writing TODO

- Cite PRF 2025 / JFM 2025 vortex-gust DNS papers for the data source.
- Confirm the D=1.5 cases all come from the run3 source group; reword
  3.1 if not.
- Add Figure 1 (the architecture diagram: encoder + predictor +
  observable head + 5-term vs 2-term loss switch).
- Decide whether to keep the PR(z) diagnostic-suite description here
  or move it into Section 4.
