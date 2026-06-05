---
name: jfm_project_writing_style
description: Rewrite and review vortex-gust / latent-ROM manuscripts so they match the narrative economy, physical interpretation, and JFM style of the project papers. Use for Codex or editor sessions that touch paper/main.tex, paper/sections/*.tex, TikZ method figures, captions, abstracts, and JFM submission polish.
---

# JFM Project Writing Style Skill

Use this skill when editing the vortex-jepa manuscript or related papers for Journal of Fluid Mechanics quality. The goal is not to make the prose ornate. The goal is to make a complex machine-learning analysis read like a fluid-mechanics paper with one clear physical idea, controlled evidence, and bounded conclusions.

## Calibration set

Treat these project papers as the style anchors:

1. **A cyclic perspective on transient gust encounters through the lens of persistent homology**. Learn its single conceptual question: transient gust encounters look complicated in physical space but can be read as a simple cycle in state space. It opens with a physical staging figure, asks one question, then uses persistent homology and an autoencoder as instruments for that question.
2. **Data-driven transient lift attenuation for extreme vortex gust-airfoil interactions**. Learn its figure-led pipeline: low-dimensional manifold, sparse dynamics, phase-amplitude analysis, then control, followed by physical-space interpretation with vorticity and lift-element fields.
3. **Observable-augmented manifold learning for multi-source turbulent flow data**. Learn the restrained JFM logic: identify a manifold, explain why ordinary autoencoders are geometrically non-unique, add a physics observable, then show that similar reconstruction error can hide very different latent physics.
4. **Using optimal transport aligned latent embeddings for separated flow analysis**. Learn how to introduce a mathematical metric from a physical need, but keep claims precise: OT gives a physically meaningful field-distance comparison; do not call a correlation or ordering preservation an isometry.
5. **Grasping extreme aerodynamics on a low-dimensional manifold**. Learn the strong, elegant broad motivation and the low-dimensional-manifold narrative, but do not copy the level of Nature-style uplift into JFM. For JFM, replace ambition with mechanism and evidence.
6. **Vortical similarities across laminar and turbulent extreme gust encounters**. Learn concision: one similarity claim, flow structures first, method second, boundary of applicability clear.
7. **Causal analysis of a turbulent shear flow model** and related causality papers. Learn definitional discipline: causal/interventional words require tests. If the present model fails a direct intervention test, say so early and cleanly.
8. **The balance between compactness and forecast accuracy of data-driven latent-space ROMs in controlled wake flows**. Learn the compactness-versus-forecast tradeoff, but avoid importing unfinished or report-like style.

## The desired spine for the current paper

The manuscript should read as:

1. Extreme gust ROMs fail if they only encode force signatures.
2. Wake closure is the hard physical endpoint.
3. Predictive, reconstructive, and linear latents are compared under a matched protocol.
4. The predictive latent wins first as a representation of wake structure; the Markov forecast is confirmation.
5. Controls show that the gain requires the predictive objective plus wake-observable supervision, not architecture alone.
6. Drift, topology, and OT-distance alignment explain the mechanism.
7. Physical-space diagnostics connect the scalar wake result to LEV and shear-layer organisation.
8. Wall-pressure recovery makes the state observable, but the conditional predictor is not a validated controller.
9. Numerical simulation metadata remain a submission gate until collaborators fill the DNS resolution rows.

Every section should advance this spine. If a paragraph does not either establish, test, explain, physically interpret, or bound the spine, move it to the appendix or delete it.

## JFM-style rules

### 1. Start from physics, not architecture

Open sections with the flow phenomenon or modelling obstacle, not the neural-network component.

Weak:
> The JEPA encoder uses a CNN stem and transformer blocks.

Better:
> The representation must retain the wake reorganisation that follows LEV roll-up while remaining predictable under recursive rollout. We therefore use an unconditional encoder and place the gust parameters only in the predictor.

### 2. Use one conceptual unit per numbered subsection

Good main-text subsection count for this paper:

- Methods: 3 subsections maximum.
- Results: 5 subsections maximum.
- Discussion: 2 subsections maximum.

Avoid subsections for individual diagnostics. Horizon, drift, topology, and OT belong to one mechanism subsection unless the manuscript becomes unreadably long.

### 3. Avoid checklist prose

In main text, avoid `\paragraph{...}` and paragraphs beginning with bold labels such as **Distance.**, **Topology.**, **Transport geometry.**, **Which gust axes...**, or **Phase relative...**. Convert them to flowing prose.

Pattern:

- First sentence: physical or logical transition.
- Second sentence: method/evidence.
- Third sentence: interpretation.
- Fourth sentence, when needed: caveat or bridge to next diagnostic.

### 4. Make every number earn its place

For each number in the main text, ask:

- Does it support the primary claim or a necessary caveat?
- Does it identify split, horizon, mode, and dimension?
- Does uncertainty or paired/case-clustered status matter?

Prefer one headline number plus a table reference over a list of all numbers. Move exhaustive sweeps to appendices.

### 5. Preserve claim hierarchy

For the current paper:

- Primary claim: representational wake-enstrophy closure.
- Confirming claim: Markov forecast closure at H=16.
- Mechanism: rollout drift, single-cycle topology, OT-distance matrix alignment.
- Boundary: not counterfactual, not closed-loop validated, not simulation-metadata complete until Table 1 is filled.

Do not let the abstract or conclusion make the forecast claim sound more decisive than the representational claim.

### 6. Keep geometry language precise

Allowed:

- “aligned with the OT-distance matrix”
- “preserves the ordering of transport distances”
- “empirical distance-matrix alignment”
- “transport-consistent in this empirical sense”

Avoid unless mathematically proved or directly computed:

- “isometry”
- “geodesic parameterisation”
- “one predictor step is a transport-consistent move” as a literal claim
- “the latent metric is the OT geometry”

### 7. Keep controller/deployment language bounded

Allowed:

- “observable from sparse wall pressure”
- “deployment-relevant state estimate”
- “necessary ingredient for future closed-loop use”
- “conditional forward-closure model”

Avoid:

- “controller would plan against” in the introduction or contribution list
- “validated controller”
- “interventional model”
- “causal model”
- any implication that c=(G,D,Y) is known in deployment without estimation or marginalisation

The abstract and conclusion should contain the boundary: the model is a conditional forward-closure model, not a validated counterfactual controller.

### 8. Do not fabricate numerical-simulation details

A collaborator will fill the DNS/solver-resolution metadata. Until then:

- Leave `[PENDING]` rows visible.
- Do not hide the author-fill checklist.
- Do not say “all scales are resolved” as an unqualified claim.
- Use bounded wording: “The simulations are performed with SOD2D without a subgrid-scale model; the solver-resolution metadata required to substantiate the DNS designation are listed in Table 1 and will be supplied before submission.”

### 9. Clarify matched protocol wording

Avoid implying that the same predictor weights are shared across incompatible latent spaces.

Use:
> the same predictor architecture and training protocol, trained separately for each latent family

Use:
> the same probe family and fitting protocol

Avoid:
> the same/shared predictor
> a single shared autoregressive predictor
> the same probes

### 10. Figure and caption rules

Main figures should be evidence, not decoration.

- Use one compact method figure that shows: unconditional encoder, predictor conditioned on c=(G,D,Y), representation closure, and forecast closure.
- Keep the predictive-vs-reconstructive training comparison either as a small second panel or in the appendix if it competes with the evaluation-protocol figure.
- Captions must state what the reader should conclude, but not introduce unsupported claims.
- Captions must distinguish training heads from post-hoc diagnostics: lift and wake heads read z during JEPA training with small weights; the visualisation decoder is trained only after freezing the encoder.
- Core tables and figures must appear close to first mention. Use `\FloatBarrier` when necessary.

## Abstract template for this paper

Single paragraph, <=250 words, no citations. Recommended order:

1. Physical problem and why wake closure is hard.
2. The question: which reduced state remains physically closed under forward propagation?
3. Dataset and configuration, with numerical-simulation claim bounded if Table 1 is pending.
4. Matched comparison: predictive JEPA, reconstructive AE, POD, same architecture/protocol, H=16 closure.
5. Primary result: representational wake-enstrophy closure.
6. Confirming result: forecast closure, explicitly secondary/consistent if needed.
7. Mechanism: drift, topology, OT-distance alignment.
8. Controls: predictive objective plus wake supervision.
9. Observability and boundary: recoverable from wall pressure, but not a validated counterfactual controller.

## Introduction template

Use seven paragraphs, each with a job:

1. Physical setting: extreme gust-airfoil interactions and wake reorganisation.
2. ROM ladder: POD/DMD, nonlinear AE, observable augmentation.
3. Information/geometry gap: reconstruction does not enforce predictable latent geometry.
4. Predictive alternative: JEPA as instrument, not subject.
5. Conditioning split and boundary: c enters predictor only; model is conditional, not causal/controller.
6. Thesis paragraph: forward physical closure and geometric mechanism.
7. Contributions: three prose paragraphs, not a numbered list unless the journal style demands it.

## Results template

Preferred flow:

1. **Forward closure and conditioning floor.** Establish the result and the floor it clears.
2. **Controls.** Handle objective/architecture/wake-head confounds before mechanism.
3. **Mechanism.** Horizon dependence, drift, topology, OT-distance alignment.
4. **Physical-space success/failure.** LEV/shear-layer structure, scale decomposition, parameter/phase boundaries.
5. **Wall-pressure observability.** State recovery and pre-impact prediction, bounded as deployment relevance.

Do not put controls after topology/OT. A skeptical referee must see confound control before the mechanism.

## Discussion template

Two subsections are enough:

1. **Interpretation and scope.** What the predictive latent retains, what the mechanism means, why it matters for fluid mechanics.
2. **Limitations and outlook.** DNS metadata gate, mid-plane proxy, c-estimation/marginalisation, no counterfactual controller, future closed-loop work.

## Rewrite patterns

### From ML-first to physics-first

Before:
> We train an autoregressive transformer on latent trajectories.

After:
> The rollout must preserve the LEV-driven wake reorganisation over several recursive steps. We therefore train the latent predictor autoregressively and evaluate closure at a fixed post-impact horizon.

### From defensive to elegant

Before:
> Because the predictive encoder differs in objective, architecture and auxiliary heads, we cannot isolate the objective in the main result.

After:
> The headline comparison is deliberately a family comparison. We then use matched controls to ask which part of that family difference is responsible for the wake gain.

### From overclaim to bounded mechanism

Before:
> The predictive latent is an isometry of the OT geometry.

After:
> Distances in the predictive latent preserve the ordering of the OT-distance matrix more faithfully than those in the reconstructive latent, consistent with a rollout that remains closer to the encoded manifold.

### From controller overreach to deployment relevance

Before:
> The latent is what a controller would plan against.

After:
> The latent is a deployment-relevant state estimate: it is recoverable from wall pressure, while the conditional predictor would still require estimating or marginalising over c before closed-loop use.

## Red-flag search list

Search every PR for:

- `Abstract must`
- `Focus on Fluids`
- `Rapids articles`
- `isometry`, `isometric`
- `OT geodesic`, `geodesic`
- `controller would plan against`
- `validated controller`
- `interventional`, `causal`, `counterfactual`
- `all scales are resolved`
- `same autoregressive predictor`, `single shared`, `shared predictor`, `same probes`
- `AdaLN-Zero on (c`, `within-encounter phase`
- `impact-frame floor`
- `attached to the frozen encoder`
- `\paragraph`, `\subparagraph`, `\textbf{` at paragraph starts
- em-dash character `—`

## Build and style gate

Before a PR is ready:

1. Build the paper with `latexmk -pdf -halt-on-error -interaction=nonstopmode main.tex`.
2. Extract PDF text with `pdftotext` and grep for template artifacts.
3. Run the project convention checker if available.
4. Run `scripts/check_jfm_project_style.py paper` from this skill package or the repository copy.
5. Inspect pages around abstract, Figures 1-3, Tables 4-7, mechanism figures, and conclusion.
6. Report remaining submission gates honestly: DNS/Table 1 metadata, funding, CRediT, DOI, license.

## Codex operating instruction

When using this skill in Codex, ask for a small, reviewable PR. Do not request a full rewrite unless the manuscript has lost its spine. Require the PR body to include: summary, files changed, style changes, figure/TikZ changes, checks run, and remaining submission gates.
