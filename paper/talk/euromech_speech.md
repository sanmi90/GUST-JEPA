# Speaking script: A predictive latent dynamics model for extreme gust encounters

EUROMECH Colloquium, Data-driven active control in flows: from model-based to reinforcement learning.

Target: about 22 to 23 minutes spoken, leaving buffer in a 25-minute slot. Pace is roughly 150 words per minute. Numbers match `HEADLINE_NUMBERS.md` and the slide bullets. Slide numbers refer to the deck `euromech_gust_jepa.pptx` (39 slides: main talk 1 to 30, backups 31 to 39).

Structure of the open: motivation (2), objective (3), prior work and the research question (4), data (5), then what a JEPA is (6 to 8), protocol (11), and results (12 onward). Do not show model results before the method.

Delivery notes:
- The JEPA primer is slides 6 and 7. The audience is mixed fluids and machine learning, so this is where you win or lose the room. Slow down.
- Slides 9 and 10 are an illustrated alternative to the box diagrams on 6 to 8. They are optional. If short on time, skip from 8 to 11.
- Four animations carry the story: slide 2 (DNS gust and lift, motivation), slide 15 (scalars tracking the rollout), slide 16 (the same forecast as decoded fields), slide 26 (state recovered from wall pressure). Let each loop once while you talk.
- The two diagnostic slides (21 and 22) each define an unfamiliar metric, so slow down. A third diagnostic, the optimal-transport alignment, is in the backup (slide 31); if asked, be honest about its metric-dependent caveat.
- Pause on the headline numbers on slide 12.

---

## Slide 1, Title [~0:30]

Good morning, and thank you. The theme of this colloquium runs from model-based control to reinforcement learning, and almost every method on that spectrum shares one dependency: a reduced model of the flow that you can roll forward in time and actually trust. This talk is about how to learn such a model for one of the hardest cases, extreme gust encounters on a wing, and about a deceptively simple question, which reduced state is worth planning against. This is joint work with Alberto Solera-Rico, Arnau Miro and Oriol Lehmkuhl, across INTA, Carlos III, UPC and the Barcelona Supercomputing Center.

## Slide 2, The control problem (motivation) [~1:10]

Let me start with the problem itself, with no model yet, just the physics. Small and micro air vehicles increasingly fly where the gust velocity is comparable to or larger than their own flight speed: urban canyons, the wakes of buildings and ships, mountainous terrain. In that regime the gust ratio G is above one. Watch what happens on the right: a discrete vortex strikes the wing, a leading-edge vortex forms and sheds, and the lift swings strongly, here from about minus one to plus three in C_L over a fraction of a convective time, and this is only a moderate gust, G equal to minus two, from inside the training set. This is fast, large, and nonlinear, and classical gust models do not capture it. If you want to reject gusts like this actively, with a model-based controller or a reinforcement-learning agent, you need a forward model of the flow you can trust under rollout. So the question behind this talk is, what reduced state should that forward model be built on.

## Slide 3, Objective [~0:55]

Here is the objective, and the trap inside it. The load transient is built by the leading-edge vortex and the wake it leaves, so the discriminating information about what happens next lives in the wake, not in the integrated forces, lift and drag. We make that precise with a Gaussian scale split, following Motoori and Goto: low-pass the vorticity to separate the large-scale, load-bearing part, the LEV and shear layer, from the fine turbulence, and integrate its square over the wake to get a large-scale wake enstrophy. Our objective is then sharp: find a reduced state that stays forward-closed, meaning every observable remains recoverable as you propagate the state, and in particular one that keeps the wake, not just the forces. And we will compare candidate states, predictive, reconstructive and linear, under one matched protocol.

## Slide 4, The opening for JEPA (prior work and the question) [~0:55]

Why is this not already solved. Most reduced-order models for flows, from POD and DMD to nonlinear autoencoders, are trained to reconstruct the field. There is a known issue with that: reconstruction fixes the latent only up to a smooth invertible change of coordinates, so the geometry of the latent is essentially unconstrained, nothing in a reconstruction loss forces the state to be predictable. So we ask a different question, the one on the slide: which reduced state stays physically closed when you propagate it forward, so that every observable remains recoverable along the rollout. That property, forward closure, is the whole talk, and it is exactly what a planner or an RL agent needs.

## Slide 5, Data [~0:50]

The testbed is direct numerical simulation, with the SOD2D solver and no subgrid-scale model, of a NACA 0012 at fourteen degrees and Reynolds number five thousand. We perturb it with Taylor vortices parametrised by three numbers: the gust strength G, the core diameter D, and the wall-normal offset Y. The figure shows that envelope in three dimensions: eighty-four cases, with two held-out sets, test b for interpolation inside the cloud, and test c, the strong gusts at G equal to four, which you can see sitting well outside the training cloud, for extrapolation. From each impact-centred encounter we read six physical observables: lift, drag, the impulse, the wake enstrophy, and the positive and negative circulation.

## Slide 6, What is a JEPA (1 of 2) [~1:00]

Since not everyone uses JEPAs, let me spend a minute, because it is the heart of the method. A JEPA, a joint-embedding predictive architecture, comes from Yann LeCun's world-model program. The structure is simple: an encoder maps the input to a latent, a predictor advances that latent in time, and, crucially, the loss lives entirely in latent space. The field is never reconstructed. To stop the encoder from cheating by collapsing everything to a constant, an anti-collapse regulariser replaces the decoder. Two things follow. First, the encoder is free to throw away any detail that is not predictive of the future. Second, the predictor cannot take pixel-level shortcuts, because no field-space signal ever reaches it.

## Slide 7, What is a JEPA (2 of 2): the recipe and why for control [~0:55]

In one line, the training objective is: minimise the latent prediction error, plus the anti-collapse term, and nothing else. JEPAs have been demonstrated on images and video, I-JEPA and V-JEPA, and the from-pixels variants, LeWM, LeJEPA, PLDM, but so far almost entirely on gridworlds and toy visual tasks. Why should you care for control? Because this is, by design, a learned world-model: it gives you a compact latent you can roll out cheaply, it keeps the control-relevant dynamics, and, as I will show, it is recoverable from sensors. To our knowledge this is the first end-to-end JEPA trained on a parametric fluid-dynamics problem.

## Slide 8, Our architecture for gust encounters [~1:00]

Here is our instantiation. The encoder is a hybrid convolutional and vision-transformer network, and it is unconditional: it sees only the mid-plane vorticity field and maps it to a latent of dimension thirty-two or sixty-four. The predictor is an autoregressive transformer, and this is where the gust parameters enter, only the predictor, through adaptive layer norm, with rotary positions and a causal mask. That split is deliberate: the encoder is a pure state map, and the dynamics is where the conditioning lives. The predictor is the latent dynamics model, the object a controller would plan against or an RL agent would treat as its world-model. There is also a visualisation decoder, but it is trained afterwards on the frozen encoder, never inside the loss; it is only for looking at fields. And we compare, at matched latent dimension, against a reconstructive autoencoder in the Fukami lineage and against POD.

## Slides 9 and 10, Illustrated alternative: the two routes [optional, ~0:40 total]

If you want the picture rather than the boxes: the reconstructive route, the Fukami and Taira lineage, is an encoder, a small latent that also feeds a lift head, and then a decoder that must rebuild the whole field, so its loss is in pixel space. Our route keeps the same encoder but replaces the decoder with a predictor that advances the latent, and the loss is the mismatch between the predicted next latent and the true next latent, with the anti-collapse term in place of the decoder. Same encoder, different thing asked of the state. If time is tight, skip these.

## Slide 11, Forward-closure protocol [~1:00]

How do we measure forward closure fairly. We take the latent at impact, roll the predictor recursively out to sixteen frames after impact, and at each step we probe the six observables from the predicted latent. The fairness is the important part: the predictor architecture and the probe family are identical across all three encoder families, trained and fitted separately for each, so any difference in closure is a property of the encoder, not of the dynamics model or the readout. And we put a floor under it: can the three gust numbers G, D and Y alone predict each observable, with a kernel-ridge fit? A latent that only matches that floor has merely re-encoded its known inputs; beating the floor proves the latent carries flow state beyond the parameters. Everything is reported with bootstrap intervals, three encoder seeds, and five-fold probe cross-validation, on held-out test b and test c.

## Slide 12, Main result: forward closure [~1:20]

This is the headline. Averaged over the six observables, the forward-closure R-squared from the rolled-out latent is zero point eight four for the predictive JEPA, against zero point four three for the reconstructive autoencoder and zero point five six for POD. And the discriminator is exactly where I told you it would be: the wake. On wake enstrophy the predictive latent reaches zero point nine three, while the reconstructive and linear baselines are at zero point two eight and zero point three seven. In error terms, the predictive model's wake-enstrophy error at the horizon is about two and a half times lower than the autoencoder and three times lower than POD. POD, to be fair, stays competitive on the impulse, a smooth low-rank quantity, which is exactly why we track it alongside enstrophy: the impulse is the easy wake integral, the enstrophy is the hard, discriminating one.

## Slide 13, Forward closure at H = 8 [~0:30]

A quick robustness check: the same comparison rolled out to eight frames instead of sixteen. The family ordering is unchanged, the predictive latent is still lowest on the wake and closest to the floor. The advantage is not an artefact of one horizon.

## Slide 14, Forward closure at H = 4 [~0:30]

And four frames out, the same picture, with the errors smaller as you would expect closer to impact. The wake stays the discriminating observable at every horizon we test, not only at sixteen.

## Slide 15, Forward closure in action: the rollout tracks the encounter [~0:55]

Let me make that concrete. This is one held-out encounter, a representative low-error case, not the hardest one. The predictor is rolled forward from impact, and at every step we read two of the observables off the rolled latent with a fixed linear probe: the wake enstrophy on top, the lift below. Black is the simulation, orange is the prediction from the rollout, and the dashed green is the reconstruction from the encoded latent, the best the representation could do. Watch the orange track the black through the impact and the lift dip. It holds through the closure horizon and only drifts gently at long times, which is honest and expected.

## Slide 16, The same forecast in physical space [~0:50]

And here is the same forecast as fields. We decode the rolled latent every frame with the frozen visualisation decoder: simulation on the left, the encoded-latent reconstruction in the middle, the prediction from the rollout on the right. At impact the prediction equals the reconstruction, and they diverge as the horizon grows. The leading-edge vortex and the shear layer are kept; the fine-scale wake turbulence is not, because at dimension sixty-four the state cannot carry it, and it does not need to for closure. Structural similarity of the prediction is about zero point seven seven at impact and zero point five three at the horizon, so the rollout stays on the manifold.

## Slide 17, It is the wake, not the forces [~1:00]

Now why. If you ask each family how it encodes the forces, lift and drag, you find they are carried redundantly, by many coordinates, in every family. Forces are easy. The wake is different. Only the predictive latent carries the future wake as a distributed, collective code: the full latent forecasts it at rank correlation zero point eight three, while the single best coordinate only reaches zero point four four. In the reconstructive and linear latents, the best single coordinate is already as good as the whole latent, which means there is no collective wake structure to find. And this clears the conditioning-only floor, so it is the state doing the work, not the gust parameters in disguise.

## Slide 18, The wake in physical space: the Gaussian scale split [~0:45]

Here is that same statement in physical space, using the scale split from the objective slide. The top row is the large-scale vorticity sixteen frames after impact for the strongest test gust: the simulation, then the predictive and the reconstructive reconstructions. The predictive reconstruction keeps the leading-edge vortex and the wake; the reconstructive one smooths it away. The bottom panel tracks the large-scale wake enstrophy through the encounter, with bands showing the spread across test b encounters. So it is the wake is not just a number, you can see the predictive state keeping the load-bearing structure that the reconstructive state loses.

## Slide 19, Latent coordinates group by physical function [~0:50]

We can go one level finer. Take the sixty-four predictive-latent coordinates and profile each one by how strongly it correlates with each of nine physical descriptors: the gust, the forces, the wake enstrophy, the circulations, the wake thickness, the centroid. Cluster those profiles and the latent organises itself into functional groups: about fifty-one coordinates form a wake-vorticity block, eleven form a gust-forcing block, and two are essentially silent. On their own, the wake groups forecast the future wake at about zero point seven, the forcing group at zero point four five, against zero point eight three for all sixty-four together, the collective code again. I want to be careful: this is a descriptive reading of correlations, not a causal decomposition.

## Slide 20, Controls: objective and supervision, not architecture [~1:10]

A skeptic will say the predictive encoder differs from the reconstructive one in three ways at once: the objective, the architecture, and the auxiliary wake supervision. So before any mechanism, we isolate the cause. We run a two-by-two: predictive versus reconstructive objective, crossed with a CNN versus a CNN-plus-transformer architecture, with the auxiliary heads matched in all four cells. The predictive objective wins at both architectures, wake R-squared around zero point four six and zero point four five, against zero point one six and zero point two nine, and the two architecture columns do not separate. So it is the objective, not the transformer. The honest other half: if you remove the wake-observable head from the predictive model, wake closure collapses below the floor, to minus one. So the result needs the predictive objective and the wake supervision together.

## Slide 21, Diagnostic 1: latent drift [~1:15]

Now two diagnostics that explain the mechanism, each with a metric you may not see every day; a third, an optimal-transport check, sits in the backup. The first is latent drift. Why it matters: a planner or an RL agent does not query the model at nice training states, it queries it at the states its own rollout reaches, so one-step error is not enough. To measure it we use the Mahalanobis distance, a covariance-aware distance from a reference cloud; a value around one means one standard deviation away. We take the rolled-out latent and ask how far it sits from the distribution of latents encoded directly from the simulation, as a ratio. The result is striking: the reconstructive rollout drifts about ten times further out than its own encoded states, it leaves the manifold, whereas the predictive and the linear rollouts stay inside, ratios around zero point eight five. So the reconstructive failure is not a probe failure, it is the rollout walking off into a region where nothing is valid. A complementary optimal-transport check, in the backup, makes the same point geometrically: within an encounter, the predictive latent's distances track the physical advection of the flow more faithfully than the reconstructive latent's.

## Slide 22, Diagnostic 2: topology of the encounter [~1:05]

The second diagnostic is topological. Shedding and gust encounters are cyclic, so a faithful state should trace one loop per encounter, not a tangle. We quantify that with persistent homology, which grows a distance scale over the latent trajectory and counts the loops that survive over a long range of scales, the genuine cycles rather than noise. On the encoded latents of the forty-two test b encounters, the predictive latent has a median of one persistent loop, the reconstructive latent a median of three and a half, a Mann-Whitney p of about four times ten to the minus eight; and at case level all ten cases have fewer loops for the predictive latent. So the predictive state is a single clean cycle and the reconstructive one fragments. To be precise, this is the topological count; it is not a claim that the PCA picture visually closes, and in fact the orbit only partially returns within the window.

## Slide 23, Decoded reconstructions [~0:35]

Pulling the three families into physical space side by side: predictive, reconstructive, POD, against the simulation, across the held-out sets. The predictive decode is blurrier pixel by pixel, but it preserves the transported large-scale wake. The reconstructive decode is sharp at the instant yet drifts under rollout. That is the trade the predictive objective makes on purpose.

## Slide 24, Sparse sensor placement [~0:35]

Now to observability, which is what makes this deployable. We place sparse wall-pressure sensors on the airfoil and select them with two target-aware criteria, TCSI and qDEIM, against a uniform baseline, at two, four, eight and sixteen taps. The map from those sensors to the latent is a kernel ridge regression, cross-validated to guard against small-sample overfitting.

## Slide 25, Flow recovered from sparse wall pressure [~0:55]

And it works. The model here is a kernel-ridge regressor, with an RBF kernel, from the K wall-pressure taps over a pre-impact window to the impact-frame latent, which the frozen decoder then renders as a field. Eight taps already recover the leading-edge vortex and the shear layer; two taps coarsen it but keep the gross wake structure. We benchmark against the oracle decode, the decode of the simulation-encoded latent, so you see the ceiling alongside the estimate. So the predictive state is reconstructible from a few wall sensors, a deployment-relevant observability result.

## Slide 26, Sparse-sensor state estimation in action [~0:50]

Here that is, animated, and note there is no predictor in this slide. This is a per-frame MLP that maps a causal six-frame window of sixteen wall-pressure taps to the latent, decoded to a field, frame by frame: simulation on the left with the taps marked, the field recovered from pressure alone on the right. On a held-out encounter, held out by whole encounter, the latent is recovered at R-squared about zero point seven two, structural similarity around zero point five. Wall pressure fixes the near-body leading-edge vortex and shear layer; the far wake is simply not observable from the surface, which is the honest limit. This is state estimation, not forecasting, the complement to the rollout.

## Slide 27, Control relevance: observable and forecastable [~1:15]

This is the part that speaks most directly to this audience, and it puts the two pieces together. First, the predictive state is observable from sparse wall pressure: from the latent we recover the gust strength and core diameter on held-out cases, R-squared around zero point four six and zero point eight. Second, the figure on the right, we can predict the impact lift ahead of time. If you roll the latent forward and then probe, the predictor-in-loop gives R-squared zero point three five a full ten frames before impact, where reading the pressure sensors directly gives essentially nothing, zero point one three; the oracle ceiling is zero point six eight. And the reconstructive latent's own oracle is negative, so it never had the pre-impact information; a representation failure, not a probe failure. So you get both ingredients a controller consumes: a state estimate from cheap sensors, and a short-horizon forecast.

## Slide 28, Conclusions [~0:55]

To bring it together. A predictive objective, trained with wake supervision, gives you a latent dynamics model that is forward-closed, with wake R-squared around zero point seven five at the representation level and zero point eight four in the mean; that stays on its manifold under rollout; and that traces a single clean cycle. It is compact, dimension thirty-two is as good as sixty-four, and observable from sparse wall pressure. The thread through all of it is the wake, the thing that force-only and reconstruction-only states throw away. It is a conditional forward-closure model, a substrate for model-based control and RL world-models, but not yet a validated controller. Where it goes next: because the gust enters only as a conditioning channel, replacing it with an actuation channel is a change of input, not of architecture, so the same machinery applies to control inputs; from there, closing the loop, three-dimensional observability at the strong gusts, and carrying the recipe to other parametric flows.

## Slide 29, Acknowledgements [~0:15]

Thank you to my collaborators and to the funders listed here, and thank you all for your attention. I would be glad to take questions.

---

## Backups (slides 30 to 39), for questions

- Backup divider (30).
- Transport geometry, optimal transport (31): OT field distance after Tran, Yeh & Taira (JFM 2026), signed +/- vorticity split + unbalanced Sinkhorn; post-hoc within-encounter Spearman 0.63 (predictive) vs 0.45 (reconstructive); pooled across encounters it reverses, so trajectory-local only.
- Dataset and protocol (32): split v2, eighty-four cases, 226 / 42 / 24 encounters, the omega pipeline, the reporting protocol.
- d = 32 vs d = 64 (33): halving the latent keeps the representation and every mechanism diagnostic; only in-distribution forecast sharpness drops (wake closure 0.45 to 0.21).
- Seed variance (34): the reconstructive transformer cell has large seed variance, consistent with the drift mechanism.
- Parameter observability (35): z to (G, D, Y) from the rolled-out latent; G 0.46, D 0.80, Y 0.10 on test b; negative on test c.
- SSIM convention (36): Wang convention, data range about 8.31, decoder SSIM about 0.71.
- Conditioning-only floor (37): collapses on test b, negative on five of six observables on test c.
- Model detail, encoder and predictor (38): hybrid CNN plus six-layer ViT encoder; six-layer transformer predictor, AdaLN-Zero, RoPE, causal mask, scheduled-sampling rollout, SIGReg.
- Phase-amplitude reading (39): phase-amplitude decomposition of the encounter cycle in the predictive latent; connection to sensitivity-function control for these flows.

### Likely questions and one-line answers

- "Is this a controller?" No. It is a conditional forward-closure model; making it a closed-loop controller is the next step and would put actuation where the gust now enters.
- "Why not just reconstruct better?" Reconstruction does not constrain the latent geometry to be predictable; the drift diagnostic shows the reconstructive rollout leaves its manifold.
- "Your transport result, is it solid?" Within each encounter, yes (0.63 vs 0.45); pooled across encounters it reverses, so I claim only trajectory-local transport consistency, not a global isometry.
- "Does it extrapolate?" Partially. The wake forecast survives at G = +4, but a single mid-plane slice stops being observable there, a sensing limit, not a parametric one.
- "How is the field recovered from pressure on slide 27?" A causal six-frame window of sixteen wall-pressure taps into a small MLP to the latent, then the frozen decoder; per-frame, no predictor, held out by whole encounter.
- "Cost?" The latent is two to three orders of magnitude cheaper to evolve than the field, which is what makes planning realistic.
