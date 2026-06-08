# Speaking script: An unconditioned predictive latent for extreme gust encounters

EUROMECH Colloquium, Data-driven active control in flows: from model-based to reinforcement learning.

Target: about 22 to 23 minutes spoken, leaving buffer in a 25-minute slot. Pace is roughly 150 words per minute. The model in this talk is fully unconditioned: no gust parameters enter the encoder or the predictor. Slide numbers refer to the deck `euromech_gust_jepa.pptx` (38 slides: main talk 1 to 28, backups 29 to 38).

Structure of the open: motivation (2), objective (3), prior work and the research question (4), data (5), then what a JEPA is (6 to 8), protocol (11), and results (12 onward). Do not show model results before the method.

Delivery notes:
- The JEPA primer is slides 6 and 7. The audience is mixed fluids and machine learning, so this is where you win or lose the room. Slow down.
- Slides 9 and 10 are an illustrated alternative to the box diagrams on 6 to 8. They are optional. If short on time, skip from 8 to 11.
- Four animations carry the story: slide 2 (DNS gust and lift, motivation), slide 13 (scalars tracking the rollout), slide 14 (the same rollout as decoded fields), slide 24 (state recovered from wall pressure). Let each loop once while you talk.
- The headline (12) is REPRESENTATIONAL: what the encoded latent carries, not a forecast number. The rollout is shown qualitatively (13, 14, 26); we do not headline a forward-closure R-squared.
- The two diagnostic slides (19 and 20) each define an unfamiliar metric, so slow down. The optimal-transport alignment is in the backup (slide 30).

---

## Slide 1, Title [~0:30]

Good morning, and thank you. The theme of this colloquium runs from model-based control to reinforcement learning, and almost every method on that spectrum shares one dependency: a reduced model of the flow that you can roll forward in time and actually trust. This talk is about how to learn such a model for one of the hardest cases, extreme gust encounters on a wing, and about a deceptively simple question, which reduced state is worth planning against. This is joint work with Alberto Solera-Rico, Arnau Miro and Oriol Lehmkuhl, across INTA, Carlos III, UPC and the Barcelona Supercomputing Center.

## Slide 2, The control problem (motivation) [~1:10]

Let me start with the problem itself, with no model yet, just the physics. Small and micro air vehicles increasingly fly where the gust velocity is comparable to or larger than their own flight speed: urban canyons, the wakes of buildings and ships, mountainous terrain. In that regime the gust ratio G is above one. Watch what happens on the right: a discrete vortex strikes the wing, a leading-edge vortex forms and sheds, and the lift swings strongly, here from about minus one to plus three in C_L over a fraction of a convective time, and this is only a moderate gust, G equal to minus two, from inside the training set. This is fast, large, and nonlinear, and classical gust models do not capture it. If you want to reject gusts like this actively, with a model-based controller or a reinforcement-learning agent, you need a forward model of the flow you can trust under rollout. So the question behind this talk is, what reduced state should that forward model be built on.

## Slide 3, Objective [~0:55]

Here is the objective, and the trap inside it. The load transient is built by the leading-edge vortex and the wake it leaves, so the discriminating information about what happens next lives in the wake, not in the integrated forces, lift and drag. We make that precise with a Gaussian scale split, following Motoori and Goto: low-pass the vorticity to separate the large-scale, load-bearing part, the LEV and shear layer, from the fine turbulence, and integrate its square over the wake to get a large-scale wake enstrophy. Our objective is then sharp: find a reduced state that faithfully keeps the wake, the structure reconstruction throws away, and that is recoverable from sensors. And we will compare candidate states, predictive, reconstructive and linear, under one matched protocol.

## Slide 4, The opening for JEPA (prior work and the question) [~0:55]

Why is this not already solved. Most reduced-order models for flows, from POD and DMD to nonlinear autoencoders, are trained to reconstruct the field. There is a known issue with that: reconstruction fixes the latent only up to a smooth invertible change of coordinates, so the geometry of the latent is essentially unconstrained, nothing in a reconstruction loss forces the state to keep the dynamically relevant structure. So we ask a different question, the one on the slide: which reduced state faithfully encodes the wake that reconstruction loses, so that the wake remains readable from the state and recoverable from sensors. That is the whole talk, and it is exactly the kind of state a planner or an RL agent needs.

## Slide 5, Data [~0:50]

The testbed is direct numerical simulation, with the SOD2D solver and no subgrid-scale model, of a NACA 0012 at fourteen degrees and Reynolds number five thousand. We perturb it with Taylor vortices parametrised by three numbers: the gust strength G, the core diameter D, and the wall-normal offset Y. The figure shows that envelope in three dimensions: eighty-four cases, with two held-out sets, test b for interpolation inside the cloud, and test c, the strong gusts at G equal to four, which you can see sitting well outside the training cloud, for extrapolation. From each impact-centred encounter we read five physical observables: lift, drag, the wake enstrophy, and the positive and negative circulation.

## Slide 6, What is a JEPA (1 of 2) [~1:00]

Since not everyone uses JEPAs, let me spend a minute, because it is the heart of the method. A JEPA, a joint-embedding predictive architecture, comes from Yann LeCun's world-model program. The structure is simple: an encoder maps the input to a latent, a predictor advances that latent in time, and, crucially, the loss lives entirely in latent space. The field is never reconstructed. To stop the encoder from cheating by collapsing everything to a constant, an anti-collapse regulariser replaces the decoder. Two things follow. First, the encoder is free to throw away any detail that is not predictive of the future. Second, the predictor cannot take pixel-level shortcuts, because no field-space signal ever reaches it.

## Slide 7, What is a JEPA (2 of 2): the recipe and why for control [~0:55]

In one line, the training objective is: minimise the latent prediction error, plus the anti-collapse term, and nothing else. JEPAs have been demonstrated on images and video, I-JEPA and V-JEPA, and the from-pixels variants, LeWM, LeJEPA, PLDM, but so far almost entirely on gridworlds and toy visual tasks. Why should you care for control? Because this is, by design, a learned world-model: it gives you a compact latent you can roll out cheaply, it keeps the control-relevant dynamics, and, as I will show, it is recoverable from sensors. To our knowledge this is the first end-to-end JEPA trained on a parametric fluid-dynamics problem.

## Slide 8, Our architecture for gust encounters [~1:00]

Here is our instantiation, and one design choice I want to be explicit about: the model is fully unconditioned, the gust parameters G, D and Y enter nowhere. The encoder is a hybrid convolutional and vision-transformer network: it sees only the mid-plane vorticity field and maps it to a latent of dimension thirty-two or sixty-four, a pure state map. The predictor is an autoregressive transformer with rotary positions and a causal mask, and it advances the latent from its own history alone, no gust input. It is the latent dynamics model, the object a controller would plan against or an RL agent would treat as its world-model. There is also a visualisation decoder, but it is trained afterwards on the frozen encoder, never inside the loss; it is only for looking at fields. And we compare, at matched latent dimension, against a reconstructive autoencoder in the Fukami lineage and against POD.

## Slides 9 and 10, Illustrated alternative: the two routes [optional, ~0:40 total]

If you want the picture rather than the boxes: the reconstructive route, the Fukami and Taira lineage, is an encoder, a small latent that also feeds a lift head, and then a decoder that must rebuild the whole field, so its loss is in pixel space. Our route keeps the same encoder but replaces the decoder with a predictor that advances the latent, and the loss is the mismatch between the predicted next latent and the true next latent, with the anti-collapse term in place of the decoder. Same encoder, different thing asked of the state. If time is tight, skip these.

## Slide 11, Closure protocol [~1:00]

How do we compare the states fairly. The primary measure is representational: take the latent encoded directly from the held-out field at a frame, and ask whether a fixed linear probe can read each of the six observables off it, in particular the wake. Separately, qualitatively, we roll the predictor forward to show the latent stays usable under rollout. The fairness is the important part: the probe family is identical across all three encoder families, fitted separately for each, so any difference is a property of the encoder, not the readout. And we put a floor under it: can the three gust numbers G, D and Y alone predict each observable, with a kernel-ridge fit? A latent that only matches that floor has merely re-encoded inputs the model never even sees; beating it proves the latent carries genuine flow state. Everything is reported with bootstrap intervals, three encoder seeds, and five-fold probe cross-validation, on held-out test b and test c.

## Slide 12, Main result: the latent keeps the wake [~1:20]

This is the headline, and it is about what the unconditioned state carries. We probe the encoded latent, no rollout, for each observable sixteen frames after impact. On wake enstrophy, the discriminator, the unconditioned predictive latent reaches R-squared zero point seven one, essentially matching the conditioned model at zero point seven five, while the reconstructive autoencoder and POD are far lower, around zero point zero six and below. Forces and circulations are read off cleanly too, lift at zero point eight eight, the circulations around zero point eight. So removing the gust parameters entirely costs almost nothing in what the latent encodes: the predictive objective alone builds a state that keeps the wake, the load-bearing structure reconstruction smooths away.

## Slide 13, The rollout tracks the encounter [~0:55]

Let me show the dynamics qualitatively. This is one held-out encounter, a representative case, gust strength plus one point five. The predictor is rolled forward from impact, with no gust input, and at every step we read two observables off the rolled latent with a fixed linear probe, the wake enstrophy on top and the lift below. Black is the simulation, orange the prediction from the rollout, dashed green the reconstruction from the encoded latent. Watch the orange track the black through impact and the lift dip. I am not putting a closure number on this; the point is qualitative, the rolled latent stays on the wake and tracks the encounter before drifting gently at long times.

## Slide 14, The rollout in physical space [~0:50]

And here is that same rollout as fields. We decode the rolled latent every frame with the frozen visualisation decoder: simulation on the left, the encoded-latent reconstruction in the middle, the rollout prediction on the right. At impact prediction equals reconstruction, and they diverge as the horizon grows. The leading-edge vortex and the shear layer are kept; the fine-scale turbulence is not, because at dimension sixty-four the state cannot carry it. Structural similarity of the prediction is about zero point seven five at impact and zero point six three sixteen frames out, so the rollout stays on the manifold.

## Slide 15, It is the wake, not the forces [~1:00]

Now why. If you ask the unconditioned latent how it encodes the forces, lift and drag, you find they are carried redundantly, by many coordinates. Forces are easy. The wake is different. The predictive latent carries the future wake as a distributed, collective code: the full latent forecasts it at about zero point eight four, while the single best coordinate only reaches about zero point four eight. In the reconstructive and linear latents, the best single coordinate is already as good as the whole latent, so there is no collective wake structure to find. And this clears the parameter-only floor, so it is the state doing the work, not gust parameters in disguise, which here the model never saw anyway.

## Slide 16, The wake in physical space: the Gaussian scale split [~0:45]

Here is that same statement in physical space, using the scale split from the objective slide. The top row is the large-scale vorticity sixteen frames after impact for the strongest test gust: the simulation, then the predictive and the reconstructive encode-then-decode reconstructions. The predictive reconstruction keeps the leading-edge vortex and the wake; the reconstructive one smooths it away. The bottom panel tracks the large-scale wake enstrophy through the encounter, with bands showing the spread across test b encounters. So it is the wake is not just a number; you can see the unconditioned predictive state keeping the load-bearing structure the reconstructive state loses.

## Slide 17, Latent coordinates group by physical function [~0:50]

We can go one level finer. Take the sixty-four predictive-latent coordinates and profile each by how strongly it correlates with nine physical descriptors: the gust, the forces, the wake enstrophy, the circulations, the wake thickness, the centroid. Cluster those profiles and the latent organises itself into functional groups: three wake-and-circulation blocks and one small force block. The wake groups alone forecast the future wake at about zero point seven to zero point eight, against zero point eight four for all sixty-four together, the collective code again. And this persists with no conditioning, so the organisation is intrinsic to the predictive objective. I want to be careful: this is a descriptive reading of correlations, not a causal decomposition.

## Slide 18, Controls: objective and supervision, not architecture (conditioned-model control) [~1:10]

A skeptic will say the predictive encoder differs from the reconstructive one in several ways at once. So we isolate the cause, and I will flag that this particular control was run on the conditioned model. We run a two-by-two: predictive versus reconstructive objective, crossed with a CNN versus a CNN-plus-transformer architecture, auxiliary heads matched. The predictive objective wins at both architectures, wake R-squared around zero point four six and zero point four five against zero point one six and zero point two nine, and the architecture columns do not separate. So it is the objective, not the transformer. The honest other half: remove the wake-observable head and wake closure collapses below the floor. So the result needs the predictive objective and the wake supervision together.

## Slide 19, Diagnostic 1: latent drift [~1:10]

Now two diagnostics that explain the mechanism. The first is latent drift. Why it matters: a planner queries the model at the states its own rollout reaches, so a state that walks off the data manifold is useless even if one-step error is small. We roll the unconditioned predictor forward and ask how far the rolled latent strays from the true encoded latent, relative to its own scale, as a function of horizon. The drift grows gracefully and stays bounded, and matches the conditioned model closely, so removing the gust parameters does not destabilise the rollout. A complementary optimal-transport check, in the backup, makes the same point geometrically.

## Slide 20, Diagnostic 2: topology of the encounter [~1:05]

The second diagnostic is topological. Shedding and gust encounters are cyclic, so a faithful state should trace one loop per encounter, not a tangle. We quantify that with persistent homology, which grows a distance scale over the latent trajectory and counts the loops that survive over a long range of scales, the genuine cycles rather than noise. On the encoded latents of the forty-two test b encounters, the unconditioned predictive latent has a median of one persistent loop, the reconstructive latent a median of about four, a Mann-Whitney p of about five times ten to the minus nine. So the predictive state is a single clean cycle and the reconstructive one fragments. To be precise, this is the topological count, the number of genuine loops, not a claim that the PCA picture visually closes.

## Slide 21, Decoded reconstructions [~0:35]

Pulling the families into physical space side by side: predictive, reconstructive, POD, against the simulation, across the held-out sets. The unconditioned predictive decode is blurrier pixel by pixel, but it preserves the transported large-scale wake. The reconstructive decode is sharp at the instant yet loses the wake structure. That is the trade the predictive objective makes on purpose.

## Slide 22, Sparse sensor placement [~0:35]

Now to observability, which is what makes this deployable. We place sparse wall-pressure sensors on the airfoil and select them with two target-aware criteria, TCSI and qDEIM, against a uniform baseline, at two, four, eight and sixteen taps. The map from those sensors to the latent is a kernel ridge regression, cross-validated to guard against small-sample overfitting.

## Slide 23, Flow recovered from sparse wall pressure [~0:55]

And it works. A kernel-ridge regressor with an RBF kernel maps the K wall-pressure taps over a pre-impact window to the impact-frame latent, which the frozen decoder renders as a field. Eight taps already recover the leading-edge vortex and the shear layer; two taps coarsen it but keep the gross wake. We benchmark against the best-case decode, the decode of the true simulation-encoded latent, so you see the ceiling alongside the estimate. Because the encoder is unconditional, this observability is unchanged by dropping the gust parameters: pressure-to-state R-squared peaks around zero point eight eight at eight taps. So the unconditioned state is reconstructible from a few wall sensors, a deployment-relevant result.

## Slide 24, Sparse-sensor state estimation in action [~0:50]

Here that is, animated, and note there is no predictor in this slide. This is a per-frame map from a causal window of eight wall-pressure taps to the latent, decoded to a field, frame by frame: simulation on the left with the taps marked, the field recovered from pressure alone on the right. On a held-out encounter, held out by whole encounter, the latent is recovered at R-squared about zero point seven four, structural similarity around zero point six. Wall pressure fixes the near-body leading-edge vortex and shear layer; the far wake is not observable from the surface, the honest limit. This is state estimation, the complement to the rollout.

## Slide 25, Control relevance: observable ahead of impact [~1:05]

This part speaks most directly to this audience. From a causal window of sparse wall pressure we recover the impact-frame latent at a lead time before impact: the impact state at R-squared about zero point eight eight right at impact, staying above zero point eight three out to eight frames ahead, and the impact lift with an error rising gracefully from about zero point three eight at impact to zero point six out to eight frames ahead. So the unconditioned state is observable ahead of impact from a few wall sensors, the ingredient a controller consumes.

## Slide 26, Two forecast windows: early warning, then forecast [~1:00]

Let me make the timing precise, because it is the practical message. Roll the unconditioned predictor and read each observable off the rolled latent, both before impact, anticipation, and after, forecast. Two windows emerge. The wake is predictable in a wide, roughly symmetric window of about sixteen frames each side of impact, R-squared above zero point seven; so the wake gives you early warning, the structural signature of the gust is visible well before the load. The lift is the opposite: it is hard to anticipate, because before impact the lift is flat and the onset timing is uncertain, but once impact happens the lift is strongly forecastable, R-squared around zero point eight out to twelve frames. So the wake warns early and the lift forecasts late, two complementary windows a controller can use.

## Slide 27, Conclusions [~0:55]

To bring it together. A predictive objective with wake supervision, and no gust conditioning anywhere, gives you a latent that keeps the wake, with representational wake R-squared around zero point seven one, essentially the conditioned value; that stays on its manifold under rollout; that traces a single clean cycle; and that is observable from sparse wall pressure. The thread through all of it is the wake, the thing force-only and reconstruction-only states throw away, and the model recovers it with no privileged knowledge of the gust. The rollout I showed qualitatively; I am not claiming a validated forecast or a closed-loop controller. Where it goes next: add an actuation channel, the same machinery, close the loop, push to three-dimensional observability at the strong gusts, and carry the recipe to other parametric flows.

## Slide 28, Acknowledgements [~0:15]

Thank you to my collaborators and to the funders listed here, and thank you all for your attention. I would be glad to take questions.

---

## Backups (slides 29 to 38), for questions

- Backup divider (29).
- Transport geometry, optimal transport (30): OT field distance after Tran, Yeh & Taira (JFM 2026), signed +/- vorticity split + unbalanced Sinkhorn; post-hoc within-encounter Spearman 0.61 (predictive) vs 0.45 (reconstructive); pooled across encounters it reverses, so trajectory-local only.
- Dataset and protocol (31): split v2, eighty-four cases, 226 / 42 / 24 encounters, the omega pipeline, the reporting protocol.
- d = 32 vs d = 64 (32, conditioned-model control): halving the latent keeps the representation and every mechanism diagnostic.
- Seed variance (33, conditioned-model control): the reconstructive transformer cell has large seed variance, consistent with the drift mechanism.
- Parameter observability (34): z to (G, D, Y) from the rolled-out latent; G 0.46, D 0.80, Y 0.10 on test b; negative on test c, the single-slice limit.
- SSIM convention (35): Wang convention, data range about 8.31, decoder SSIM about 0.73.
- Parameter-only floor (36): collapses on test b, negative on five of six observables on test c.
- Model detail, encoder and predictor (37): hybrid CNN plus six-layer ViT encoder; six-layer transformer predictor, unconditioned (no gust parameters), RoPE, causal mask, scheduled-sampling rollout, SIGReg anti-collapse, 20k steps.
- Phase-amplitude reading (38): phase-amplitude decomposition of the encounter cycle in the predictive latent; connection to sensitivity-function control for these flows.

### Likely questions and one-line answers

- "Is this a controller?" No. It is an unconditioned forward model of the state; making it a closed-loop controller is the next step and would add an actuation channel.
- "Doesn't dropping the gust parameters hurt?" Barely, on what matters: representational wake R-squared 0.71 unconditioned vs 0.75 conditioned, both far above the 0.06 baselines. The model recovers the wake with no privileged gust knowledge.
- "Why not just reconstruct better?" Reconstruction does not constrain the latent geometry to keep the wake; the drift diagnostic shows the reconstructive rollout leaves its manifold and the wake closure is near zero.
- "Your transport result, is it solid?" Within each encounter, yes (0.61 vs 0.45); pooled across encounters it reverses, so I claim only trajectory-local transport consistency, not a global isometry.
- "Does it extrapolate?" Partially. The wake encoding survives at G = +4, but a single mid-plane slice stops being observable there, a sensing limit, not a parametric one.
- "How early can you see the gust, and how far can you forecast?" The wake gives about sixteen frames of early warning before impact; the lift is forecastable about sixteen frames after impact. Two complementary windows.
- "Cost?" The latent is two to three orders of magnitude cheaper to evolve than the field, which is what makes planning realistic.
