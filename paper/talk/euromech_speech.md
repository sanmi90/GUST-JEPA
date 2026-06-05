# Speaking script: A predictive latent dynamics model for extreme gust encounters

EUROMECH Colloquium, Data-driven active control in flows: from model-based to reinforcement learning.

Target: about 22 to 23 minutes spoken, leaving buffer in a 25-minute slot. Pace is roughly 150 words per minute. Numbers match `HEADLINE_NUMBERS.md` and the slide bullets. Slide numbers refer to the deck `euromech_gust_jepa.pptx` (39 slides: main talk 1 to 30, backups 31 to 39).

Delivery notes:
- The JEPA primer is slides 6 and 7. The audience is mixed fluids and machine learning, so this is where you win or lose the room. Slow down.
- Slides 9 and 10 are an illustrated alternative to the box diagrams on 6 to 8. They are optional. If you are short on time, skip straight from 8 to 11.
- Three animations carry the story: slide 15 (scalars tracking the rollout), slide 16 (the same forecast as decoded fields), slide 26 (state recovered from wall pressure). Each plays in about five seconds; let it loop once while you talk, then move on.
- The three diagnostic slides (20 to 22) each define an unfamiliar metric, so slow down there.
- Pause on the headline numbers on slide 12.

---

## Slide 1, Title [~0:30]

Good morning, and thank you. The theme of this colloquium runs from model-based control to reinforcement learning, and almost every method on that spectrum shares one dependency: a reduced model of the flow that you can roll forward in time and actually trust. This talk is about how to learn such a model for one of the hardest cases, extreme gust encounters on a wing, and about a deceptively simple question, which reduced state is worth planning against. This is joint work with Alberto Solera-Rico, Arnau Miro and Oriol Lehmkuhl, across INTA, Carlos III, UPC and the Barcelona Supercomputing Center.

## Slide 2, The control problem [~1:00]

Let me start with why this matters for control. Small and micro air vehicles increasingly fly where the gust velocity is comparable to or larger than their own flight speed: urban canyons, the wakes of buildings and ships, mountainous terrain. In that regime the gust ratio G is above one, and a discrete vortex hitting the wing produces load transients that are large, fast, and not captured by classical gust models. If you want to reject those gusts actively, with a model-based controller or a reinforcement-learning agent, you need a forward model of the flow: something that takes the current state and tells you where it is going, fast enough and faithfully enough to plan against. So the question behind this talk is, what reduced state should that forward model be built on.

## Slide 3, Why it is hard: it is the wake, not just the forces [~1:00]

Here is why this is genuinely hard. The load transient in these encounters is built by a leading-edge vortex: it forms, grows, and sheds, and it leaves a wake behind it. The discriminating information about what happens next lives in that wake, not only in the integrated forces, lift and drag. That is the trap. You can build a reduced model that reproduces the lift curve beautifully and still throw away the wake. Instantaneously it looks fine, but the moment you roll it forward to plan, it fails, because the state it carries does not contain the structure that determines the next instant. Keep that phrase in mind: it is the wake, not just the forces.

## Slide 4, The opening for JEPA [~0:50]

Most reduced-order models for flows, from POD and DMD to nonlinear autoencoders, are trained to reconstruct the field. And there is a known issue with that: reconstruction fixes the latent only up to a smooth invertible change of coordinates, so the geometry of the latent is essentially unconstrained. Nothing in a reconstruction loss forces the state to be predictable. So we ask a different question. Instead of a state that reconstructs the field, what about a state that is predictive by construction, and that we then plan against? Concretely: which reduced state stays physically closed when you propagate it forward, meaning every observable you care about can still be read off along the rollout. That property, forward closure, is the whole talk.

## Slide 5, Data [~0:50]

The testbed is direct numerical simulation, with the SOD2D solver and no subgrid-scale model, of a NACA 0012 at fourteen degrees and Reynolds number five thousand. We perturb it with Taylor vortices parametrised by three numbers: the gust strength G, the core diameter D, and the wall-normal offset Y, where the vortex passes relative to the wing. That gives eighty-four cases. Each encounter is centred on the impact instant, and from each we extract six physical observables: lift, drag, the impulse, the wake enstrophy, and the positive and negative circulation. We hold out two sets: test b for interpolation inside the envelope, and test c, the strong gusts at G equal to four, for extrapolation.

## Slide 6, What is a JEPA (1 of 2) [~1:00]

Since not everyone uses JEPAs, let me spend a minute, because it is the heart of the method. A JEPA, a joint-embedding predictive architecture, comes from Yann LeCun's world-model program. The structure is simple: an encoder maps the input to a latent, a predictor advances that latent in time, and, crucially, the loss lives entirely in latent space. The field is never reconstructed. To stop the encoder from cheating by collapsing everything to a constant, an anti-collapse regulariser replaces the decoder. Two things follow. First, the encoder is free to throw away any detail that is not predictive of the future. Second, the predictor cannot take pixel-level shortcuts, because no field-space signal ever reaches it.

## Slide 7, What is a JEPA (2 of 2): the recipe and why for control [~0:55]

In one line, the training objective is: minimise the latent prediction error, plus the anti-collapse term, and nothing else. JEPAs have been demonstrated on images and video, I-JEPA and V-JEPA, and the from-pixels variants, LeWM, LeJEPA, PLDM, but so far almost entirely on gridworlds and toy visual tasks. Why should you care for control? Because this is, by design, a learned world-model: it gives you a compact latent you can roll out cheaply, it keeps the control-relevant dynamics, and, as I will show, it is recoverable from sensors. To our knowledge this is the first end-to-end JEPA trained on a parametric fluid-dynamics problem.

## Slide 8, Our architecture for gust encounters [~1:00]

Here is our instantiation. The encoder is a hybrid convolutional and vision-transformer network, and it is unconditional: it sees only the mid-plane vorticity field and maps it to a latent of dimension thirty-two or sixty-four. The predictor is an autoregressive transformer, and this is where the gust parameters enter, only the predictor, through adaptive layer norm, with rotary positions and a causal mask. That split is deliberate: the encoder is a pure state map, and the dynamics is where the conditioning lives. The predictor is the latent dynamics model, the object a controller would plan against or an RL agent would treat as its world-model. There is also a visualisation decoder, but it is trained afterwards on the frozen encoder, never inside the loss; it is only for looking at fields. And we compare, at matched latent dimension, against a reconstructive autoencoder in the Fukami lineage and against POD.

## Slides 9 and 10, Illustrated alternative: the two routes [optional, ~0:40 total]

If you want the picture rather than the boxes: the reconstructive route, the Fukami and Taira lineage, is an encoder, a small latent that also feeds a lift head, and then a decoder that must rebuild the whole field, so its loss is in pixel space. Our route keeps the same encoder but replaces the decoder with a predictor that advances the latent, and the loss is the mismatch between the predicted next latent and the true next latent, with the anti-collapse term in place of the decoder. Same encoder, different thing asked of the state. If time is tight, skip these and go to the protocol.

## Slide 11, Forward-closure protocol [~1:00]

How do we measure forward closure fairly. We take the latent at impact, roll the predictor recursively out to sixteen frames after impact, and at each step we probe the six observables from the predicted latent. The fairness is the important part: the predictor architecture and the probe family are identical across all three encoder families, trained and fitted separately for each, so any difference in closure is a property of the encoder, not of the dynamics model or the readout. And we put a floor under it: a kernel-ridge regressor from the gust parameters alone to each observable. The latent has to beat that floor, otherwise we would just be rediscovering the parameters, not the state. Everything is reported with bootstrap intervals, three encoder seeds, and five-fold cross-validation on the probes, on held-out test b and test c.

## Slide 12, Main result: forward closure [~1:20]

This is the headline. Averaged over the six observables, the forward-closure R-squared from the rolled-out latent is zero point eight four for the predictive JEPA, against zero point four three for the reconstructive autoencoder and zero point five six for POD. And the discriminator is exactly where I told you it would be: the wake. On wake enstrophy the predictive latent reaches zero point nine three, while the reconstructive and linear baselines are at zero point two eight and zero point three seven. In error terms, the predictive model's wake-enstrophy error at the horizon is about two and a half times lower than the autoencoder and three times lower than POD. POD, to be fair, stays competitive on the integrated impulse, a smooth low-rank quantity. But on the wake, the structure that governs the transient, only the predictive state stays closed.

## Slide 13, Forward closure at H = 8 [~0:30]

A quick robustness check: the same comparison rolled out to eight frames instead of sixteen. The family ordering is unchanged, the predictive latent is still lowest on the wake and closest to the floor. The advantage is not an artefact of one horizon.

## Slide 14, Forward closure at H = 4 [~0:30]

And four frames out, the same picture, with the errors smaller as you would expect closer to impact. The wake stays the discriminating observable at every horizon we test, not only at sixteen.

## Slide 15, Forward closure in action: the rollout tracks the encounter [~0:55]

Let me make that concrete. This is one held-out encounter, a representative low-error case, not the hardest one. The predictor is rolled forward from impact, and at every step we read two of the observables off the rolled latent with a fixed linear probe: the wake enstrophy on top, the lift below. Black is the simulation, orange is the prediction from the rollout, and the dashed green is the reconstruction from the encoded latent, which is the best the representation could do. Watch the orange track the black through the impact and the lift dip. It holds through the closure horizon and only drifts gently at long times, which is honest and expected.

## Slide 16, The same forecast in physical space [~0:50]

And here is the same forecast as fields, because scalars only get you so far with a fluids audience. We decode the rolled latent every frame with the frozen visualisation decoder: simulation on the left, the encoded-latent reconstruction in the middle, the prediction from the rollout on the right. The leading-edge vortex and the shear layer are kept; the fine-scale wake turbulence is not, because at dimension sixty-four the state cannot carry it, and it does not need to for closure. Structural similarity of the prediction is about zero point seven seven at impact and zero point five three at the horizon. The prediction tracks the reconstruction, which tells you the rollout stays on the manifold.

## Slide 17, It is the wake, not the forces [~1:00]

Let me unpack why. If you ask each family how it encodes the forces, lift and drag, you find they are carried redundantly, by many coordinates, in every family. Forces are easy. The wake is different. Only the predictive latent carries the future wake as a distributed, collective code: the full latent forecasts it at rank correlation zero point eight three, while the single best coordinate only reaches zero point four four. In the reconstructive and linear latents, the best single coordinate is already as good as the whole latent, which means there is no collective wake structure to find. And this clears the conditioning-only floor, so it is the state doing the work, not the gust parameters in disguise.

## Slide 18, Latent coordinates group by physical function [~0:50]

We can go one level finer. Take the sixty-four predictive-latent coordinates and profile each one by how strongly it correlates with each of nine physical descriptors: the gust, the forces, the wake enstrophy, the circulations, the wake thickness, the centroid. Cluster those profiles and the latent organises itself into functional groups: about fifty-one coordinates form a wake-vorticity block, eleven form a gust-forcing block, and two are essentially silent. On their own, the wake groups forecast the future wake at about zero point seven, the forcing group at zero point four five, against zero point eight three for all sixty-four together, which is the collective code again. I want to be careful: this is a descriptive reading of correlations, not a causal decomposition.

## Slide 19, Controls: objective and supervision, not architecture [~1:10]

A skeptic will say the predictive encoder differs from the reconstructive one in three ways at once: the objective, the architecture, and the auxiliary wake supervision. So before any mechanism, we isolate the cause. We run a two-by-two: predictive versus reconstructive objective, crossed with a CNN versus a CNN-plus-transformer architecture, with the auxiliary heads matched in all four cells. The predictive objective wins at both architectures, wake R-squared around zero point four six and zero point four five, against zero point one six and zero point two nine, and the two architecture columns do not separate. So it is the objective, not the transformer. The honest other half: if you remove the wake-observable head from the predictive model, wake closure collapses below the floor, to minus one. So the result needs the predictive objective and the wake supervision together.

## Slide 20, Diagnostic 1: latent drift [~1:15]

Now three diagnostics that explain the mechanism, each with a metric you may not see every day. The first is latent drift. Why it matters: a planner or an RL agent does not query the model at nice training states, it queries it at the states its own rollout reaches, so one-step error is not enough. To measure it we use the Mahalanobis distance, a covariance-aware distance from a reference cloud; a value around one means one standard deviation away. We take the rolled-out latent and ask how far it sits from the distribution of latents encoded directly from the simulation, as a ratio. The result is striking: the reconstructive rollout drifts about ten times further out than its own encoded states, it leaves the manifold, whereas the predictive and the linear rollouts stay inside, ratios around zero point eight five. So the reconstructive failure is not a probe failure, it is the rollout walking off into a region where nothing is valid.

## Slide 21, Diagnostic 2: topology of the encounter [~1:00]

The second diagnostic is topological. Shedding and gust encounters are cyclic, so a faithful dynamics model should keep the trajectory as a single closed loop as you roll it out. We quantify that with persistent homology, which counts topological loops and tracks how long each persists as you grow a scale parameter. One long-lived loop means a clean cycle; many short-lived ones mean the trajectory has fragmented. The predictive latent gives a single persistent cycle. The reconstructive latent fragments. So the predictive state is a coherent object to integrate forward, and the reconstructive one is not.

## Slide 22, Diagnostic 3: transport geometry [~1:10]

The third diagnostic asks whether motion in the latent corresponds to physically meaningful motion of the flow. A Euclidean or pixel distance between two vorticity fields ignores where the structures actually move, but control cares exactly about advection of the leading-edge vortex and the shear layer. So we use an optimal-transport distance, the least cost to rearrange one vorticity field into another. Then we correlate, per encounter, the latent distance matrix with the optimal-transport distance matrix, with a rank correlation. The predictive latent reaches zero point six three, against zero point four five for the reconstructive one. To be precise, this is order-preservation of transport distances, not a metric isometry. But it means a step in the predictive latent corresponds to a physically sensible change in the field, which is what you want from a dynamics model.

## Slide 23, Decoded reconstructions [~0:40]

Pulling the three families into physical space side by side: predictive, reconstructive, POD, against the simulation, across the held-out sets. The predictive decode is blurrier pixel by pixel, but it preserves the transported large-scale wake. The reconstructive decode is sharp at the instant yet drifts under rollout. That is the trade the predictive objective makes on purpose: it gives up some instantaneous sharpness to keep the structure that survives forward in time.

## Slide 24, Sparse sensor placement [~0:35]

Now to observability, which is what makes this deployable. We place sparse wall-pressure sensors on the airfoil and select them with two target-aware criteria, TCSI and qDEIM, against a uniform baseline. The map from those sensors to the latent is a kernel ridge regression, cross-validated to guard against small-sample overfitting, at two, four, eight and sixteen taps.

## Slide 25, Flow recovered from sparse wall pressure [~0:55]

And it works. We estimate the impact-frame predictive latent from K wall-pressure taps and decode it to a field. Eight taps already recover the leading-edge vortex and the shear layer; two taps coarsen it but keep the gross wake structure. We benchmark against the oracle decode, the decode of the simulation-encoded latent, so you see the ceiling alongside the estimate. So the predictive state is reconstructible from a few wall sensors, a deployment-relevant observability result.

## Slide 26, Sparse-sensor state estimation in action [~0:50]

Here that is, animated, and note there is no predictor in this slide. This is a per-frame map from sixteen wall-pressure taps straight to the latent, decoded to a field, frame by frame: simulation on the left with the taps marked, the field recovered from pressure alone on the right. On a held-out encounter the latent is recovered at R-squared about zero point seven two, structural similarity around zero point five. Wall pressure fixes the near-body leading-edge vortex and shear layer; the far wake is simply not observable from the surface, which is the honest limit. This is state estimation, not forecasting, the complement to the rollout.

## Slide 27, Control relevance: observable and forecastable [~1:15]

This is the part that speaks most directly to this audience, and it puts the two pieces together. First, the predictive state is observable from sparse wall pressure: from the latent we recover the gust strength and core diameter on held-out cases, R-squared around zero point four six and zero point eight. Second, the figure on the right, we can predict the impact lift ahead of time. If you roll the latent forward and then probe, the predictor-in-loop gives R-squared zero point three five a full ten frames before impact, where reading the pressure sensors directly gives essentially nothing, zero point one three; the oracle ceiling is zero point six eight. And the reconstructive latent's own oracle is negative, so it never had the pre-impact information; a representation failure, not a probe failure. So you get both ingredients a controller consumes: a state estimate from cheap sensors, and a short-horizon forecast.

## Slide 28, Takeaways [~0:55]

To bring it together. A predictive objective, trained with wake supervision, gives you a latent dynamics model that is forward-closed, with wake R-squared around zero point seven five at the representation level and zero point eight four in the mean; that stays on its manifold under rollout; that is a single clean cycle; and whose geometry is aligned with the physical transport of the flow. It is also compact, dimension thirty-two is as good as sixty-four with a participation ratio around one point seven, and it is observable from sparse wall pressure. In short, it is a conditional forward-closure model: a substrate you can build model-based control or an RL world-model on. And the thread through all of it is the wake, the thing that force-only and reconstruction-only states throw away.

## Slide 29, Outlook [~0:40]

Where this goes next. Because the gust enters only as a conditioning channel, replacing it with an actuation channel is a change of input, not of architecture, so the same forward-closure machinery applies to control inputs. From there, closed-loop control or an RL world-model on the latent. The strong-gust case at G equal to four also tells us where a single mid-plane slice stops being observable, which points to three-dimensional sensing. And the recipe, judge a reduced state by forward closure rather than reconstruction, is not specific to this flow.

## Slide 30, Acknowledgements [~0:15]

Thank you to my collaborators and to the funders listed here, and thank you all for your attention. I would be glad to take questions.

---

## Backups (slides 31 to 39), for questions

- Backup divider (31).
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
- "Does it extrapolate?" Partially. The wake forecast survives at G = +4, but a single mid-plane slice stops being observable there, which is a sensing limit, not a parametric one.
- "How is the field recovered from pressure on slide 26?" A causal six-frame window of sixteen wall-pressure taps into a small MLP to the latent, then the frozen decoder; per-frame, no predictor. Held out by whole encounter.
- "Cost?" The latent is two to three orders of magnitude cheaper to evolve than the field, which is what makes planning realistic.
