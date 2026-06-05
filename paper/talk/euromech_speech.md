# Speaking script: A predictive latent dynamics model for extreme gust encounters

EUROMECH Colloquium, Data-driven active control in flows: from model-based to reinforcement learning.

Target: about 22 to 23 minutes spoken, leaving buffer in a 25-minute slot. Pace is roughly 150 words per minute. Numbers match `HEADLINE_NUMBERS.md` and the slide bullets. Slide numbers refer to the deck `euromech_gust_jepa.pptx`. Backups (slides 20 to 29) are for questions.

Delivery notes: the three "diagnostic" slides (12 to 14) define an unfamiliar metric each, so slow down there. The audience is mixed fluids and machine learning, so the JEPA primer (slide 6) is where you win or lose the room. Pause on the headline numbers.

---

## Slide 1, Title [~30 s]

Good morning, and thank you. The theme of this colloquium runs from model-based control to reinforcement learning, and almost every method on that spectrum shares one dependency: a reduced model of the flow that you can roll forward in time and actually trust. This talk is about how to learn such a model for one of the hardest cases, extreme gust encounters on a wing, and about a deceptively simple question, which reduced state is worth planning against. This is joint work with Alberto Solera-Rico, Arnau Miro and Oriol Lehmkuhl, across INTA, Carlos III, UPC and the Barcelona Supercomputing Center.

## Slide 2, The control problem [~1:15]

Let me start with why this matters for control. Small and micro air vehicles increasingly fly where the gust velocity is comparable to or larger than their own flight speed: urban canyons, the wakes of buildings and ships, mountainous terrain. In that regime the gust ratio G is above one, and a discrete vortex hitting the wing produces load transients that are large, fast, and not captured by classical gust models. If you want to reject those gusts actively, with a model-based controller or a reinforcement-learning agent, you need a forward model of the flow: something that takes the current state and tells you where it is going, fast enough and faithfully enough to plan against. So the question behind this talk is, what reduced state should that forward model be built on. Everything that follows is an attempt to answer that with evidence rather than intuition.

## Slide 3, Why it is hard [~1:15]

Here is why this is genuinely hard. The load transient in these encounters is built by a leading-edge vortex: it forms, grows, and sheds, and it leaves a wake behind it. The discriminating information about what happens next lives in that wake, not only in the integrated forces, lift and drag. That is the trap. You can build a reduced model that reproduces the lift curve beautifully and still throws away the wake. Instantaneously it looks fine, but the moment you roll it forward to plan, it fails, because the state it carries does not contain the structure that determines the next instant. At fourteen degrees angle of attack and Reynolds number five thousand the flow is massively separated and couples nonlinearly with the natural shedding, so this is not a small perturbation problem. Keep that phrase in mind: it is the wake, not just the forces.

## Slide 4, The opening for JEPA [~1:00]

Now, most reduced-order models for flows, from POD and DMD to nonlinear autoencoders, are trained to reconstruct the field. And there is a known issue with that: reconstruction fixes the latent only up to a smooth invertible change of coordinates, so the geometry of the latent is essentially unconstrained. Nothing in a reconstruction loss forces the state to be predictable. So we ask a different question. Instead of a state that reconstructs the field, what about a state that is predictive by construction, and that we then plan against? Concretely: which reduced state stays physically closed when you propagate it forward, meaning every observable you care about can still be read off along the rollout. That property, forward closure, is the whole talk.

## Slide 5, Data [~1:00]

The testbed is direct numerical simulation, with the SOD2D solver and no subgrid-scale model, of a NACA 0012 at fourteen degrees and Reynolds number five thousand. We perturb it with Taylor vortices parametrised by three numbers: the gust strength G, the core diameter D, and the wall-normal offset Y, where the vortex passes relative to the wing. That gives eighty-four cases. Each encounter is centred on the impact instant, and from each we extract six physical observables: lift, drag, the impulse, the wake enstrophy, and the positive and negative circulation. We hold out two sets: test b for interpolation inside the envelope, and test c, the strong gusts at G equal to four, for extrapolation.

## Slide 6, What is a JEPA [~1:45]

Since not everyone uses JEPAs, let me spend a minute here, because it is the heart of the method. A JEPA, a joint-embedding predictive architecture, comes from Yann LeCun's world-model program. The structure is: an encoder maps the input to a latent, a predictor advances that latent in time, and, crucially, the loss lives entirely in latent space. The field is never reconstructed. To stop the encoder from cheating by collapsing everything to a constant, an anti-collapse regulariser replaces the decoder. Two things follow from that design. First, the encoder is free to throw away any detail that is not predictive of the future. Second, the predictor cannot take pixel-level shortcuts, because no field-space signal ever reaches it. JEPAs have been demonstrated on images and video, I-JEPA and V-JEPA, and the from-pixels variants, LeWM, LeJEPA, PLDM, but so far almost entirely on gridworlds and toy visual tasks. Why should you care for control? Because this is, by design, a learned world-model: it gives you a compact latent you can roll out cheaply, it keeps the control-relevant dynamics, and, as I will show, it is recoverable from sensors. To our knowledge this is the first end-to-end JEPA trained on a parametric fluid-dynamics problem.

## Slide 7, Our architecture [~1:15]

Here is our instantiation. The encoder is a hybrid convolutional and vision-transformer network, and it is unconditional: it sees only the mid-plane vorticity field and maps it to a latent of dimension thirty-two or sixty-four. The predictor is an autoregressive transformer, and this is where the gust parameters enter, only the predictor, through adaptive layer norm, with rotary positions and a causal mask. That split is deliberate: the encoder is a pure state map, and the dynamics is where the conditioning lives. The point I want you to take away is that the predictor is the latent dynamics model, the object a controller would plan against or an RL agent would treat as its world-model. There is also a visualisation decoder, but it is trained afterwards on the frozen encoder, never inside the loss; it is only for looking at fields. And we compare, at matched latent dimension, against a reconstructive autoencoder in the Fukami lineage and against POD.

## Slide 8, Forward-closure protocol [~1:15]

How do we measure forward closure fairly. We take the latent at impact, roll the predictor recursively out to sixteen frames after impact, and at each step we probe the six observables from the predicted latent. The fairness is the important part: the predictor architecture and the probe family are identical across all three encoder families, trained and fitted separately for each, so any difference in closure is a property of the encoder, not of the dynamics model or the readout. And we put a floor under it: a kernel-ridge regressor from the gust parameters G, D, Y alone to each observable. The latent has to beat that floor, otherwise we would just be rediscovering the parameters, not the state. Everything is reported with bootstrap confidence intervals, three encoder seeds, and five-fold cross-validation on the probes, on held-out test b and test c.

## Slide 9, Main result [~1:30]

This is the headline. Averaged over the six observables, the forward-closure R-squared from the rolled-out latent is zero point eight four for the predictive JEPA, against zero point four three for the reconstructive autoencoder and zero point five six for POD. And the discriminator is exactly where I told you it would be: the wake. On wake enstrophy the predictive latent reaches zero point nine three, while the reconstructive and linear baselines are at zero point two eight and zero point three seven. In terms of error, the predictive model's wake-enstrophy error at the horizon is about two and a half times lower than the autoencoder and three times lower than POD. POD, to be fair, stays competitive on the integrated impulse, which is a smooth, low-rank quantity. But on the wake, the structure that governs the transient, only the predictive state stays closed.

## Slide 10, It is the wake, not the forces [~1:15]

Let me unpack why. If you ask each family how it encodes the forces, lift and drag, you find they are carried redundantly, by many coordinates, in every family. Forces are easy. The wake is different. Only the predictive latent carries the future wake as a distributed, collective code: the full latent forecasts it at rank correlation zero point eight three, while the single best coordinate only reaches zero point four four. In the reconstructive and linear latents, the best single coordinate is already as good as the whole latent, which means there is no collective wake structure to find. And this clears the conditioning-only floor, so it is the state doing the work, not the gust parameters in disguise.

## Slide 11, Controls [~1:15]

A skeptic will say: the predictive encoder differs from the reconstructive one in three ways at once, the objective, the architecture, and the auxiliary wake supervision. So before any mechanism, we isolate the cause. We run a two-by-two: predictive versus reconstructive objective, crossed with a CNN versus a CNN-plus-transformer architecture, with the auxiliary heads matched in all four cells. The predictive objective wins at both architectures, wake R-squared around zero point four six and zero point four five, against zero point one six and zero point two nine, and the two architecture columns do not separate. So it is the objective, not the transformer. The honest other half: if you remove the wake-observable head from the predictive model, wake closure collapses below the floor, to minus one. So the result needs the predictive objective and the wake supervision together. That is the claim, stated precisely.

## Slide 12, Diagnostic 1, latent drift [~1:30]

Now three diagnostics that explain the mechanism, and each uses a metric you may not see every day, so let me define them. The first is latent drift. The reason it matters: a planner or an RL agent does not query the model at nice training states, it queries it at the states its own rollout reaches, so one-step error is not enough. To measure it we use the Mahalanobis distance, which is just a covariance-aware distance from a reference cloud of points; a value of about one means one standard deviation away. We take the rolled-out latent and ask how far it sits from the distribution of latents encoded directly from the DNS, as a ratio. The result is striking: the reconstructive rollout drifts about ten times further out than its own encoded states, it leaves the manifold, whereas the predictive and the linear rollouts stay inside, ratios around zero point eight five. So the reconstructive failure is not a probe failure, it is the rollout walking off into a region where nothing is valid.

## Slide 13, Diagnostic 2, topology [~1:15]

The second diagnostic is topological, and again let me define the tool. Shedding and gust encounters are cyclic, so a faithful dynamics model should keep the trajectory as a single closed loop as you roll it out. We quantify that with persistent homology, which counts topological loops, one-cycles, and tracks how long each persists as you grow a scale parameter. One long-lived loop means a clean cycle; many short-lived ones mean the trajectory has fragmented. The predictive latent gives a single persistent cycle. The reconstructive latent fragments. So the predictive state is a coherent object to integrate forward, and the reconstructive one is not.

## Slide 14, Diagnostic 3, transport geometry [~1:15]

The third diagnostic asks whether motion in the latent corresponds to physically meaningful motion of the flow. A Euclidean or pixel distance between two vorticity fields ignores where the structures actually move, but control cares exactly about advection of the leading-edge vortex and the shear layer. So we use an optimal-transport distance, the least cost to rearrange one vorticity field into another, which is sensitive to transport. Then we correlate, per encounter, the latent distance matrix with the optimal-transport distance matrix, using a rank correlation. The predictive latent reaches zero point six three, against zero point four five for the reconstructive one. I want to be precise: this is order-preservation of transport distances, not a metric isometry. But it means that a step in the predictive latent corresponds to a physically sensible change in the field, which is exactly what you want from a dynamics model.

## Slide 15, Physical space [~1:00]

To make this concrete in physical space, we decode the latents back to fields, using that frozen-encoder decoder, purely as a diagnostic. The large-scale wake and leading-edge vortex of the predictive decode track the simulation closely, Pearson correlation about zero point eight nine at impact and zero point nine one at the horizon. The reconstructive decode is actually sharper pixel by pixel, but it loses the transported wake under rollout. That is the trade the predictive objective makes on purpose: it gives up some instantaneous sharpness to keep the structure that survives forward in time.

## Slide 16, Control relevance [~1:30]

This is the part that speaks most directly to this audience. First, the predictive state is observable from sparse wall pressure: from the latent we can recover the gust strength and the core diameter on held-out cases, R-squared around zero point four six and zero point eight. Second, and this is the figure on the right, we can predict the impact lift ahead of time. If you roll the latent forward and then probe, the predictor-in-loop gives R-squared zero point three five a full ten frames before impact, where reading the pressure sensors directly gives essentially nothing, zero point one three; the oracle ceiling is zero point six eight. And note the reconstructive latent's own oracle is negative, so it never had the pre-impact information; that is a representation failure, not a probe failure. So you get both ingredients a controller consumes: a state estimate from cheap sensors, and a short-horizon forecast.

## Slide 17, Takeaways [~1:00]

So, to bring it together. A predictive objective, trained with wake supervision, gives you a latent dynamics model that is forward-closed, with wake R-squared around zero point seven five at the representation level and zero point eight four in the mean; that stays on its manifold under rollout; that is a single clean cycle; and whose geometry is aligned with the physical transport of the flow. It is also compact, dimension thirty-two is as good as sixty-four with a participation ratio around one point seven, and it is observable from sparse wall pressure. In short, it is a conditional forward-closure model: a substrate you can build model-based control or an RL world-model on. And the thread through all of it is the wake, the thing that force-only and reconstruction-only states throw away.

## Slide 18, Outlook [~0:45]

Where this goes next. Because the gust enters only as a conditioning channel, replacing it with an actuation channel is a change of input, not of architecture, so the same forward-closure machinery applies to control inputs. From there, closed-loop control or an RL world-model on the latent. The strong-gust case at G equal to four also tells us where a single mid-plane slice stops being observable, which points to three-dimensional sensing. And the recipe, judge a reduced state by forward closure rather than reconstruction, is not specific to this flow. Thank you, I would be glad to take questions.

---

## Backups (slides 20 to 29), for questions

- Dataset and protocol (20): split v2, eighty-four cases, 226 / 42 / 24 encounters, the omega pipeline, the reporting protocol.
- d = 32 vs d = 64 (21): halving the latent keeps the representation and every mechanism diagnostic; only in-distribution forecast sharpness drops (wake closure 0.45 to 0.21).
- Seed variance (22): the reconstructive transformer cell has large seed variance, consistent with the drift mechanism.
- Reconstructions (23): decoded fields, predictive vs reconstructive vs POD vs DNS.
- Sensor placement (24): TCSI / qDEIM sparse placement; the pressure-to-latent map is kernel ridge, cross-validated.
- Parameter observability (25): z to (G, D, Y); G 0.46, D 0.80, Y 0.10 on test b; negative on test c.
- SSIM convention (26): Wang convention, data range about 8.31, decoder SSIM about 0.71.
- Conditioning-only floor (27): collapses on test b, negative on five of six observables on test c.
- Predictor detail (28): six-layer transformer, AdaLN-Zero, RoPE, causal mask, scheduled-sampling rollout, SIGReg.
- Phase-amplitude (29): connection to sensitivity-function control for these flows.

### Likely questions and one-line answers

- "Is this a controller?" No. It is a conditional forward-closure model; making it a closed-loop controller is the next step and would put actuation where the gust now enters.
- "Why not just reconstruct better?" Reconstruction does not constrain the latent geometry to be predictable; the drift diagnostic shows the reconstructive rollout leaves its manifold.
- "Does it extrapolate?" Partially. The wake forecast survives at G = +4, but a single mid-plane slice stops being observable there, which is a sensing limit, not a parametric one.
- "Cost?" The latent is two to three orders of magnitude cheaper to evolve than the field, which is what makes planning realistic.
