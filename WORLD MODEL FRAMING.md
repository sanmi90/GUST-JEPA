# The world-model framing: gust as intervention

This is the conceptual hook you asked about, worked out carefully against the
two LeCun slides and against what your architecture actually is. It is strong,
and it is defensible, but only if stated with two specific caveats. I give the
correspondence first, then the paper-ready prose, then the caveats, then where
it should live in the manuscript.

## 1. The correspondence is exact, term for term

LeCun’s “action-conditioned world model” slide (IMG_2229) draws:

```
            Enc(x) --> s_x --> Pred(s_x) --> s_tilde_y
                                  ^               |
                                  |               v
                                  a          D(s_y, s_tilde_y)
                                                  ^
            Enc(y) --> s_y --------------------- |
```

with the captions “abstract representation of the state of the system”,
“action / intervention”, “prediction of the outcome in representation space”,
and “the model does not predict every detail of the outcome but performs
predictions in an abstract representation space”.

Your architecture, as locked in CLAUDE.md (D6, D16, D37) and reported in D124
and D129/131, is:

|LeCun slide                        |Your model                                             |Evidence                                                    |
|-----------------------------------|-------------------------------------------------------|------------------------------------------------------------|
|`x` (observation of the system)    |mid-plane vorticity field `omega_z(192,96)` at time `t`|CLAUDE.md encoder input                                     |
|`Enc(x)`                           |unconditional hybrid CNN+ViT encoder `E_theta`         |D3, D6; encoder never sees `c`                              |
|`s_x` (abstract state)             |latent `z_t in R^d`, `d in {32,64}`                    |D4, D95                                                     |
|`a` (action / intervention)        |gust parameters `c = (G, D, Y)`                        |D16, D37; `c` enters ONLY the predictor                     |
|`Pred(s_x)` conditioned on `a`     |autoregressive transformer with AdaLN-Zero on `c`      |D16; AdaLN-Zero is literally how `a` modulates the predictor|
|`s_tilde_y` (predicted outcome)    |rolled-out latent `z_hat_{t+H}`                        |D101, D124                                                  |
|`Enc(y)`                           |same encoder applied to the future field               |shared-weight target embedding                              |
|`s_y` (true outcome in repr. space)|`z_{t+H}` encoded from the DNS future field            |D124                                                        |
|`D(s_y, s_tilde_y)`                |latent-space prediction loss (no field reconstruction) |D1, “Things to NOT do”: no recon loss in JEPA objective     |

The single most important line on the slide, “the model does not predict every
detail of the outcome but performs predictions in an abstract representation
space”, is exactly the design decision in your “Things to NOT do” list: do not
add reconstruction loss to the JEPA encoder objective. You did not adopt this to
imitate a slide; you adopted it for the reasons in D1, and it happens to be the
defining property LeCun uses to separate world models from the alternatives.

## 2. The gust is an intervention in the precise (causal) sense

The reason the analogy is more than a loose metaphor: a vortex gust is literally
a `do()` operation on the flow. You take the post-stall limit-cycle wake and you
intervene on it by releasing a Taylor vortex of strength `G`, size `D`, at
offset `Y`. The encounter is the outcome of that intervention. LeCun’s slide
labels the channel “Action, Intervention” with a circle `a`; your `c = (G,D,Y)`
is the parametrisation of that intervention. The slide’s own example system is a
jet engine, an aerodynamic system; your system is an airfoil in a gust. The
mapping is not strained.

This also resolves cleanly into the control story you already have. Right now
`a = c` is an exogenous intervention (the gust, which you observe but do not
choose). At deployment, an actuation channel (a synthetic jet, a flap deflection,
a Gurney flap schedule) is a second intervention you do choose, and it enters at
exactly the same place in the diagram, the predictor conditioning, with no change
to the encoder or the loss. So the world-model framing is what licenses the claim
in your Discussion that “replacing the parametric conditioning with an actuation
channel is a change of input rather than of architecture”. The predictor is
already an action-conditioned world model; today the action is the disturbance,
tomorrow it is the control.

## 3. The “what world models should NOT be” slide gives you your baselines

The first slide (IMG_2230) lists what world models should not be: world
simulators, digital twins, generative models, video-generation systems. This is
not decoration; it maps onto your baseline taxonomy and explains the result.

The reconstructive autoencoder (Fukami) is trained to reproduce the field. In
LeCun’s terms it is being pushed toward the “world simulator / generative model”
end, spending capacity predicting every detail. POD is a linear “digital twin”
basis: an energy-optimal reconstruction of the field with no notion of what is
predictive. The slide’s thesis is that neither is the right object for forward
prediction, because forcing the model to reconstruct every detail of the outcome
is the wrong objective. Your D131(6) drift result is the empirical version of
that thesis: the reconstructive latent leaves its training manifold by 9.9x under
rollout, precisely because its objective never required the latent geometry to be
forward-predictable. So the LeCun taxonomy is not just a framing, it is a
prediction about which baselines should fail and why, and your data confirm it.

## 4. Paper-ready prose (drop-in, no emdashes)

A short version for the Introduction, after the JEPA paragraph:

> This places the present model within the action-conditioned world-model
> framework articulated by LeCun (2022): a system is observed through an encoder
> into an abstract state, an intervention is applied, and a predictor forecasts
> the outcome of that intervention in the representation space rather than in the
> space of the raw observation. The defining property of such a model is that it
> does not reconstruct every detail of the outcome; it predicts only in the
> abstract space. The vortex gust is the intervention. A discrete vortex of
> strength G, size D, and offset Y is released into the post-stall wake, and the
> encounter is the outcome of that intervention; the encoder maps the vorticity
> field to an unconditional latent state, and the predictor advances that state
> conditioned on the gust parameters. The reconstructive and linear baselines we
> compare against are, in this framing, the objects the world-model view argues
> against: a reconstruction-trained autoencoder is pushed toward reproducing
> every detail of the field, and a proper orthogonal decomposition is an
> energy-optimal reconstruction basis, neither of which is constrained to make
> the latent state forward-predictable. The drift mechanism of
> Section [results-drift] is the empirical consequence.

A single sentence for the Discussion, control subsection:

> Because the predictor is an action-conditioned world model in which the present
> intervention is the gust, an actuation channel enters at the same point in the
> model, the predictor conditioning, and the same forward-closure machinery
> applies with the control as the action rather than the disturbance.

## 5. The two caveats (state these, or a reviewer will)

First, “action” versus “disturbance”. In reinforcement learning an action is
chosen by an agent; your gust is an exogenous, observed disturbance. LeCun’s
slide explicitly writes “Action, Intervention”, and the causal-intervention
reading is correct, so the framing holds, but do not call `c` an “action”
without the word “intervention” alongside it, and do not imply the model is
already a controller. It is the world model a controller would use. Make that
distinction explicit; it is the honest and the stronger statement.

Second, do not overclaim “causal”. The slide says “a causal model”. Your model is
conditioned on the true generative parameters of the intervention, which is a
strong and legitimate sense in which it is interventional, but you have not done
counterfactual identification or a do-calculus analysis, and your D124(c)
Wu-impulse check already shows one place where the mid-plane data cannot support
a physical-consistency claim. So frame it as “action-conditioned / interventional
world model”, which is exactly LeCun’s title, and avoid the stronger unmodified
word “causal” in your own claims. The SHAP-and-intervention analyses (D121, D126)
are the closest you get to a causal probe, and they belong in the interpretability
section, not in support of a headline causal claim.

## 6. Where it goes

The framing is worth one paragraph in the Introduction (it motivates the whole
design and the baseline choice) and one sentence in the Discussion control
subsection (it licenses the actuation argument). It is NOT worth a standalone
section: the reference papers you are matching (Fukami, Tran, Smith, Odaka) are
physics-first and would not foreground an ML framing. Used sparingly, it gives
the paper a clean conceptual spine that a fluids reader and an ML reader both
recognise; overused, it would read as grafted-on. One figure is justified: a
small schematic that overlays your encoder/predictor/gust onto the
Enc/Pred/D(s,s_tilde) diagram, which you can build from the existing
fig1_jepa_architecture and fig2_predictive_vs_reconstructive TikZ sources.