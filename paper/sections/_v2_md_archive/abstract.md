# Abstract

We test whether a predictive latent training objective produces a reduced-order
representation of a parametric gust-vortex airfoil interaction that is more
useful for forward-time forecasting than a reconstructive one. The dataset is a
direct numerical simulation of a NACA 0012 airfoil at angle of attack
$\alpha = 14^\circ$ and chord Reynolds number $\text{Re} = 5000$ perturbed by
Taylor vortices parametrised by gust strength $G$, vortex core diameter $D$,
and wall-normal offset $Y$, with 84 cases split into 226 training, 28
validation, 42 stratified test, and 24 out-of-distribution extrapolation
encounters. A Joint Embedding Predictive Architecture is compared head-to-head
with the Fukami observable-augmented autoencoder and a proper orthogonal
decomposition basis at matched latent dimensions $d \in \{16, 32, 64\}$ under
a fairness protocol in which each baseline shares an identical autoregressive
transformer predictor, the identical training recipe, the identical
conditioning on $(G, D, Y)$, and the identical probe family; the encoder is
the only thing that varies. The headline metric is the closure of the
predictor rollout to six physical observables at horizon $H = 16$.
The predictive latent achieves a mean coefficient of determination of $0.835$
at $d = 64$, against $0.561$ for the linear basis and $0.427$ for the
reconstructive autoencoder at the same dimension; the gap is largest on the
spatially distributed wake observables. The matched-capacity result at
$d = 32$ retains a mean $R^2$ of $0.808$. The mechanism behind the gap is a
latent drift diagnostic that shows the reconstructive autoencoder rollout
goes one order of magnitude out of distribution at $H = 16$ while the
predictive latent stays inside its training manifold. A conditioning-only
floor establishes that the generalisation gap to the autoencoder is not
explained by the predictor's parametric conditioning. The predictor is the
latent dynamics function required by model-based control and the
architectural pathway to a closed-loop deployment is sketched.
