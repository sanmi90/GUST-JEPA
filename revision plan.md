# Bringing “Predictive latent dynamics for parametric vortex-gust airfoil interactions” to the JFM bar

This is a concrete, paste-ready revision document. It is built from a close reading of the seven papers in the project, which define the quality bar:

- Fukami, Smith & Taira, *Phys. Rev. Fluids* **10**, 084703 (2025), the physics source and the dataset characterisation (ref [7]).
- Smith, Fukami, Sedky, Jones & Taira, *JFM* **980**, A18 (2024), persistent homology / cyclic latent (ref [19]).
- Tran, Yeh & Taira, *JFM* **1027**, A24 (2026), optimal-transport-aligned latent embeddings (ref [23]).
- Odaka, Lopez-Doriga & Taira, *JFM* **1031**, R3 (2026), scale decomposition and large-scale structures (ref [18]).
- Fukami & Taira, *JFM* **1010**, R4 (2025), observable-augmented manifold learning (ref [9]).
- Fukami, Nakao & Taira, *JFM* **992**, A17 (2024), phase-amplitude control (ref [6]).
- Wang, Kou, Noack & Zhang, *JFM* **1035**, A18 (2026), causal analysis of a shear-flow model (not currently cited; relevant to the interventional framing).

I disagree with the external critique on one substantive point and say so up front: the wall-pressure result should be **promoted to a main result**, not buried in an appendix. I explain why below and rewrite it accordingly. The only piece that should be demoted is the closed-loop control pilot, which is a different thing from pressure observability.

-----

## Part I. Honest verdict: where the draft stands against the JFM bar

The scientific core is publishable and, in places, the writing is denser and more sophisticated than the reference R-papers. The matched-predictor protocol, the drift mechanism, the topology and transport diagnostics, and the 2x2 controls are a coherent and original contribution. The draft is **not** below the JFM bar on ambition or rigour of idea.

It is below the bar on five things that a JFM reviewer in this exact community (the Taira group and adjacent) will attack, because every one of the reference papers does these things and the draft does not yet:

1. **The methods are not reproducible as written.** Every reference paper states the solver, domain, grid, near-wall resolution, time step, Mach number, and boundary conditions explicitly (Fukami-Smith-Taira give cell counts of 8/20/40 million, $y^+$, domain $[-20,25]\times[-20,20]\times[0,1]$; Tran et al. give $\Delta t u_\infty/L_c = 4.14\times10^{-5}$, CFL 0.84; Odaka et al. give 31 million control volumes, $n_\text{airfoil}=480$, $y^+_0=0.24$). The draft has a placeholder that says these “will be supplied by the simulation collaborators before submission.” That cannot be submitted. See Part II.2.
1. **The observables that carry the headline are never defined.** The reference papers define every diagnostic with an equation: Odaka et al. write out the force-element decomposition and the Gaussian scale filter; Tran et al. write out the unbalanced-OT cost. The draft says the wake observables “integrate (signed) vorticity over the wake region” and stops there. The wake-enstrophy claim is the whole paper and the reader cannot reproduce the metric. See Part II.3.
1. **The paper contradicts itself about the controls.** The abstract says the controls were run and attribute the gain; §4.5 reports them; the Outlook says running them “is the immediate next step.” This is internally inconsistent and is the single most dangerous thing in the manuscript. See Part II.1.
1. **It is physics-light where JFM is physics-led.** The reference papers reach the leading-edge vortex (LEV), trailing-edge vortex, tip vortices, arch and hairpin vortices, the gust-induced wall-normal vorticity flux, and the pressure cores within the first few pages. The draft reaches physical structures only in §4.6, and even there the argument is mostly about latent geometry. To match the venue, the wake-enstrophy advantage has to be turned into a measured statement about the LEV and shear layer. See Part V.
1. **Specific over-claims that the tables do not support**, chiefly the conditioning floor (the wake-enstrophy floor $R^2=0.482$ sits above the JEPA forecast $R^2=0.449$, so “parameters alone cannot generalise” is too strong for that observable) and the “ordering is consistent across every observable” sentence (Fukami $d=32$ beats JEPA on $I_y$ and positive circulation in forecast). See Part II.4 and Part III.

There is also a sixth issue that is small but will be caught: the **case-versus-encounter accounting does not add up** (84 cases, but 226+28+42+24 = 320 encounters). See Part II.5.

Everything else is presentation: the abstract is 308 words against the 250-word JFM limit, the figures have clipped labels and inconsistent legends, and the architecture lives in prose where the venue expects a layer table.

If items 1 to 6 are fixed and the physics depth of Part V is added, the paper is at or above the level of the R-papers in the project and competitive with the A-papers.

-----

## Part II. Blocking fixes (must be done before submission)

### II.1 Resolve the controls contradiction: one claim, used everywhere

Adopt this single sentence and use it verbatim wherever the controls are credited:

> The predictive objective improves forward wake closure over a reconstructive objective when both are trained with matched wake-observable supervision and at matched architecture; the wake head is a necessary part of the successful configuration, and the CNN-versus-CNN+ViT architecture is not the driver.

Then make these edits:

|Location                                |Action                                                                                                                                                                                                                                                        |
|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|Abstract                                |Keep “matched … 2x2 control”; add “with wake supervision” wherever the objective is credited.                                                                                                                                                                 |
|End of Introduction (third contribution)|Replace “set out the matched-architecture and matched-auxiliary controls required to attribute the gain” with “perform matched objective-architecture controls (§4.5) that attribute the wake gain to the predictive objective trained with wake supervision.”|
|§4.5                                    |Keep as the causal-attribution section. This is where the result lives.                                                                                                                                                                                       |
|§5.2                                    |Shorten. State the two-part conclusion once; do not re-run the full §4.5 argument.                                                                                                                                                                            |
|§5.5 Outlook                            |Delete the sentence “The immediate next step is to run those controls and the horizon, topology, transport, and phase-amplitude analyses set out above …”. They are already reported. Replace with the forward-looking sentence about other datasets only.    |
|§6 Conclusions                          |Replace “set out the matched controls that would isolate the objective” with “used matched controls that isolate the objective from the architecture and identify the wake supervision as a necessary component.”                                             |

### II.2 Complete the numerical-method subsection

Insert a full subsection **before** the “Data, fields, and observables” paragraph in §2.2. The structural facts and definitions below are fixed; the solver-resolution numbers are supplied by the authors and inserted into the marked block. The runs are confirmed DNS with no subgrid model, and the encoder reads the mid-plane $\omega_z$ slice.

> **2.2 Numerical method.** The flow fields are direct numerical simulations computed with the GPU-resident spectral-element solver SOD2D [10], which solves the compressible Navier-Stokes equations at low Mach with a continuous-Galerkin spectral-element discretisation of polynomial order $p=4$ and no subgrid-scale model, so all scales of the separated wake are resolved at this Reynolds number. The leading edge is at the origin and the spanwise boundaries are periodic. Lift and drag follow $C_L = F_L/(\tfrac12 \rho u_\infty^2 c)$ and $C_D = F_D/(\tfrac12 \rho u_\infty^2 c)$ from integrating pressure and viscous stresses over the surface. The gust is a Taylor vortex [cite the Taylor reference] released upstream of the leading edge with azimuthal profile
> $$ u_\theta(r) = u_{\theta,\max},\frac{r}{R},\exp!\left[\tfrac12\Bigl(1-\tfrac{r^2}{R^2}\Bigr)\right], $$
> so that the gust ratio is $G \equiv u_{\theta,\max}/u_\infty$, the core diameter $D \equiv 2R/c$, and the wall-normal offset $Y \equiv y_0/c$. Convective time $t$ is in chords with $t=0$ when the vortex centre reaches the leading edge.
> 
> *[Authors to insert the solver-resolution details here: free-stream Mach number $M_\infty$; the computational domain $x/c$, $y/c$, and spanwise extent $L_z/c$; element and solution-point counts; minimum wall-normal spacing $\Delta n_\text{min}/c$ and the corresponding $\Delta n^+$ in the baseline wake; the time step $\Delta t,u_\infty/c$ and maximum CFL; the gust release station $x_0/c$; and the grid and time-step sensitivity check. These DNS details are not yet filled in and will be added by the authors.]*

These values are deferred to the authors, but two facts about the subsection are settled and should be stated as written above:

- **DNS, confirmed.** The runs are direct numerical simulations with no subgrid model. State this explicitly and, once the resolution numbers are inserted, justify “DNS” by reporting $\Delta n^+$ in the baseline wake. The source [7] at the same $Re$ and $\alpha$ was an LES, so a reviewer will expect the resolution shown that makes DNS credible here; the authors should supply it in the block above.
- **Mid-plane slice, confirmed.** The encoder consumes the mid-plane $\omega_z(192,96)$ slice. State it in §2.2 and §3.1, and name the span average once as the alternative a reader familiar with the source paper might assume. The $\chi_{3D}$ diagnostic (Part V.2) measures exactly what the mid-plane slice discards at $|G|=4$, which is the physical reason test_c degrades.

The cache subdomain you state, $x/c\in[-1.5,4.5]$, $y/c\in[-1.5,1.5]$, matches the data-driven subdomain in the source paper exactly, so cite that lineage: “The fields are cached on the analysis subdomain of [7].”

### II.3 Define the six observables with equations

Add this block immediately before Table 2 (or at the end of §2.2). The reference papers never present a force or wake diagnostic without its formula; this paragraph is the minimum.

> The six observables are defined on each cached frame as follows. The body forces are the lift and drag coefficients $C_L, C_D$ above. On the wake window $\Omega_w = {(x,y): x/c \in [\text{FILL}], y/c \in [\text{FILL}]}$, with the airfoil and one adjacent cell layer masked out, the wake enstrophy is
> $$ \Omega_w(t) = \int_{\Omega_w} \omega_z^2(x,y,t),\mathrm{d}A, $$
> and the signed circulations are
> $$ \Gamma^{+}(t) = \int_{\Omega_w} \max(\omega_z,0),\mathrm{d}A, \qquad \Gamma^{-}(t) = \int_{\Omega_w} \min(\omega_z,0),\mathrm{d}A. $$
> The wall-normal hydrodynamic impulse per unit span is
> $$ I_y(t) = -\int_{\Omega_f} x,\omega_z(x,y,t),\mathrm{d}A $$
> over the fluid region $\Omega_f$ [state your exact sign convention and region]. All wake observables use the same fixed mask for every case; values are reported in [dimensional / chord-normalised] units. The $H=16$ frame is the sixteenth cached frame after impact, $t = 16\Delta t = 0.8,c/u_\infty$ after $t=0$.

State four things in words alongside: the wake-window bounds, that the airfoil mask is excluded, that the same mask is used for every case, and how $H=16$ relates to impact. These are the exact questions a reviewer writes in the margin.

### II.4 Fix the conditioning-floor over-claim

The current sentence in §4.1, “The parameters alone cannot generalise outside the training envelope, so the held-out closure is doing representational work that the conditioning cannot replicate,” is contradicted by your own Table 4: for wake enstrophy on test_b the floor is $R^2=0.482$, above the JEPA $H=16$ forecast of $0.449$. Replace with:

> The conditioning floor excludes the gust parameters as the source of the force and circulation gains and confirms that parameter-only interpolation fails on the extrapolation set test_c, where it is negative for five of six observables. For wake enstrophy on test_b the parameter floor ($R^2=0.482$) is about level with the $H=16$ predictive forecast ($0.449$), while the predictive representational closure clears it comfortably ($R^2=0.754$, Table 2a). The wake-forecast claim therefore rests on the paired model comparison (Table 7) and the drift mechanism (§4.3), not on the conditioning floor alone.

Update the Table 4 caption to match (it currently already half-admits this; make it consistent with the body).

### II.5 Reconcile the case/encounter accounting

State the relationship explicitly. As written, “84 cases” and “226 + 28 + 42 + 24 = 320 encounters” cannot both be primitive counts unless each case yields multiple encounters. Add one sentence to §2.2:

> The 84 simulation cases (distinct $(G,D,Y)$ parameter points) are separated in preprocessing into 320 encounter windows of 120 frames each, giving the 226 / 28 / 42 / 24 split (training, held-out training encounters as test_a, in-distribution test_b, and $|G|=4$ test_c) used identically by every model.

This is confirmed: a case contains several encounters that preprocessing already separated, and $226+28+42+24 = 320$, so there is no arithmetic problem, only a definition the current draft omits. Confirm the per-case count from the split manifest (Track B of the session plan) and make the abstract and §2.2 numbers agree with it.

If in fact there are 84 encounters and the split numbers are something else, then the abstract and §2.2 numbers are wrong and must be corrected. Either way, make cases, encounters, and the split arithmetically consistent on the page.

-----

## Part III. Paste-ready rewrites of the core text

### III.1 Abstract (single paragraph, about 250 words once symbols are counted; trim the bracketed numbers if you need it exact)

> Vortex-gust encounters at large gust ratio produce transient aerodynamic loads that are hard to compress into a forecastable reduced state. We ask whether a latent representation trained for prediction, rather than reconstruction, provides a better one. The data are direct numerical simulations (SOD2D) of a NACA 0012 at $\alpha=14^\circ$ and $Re=5000$, perturbed by a Taylor-vortex gust of strength $G$, core diameter $D$, and wall-normal offset $Y$; a fixed split gives 226 training, 28 validation, 42 in-distribution and 24 $|G|=4$ extrapolation encounters. At matched latent dimension we compare a joint-embedding predictive architecture, a reconstructive observable-augmented autoencoder, and proper orthogonal decomposition, under one autoregressive predictor, conditioning on $(G,D,Y)$, and probe family, so only the encoder varies. Each state is judged by forward physical closure of six observables spanning body forces, integrated impulse, and wake vorticity, sixteen frames after impact, not by reconstruction. The predictive latent is the only representation with positive wake-enstrophy forecast $R^2$ at this horizon (0.45 at $d=64$; $-0.48$ and $-0.09$ for the autoencoder and POD), and a per-encounter paired test gives it the smaller wake-enstrophy error on 31 of 42 encounters under direct encoding and 27 of 42 under rollout. The mechanism is geometric: the reconstructive rollout leaves its encoded training manifold by an order of magnitude, whereas the predictive and linear rollouts stay within it and preserve a single cyclic encounter whose metric tracks the optimal-transport geometry of vorticity transport. Matched objective-architecture controls attribute the wake gain to the predictive objective trained with wake-observable supervision, not to the architecture. The same predictive state is the most recoverable of the three from sparse wall pressure, so it is observable as well as forecastable.

Note the deliberate change: the abstract now ends on **pressure observability and forecastability**, not on model-based control. This keeps the result the user values and removes the control over-claim.

### III.2 Introduction, contributions paragraph

> We make three contributions. First, we introduce a matched-predictor evaluation protocol for reduced states of vortex-gust airfoil encounters: a predictive encoder, a reconstructive observable-augmented autoencoder, and a POD basis are compared at matched latent dimension under one autoregressive predictor, one parametric conditioning on $\mathbf{c}=(G,D,Y)$, and one probe family, so that forecast quality is attributable to the encoder alone. The figure of merit is forward physical closure of body forces, integrated impulse, and wake vorticity at horizon $H=16$, not reconstruction error. Second, we identify the mechanism that controls closure. The reconstructive rollout leaves its encoded training manifold by an order of magnitude, whereas the predictive latent stays in distribution and preserves a single cyclic encounter geometry whose metric, without being trained to, tracks the optimal-transport geometry of vorticity transport more faithfully than the reconstructive latent does. Third, we separate objective, architecture, and auxiliary supervision with matched controls: the predictive objective improves wake closure at both CNN and CNN+ViT architectures when the wake supervision is held in common, while removing the wake head collapses the wake forecast. The resulting claim is therefore specific, that a predictive objective combined with wake-observable supervision produces a forecastable reduced state, and we show that this same state is the most recoverable of the three from sparse wall-pressure measurements, so it is observable as well as forecastable.

The phrase “without being trained to” is worth keeping: Tran et al. [23] *train* the latent to be OT-aligned; you get OT-alignment as a by-product of the predictive objective. That contrast strengthens your paper and you should state it.

### III.3 Results, the mixed-ordering sentence (§4.1)

Delete “All six observables are reported in both modes, not a selected subset, so the consistency of the ordering across every observable is itself part of the evidence.” Replace with:

> The ordering is not uniform across all six observables, and reporting all of them rather than a selected subset makes that explicit. The predictive latent is strongest where the dynamics require spatially distributed wake information, the wake enstrophy and the negative circulation, while the integrated impulse and the positive circulation remain competitive across families (the reconstructive $d=32$ latent even leads on $I_y$ and positive circulation under forecast). This mixed ordering is informative: the predictive objective does not dominate every scalar readout, it improves the observables most directly tied to wake-vorticity redistribution, which is the quantity a force-only representation can miss.

### III.4 Results, make the paired test the headline (§4.1)

The marginal bootstrap interval on the forecast wake $R^2$ ($[-0.96,0.79]$) is wide and currently sits in the same sentence as the point estimate, which undercuts the result. Lead with the paired statistic and demote the marginal interval to a parenthetical:

> On held-out encounters the predictive latent is the only representation whose wake-enstrophy forecast clears the predict-the-mean floor at $H=16$ ($R^2=0.449$ at $d=64$). The load-bearing test is the per-encounter paired comparison, which cancels the encounter-to-encounter difficulty that every model shares: the predictive latent carries the smaller wake-enstrophy error on 31 of 42 encounters under direct encoding (paired mean improvement $43.1$, $95%$ CI $[23.5,66.0]$, one-sided sign $p=1.4\times10^{-3}$) and on 27 of 42 under forecast ($32.0$, $[10.8,54.8]$, $p=4.4\times10^{-2}$). The marginal one-observable-at-a-time bootstrap intervals are wide ($R^2=0.449$ has interval $[-0.96,0.79]$) because they are dominated by that shared difficulty; the paired test removes it and is the appropriate statistic.

### III.5 Discussion, scope paragraph (§5.2), shortened

> We are deliberate about scope. The predictive encoder differs from the reconstructive baseline in three ways at once: objective, CNN+ViT architecture, and wake supervision. The conditioning floor removes the shared parametric conditioning; the matched 2x2 controls of §4.5 separate the other two. They give a two-part answer. The predictive objective improves forward wake closure over the reconstructive objective at both a CNN and a CNN+ViT encoder with the auxiliary heads matched, and the two architecture columns do not separate, so the gain is the objective and not the architecture. The wake supervision is a necessary part of the configuration: removing the wake head from the predictive model collapses wake closure below the floor. The honest claim is that the predictive objective improves forward closure when trained with wake supervision, not that the objective in isolation does, and not that the wake head attached to any objective does, since the reconstructive cells carry the same head and do not reach the predictive closure. POD remains useful: it is the most competitive family on the integrated impulse and is amplitude-accurate in reconstruction, so the result is not that POD is obsolete but that its rollout does not close the wake observables.

### III.6 Conclusions (§6)

> We compared a predictive latent, a reconstructive observable-augmented autoencoder, and a POD basis as reduced states for parametric vortex-gust airfoil interactions at $Re=5000$, under a protocol in which one autoregressive predictor, one conditioning, and one probe family were held fixed so that only the encoder varied, and judged each by forward physical closure rather than by reconstruction. The main result is not that the predictive latent reconstructs the flow most accurately, but that it gives the most forecastable state for wake-vorticity observables. On held-out encounters it reduces direct-encoding wake-enstrophy error by factors of 2.4 and 3.0 relative to the reconstructive autoencoder and POD, and it is the only $d=64$ representation with positive $H=16$ wake-enstrophy forecast $R^2$. The improvement is paired by encounter and is explained by rollout geometry: the reconstructive latent leaves its own encoded manifold, whereas the predictive latent stays in distribution and preserves a single cyclic encounter structure whose metric is aligned with vorticity transport. Matched controls show the gain is the predictive objective and not the architecture, with wake-observable supervision as a necessary component. The same predictive state is the most recoverable of the three from sparse wall pressure, so it is observable as well as forecastable, which is the property a downstream estimator or controller would exploit. The remaining limitations are the weak resolution of the wall-normal offset, the three-dimensional observability boundary at $|G|=4$, and a rollout error that still prevents a closed-loop control demonstration.

-----

## Part IV. Pressure observability: promote it (my disagreement with the external critique)

The external critique recommends moving the pressure material to Appendix B. I think that is the wrong call, and the right call is to **split the appendix into two ideas that are currently fused**:

1. **Pressure observability of the latent state.** This is a genuine main result and reinforces the central thesis. The argument the paper makes about forecastability has an exact deployment mirror: the predictive state is not only the one whose rollout closes the wake, it is also the one most recoverable from sparse wall pressure ($R^2 = 0.89$ at $K=8$ and $d=64$, against $0.58$ for the reconstructive AE and $0.32$ for POD), and the contrast is sharpest at matched capacity. The reconstructive $d=3$ latent is the *easiest* to recover yet gives the *worst* lift estimate, which is a clean, counter-intuitive, and quotable result. In this lineage (Mousavi & Eldredge 2025 did sparse-sensor estimation with uncertainty on a Fukami-Taira latent; sensor-based reconstruction is a Taira-group staple), pressure recoverability is a natural and welcome contribution, not an aside. It says: a predictive objective produces a state that is **both forecastable and observable**, the two properties a reduced-order model needs to be useful, and a reconstruction objective produces a state that is neither in the way that matters.
1. **The closed-loop control pilot.** This is the only part that should be demoted, and the reason is internal: the loop misses its tolerance, and it misses it even with an oracle latent (18% within band against an 80% target, oracle and estimated alike). That is an honest negative result about the *predictor rollout*, not about pressure or about the representation. It belongs in one short, clearly-labelled paragraph as a limitation.

Concrete actions:

- Add a short main-text subsection (about half a page) titled **“The predictive state is the most observable from the wall.”** Promote one panel: predictive-state recovery $R^2$ versus sensor count across families (current Fig. 13a), plus the impact-$C_L$ contrast (current Fig. 13b). Keep the sentence “the energy-optimal POD coordinates are the hardest to read from pressure, the deployment counterpart of the representation result.”
- Keep the *placement* result (TCSI clusters at the leading edge, decisive at $K=2$, methods converge by $K\ge4$) in the appendix; it is supporting, not headline.
- Reduce the closed-loop pilot to one paragraph in §5, labelled as a limitation, with the oracle-equals-estimator point as the key sentence. Remove “model-based control” from any headline claim; “compatible with model-based control, with the rollout as the current bottleneck” is the strongest honest phrasing.
- Add to the abstract (done in III.1) and the conclusion (done in III.6) the single clause that the predictive state is the most pressure-recoverable. That one clause does more for the paper than the entire closed-loop pilot.

Suggested main-text paragraph for the observability subsection:

> At deployment the vorticity field is not available; only the wall pressure is. We therefore ask which representation is most recoverable from a sparse set of $K$ wall-pressure taps over a pre-impact window, using one kernel-ridge estimator across families for a uniform comparison. The predictive state is markedly the most recoverable: with $K=8$ taps placed by a target-conditioned greedy selection, the held-out test_b state $R^2$ is $0.89$ at $d=64$ and $0.87$ at $d=32$, against $0.58$ to $0.67$ for the reconstructive autoencoder and $0.32$ to $0.49$ for POD, and the contrast is sharpest at matched capacity. This is the deployment mirror of the forecast result of §4.1: the same predictive objective that produces a forecastable wake state produces a state that is also legible from the wall, whereas the energy-optimal POD coordinates, optimal for reconstruction, are the hardest of the three to read from pressure. Recoverability is not estimation quality, and the distinction is itself informative: the three-dimensional reconstructive latent is the easiest to recover yet routing the impact-lift estimate through it gives the worst $C_L$ error, while the lift is read most accurately straight off the pressure. The deployment value of the predictive latent is the forecastable, observable state it hands to the predictor, not the instantaneous lift, which needs no latent.

-----

## Part V. Physics depth: turn the wake-enstrophy advantage into a statement about the LEV and shear layer

This is what moves the paper from a strong ML-on-fluids paper to a JFM fluid-mechanics paper. In Odaka et al. and Fukami-Smith-Taira, the lift peaks are *explained* by the LEV, the trailing-edge vortex, the tip vortices, and the gust-induced wall-normal vorticity flux, with the large-scale structures extracted by the same $\sigma/c=0.05$ Gaussian filter you already use. Your §4.6 already does the scale decomposition; you need one more diagnostic so the wake-enstrophy number becomes a vortex.

### V.1 Leading-edge-vortex tracking (new diagnostic)

Procedure (Python; you have everything in the cache):

```python
import numpy as np
from scipy.ndimage import gaussian_filter, label, center_of_mass

def lev_diagnostics(wz, dx, dy, sigma_c, x, y):
    # wz: (Nx,Ny) spanwise vorticity (mid-plane or span-averaged), physical units
    # large-scale field via the same sigma/c = 0.05 filter used in 4.6
    wzL = gaussian_filter(wz, sigma=sigma_c/dx)        # match 4.6 exactly
    # threshold the large-scale negative (suction-side LEV) lobe
    thr = 0.5 * np.nanmax(np.abs(wzL))
    mask = (np.abs(wzL) > thr) & (X < x_te)            # restrict to LE/wing region
    lbl, n = label(mask)
    # take the strongest connected lobe as the LEV
    sizes = [np.sum(np.abs(wzL[lbl==k])) for k in range(1, n+1)]
    k = 1 + int(np.argmax(sizes))
    cyx = center_of_mass(np.abs(wzL)*(lbl==k))
    x_lev = x[int(round(cyx[0]))]; y_lev = y[int(round(cyx[1]))]
    peak  = np.nanmax(np.abs(wzL[lbl==k]))
    gamma_lev = np.sum(wzL[lbl==k]) * dx * dy           # signed LEV circulation
    return x_lev, y_lev, peak, gamma_lev
```

Compute $(x_\text{LEV}, y_\text{LEV})$, peak $|\omega_z|$, and $\Gamma_\text{LEV}$ for the simulation, the predictive decode, the reconstructive AE decode, and the POD reconstruction, at impact and at $H=16$, on test_b.

Deliverable figure (one panel each):

- $x_\text{LEV}$ error versus horizon, three families.
- $\Gamma_\text{LEV}$ error versus horizon, three families.
- a scatter of wake-enstrophy error against LEV-circulation error, to show they co-vary.

Earned sentence (add to §4.6):

> The wake-enstrophy advantage is not only a latent-space property. The predictive decode places the large-scale leading-edge vortex within $[\text{FILL}],c$ of the simulation centroid at $H=16$ and recovers its circulation to within $[\text{FILL}]%$, whereas the reconstructive decode loses the LEV (centroid error $[\text{FILL}],c$, circulation retained $[\text{FILL}]%$); the per-encounter wake-enstrophy error tracks the LEV-circulation error (Spearman $[\text{FILL}]$). The scalar wake enstrophy is therefore standing in for the location and strength of the leading-edge vortex and shear layer that produce the lift transient.

This is the single highest-value physics addition. It connects your wake observable to the structure that the source papers spend their figures on.

### V.2 A measured observability boundary at $|G|=4$ (replaces a defensive paragraph with a number)

Right now test_c is explained by a verbal appeal to three-dimensionality. The source paper shows it (the bounding vortex pair becomes unstable before re-impingement; spanwise instabilities appear for $|G|\ge4$). Give it a number from your own 3D fields:

$$ \chi_{3D}(t) = \frac{\displaystyle\int \lVert \boldsymbol{\omega}(x,y,z,t) - \langle\boldsymbol{\omega}\rangle_z(x,y,t)\rVert^2,\mathrm{d}V}{\displaystyle\int \lVert \boldsymbol{\omega}(x,y,z,t)\rVert^2,\mathrm{d}V}, $$

the fraction of enstrophy not captured by the spanwise mean (the spanwise-fluctuating, three-dimensional content the mid-plane encoder cannot see).

```python
# omega: (3, Nx, Ny, Nz) full vorticity; or use scalar wz if that is what you cache in 3D
mean_z = omega.mean(axis=-1, keepdims=True)
num = np.sum((omega - mean_z)**2)
den = np.sum(omega**2)
chi3d = num/den
```

Plot $\max_t \chi_{3D}$ against $|G|$ for the training range and for $|G|=4$. The expected jump at $|G|=4$ converts the test_c degradation from an excuse into a measured limit. Put it as an inset to Figure 1 or Figure 5 and rewrite the §2.1 / §5.3 sentences to cite the number rather than the source paper alone.

This also resolves the mid-plane-versus-span-average ambiguity of Part II.2: $\chi_{3D}$ is exactly the quantity that says how much the mid-plane (or span-mean) representation throws away.

-----

## Part VI. New analyses with exact specifications

These are the runs that close the remaining reviewer gaps. They need your data and your training pipeline; the specs are exact so they can be executed directly.

### VI.1 Seed robustness for the headline models

You report three seeds only for the §4.5 controls. Run three seeds for the headline $d=64$ and $d=32$ JEPA/AE/POD comparison (POD is deterministic; the seed variation is over encoder retrains for JEPA and AE, and over the probe fit for POD).

Deliverables:

- A version of Table 2 with mean $\pm$ standard deviation over three seeds.
- The per-seed paired JEPA-minus-AE wake-enstrophy improvement.
- One sentence: “The paired wake-enstrophy improvement is positive in all three seeds.”

This is the cheapest high-value addition after the methods section. The reconstructive CNN+ViT cell in Table 5 already shows large seed variance ($\pm0.27$); a reviewer will ask whether the headline is similarly fragile, and you want the answer in the paper.

### VI.2 Stronger conditioning floor

Because the parameter-only floor is near JEPA on wake enstrophy (Part II.4), strengthen Table 4 so it proves what it claims. Add, for each observable and split:

- $\mathbf{c}$-only (current).
- phase-only (the within-encounter phase or time index $\tau$).
- $\mathbf{c} + \tau$.
- nearest-neighbour-in-parameter-space.
- kernel ridge with leave-one-case-out cross-validation (not leave-one-encounter-out, to avoid leakage across encounters of the same case).

```python
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import LeaveOneGroupOut   # group = case id
# X = [G, D, Y, tau];  y = observable at impact;  groups = case_id
logo = LeaveOneGroupOut()
# fit KRR per fold, collect held-out R^2 -> this is the honest floor
```

Then rewrite Table 4 around what the floor rules out (gust parameters as the source of the force and circulation gains; parameter interpolation failing on test_c) rather than the over-strong “parameters cannot generalise.”

### VI.3 Error maps over $(G, D, Y, \phi)$

You state the latent resolves $G$ and $D$ but weakly resolves $Y$, then defer the per-stratum table. Do not defer it; the test set is small, so use continuous trend plots instead of bins. Define the per-encounter paired improvement $\Delta e = e_\text{AE} - e_\text{JEPA}$ on wake enstrophy and plot it against, in four panels:

- $G$, $D$, $Y$, and the baseline shedding phase at impact $\phi$.

Overlay LOWESS curves with bootstrapped bands:

```python
import statsmodels.api as sm
lo = sm.nonparametric.lowess(delta_e, G, frac=0.6, return_sorted=True)
# bootstrap the LOWESS over encounters for a band
```

The shedding phase at impact is the most interesting axis and is the one the source papers care about (the timing relative to the natural cycle). Computing $\phi$ at impact from the baseline limit cycle (you already have the Hilbert-phase machinery in §4.4) and showing where the predictive advantage concentrates in phase is a result, not a table.

### VI.4 (Stretch, “exceed” not “match”) A measured interventional test

Your §1 frames the model as an interventional world model but explicitly does not test it (“we do not perform counterfactual identification”). The Wang-Kou-Noack-Zhang causal paper in the project is the template for turning that framing into a measurement. A minimal version that fits your setup:

- Take a held-out encounter. Perturb the conditioning $\mathbf{c} \to \mathbf{c} + \delta\mathbf{c}$ along one axis (say $\delta G$). Roll the predictor forward and read the observables.
- Compare the predicted *change* in each observable, $\Delta \hat{q}(\delta G)$, to the *measured* change between matched simulation encounters that differ only in $G$ by $\delta G$.
- Report the correlation between predicted and measured response across the parameter grid.

If the predictor’s response to a parameter intervention matches the simulation’s, you have earned the “interventional world model” language as a result rather than a metaphor, and you can cite the GGC paper as the causal-inference reference. If it does not match, that is itself worth reporting and you should soften §1 to “conditional forward model.” Either outcome is publishable and removes the current rhetorical exposure.

-----

## Part VII. Figure plan (physics-led, JFM conventions)

The reference papers are figure-led: the reader sees the configuration and the physical mechanism early (Odaka Fig. 1-2, Fukami-Smith-Taira Fig. 1-2). Your first flow-physics panels are very late and the pressure figures dilute the story. Reorder to five main figures plus appendix figures.

|New figure                                   |Built from                                                              |Concrete change                                                                                                                                                                                                                     |
|---------------------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|**Fig 1: flow and dataset**                  |current Fig. 1 + one vorticity snapshot sequence + the $\chi_{3D}$ inset|Lead with the configuration: airfoil, Taylor vortex, $G,D,Y$, impact definition, the four stages, and the parameter-space split. Add the $\chi_{3D}$-versus-$                                                                       |
|**Fig 2: matched protocol**                  |current Figs. 2-3 + Table 1                                             |One figure: JEPA / AE / POD feeding the same predictor and the same probes. Fold the predictive-vs-reconstructive training contrast in as a sub-panel rather than a separate figure.                                                |
|**Fig 3: headline closure**                  |Table 2 + Fig. 4 + Table 7                                              |A compact heatmap or forest plot of held-out $R^2$ across families and observables, plus the paired per-encounter wake panel (JEPA vs AE error). Make the paired result visually primary.                                           |
|**Fig 4: drift mechanism**                   |Table 3 + Figs. 6-7                                                     |Horizon dependence, Mahalanobis drift ratio, and the persistent-$H_1$ generator count in one figure.                                                                                                                                |
|**Fig 5: transport and physical wake**       |Figs. 8-11 + the new LEV-tracking panels (Part V.1)                     |Combine the vorticity stages, the OT field distance, the large-scale wake enstrophy, and the LEV-centroid/circulation error. The reader should leave this figure seeing the LEV and shear-layer mechanism, not only latent geometry.|
|**App B: pressure placement and closed loop**|Figs. 12, 14, 15                                                        |Keep placement and closed-loop here. Promote only the cross-family recoverability panel (Fig. 13) into the new main observability subsection (Part IV).                                                                             |

Specific repairs (these are defects a reviewer will list):

- **Fig. 9**: the x-axis label is clipped (“simulation OT-geodesic distance (nor…”). Fix.
- **Fig. 8**: the baseline cycle is tiny and visually disconnected from the gust trajectory. Normalise the axes or add an inset so the “return to orbit” is legible. The orbit-phase panel (b) has a confusing wrap; label the wrap explicitly.
- **Fig. 6**: the bottom-right panel has missing early-horizon curves; label unavailable horizons rather than leaving blank space.
- **All vorticity panels**: same spatial extent, same colourbar, same ticks (or none), consistent airfoil scaling, with $x/c, y/c$ labelled at least once per figure. The reference papers are rigid about this.
- **Legends**: pick one label. The draft uses “recon.”, “reconstructive”, and “Fukami” for the same family, and “JEPA” / “predictive” interchangeably. Choose “predictive (JEPA)” and “reconstructive AE” everywhere.
- **Format**: vector PDF or EPS, editable labels, RGB, minimum line width 0.5 pt, captions beneath, figures cited in order.
- **Graphical abstract**: JFM now requires one. Make it a single clean image: airfoil + incoming vortex + three small latent rollouts labelled by colour only, no dense text.

Two venue-convention additions the reference papers have and you lack:

- An **architecture table** for the encoder and the predictor (layer, output size), in the style of Tran et al. Table 1. Your Appendix A prose can become two compact tables.
- The **scale-decomposition equations** (Gaussian filter, large/small split) written out in §4.6, in the style of Odaka et al. (3.1)-(3.4), since you use exactly that method.

-----

## Part VIII. Compliance and submission

- The abstract must be a single paragraph of at most 250 words (the rewrite in III.1 is at the limit).
- Remove the manually entered “Key words” line unless the JFM class you use requires it; JFM keywords are chosen during online submission. (The reference R-papers do print a key-words line, so check the template you adopt and be consistent.)
- Add competing-interests, data-availability, funding, and author-contributions statements before the references; JFM lists these as required disclosure sections.
- Audit arXiv and under-review citations ([2], [3], [4], [13], [14], [20], [21]). JFM discourages citing unreviewed preprints when a peer-reviewed version exists; [14] LeWM in particular is a 2026 preprint your central regulariser depends on, so either cite the published version or state clearly that it is a preprint and self-contain the method (you partly do this in Appendix A; make the SIGReg description fully self-contained so the paper does not depend on an unreviewed reference for reproducibility).
- Prepare an AI-use declaration if any AI tool generated or edited manuscript text, figures, or analysis, with tool, version, dates, and a description of use.

-----

## Part IX. Priority order

The fastest path to a JFM-ready version, in order:

1. Fix the controls contradiction (II.1) and the conditioning-floor over-claim (II.4). Half a day, zero new computation, removes the two most dangerous internal problems.
1. Complete the numerical-method subsection and settle DNS-versus-LES and mid-plane-versus-span-average (II.2). Reconcile the case/encounter counts (II.5). Define the six observables (II.3).
1. Run the seed robustness for the headline models (VI.1) and the stronger conditioning floor (VI.2).
1. Add the LEV-tracking physics (V.1) and the $\chi_{3D}$ boundary number (V.2). This is the change that earns the JFM label.
1. Promote pressure observability to a main subsection and demote the closed-loop pilot to one paragraph (Part IV).
1. Rebuild the first five figures around the physical mechanism and repair the defects (Part VII).
1. Rewrite abstract, contributions, results claims, discussion scope, and conclusion (Part III), then the compliance pass (Part VIII).

Items 1, 2, and 7 are writing and can be done now from this document. Items 3 to 6 need your data and pipeline; the specifications above are exact enough to execute directly.