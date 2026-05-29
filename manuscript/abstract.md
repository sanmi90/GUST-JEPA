# Abstract

Vortex-gust impacts on a stalled NACA 0012 at chord Reynolds number 5000
demand a reduced-order model that generalises across held-out gust parameters
from a single impact-frame snapshot. Reconstruction-trained autoencoders
(Fukami and Taira, Phys. Rev. Fluids 10, 084703, 2025; Solera-Rico et al.,
Nat. Commun. 15, 1361, 2024) reach low pixel error but do not close the
forecast loop on physical observables across unseen cases. We train an
end-to-end Joint-Embedding Predictive Architecture on mid-plane vorticity at
latent dimension 64, with SIGReg anti-collapse (Balestriero and LeCun,
arXiv:2511.08544, 2025) and gust conditioning on the predictor only. Against
Fukami and snapshot POD on identical preprocessing and on train, val, test_b
and test_c splits, JEPA forecasts wake enstrophy 1.5 times more accurately
than Fukami and 3.7 times more accurately than POD at matched dimension 64,
sixteen-frame Markov horizon. Impact-frame kernel-ridge probes recover gust
strength, diameter and offset at R^2 of 0.96, 0.91 and 0.64 on test_b
(bootstrap n = 2000, 3-seed standard deviation under 0.06, 5-fold CV);
test_c recovery is partial. A frozen-encoder decoder reaches test_a SSIM
0.71 (Wang K1 = 0.01, K2 = 0.03; L from p99.9 of normalised target).
Closed-loop pressure sensing fails its C_L gate and offset Y is the hardest
parameter. Wake-aware predictive states close Markov forecasts on
structure-rich observables where reconstruction-trained latents do not.
