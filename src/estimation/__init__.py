"""State-estimation harness for the Session 32 predict-correct filter (Track B).

Modules:
- :mod:`src.estimation.obs_operator` -- the observation operator ``H: z_pool ->
  p_K`` (K=8 qDEIM wall-pressure taps) and its obs-noise covariance R. H is
  pressure-only by construction; observable probes (E_w, C_L) never enter it.
- :mod:`src.estimation.enkf` -- the ensemble Kalman filter (stochastic +
  deterministic square-root), with per-member rolling analysis context, field-free
  init, additive process noise, and multiplicative inflation.
- :mod:`src.estimation.metrics` -- closure, innovation whiteness, NIS coverage,
  analysis Mahalanobis ratio, divergence flag, CRPS and spread-skill.
"""

from __future__ import annotations
