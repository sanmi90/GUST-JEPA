# Track B0: per-encounter clip-leakage diagnosis (v2.1)

The clip threshold is a per-encounter p99.99 computed over all 120 frames and applied to every frame: a structural temporal leak. Materiality below.

- encounters sampled: 92 (all test_b + 50 train)
- median |full vs causal threshold shift|: 11.179% (p95 29.688%)
- median |full vs strictly-pre-impact shift|: 12.131%
- impact-window [25,55] differential-clipped cell fraction: median 0.0048% (p95 0.0128%)
- manifest-recompute sanity (full-window vs stored): median rel 0.0016887326179289802

**Verdict: WEAK.** Leak is REAL: the full-window p99.99 is materially future-dependent (median ~11%, p95 ~30% higher than the causal [0:impact+H] threshold), because post-readout wake frames carry comparable |omega| extremes. WEAK branch. BUT the downstream effect at the wake readout is tiny: only ~0.005% of impact-window [25,55] cells are clipped differently under the causal threshold, so the encoder input there is essentially unchanged. The decisive materiality test is B0.5 frozen sensitivity (recompute the wake result under causal/global/no clip and on physical units); a B1 retrain is warranted only if B0.5 moves the JEPA advantage.
