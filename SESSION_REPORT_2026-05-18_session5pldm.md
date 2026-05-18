# Session report -- 2026-05-18 (Session 5.PLDM)

Conditional follow-up to Session 5. Triggered by D27's TRIVIAL-dominant
outcome and the always-record PLDM-priority-on-TRIVIAL rule (D29). The
session ran on the same 2026-05-18 calendar day as the Session 5 report
(`SESSION_REPORT_2026-05-18.md`); this file uses the `_session5pldm`
suffix to disambiguate.

## Starting state

| Item | State |
|---|---|
| Last commit on `origin/main` | `46300d2` (Session 5) |
| `src/baselines/` | did not exist |
| `src/training/` | scheduled_sampling.py, diagnostics.py, auto_fallback.py, sanity_checks.py, train_jepa.py |
| `src/models/` | encoder.py, predictor.py, jepa.py, sigreg.py, vicreg.py, adaln.py, rope.py |
| `tests/` | 79 tests passing, 1 slow integration test gated by `--runslow` |
| Partition v1 | 49 cases / 238 encounters; split SHA256 `7f8f6042...875e2331` |
| HANDOFF.md decision log | last entry was D33 (then D32 numerically; D27/D28/D29 from Session 5) |

## What ran end-to-end

1. **Read arXiv:2502.14819 and verified the loss formulation.** The arxiv
   MCP plugin downloaded the full paper (102k chars). The content was too
   large for the main context; a `general-purpose` subagent sliced the
   file via Python `read()[A:B]` and reported back the loss equations
   verbatim with section/equation citations. The main agent then
   independently verified the agent's key claim (the term count) by direct
   `grep` on the saved file. The verified-from-paper loss is:

   ```
   L_JEPA = L_sim + alpha * L_var + beta * L_cov + delta * L_time-sim + omega * L_IDM
   ```

   FIVE terms, not seven. The L_sim is the multi-step rollout MSE from
   Section 3.3 Equation 3 (sum over the prediction horizon H, divided by
   batch N only). The four anti-collapse terms are in Appendix D.1.1.
   **There are no var(dz) or cov(dz) terms** -- D8's "term 5" and "term 6"
   were spurious.

   Paper-side hyperparameters by environment (Tables 13-17, Appendix J.2):

   | Environment        | alpha | beta | delta | omega |
   |--------------------|-------|------|-------|-------|
   | Two-Rooms          |  4.0  |  6.9 |  0.75 | 0.0   |
   | Diverse PointMaze  | 35.0  | 12.0 |  0.1  | 5.4   |
   | Ant-U-Maze         | 26.2  |  0.5 |  8.1  | 0.58  |

   None matched our regime cleanly; the smoke used all-1.0 defaults.

2. **TDD landed for `src/baselines/pldm.py`** with `tests/test_pldm_loss.py`
   (13 tests). Tests cover the shape contract, perfect-rollout L_sim, the
   variance hinge on unit-Gaussian vs collapsed z, off-diagonal covariance
   on independent vs correlated dims, temporal smoothness on static vs
   white-noise z, IDM at init vs after training, total-equals-weighted-sum,
   gradient flow through each term, and bf16 dtype promotion. Two
   thresholds were re-calibrated from the plan defaults: the cov test
   needed N=1024 to drop the empirical floor below 0.1, and the time-sim
   test expectation switched from sum-scaling (~64) to mean-scaling (~2.0)
   per the loss's `.mean()` reduction.

3. **`src/baselines/pldm.py`** implements `PLDMLoss(d, c_dim, lambda_var,
   lambda_cov, lambda_time_sim, lambda_idm, gamma, eps, idm_hidden)`.
   Five terms with the formulas verified verbatim against Appendix D.1.1.
   The IDM is a 3-layer MLP `(2*d) -> 128 -> 128 -> c_dim`. All four
   anti-collapse terms run in fp32 even under bf16 autocast (same
   convention as SIGReg and VICReg).

4. **TDD landed for `src/models/pldm_wrapper.py`** with
   `tests/test_pldm_wrapper.py` (5 integration tests): shape contract,
   loss decomposition, gradient flow through encoder + predictor + IDM,
   the `prediction_horizon` wiring, and a bf16 autocast smoke test on the
   RTX 6000.

5. **`src/models/pldm_wrapper.py`** composes `HybridCNNViTEncoder` +
   `AutoregressivePredictor` + `PLDMLoss`. The forward pass encodes
   `omega` to `z`, rolls out `z[:, :1, :]` for `prediction_horizon`
   steps via `predictor.rollout`, calls `PLDMLoss` on
   `z[:, :H+1, :]` and the rolled-out `z_hat`, and returns the five-term
   loss dict. Architecturally identical to JEPA so the contrast isolates
   the loss.

6. **`src/training/train_baseline.py`** is the argparse training
   entrypoint for matched-capacity baselines, dispatching on
   `--baseline {pldm, fukami_ae, solera_rico, pod}`. Only `pldm` is
   implemented; the others raise `NotImplementedError` until their
   sessions land. The IDM MLP parameters live in the predictor optimizer
   group (closest semantically). W&B contract mirrors `train_jepa.py`
   with the four PLDM lambdas replacing `lambda_sigreg` (which is set
   to None and kept in the config so the four required keys per
   CLAUDE.md "Logging" remain emitted). Side JSONL logger writes
   `<output_dir>/metrics.jsonl` for offline analysis.

7. **Test suite green** -- `pytest tests/`: 97 passed, 1 skipped in
   215 s. 18 new tests this session (13 in test_pldm_loss.py, 5 in
   test_pldm_wrapper.py).

8. **Run PLDM-A** -- 5000 iterations on the 5-case smoke subset,
   seed 0, RTX PRO 6000 Blackwell Max-Q. Wall clock approximately
   55 minutes. Final state:

   - PR = 5.97 (out of d=32; below the 16 healthy threshold and the
     9.6 fallback floor)
   - r2_overall = 0.970 (highest of any variant in the session)
   - r2_G = 0.986, r2_D = 0.970, r2_Y = 0.953
   - L_sim = 0.014, L_var = 0.510, L_cov = 0.102,
     L_time_sim = 0.002, L_idm = 0.0005

   Trajectory: PR climbs slowly 1.4 -> 2.3 -> 4 -> 5-6 across the run,
   ending at 5.97. r2_overall sits at 0.95-0.97 by iter 1000 and stays
   there. L_time_sim and L_idm both drop to near-zero by iter 500 and
   stay there.

9. **PLDM-B (LayerNorm)** deferred. Optional per the plan; given that
   the Session 5 Run B (SIGReg + LN) degraded probe r2 rather than
   recovering PR, running PLDM-B was unlikely to change the
   DATA_SCALE_BOUND conclusion. The decision can be revisited in
   Session 5.5 if the case-count expansion produces ambiguous PLDM
   behaviour.

10. **Analysis notebook extended.** `notebooks/01_smoke_5k_analysis.ipynb`
    gained Section 7 (4 markdown + 4 code cells): PLDM loss trajectories
    (5-panel, sym-log), latent-health diagnostics with PLDM overlay,
    5-variant 2x2 quadrant table, and a PLDM-specific decision string
    that distinguishes REGIME_CONFIRMED / DATA_SCALE_BOUND /
    PLDM_PARTIAL. The notebook executes end-to-end and prints
    `DATA_SCALE_BOUND: PLDM-A PR=5.97 (<= 16); both regularisers
    collapse on 5 cases. r2=0.970. -> Session 5.5 (expand to 10-12
    cases) on BOTH SIGReg and PLDM.`

11. **HANDOFF.md D30 / D31 appended** -- D30 records the full Session
    5.PLDM execution AND the 7-to-5-term correction with paper citations
    and verification trail. D31 records the DATA_SCALE_BOUND outcome
    with the five-variant comparison table and the methodological
    reading (at the 5-case scale the only consistent local minimum is
    `z = f(c)` plus noise; different regularisers produce different
    *forms* of that minimum but none escape it).

12. **CLAUDE.md "Baselines to implement"** updated: the PLDM description
    now reads "5-term VICReg-derived objective (four tunable weights)"
    with a cross-reference to D30 and the regime-dependent paper claim
    framing.

## Files

### New

| Path | Purpose |
|---|---|
| `src/baselines/__init__.py` | Package root |
| `src/baselines/pldm.py` | 5-term PLDM loss |
| `src/models/pldm_wrapper.py` | Encoder + predictor + 5-term loss composition |
| `src/training/train_baseline.py` | argparse entrypoint dispatching on `--baseline` |
| `tests/test_pldm_loss.py` | 13 unit tests for PLDMLoss |
| `tests/test_pldm_wrapper.py` | 5 integration tests for PLDMWrapper |
| `SESSION_REPORT_2026-05-18_session5pldm.md` | This report |

### Modified

| Path | Change |
|---|---|
| `HANDOFF.md` | D30, D31 appended; D8's 7-term framing now overridden by D30 |
| `CLAUDE.md` | "Baselines" item 4 PLDM description: 5-term, four tunable, regime-dependent claim |
| `notebooks/01_smoke_5k_analysis.ipynb` | Section 7 + PLDM-specific decision string |

## Decision string

`DATA_SCALE_BOUND: PLDM-A PR=5.97 (<= 16); both regularisers collapse
on 5 cases. r2=0.970. -> Session 5.5 (expand to 10-12 cases) on BOTH
SIGReg and PLDM.`

## Five-variant comparison (this session and Session 5 together)

| Variant            | Anti-collapse     | Proj | PR    | r2    | Quadrant      |
|--------------------|-------------------|------|-------|-------|---------------|
| A: SIGReg + BN     | 2-term LeWM       | BN   |  1.025| 0.779 | TRIVIAL       |
| B: SIGReg + LN     | 2-term LeWM       | LN   |  1.135| 0.452 | DEAD          |
| C: VICReg + BN     | 2-term VICReg     | BN   | 17.463| 0.887 | TRIVIAL_LITE  |
| D: VICReg + LN     | 2-term VICReg     | LN   |  7.588| 0.803 | TRIVIAL       |
| PLDM-A             | 5-term VICReg+IDM | BN   |  5.966| 0.970 | TRIVIAL       |

## Suggested next steps

**Session 5.5: expand the case subset to 10-12 cases and re-run.**
The methodological question Session 5.5 answers: does PR / r2 land in
HEALTHY when the encoder has more training samples (~30 sub-trajectories
instead of 16)? Three outcomes are interesting:

- HEALTHY (any variant): the trivial-solution failure was data-scale-bound;
  Session 6 proceeds with the winning variant and the paper reports the
  case-count transition.
- TRIVIAL/DEAD persists: the failure is structural, not data-scale-bound;
  Session 6 reads as "no 2-term or 5-term loss escapes collapse-to-c on
  this regime" and the paper's contribution sharpens to the diagnostic
  itself plus an alternative-architecture proposal (e.g., symmetry
  augmentation per Open Q6, phi_t conditioning per D16 alternative).
- Mixed: one variant clears, others don't. Triggers a 2-variant
  expansion (just the winner) at the next case count.

Session 5.5 should run SIGReg + BN (Run A on the new subset) AND
PLDM-A (PLDM on the new subset). VICReg + BN is the obvious third if
budget allows; VICReg + LN and SIGReg + LN are skipped unless the
first three are ambiguous.

After Session 5.5, the next gating step is Hydra + lambda bisection
(Session 6) once a healthy variant exists.

## Open items

- The "7-term VICReg" framing in `SESSION5_PLDM_BASELINE.md` (the plan
  for this session) is left in place as a historical record of how the
  plan was written under the D8 misreading. D30 supersedes; future
  documents should cite D30 / arXiv:2502.14819 Appendix D.1.1
  directly.
- The IDM term's interaction with the static-c adaptation deserves
  more thought. The paper's IDM predicts a per-step action that varies
  meaningfully across time; ours predicts a static c that is constant.
  An IDM that is too easy to satisfy may actually PRESSURE the encoder
  toward collapse-to-c (the easiest way to make c decodable from any
  pair is to have z be constant per case). Consider a stricter IDM
  variant for Session 5.5: predict (c, t) -> z_{t+1} - z_t, or
  predict a per-step quantity from the dataset (e.g., the vortex
  centroid position) instead of the static c.
- The predictor architectural difference (PLDM's single-step GRU vs
  ours' AdaLN-Zero transformer) is a known confound. Per the plan, this
  is the second-order ablation; first the data-scale question gets
  answered by Session 5.5.
- The default PLDM weights (all 1.0) are not paper-grade. Once Session
  5.5 produces a healthy run, a lambda bisection (Session 6 scope) on
  alpha / beta / delta / omega should be run with the same protocol as
  the SIGReg lambda bisection.
