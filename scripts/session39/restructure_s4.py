#!/usr/bin/env python3
"""Session 39 comparison-led restructure of paper/sections/section_4_results.tex.

Reassembles the frozen Results wrapper into the six redesign beats
(paper_redesign.md section 4) by SLICING the original by line range, so every
number-bearing prose block is preserved byte-for-byte. Only the connective
tissue (header, preamble, subsection headings, the POD VRMSE-vs-SSIM caveat of
precision fix 2.3) is newly authored here. Reads the frozen snapshot, writes the
destination, so it is safe to re-run.

New order: 4.1 non-collapsed state (cube + carries); 4.2 representation across
dimension (dimension + decode floor + VRMSE caveat + distributed code); 4.3 the
physics the latent holds (atlas/DMD + anisotropic subspace + tab:mechanism),
placed BEFORE forecast per D-G; 4.4 forecastability (shared forecaster + rollout
multi-step para + fig:mechanism_hroll); 4.5 what wall pressure observes; 4.6
estimating the encounter from the wall.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = Path("/tmp/claude-1004/-home-carlos-GUST-JEPA/"
           "9712ce3c-0db2-4785-83d6-00d954aeaf98/scratchpad/s4_orig.tex")
DST = REPO / "paper" / "sections" / "section_4_results.tex"

lines = SRC.read_text().split("\n")


def seg(a: int, b: int) -> str:
    """Original lines a..b inclusive (1-indexed), joined verbatim."""
    return "\n".join(lines[a - 1:b])


HEADER = r"""% Results, Session 39 comparison-led restructure (paper_redesign.md section 4,
% 2026-07-11). Six beats in the redesign order: 4.1 a non-collapsed predictive
% state, 4.2 representation quality across dimension, 4.3 the physics the latent
% holds (BEFORE forecastability, decision D-G), 4.4 forecastability, 4.5 what
% wall pressure observes, 4.6 estimating the encounter from the wall. Every
% frozen number-bearing block is preserved verbatim and re-parented by
% scripts/session39/restructure_s4.py; only the connective tissue (preamble,
% subsection headings, the POD VRMSE-vs-SSIM caveat of precision fix 2.3) is
% newly authored. The published-recipe DMD prose already scopes precision fix
% 2.2 ("Fukami-lineage reconstruction, not reconstruction in general").
% British spelling; no em-dashes."""

PREAMBLE = r"""We evaluate three families of reduced-order state at matched pooled dimension:
a predictive coefficient state, a matched-supervision reconstruction
autoencoder, and a linear proper orthogonal basis, with the published-recipe
reconstruction and a $\beta$-variational autoencoder retained as reference
recipes. The controlled representational comparisons
(\S\ref{sec:res_construct} to \S\ref{sec:res_wallobs}) are reported on the
\TestSplit{} encounters, with the \BoundarySplit{} set held in reserve so that
the $|G| = 4$ extrapolation enters only as a final reported number and never as
a selection signal. The estimator and the operating envelope
(\S\ref{sec:res_estimation}) are characterised with a frozen filter across all
encounters, training, \ValSplit{}, \TestSplit{} and \BoundarySplit{} sets
alike, and the per-encounter paired tests on the \TestSplit{} and
\BoundarySplit{} sets are given in Appendix~\ref{app:reg}. Every quantity quoted
below is a held-out value at the readout horizon unless stated otherwise. One
convention holds throughout: the \PredState{} (conditioning-cube cell CLW) is
the single predictive representative carried into the forecasting and estimation
comparisons, and the \LiftState{} is the lift-sharpened variant whose accuracy
ceiling the construction section quantifies."""

SUB_41 = r"""\subsection{A non-collapsed predictive state}
\label{sec:res_construct}"""

SUB_42 = r"""\subsection{Representation quality across dimension}
\label{sec:res_compress}"""

VRMSE_CAVEAT = r"""The linear basis is not disadvantaged in field reconstruction by this
comparison, and it is worth saying why before the wake result is read. By field
VRMSE the proper orthogonal basis is the most accurate of the ten families
($\XvrmsePod$, against $\XvrmseJepaWake$ for the predictive state), and its
decoded similarity matches the nonlinear states ($\XssimPod$ against
$\XssimJepaWake$ SSIM). VRMSE rewards an energy-optimal linear basis and
penalises a nonlinear state that carries a high decode floor, so we route
field-fidelity claims through SSIM and the decoded snapshots and reserve the
wake-observable readability for the nonlinear-content claim. There the linear
basis reaches only $\XwakePod$ at $d = 32$, and no linear projection recovers
the wake at any tested dimension. The proper orthogonal basis holds the shedding
clock (\S\ref{sec:res_latent_physics}) and stays competitive on the forces; the
finding is not that it is obsolete but that no linear projection reaches the
wake."""

SUB_43 = r"""\subsection{The physics the latent state holds}
\label{sec:res_latent_physics}
\label{sec:res_flow_physics} % v2.1 alias
\label{sec:res_param_phase} % v2.1 alias
\label{sec:res_physical} % v2.1 alias"""

SUB_44 = r"""\subsection{Forecastability}
\label{sec:res_forecast}"""

SUBSUB_ROLLOUT = r"""\subsubsection{Why the state stays usable under rollout}
\label{sec:res_rollout}
\label{sec:res_controls} % v2.1 alias
\label{sec:res_horizon} % v2.1 alias
\label{sec:res_drift} % v2.1 alias"""

SUB_46 = r"""\subsection{Estimating the encounter from the wall}
\label{sec:res_estimation}"""

parts = [
    HEADER,
    "",
    PREAMBLE,
    "",
    # ---- 4.1 A non-collapsed predictive state ----
    SUB_41,
    "",
    seg(37, 37),      # \input s4_a (the cube)
    "",
    seg(41, 98),      # sec:res_carries
    "",
    seg(100, 100),    # \input tab:closure
    "",
    seg(102, 105),    # dropped fig:readability_matrix comment
    "",
    # ---- 4.2 Representation quality across dimension ----
    SUB_42,
    "",
    seg(156, 156),    # \input s4_b2 (dimension + probe dilution)
    "",
    seg(39, 39),      # \input s4_b (decode floor), moved out of 4.1
    "",
    VRMSE_CAVEAT,
    "",
    seg(158, 180),    # sec:res_code (distributed wake code + decode/estimate trade)
    "",
    # ---- 4.3 The physics the latent state holds (before forecast, D-G) ----
    SUB_43,
    "",
    seg(113, 138),    # physics prose (atlas + DMD); opener reworded post-hoc
    "",
    seg(190, 207),    # protected anisotropic subspace (para 1 of old rollout)
    "",
    seg(226, 226),    # \input tab:mechanism
    "",
    seg(140, 151),    # fig:atlas
    "",
    # ---- 4.4 Forecastability ----
    SUB_44,
    "",
    seg(182, 182),    # \input s4_c (shared forecaster, direct-vs-AR, cond null)
    "",
    SUBSUB_ROLLOUT,
    "",
    seg(209, 224),    # multi-step vs single-step para (para 2 of old rollout)
    "",
    seg(228, 238),    # fig:mechanism_hroll
    "",
    # ---- 4.5 What wall pressure observes ----
    seg(240, 273),    # wallobs subsection heading + prose + gate-O
    "",
    seg(275, 314),    # sec:res_delays
    "",
    seg(316, 316),    # \input tab:recovery
    "",
    seg(318, 329),    # fig:trade
    "",
    # ---- 4.6 Estimating the encounter from the wall ----
    SUB_46,
    "",
    seg(334, 386),    # tracking + tab:family_filter + handoff
    "",
    seg(388, 399),    # fig:hero
    "",
    seg(401, 529),    # envelope block (tab:envelope, fig:cl_envelope_traces,
                      # tab:filter_error, fig:envelope)
    "",
    seg(531, 531),    # \input s4_d (assimilation grid)
    "",
]

DST.write_text("\n".join(parts))
print(f"wrote {DST} ({len(DST.read_text().splitlines())} lines)")
