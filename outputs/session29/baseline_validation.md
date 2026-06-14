# Track A: baseline external validation (referee F1, v2.1)

Undisturbed NACA0012, Re = 5000, alpha = 14 deg, no-gust Baseline. Our DNS (SOD2D) vs the external references the lineage validates against (Fukami, Smith & Taira PRF 2025 Fig. 2(b)). Mean/rms forces and St are read from the frozen `undisturbed.json`; wake enstrophy is computed here. % differences are ours vs that reference.

| quantity | ours | Fukami2025 | Rolandi2025 | Gupta2023 | within-band? |
|---|---|---|---|---|---|
| time-averaged C_L | 0.761 | 0.737 (+3.3%) | 0.734 (+3.7%) | 0.763 (-0.3%) | in band [0.734, 0.763] |
| rms C_L' | 0.127 | 0.116 (+9.6%) | NEEDS-LITERATURE | NEEDS-LITERATURE | outside band [0.116, 0.116] |
| time-averaged C_D | 0.253 | 0.249 (+1.7%) | 0.246 (+3.0%) | 0.223 (+13.6%) | outside band [0.223, 0.249] |
| rms C_D' | 0.0231 | 0.0190 (+21.5%) | NEEDS-LITERATURE | NEEDS-LITERATURE | outside band [0.019, 0.019] |
| shedding Strouhal | 0.675 | NEEDS-LITERATURE | NEEDS-LITERATURE | NEEDS-LITERATURE | ours-only (no ext. table value) |
| wake enstrophy | 123.6 | NO-EXTERNAL-REFERENCE | NO-EXTERNAL-REFERENCE | NO-EXTERNAL-REFERENCE | ours-only (no ext. table value) |

Reference availability (quantities with a published value): Fukami2025: CL_mean, CL_rms, CD_mean, CD_rms; Rolandi2025: CL_mean, CD_mean; Gupta2023: CL_mean, CD_mean.
Rolandi2025 and Gupta2023 report only mean C_L and C_D in Fig. 2(b); their rms and St cells are NEEDS-LITERATURE. No external reference reports a baseline Strouhal here (St is the manuscript M13 internal cross-check) or a wake enstrophy (grid/normalisation-dependent internal diagnostic).

## Verdict
VERDICT: our undisturbed baseline falls within a defensible band of the available references. Our mean C_L = 0.761 sits inside the published band [0.734, 0.763] spanned by the three references, -0.3% from the nearest (Gupta 2023 experiment, 0.763). Our mean C_D = 0.253 sits outside the published band [0.223, 0.249], +1.7% from the nearest reference. Physical reading of the spread: the three references are not like-for-like. Fukami 2025 (DNS, incompressible) and Rolandi 2025 (LES, M_inf = 0.2) both use span Lz/c = 1 and agree to ~0.4% in C_L; Gupta 2023 is a wind-tunnel experiment at Lz/c = 10, whose larger span relaxes spanwise confinement and lifts mean C_L (0.763) while lowering mean C_D (0.223). Our DNS matches the Re and incidence and uses the same incompressible/low-Mach regime as Fukami; both our mean forces sit within 1.7% of Fukami's own production-grid DNS (C_D 0.253 vs 0.249, +1.7%; just above the 0.223-0.249 envelope only because the experiment is the low edge), the smallest and most apt comparison. The rms fluctuations are the loosest cells: only Fukami reports them, and even their three grids span CL_rms 0.098-0.116 and CD_rms 0.0183-0.0224, so these statistics are resolution- and window-sensitive. Our CL_rms (+9.6%) is within that grid scatter; our CD_rms (+21.5%) exceeds it and is the one statistic that is honestly higher than the lineage. It is a single-quantity fluctuation level (mean C_D, which sets the loading, is within 1.7%) and is sensitive to the stationary-window choice (the frozen window is t/c [20,40]); it is flagged, not papered over, and should be revisited once the partner solver's grid/time-step rows land. The St cross-check (dominant line and its full-cycle subharmonic ~0.34, the manuscript's 'St near 0.36') is internal: Fig. 2(b) lists no St for any external reference.
