# Session 28 B6 core statistics harvest

All paired tests use the D165 convention: per-encounter absolute error, delta = err(reconstructive) - err(predictive), so delta > 0 means the predictive family is better. CIs are case-clustered (cluster unit = case, n = 10000). The HEADLINE Holm survivor count uses the encounter-level one-sided sign-test p (the exact pre-registered D165 holm.json procedure); the stricter case-RESPECTING case-level one-sided sign-test p (one value per case, honouring cluster_unit = case) is reported and Holm-corrected alongside as a conservative robustness companion.

## (a) The 12-test Holm family (predictive jepa_tf_noc vs reconstructive Fukami AE)

Cell: test_b pooled, H=16, d=64, ridge; n_tests = 12.

  test                       n_enc  delta      clustered_CI            enc_p    holm_enc  surv_enc  case_p   holm_case surv_case
  forecast/CD                  42    +0.081        [-0.088, +0.251]  0.322   1   False  0.623   1   False
  forecast/CL                  42    +0.012        [-0.430, +0.424]  0.04421   0.3537   False  0.623   1   False
  forecast/Iy                  42    +0.141        [-0.307, +0.610]  0.5612   1   False  0.1719   1   False
  forecast/circ_neg            42    +0.708        [-0.010, +1.720]  0.003958   0.03958   True   0.05469   0.5469   False
  forecast/circ_pos            42    -0.128        [-0.435, +0.194]  0.9179   1   False  0.8281   1   False
  forecast/wake_enstrophy      42   +16.845       [+1.061, +32.757]  0.08207   0.4924   False  0.05469   0.5469   False
  repr/CD                      42    +0.134        [+0.008, +0.309]  0.14   0.6999   False  0.377   1   False
  repr/CL                      42    +0.552        [+0.241, +0.949]  3.439e-05   0.0004126   True   0.0009766   0.01172   True 
  repr/Iy                      42    -0.681        [-1.400, +0.074]  0.9179   1   False  0.8281   1   False
  repr/circ_neg                42    +0.944        [+0.302, +1.969]  0.0001358   0.001493   True   0.0009766   0.01172   True 
  repr/circ_pos                42    +0.398        [+0.038, +0.912]  0.04421   0.3537   False  0.1719   1   False
  repr/wake_enstrophy          42   +33.563       [+7.791, +58.720]  0.003958   0.03958   True   0.1719   1   False

Holm survivors (encounter-level sign p, HEADLINE / D165 procedure, alpha=0.05): 4/12 -> forecast/circ_neg, repr/CL, repr/circ_neg, repr/wake_enstrophy
Holm survivors (case-level sign p, STRICTER case-respecting robustness): 2/12 -> repr/CL, repr/circ_neg

## (b) Per-family verdicts on the primary cell (predictive vs each baseline)

Cell: representational wake_enstrophy R^2, H=16, test_b pooled, d=64, ridge.

  family                 obj             seed_mean  sd     paired_delta  clustered_CI            sig?
  bvae_faith             reconstructive   -0.187  0.483     +30.359       [+4.651, +59.488]  True
  bvae_match             reconstructive   +0.532  0.196      +1.465      [-14.322, +20.664]  False
  ctrl_pred_cnn          predictive       +0.789  0.062      -4.918       [-12.858, +2.681]  False
  ctrl_pred_vit_nowake   predictive       +0.105  0.133     +26.946       [+6.227, +49.415]  True
  ctrl_recon_cnn         reconstructive   +0.346  0.454     +20.318       [-3.614, +44.079]  False
  ctrl_recon_cnnvit      reconstructive   +0.127  0.554     +37.069      [+18.740, +55.233]  True
  fukami                 reconstructive   -0.253  0.273     +33.563       [+7.791, +58.720]  True
  jepa_lstm_noc          predictive       +0.690  0.052      +5.689      [-11.918, +28.405]  False
  jepa_tf_cond           predictive       +0.775  0.074      +1.921       [-7.181, +10.891]  False
  pod                    reconstructive   -0.157  0.000     +51.184      [+24.655, +75.967]  True

Predictive lead (jepa_tf_noc) seed_mean = +0.794.

## (c) Gate-GD weak-branch hardening (matched-head reconstructive AE)

Cell: representational wake_enstrophy, H=16, test_b pooled, d=64, nonlinear probes.
Predictive = jepa_tf_noc_d64_s42; matched-head = ctrl_recon_cnn.

  probe krr_rbf:
    ctrl_recon_cnn_s0    delta=  +4.978  CI=[-12.119, +19.531]  ci_includes_zero=True
    ctrl_recon_cnn_s1    delta=  +1.204  CI=[-7.533, +10.557]  ci_includes_zero=True
    ctrl_recon_cnn_s2    delta=  +3.972  CI=[-6.118, +16.129]  ci_includes_zero=True
  probe mlp_reg:
    ctrl_recon_cnn_s0    delta=  +5.404  CI=[-5.036, +14.845]  ci_includes_zero=True
    ctrl_recon_cnn_s1    delta=  +0.233  CI=[-8.580, +6.835]  ci_includes_zero=True
    ctrl_recon_cnn_s2    delta=  +1.951  CI=[-8.103, +9.575]  ci_includes_zero=True

Weak branch statistically supported: True
  - krr_rbf: 3/3 matched-head seeds have a clustered CI that includes zero (predictive not reliably better)
  - mlp_reg: 3/3 matched-head seeds have a clustered CI that includes zero (predictive not reliably better)

Weak branch SUPPORTED: the matched-head reconstructive AE is not reliably separated from the predictive family on linear/nonlinear wake decodability after case clustering; D183 lands on WEAK (linear-decodability framing).

## (d) Fig-8 trends: per-encounter wake closure vs |G| and D (lead family)

Family jepa_tf_noc; metric: per-encounter closure = -|y_pred - y_true| (representational wake, H=16); test_b + test_c pooled.

  vs absG : Spearman rho = -0.563, perm_p = 0.0022 (n_enc=66, n_cases=14)
  vs D    : Spearman rho = -0.051, perm_p = 0.8164 (n_enc=66, n_cases=14)
