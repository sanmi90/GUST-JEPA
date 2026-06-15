# Track C-full: 5-fold grouped-CV held-out wake closure (canonical, H16)

| family | median | mean | IQR | min | max | folds |
|---|---|---|---|---|---|---|
| jepa | +0.675 | +0.684 | [+0.670,+0.737] | +0.566 | +0.771 | 5 |
| ctrl_recon | +0.695 | +0.594 | [+0.583,+0.719] | +0.020 | +0.953 | 5 |

**Verdict: WEAK** (jepa median +0.675 vs matched-recon +0.695, gap -0.020).
5-fold held-out-case wake R^2 distribution; STRONG if the predictive median clears the matched reconstructive control by >0.2 across folds. Every case predicted by an encoder that never saw its case (F3).
