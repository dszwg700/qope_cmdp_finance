# Methodology

## Scope and attribution

This project is a finite-horizon engineering integration. Quantile Off-Policy
Evaluation (QOPE), the Doubly Robust (DR) quantile construction, conditional
generative nuisance approximation, and sandwich-style inference build on Xu et
al. (2022). Mediator-based reasoning under a Confounded Markov Decision Process
(CMDP) builds on Shi et al. (2022/2024). The implementation does not claim a new
identification theorem and is not a line-by-line reproduction of either paper.

Shi et al. study a marginalized infinite-horizon mean-value estimator. This
repository instead studies a finite-horizon return quantile with full-trajectory
weights and an outcome augmentation. The two must not be presented as the same
estimator.

## Observed data and target

One logged trajectory contains states `S`, actions `A`, mediators `M`, and
rewards `R` for `T` stages. With discount factor `gamma`, its remaining return
from stage `t` is

```text
G_t = R_t + gamma R_{t+1} + ... + gamma^(T-t-1) R_{T-1}.
```

For each quantile level `tau`, the estimator targets a quantile `eta_tau` of the
discounted cumulative return distribution under the target policy. The target
policy maps observable state to action probabilities. It cannot use `latent_u`.

## Cross-fitted nuisance pipeline

Outer trajectory folds separate nuisance training and estimating-equation
evaluation:

1. Fit `P(M | S,A)` for front-door-style weights, or `P(A | S)` for the behavior-weight ablation.
2. Pool stage-specific history/action rows from training trajectories.
3. Fit a conditional remaining-return distribution on the pooled rows.
4. Draw target-action and conditional-return Monte Carlo samples for held-out trajectories.
5. Form trajectory weights and the DR pinball-loss objective on held-out data.
6. Combine held-out folds and solve separately for every `tau`.

The history representation is fixed dimensional through padded state, action,
mediator, and reward histories plus masks and normalized time.

## Front-door-style trajectory weights

At a stage, the implemented mediator ratio compares the mediator probability
averaged over target-policy actions with the mediator probability under the
observed action. Step ratios are clipped to the configured interval, and the
full trajectory weight is their product. Prefix products enter the augmentation
terms.

Clipping is a fixed experiment setting, not tuned to improve published results.
Effective Sample Size (ESS) is reported as

```text
ESS = (sum_i w_i)^2 / sum_i w_i^2.
```

The code rescales weights before this calculation to avoid overflow without
changing ESS.

## Conditional outcome distributions

### OOF-GBDT primary backend

The Gradient Boosting Decision Tree (GBDT) learner fits:

1. a conditional location model `mu(X)`;
2. a conditional squared-residual/scale model;
3. a standardized empirical residual law;
4. samples `mu(X) + sigma(X) * residual_draw`.

Location residuals are Out-of-Fold (OOF) at the trajectory-group level. Final
location and scale prediction models use all training rows, while the residual
law uses predictions from models that did not see the corresponding trajectory.
A minimum conditional scale prevents numerical instability. The empirical
Cumulative Distribution Function (CDF) and a one-dimensional residual Kernel
Density Estimation (KDE) provide CDF and Probability Density Function (PDF)
queries with scale adjustment.

### MDN reference backend

The Mixture Density Network (MDN) represents the conditional return distribution
as a learned Gaussian mixture. It remains a flexible generative nuisance
reference and procedural-stability ablation. Both backends expose compatible
`fit`, `sample`, `cdf`, and `pdf` operations used by the same estimator.

## Quantile objective and analytic uncertainty

The estimator minimizes the mean cross-fitted DR pinball-loss objective using a
bounded scalar optimizer. Its estimating-equation score combines the weighted
observed indicator and outcome-model augmentation indicators.

At the fitted quantile, `score_sd` is the cross-trajectory score Standard
Deviation (SD). The derivative/density term `j0` is estimated with Gaussian KDE.
For `N` trajectories, the implemented analytic Standard Error (SE) is

```text
SE = score_sd / (sqrt(N) * j0),
```

and the unchanged Wald interval is `eta +/- 1.96 * SE`. This formula can omit
finite-sample variation introduced by learned nuisances and weighting; the
bootstrap evaluates that issue without changing the analytic formula.

## Full-refit trajectory bootstrap

The nonparametric bootstrap resamples complete trajectories. Every replicate
refits the mediator/behavior nuisance, the pooled outcome distribution, and the
QOPE estimator. Holding an original nuisance model fixed would fail to propagate
its sampling variation and is deliberately not supported.

Bootstrap-normal intervals center a normal interval at the original point
estimate using the bootstrap Standard Error. Percentile intervals use empirical
bootstrap quantiles and are retained as a robustness check.

## References

- Xu et al. (2022), [arXiv:2212.14466](https://arxiv.org/abs/2212.14466).
- Shi et al. (2022/2024), [DOI:10.1080/01621459.2022.2110878](https://doi.org/10.1080/01621459.2022.2110878).
