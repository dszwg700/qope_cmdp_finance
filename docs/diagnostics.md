# Finite-sample diagnostics

## Why diagnosis precedes formula changes

The project freezes the Data-Generating Process (DGP), policy definitions,
point-estimation logic, analytic Standard Error (SE), importance-weight clipping,
and Out-of-Fold Gradient Boosting Decision Tree (OOF-GBDT) learner. Diagnostics
compare reported uncertainty with repeated-sampling behavior before proposing
any mathematical change.

## Outcome-distribution calibration

Point-prediction Root Mean Squared Error (RMSE) only checks the conditional
location. Quantile Off-Policy Evaluation (QOPE) also needs distributional spread.
The predictive experiment therefore records:

- 50%, 80%, and 90% central interval coverage;
- empirical coverage at nominal quantiles;
- predictive Standard Deviation (SD) and Interquartile Range (IQR);
- Continuous Ranked Probability Score (CRPS);
- pinball loss.

In-sample GBDT residuals were severely compressed. OOF residuals increased
predictive SD from `0.00739` to `0.02957`, improved 90% interval coverage from
`0.292` to `0.719`, and reduced CRPS from `0.02052` to `0.01787`, while mean
RMSE remained approximately `0.0317`. Tail quantiles remain imperfect.

## Standard-error calibration

The main calibration statistic is

```text
SE ratio = mean reported SE / empirical SD across outer datasets.
```

- Below one: uncertainty is understated on average.
- Near one: reported and empirical scales agree.
- Above one: uncertainty is conservative on average.

Coverage is a binomial Monte Carlo estimate. The release reports its Monte Carlo
Standard Error (MCSE) and a Wilson 95% interval. With 30 outer datasets, Wilson
intervals remain wide; observed coverage should not be interpreted as exact.

At `N=40`, `T=20`, analytic SE ratios were `0.539`, `0.804`, and `1.032` for
`tau=0.10`, `0.25`, and `0.50`. Full-refit trajectory-bootstrap ratios were
`1.153`, `1.114`, and `1.061`. This supports the finite-sample diagnosis that
weighting and nuisance learning variation is not fully represented by the
plug-in analytic formula, especially in the lower tail.

## Importance-weight diagnostics

Each point fit and bootstrap replicate can report:

- ESS and ESS/N;
- maximum, 95th, and 99th percentile full weight;
- clipping fraction;
- `j0`, `score_sd`, and KDE bandwidth;
- bootstrap distribution skewness and extremes.

At fixed `N=40`, point-fit mean ESS/N falls from `0.0899` at `T=12` to `0.0590`
at `T=40`, a reduction of about 34%. The bootstrap-replicate median follows the
same direction. This is finite-sample weight degeneration, not evidence that
sequential change-of-measure weighting is mathematically invalid.

## Reading the release figures

- `backend_comparison.png`: paired refit runtime and incremental process memory.
- `outcome_calibration.png`: residual calibration affects distributional spread, not just mean prediction.
- `bootstrap_calibration.png`: analytic versus full-refit bootstrap SE ratio and coverage.
- `sample_size_sensitivity.png`: point error averaged over the three reported quantiles.
- `horizon_ess_sensitivity.png`: trajectory horizon versus weight concentration.

The underlying summary rows are retained in `results/`; raw bootstrap replicate
files are intentionally excluded from the release directory.
