# Reproducibility

## Environment

The confirmed research environment used Python 3.12.5 with NumPy 1.26.4,
SciPy 1.13.1, pandas 2.2.2, scikit-learn 1.5.2, and Matplotlib 3.9.2. The
Mixture Density Network (MDN) reference used TensorFlow 2.21.0, Keras 3.14.1,
and TensorFlow Probability 0.25.0.

`requirements.txt` is the primary Gradient Boosting Decision Tree (GBDT)
environment. `requirements-full.txt` adds the optional neural backend.

## Seeds and independence

Every public experiment exposes a root seed. The bootstrap derives independent
child streams from the root seed, outer-dataset index, bootstrap-replicate index,
and purpose-specific stream identifier. A bootstrap draw resamples whole
trajectory indices and applies the same index vector to state, action, mediator,
and reward arrays.

Conditional-return Monte Carlo draws use operation-local child seeds. Batched
rows receive independent draws; repeated calls with the same inputs and seed are
reproducible. Exact wall time and neural results can still vary by hardware and
TensorFlow execution details.

## Commands

Regenerate the five checked-in images from summary-level Comma-Separated Values
(CSV) files without estimator fitting:

```bash
python experiments/make_figures.py
```

Run only the two sensitivity figures:

```bash
python experiments/sensitivity_analysis.py
```

Fast GBDT bootstrap smoke test:

```bash
python experiments/bootstrap_calibration.py \
  --n 8 \
  --horizon 4 \
  --outer-trials 2 \
  --bootstrap-replicates 3 \
  --truth-trajectories 200 \
  --n-folds 2 \
  --n-mc 5 \
  --gbdt-n-estimators 10 \
  --taus 0.1,0.5 \
  --outcome-backend gbdt \
  --output-dir outputs/smoke_bootstrap
```

Small default demonstration:

```bash
python experiments/bootstrap_calibration.py \
  --n 40 --horizon 20 \
  --outer-trials 3 --bootstrap-replicates 20 \
  --outcome-backend gbdt
```

The full research calibration was:

```bash
python experiments/bootstrap_calibration.py \
  --n 40 --horizon 20 \
  --taus 0.10,0.25,0.50 \
  --outer-trials 30 --bootstrap-replicates 100 \
  --weight-type frontdoor --outcome-backend gbdt
```

The full command is documented for provenance and is intentionally not a
default. It performs 3,030 complete estimator fits and can take hours depending
on hardware.

The paired backend benchmark and predictive calibration require the full
requirements:

```bash
python experiments/backend_benchmark.py --refits 5
python experiments/outcome_calibration.py --datasets 5
```

## Output schemas

The release keeps only four compact files:

- `backend_benchmark_summary.csv`: computation and procedural variability.
- `outcome_calibration_summary.csv`: interval and quantile predictive calibration.
- `bootstrap_summary.csv`: analytic, bootstrap-normal, and percentile summaries.
- `sensitivity_summary.csv`: point error, uncertainty calibration, and ESS/N by `N`, `T`, and `tau`.

Demo commands may create trial and replicate outputs under `outputs/`. These are
ignored by version control. Large raw replicate files should be archived outside
the public repository; only reviewed summaries should be promoted to `results/`.

## Numerical provenance

Release numbers were mechanically derived from completed experiment files in
the private research workspace. The release does not contain those large raw
inputs. Every plotted value can be reconstructed from the four checked-in
summary files using `experiments/make_figures.py`.

Coverage uncertainty uses a binomial MCSE and a two-sided Wilson 95% interval.
Empirical SD uncertainty uses the approximation
`empirical_sd / sqrt(2 * (outer_trials - 1))`, which assumes independent,
approximately normal outer estimates.
