"""Trajectory-level nonparametric bootstrap for one finance CMDP setting.

Each bootstrap replicate resamples complete trajectories and then calls the
existing QOPE experiment entry point.  Consequently, cross-fitted mediator or
behavior nuisance models, the selected pooled outcome learner, and the QOPE estimator are all refit in
every replicate.  No fitted nuisance object from the original sample is reused.

The script supports one (N, T) configuration and multiple quantile levels.  It
writes replicate estimates, outer-trial intervals, and analytic-vs-bootstrap
calibration summaries without modifying estimator or analytic-CI code.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, cast

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments._estimation import qope_results, truth_for_horizon
from src.finance_cmdp_sim import generate_finance_cmdp


DEFAULT_N = 40
DEFAULT_T = 20
DEFAULT_OUTER_TRIALS = 3
DEFAULT_BOOTSTRAP_REPLICATES = 20
DEFAULT_TAUS = (0.10, 0.25, 0.50)
NORMAL_95_CRITICAL_VALUE = 1.96  # Matches the existing analytic CI exactly.
TRAJECTORY_FIELDS = ("states", "actions", "mediators", "rewards")
OUTPUT_SCHEMA_VERSION = 3
REPLICATE_WEIGHT_DIAGNOSTIC_FIELDS = (
    "ess",
    "ess_fraction",
    "weight_max",
    "weight_p95",
    "weight_p99",
    "clipping_fraction",
)
OUTER_WEIGHT_SUMMARY_STATS = ("median", "q05", "min")
RESUME_CONFIG_FIELDS = (
    "n",
    "horizon",
    "outer_trials",
    "bootstrap_replicates",
    "taus",
    "weight_type",
    "outcome_backend",
    "gamma",
    "truth_trajectories",
    "n_folds",
    "n_mc",
    "mdn_inference_batch_size",
    "seed",
    "clip_ratio",
    "nuisance_clip",
    "optimize_maxiter",
    "density_bandwidth",
    "mdn_components",
    "mdn_hidden_dims",
    "mdn_lr",
    "mdn_batch_size",
    "mdn_epochs",
    "mdn_min_sigma",
    "gbdt_n_estimators",
    "gbdt_learning_rate",
    "gbdt_max_depth",
    "gbdt_min_samples_leaf",
    "gbdt_subsample",
    "gbdt_min_sigma",
    "gbdt_residual_n_folds",
)


def _parse_csv(text: str, cast_type) -> Tuple:
    try:
        values = tuple(
            cast_type(item.strip()) for item in text.split(",") if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid comma-separated value: {text}"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def _validate_args(args: argparse.Namespace) -> None:
    positive_fields = (
        "n",
        "horizon",
        "outer_trials",
        "bootstrap_replicates",
        "truth_trajectories",
        "n_folds",
        "n_mc",
        "mdn_inference_batch_size",
        "optimize_maxiter",
        "mdn_components",
        "mdn_batch_size",
        "mdn_epochs",
        "gbdt_n_estimators",
        "gbdt_max_depth",
        "gbdt_min_samples_leaf",
        "gbdt_residual_n_folds",
        "jobs",
    )
    for field in positive_fields:
        if int(getattr(args, field)) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.outer_trials < 2:
        raise ValueError("--outer-trials must be at least 2 for empirical SD")
    if args.bootstrap_replicates < 3:
        raise ValueError(
            "--bootstrap-replicates must be at least 3 for bootstrap SE and skewness"
        )
    if args.n_folds > args.n:
        raise ValueError("--n-folds cannot exceed N")
    if args.jobs > args.outer_trials:
        raise ValueError("--jobs cannot exceed --outer-trials")
    if len(set(args.taus)) != len(args.taus):
        raise ValueError("--taus must not contain duplicates")
    if any(not 0.0 < tau < 1.0 for tau in args.taus):
        raise ValueError("all taus must lie in (0, 1)")
    if any(value <= 0 for value in args.mdn_hidden_dims):
        raise ValueError("all MDN hidden dimensions must be positive")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("--gamma must lie in (0, 1]")
    if args.clip_ratio < 1.0:
        raise ValueError("--clip-ratio must be at least 1")
    if not 0.0 < args.nuisance_clip < 1.0:
        raise ValueError("--nuisance-clip must lie in (0, 1)")
    if args.mdn_lr <= 0.0 or args.mdn_min_sigma <= 0.0:
        raise ValueError("MDN learning rate and minimum sigma must be positive")
    if args.gbdt_learning_rate <= 0.0 or args.gbdt_min_sigma <= 0.0:
        raise ValueError("GBDT learning rate and minimum sigma must be positive")
    if args.gbdt_residual_n_folds < 2:
        raise ValueError("--gbdt-residual-n-folds must be at least 2")
    if not 0.0 < args.gbdt_subsample <= 1.0:
        raise ValueError("--gbdt-subsample must lie in (0, 1]")
    if args.density_bandwidth is not None and args.density_bandwidth <= 0.0:
        raise ValueError("--density-bandwidth must be positive when supplied")


def _estimator_args(config: Mapping[str, object]) -> argparse.Namespace:
    """Namespace expected by the unchanged finance experiment fit helper."""
    return argparse.Namespace(**dict(config))


def resample_trajectories(
    data: Mapping[str, np.ndarray], indices: np.ndarray
) -> Dict[str, np.ndarray]:
    """Select the same trajectory indices across every longitudinal array."""
    indices = np.asarray(indices, dtype=int).reshape(-1)
    if indices.size == 0:
        raise ValueError("bootstrap trajectory indices must not be empty")
    n = np.asarray(data["rewards"]).shape[0]
    if np.any(indices < 0) or np.any(indices >= n):
        raise IndexError("bootstrap trajectory index is out of range")
    output: Dict[str, np.ndarray] = {}
    for field in TRAJECTORY_FIELDS:
        values = np.asarray(data[field])
        if values.shape[0] != n:
            raise ValueError(f"{field} does not share the trajectory dimension")
        output[field] = values[indices].copy()
    return output


def _seed_for(seed: int, outer_trial: int, replicate: int, stream: int) -> int:
    sequence = np.random.SeedSequence(
        [int(seed) % (2**32), int(outer_trial), int(replicate), int(stream)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _sample_skewness(values: np.ndarray) -> Optional[float]:
    """Bias-corrected Fisher-Pearson sample skewness without a SciPy dependency."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) < 3:
        return None
    centered = values - np.mean(values)
    second_moment = float(np.mean(np.square(centered)))
    if second_moment <= 0.0:
        return 0.0
    uncorrected = float(np.mean(centered**3) / second_moment**1.5)
    n = len(values)
    return float(np.sqrt(n * (n - 1)) / (n - 2) * uncorrected)


def _wilson_interval(
    successes: int,
    trials: int,
    z: float = NORMAL_95_CRITICAL_VALUE,
) -> Tuple[float, float]:
    """Two-sided Wilson score interval for a binomial coverage probability."""
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials")
    probability = successes / trials
    z_squared = z**2
    denominator = 1.0 + z_squared / trials
    center = (probability + z_squared / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(
            probability * (1.0 - probability) / trials
            + z_squared / (4.0 * trials**2)
        )
        / denominator
    )
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def _summarize_replicate_weights(
    values_by_field: Mapping[str, Sequence[float]],
) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for field in REPLICATE_WEIGHT_DIAGNOSTIC_FIELDS:
        values = np.asarray(values_by_field[field], dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise FloatingPointError(f"invalid bootstrap weight diagnostic: {field}")
        output[f"bootstrap_{field}_median"] = float(np.median(values))
        output[f"bootstrap_{field}_q05"] = float(np.quantile(values, 0.05))
        output[f"bootstrap_{field}_min"] = float(np.min(values))
    return output


def _fit_qope(
    data: Dict[str, np.ndarray],
    config: Mapping[str, object],
    estimator_seed: int,
) -> Dict[float, Dict[str, float]]:
    """Perform one complete fit and release backend-specific state afterwards."""
    try:
        return qope_results(
            data=data,
            weight_type=cast(str, config["weight_type"]),
            taus=cast(Sequence[float], config["taus"]),
            args=_estimator_args(config),
            seed=estimator_seed,
        )
    finally:
        # The returned payload contains only floats; fitted models are not reused.
        if str(config.get("outcome_backend", "mdn")) == "mdn":
            import keras

            keras.backend.clear_session()
        gc.collect()


def _run_outer_trial(
    config: Mapping[str, object],
    outer_trial: int,
    truth_by_tau: Mapping[float, float],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    n = int(config["n"])
    horizon = int(config["horizon"])
    seed = int(config["seed"])
    bootstrap_replicates = int(config["bootstrap_replicates"])
    taus = tuple(float(tau) for tau in cast(Sequence[float], config["taus"]))
    outcome_backend = str(config.get("outcome_backend", "mdn"))
    # Preserve the pre-existing method label; backend is recorded separately.
    method = f"{config['weight_type']}_qope"

    # Match the single-scenario convention in finance_cmdp_experiment.py.
    trial_seed = seed + 101 * outer_trial
    data = generate_finance_cmdp(n, horizon, trial_seed, "behavior", float(config["gamma"]))
    analytic = _fit_qope(data, config, trial_seed + 31_337)

    bootstrap_by_tau: MutableMapping[float, List[float]] = {
        tau: [] for tau in taus
    }
    weight_diagnostics_by_field: MutableMapping[str, List[float]] = {
        field: [] for field in REPLICATE_WEIGHT_DIAGNOSTIC_FIELDS
    }
    replicate_rows: List[Dict[str, object]] = []
    resample_seeds: set[int] = set()
    estimator_seeds: set[int] = set()
    progress_every = max(1, min(10, bootstrap_replicates // 5))

    for replicate in range(bootstrap_replicates):
        resample_seed = _seed_for(seed, outer_trial, replicate, stream=1)
        estimator_seed = _seed_for(seed, outer_trial, replicate, stream=2)
        if resample_seed in resample_seeds or estimator_seed in estimator_seeds:
            raise RuntimeError("unexpected seed collision in bootstrap plan")
        resample_seeds.add(resample_seed)
        estimator_seeds.add(estimator_seed)

        rng = np.random.default_rng(resample_seed)
        indices = rng.integers(0, n, size=n)
        bootstrap_data = resample_trajectories(data, indices)
        replicate_results = _fit_qope(bootstrap_data, config, estimator_seed)
        n_unique = int(np.unique(indices).size)
        reference_result = replicate_results[taus[0]]
        for field in REPLICATE_WEIGHT_DIAGNOSTIC_FIELDS:
            value = float(reference_result[field])
            if not np.isfinite(value):
                raise FloatingPointError(
                    f"non-finite {field} at outer_trial={outer_trial}, "
                    f"bootstrap_replicate={replicate}"
                )
            weight_diagnostics_by_field[field].append(value)

        for tau in taus:
            result = replicate_results[tau]
            estimate = float(result["estimate"])
            bootstrap_by_tau[tau].append(estimate)
            replicate_rows.append(
                {
                    "N": n,
                    "T": horizon,
                    "outer_trial": outer_trial,
                    "bootstrap_replicate": replicate,
                    "tau": tau,
                    "method": method,
                    "outcome_backend": outcome_backend,
                    "resample_seed": resample_seed,
                    "estimator_seed": estimator_seed,
                    "n_unique_trajectories": n_unique,
                    "bootstrap_estimate": estimate,
                    "replicate_analytic_se": float(result["se"]),
                    "j0": float(result["j0"]),
                    "score_sd": float(result["score_sd"]),
                    "bandwidth": float(result["bandwidth"]),
                    "ess": float(result["ess"]),
                    "ess_fraction": float(result["ess_fraction"]),
                    "weight_max": float(result["weight_max"]),
                    "weight_p95": float(result["weight_p95"]),
                    "weight_p99": float(result["weight_p99"]),
                    "clipping_fraction": float(result["clipping_fraction"]),
                }
            )
        if (replicate + 1) % progress_every == 0 or replicate + 1 == bootstrap_replicates:
            print(
                f"outer trial {outer_trial + 1}: bootstrap "
                f"{replicate + 1}/{bootstrap_replicates}",
                flush=True,
            )

    trial_rows: List[Dict[str, object]] = []
    outer_weight_summary = _summarize_replicate_weights(
        weight_diagnostics_by_field
    )
    for tau in taus:
        point = analytic[tau]
        estimate = float(point["estimate"])
        analytic_se = float(point["se"])
        samples = np.asarray(bootstrap_by_tau[tau], dtype=float)
        if not np.all(np.isfinite(samples)):
            raise FloatingPointError(
                f"non-finite bootstrap estimate at outer_trial={outer_trial}, tau={tau}"
            )
        bootstrap_se = float(np.std(samples, ddof=1))
        percentile_low, percentile_high = np.quantile(samples, [0.025, 0.975])
        normal_low = estimate - NORMAL_95_CRITICAL_VALUE * bootstrap_se
        normal_high = estimate + NORMAL_95_CRITICAL_VALUE * bootstrap_se
        truth = float(truth_by_tau[tau])
        centered_samples = samples - estimate
        absolute_deviations = np.abs(centered_samples)
        extreme_cutoff = float(np.quantile(absolute_deviations, 0.95))
        tau_replicate_rows = [
            row for row in replicate_rows if float(row["tau"]) == tau
        ]
        if len(tau_replicate_rows) != bootstrap_replicates:
            raise RuntimeError("bootstrap replicate rows are incomplete before summary")
        for row, centered, absolute in zip(
            tau_replicate_rows, centered_samples, absolute_deviations
        ):
            row["bootstrap_centered_deviation"] = float(centered)
            row["bootstrap_abs_deviation"] = float(absolute)
            row["bootstrap_extreme_5pct"] = int(absolute >= extreme_cutoff)
        trial_rows.append(
            {
                "N": n,
                "T": horizon,
                "outer_trial": outer_trial,
                "tau": tau,
                "method": method,
                "outcome_backend": outcome_backend,
                "truth": truth,
                "estimate": estimate,
                "empirical_error": estimate - truth,
                "analytic_se": analytic_se,
                "analytic_ci_low": float(point["ci_low"]),
                "analytic_ci_high": float(point["ci_high"]),
                "analytic_coverage": int(point["ci_low"] <= truth <= point["ci_high"]),
                "analytic_ci_width": float(point["ci_high"] - point["ci_low"]),
                "bootstrap_mean": float(np.mean(samples)),
                "bootstrap_bias": float(np.mean(samples) - estimate),
                "bootstrap_se": bootstrap_se,
                "bootstrap_estimate_skewness": _sample_skewness(samples),
                "bootstrap_estimate_min": float(np.min(samples)),
                "bootstrap_estimate_q01": float(np.quantile(samples, 0.01)),
                "bootstrap_estimate_q99": float(np.quantile(samples, 0.99)),
                "bootstrap_estimate_max": float(np.max(samples)),
                "bootstrap_estimate_range": float(np.max(samples) - np.min(samples)),
                "bootstrap_estimate_max_abs_deviation": float(
                    np.max(absolute_deviations)
                ),
                "bootstrap_percentile_ci_low": float(percentile_low),
                "bootstrap_percentile_ci_high": float(percentile_high),
                "bootstrap_percentile_coverage": int(percentile_low <= truth <= percentile_high),
                "bootstrap_percentile_ci_width": float(percentile_high - percentile_low),
                "bootstrap_normal_ci_low": normal_low,
                "bootstrap_normal_ci_high": normal_high,
                "bootstrap_normal_coverage": int(normal_low <= truth <= normal_high),
                "bootstrap_normal_ci_width": normal_high - normal_low,
                "analytic_j0": float(point["j0"]),
                "analytic_score_sd": float(point["score_sd"]),
                "analytic_bandwidth": float(point["bandwidth"]),
                "analytic_ess": float(point["ess"]),
                "analytic_ess_fraction": float(point["ess_fraction"]),
                **outer_weight_summary,
                "B": bootstrap_replicates,
            }
        )
    print(f"completed outer trial {outer_trial + 1}", flush=True)
    return trial_rows, replicate_rows


def summarize_calibration(
    trial_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    groups: MutableMapping[float, List[Mapping[str, object]]] = {}
    for row in trial_rows:
        groups.setdefault(float(row["tau"]), []).append(row)

    output: List[Dict[str, object]] = []
    interval_specs = (
        ("analytic", "analytic_se", "analytic_coverage", "analytic_ci_width"),
        (
            "bootstrap_normal",
            "bootstrap_se",
            "bootstrap_normal_coverage",
            "bootstrap_normal_ci_width",
        ),
        (
            "bootstrap_percentile",
            "bootstrap_se",
            "bootstrap_percentile_coverage",
            "bootstrap_percentile_ci_width",
        ),
    )
    for tau, group in sorted(groups.items()):
        estimates = np.asarray([float(row["estimate"]) for row in group])
        truths = np.asarray([float(row["truth"]) for row in group])
        errors = estimates - truths
        skewness_values = np.asarray(
            [float(row["bootstrap_estimate_skewness"]) for row in group],
            dtype=float,
        )
        empirical_sd = (
            float(np.std(estimates, ddof=1)) if len(estimates) > 1 else None
        )
        empirical_sd_mc_se = (
            empirical_sd / np.sqrt(2.0 * (len(estimates) - 1))
            if empirical_sd is not None
            else None
        )
        empirical_sd_relative_mc_se = (
            1.0 / np.sqrt(2.0 * (len(estimates) - 1))
            if empirical_sd is not None
            else None
        )
        for interval_method, se_field, coverage_field, width_field in interval_specs:
            mean_se = float(np.mean([float(row[se_field]) for row in group]))
            coverage_values = np.asarray(
                [int(row[coverage_field]) for row in group], dtype=int
            )
            coverage_successes = int(np.sum(coverage_values))
            coverage = coverage_successes / len(coverage_values)
            coverage_wilson_low, coverage_wilson_high = _wilson_interval(
                coverage_successes, len(coverage_values)
            )
            output.append(
                {
                    "N": int(group[0]["N"]),
                    "T": int(group[0]["T"]),
                    "tau": tau,
                    "method": group[0]["method"],
                    "outcome_backend": group[0].get("outcome_backend", "mdn"),
                    "interval_method": interval_method,
                    "truth": float(truths[0]),
                    "mean_estimate": float(np.mean(estimates)),
                    "bias": float(np.mean(errors)),
                    "abs_bias": abs(float(np.mean(errors))),
                    "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                    "empirical_sd": empirical_sd,
                    "empirical_sd_mc_se_approx": empirical_sd_mc_se,
                    "empirical_sd_relative_mc_se_approx": empirical_sd_relative_mc_se,
                    "empirical_sd_uncertainty_note": (
                        "Approximate MC SE = empirical_sd/sqrt(2*(outer_trials-1)); "
                        "assumes iid approximately normal outer estimates."
                    ),
                    "mean_reported_se": mean_se,
                    "se_ratio": (
                        mean_se / empirical_sd
                        if empirical_sd is not None and empirical_sd > 0.0
                        else None
                    ),
                    "coverage": coverage,
                    "coverage_successes": coverage_successes,
                    "coverage_mc_se": float(
                        np.sqrt(coverage * (1.0 - coverage) / len(coverage_values))
                    ),
                    "coverage_wilson_95_low": coverage_wilson_low,
                    "coverage_wilson_95_high": coverage_wilson_high,
                    "mean_ci_width": float(
                        np.mean([float(row[width_field]) for row in group])
                    ),
                    "mean_bootstrap_estimate_skewness": float(
                        np.mean(skewness_values)
                    ),
                    "mean_abs_bootstrap_estimate_skewness": float(
                        np.mean(np.abs(skewness_values))
                    ),
                    "bootstrap_estimate_min": float(
                        np.min([float(row["bootstrap_estimate_min"]) for row in group])
                    ),
                    "mean_bootstrap_estimate_q01": float(
                        np.mean([float(row["bootstrap_estimate_q01"]) for row in group])
                    ),
                    "mean_bootstrap_estimate_q99": float(
                        np.mean([float(row["bootstrap_estimate_q99"]) for row in group])
                    ),
                    "bootstrap_estimate_max": float(
                        np.max([float(row["bootstrap_estimate_max"]) for row in group])
                    ),
                    "mean_bootstrap_estimate_range": float(
                        np.mean([float(row["bootstrap_estimate_range"]) for row in group])
                    ),
                    "mean_bootstrap_estimate_max_abs_deviation": float(
                        np.mean([
                            float(row["bootstrap_estimate_max_abs_deviation"])
                            for row in group
                        ])
                    ),
                    "outer_trials": len(group),
                    "B": int(group[0]["B"]),
                }
            )
    return output


def _pearson(first: Sequence[float], second: Sequence[float]) -> Optional[float]:
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize_weight_relationships(
    trial_rows: Sequence[Mapping[str, object]],
    replicate_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Describe whether bootstrap tails coincide with degenerate weights."""
    trial_by_tau: MutableMapping[float, List[Mapping[str, object]]] = {}
    replicate_by_tau: MutableMapping[float, List[Mapping[str, object]]] = {}
    for row in trial_rows:
        trial_by_tau.setdefault(float(row["tau"]), []).append(row)
    for row in replicate_rows:
        replicate_by_tau.setdefault(float(row["tau"]), []).append(row)

    output: List[Dict[str, object]] = []
    for tau, group in sorted(replicate_by_tau.items()):
        absolute = [float(row["bootstrap_abs_deviation"]) for row in group]
        squared = [value**2 for value in absolute]
        extreme = [bool(int(row["bootstrap_extreme_5pct"])) for row in group]
        for diagnostic in REPLICATE_WEIGHT_DIAGNOSTIC_FIELDS:
            values = [float(row[diagnostic]) for row in group]
            extreme_values = [
                value for value, is_extreme in zip(values, extreme) if is_extreme
            ]
            nonextreme_values = [
                value for value, is_extreme in zip(values, extreme) if not is_extreme
            ]
            extreme_mean = float(np.mean(extreme_values)) if extreme_values else None
            nonextreme_mean = (
                float(np.mean(nonextreme_values)) if nonextreme_values else None
            )
            output.append(
                {
                    "level": "replicate",
                    "tau": tau,
                    "outcome_backend": group[0].get("outcome_backend", "mdn"),
                    "diagnostic": diagnostic,
                    "degeneration_direction": (
                        "lower" if diagnostic in {"ess", "ess_fraction"} else "higher"
                    ),
                    "outer_summary_stat": None,
                    "n": len(group),
                    "n_extreme": len(extreme_values),
                    "corr_abs_deviation": _pearson(values, absolute),
                    "corr_squared_deviation": _pearson(values, squared),
                    "extreme_mean_diagnostic": extreme_mean,
                    "nonextreme_mean_diagnostic": nonextreme_mean,
                    "extreme_to_nonextreme_ratio": (
                        extreme_mean / nonextreme_mean
                        if extreme_mean is not None
                        and nonextreme_mean is not None
                        and nonextreme_mean != 0.0
                        else None
                    ),
                    "corr_bootstrap_se": None,
                    "corr_abs_skewness": None,
                    "corr_estimate_range": None,
                    "corr_max_abs_deviation": None,
                }
            )

    for tau, group in sorted(trial_by_tau.items()):
        bootstrap_se = [float(row["bootstrap_se"]) for row in group]
        abs_skewness = [
            abs(float(row["bootstrap_estimate_skewness"])) for row in group
        ]
        estimate_range = [float(row["bootstrap_estimate_range"]) for row in group]
        max_abs_deviation = [
            float(row["bootstrap_estimate_max_abs_deviation"]) for row in group
        ]
        for diagnostic in REPLICATE_WEIGHT_DIAGNOSTIC_FIELDS:
            for statistic in OUTER_WEIGHT_SUMMARY_STATS:
                field = f"bootstrap_{diagnostic}_{statistic}"
                values = [float(row[field]) for row in group]
                output.append(
                    {
                        "level": "outer_trial",
                        "tau": tau,
                        "outcome_backend": group[0].get("outcome_backend", "mdn"),
                        "diagnostic": diagnostic,
                        "degeneration_direction": (
                            "lower"
                            if diagnostic in {"ess", "ess_fraction"}
                            else "higher"
                        ),
                        "outer_summary_stat": statistic,
                        "n": len(group),
                        "n_extreme": None,
                        "corr_abs_deviation": None,
                        "corr_squared_deviation": None,
                        "extreme_mean_diagnostic": None,
                        "nonextreme_mean_diagnostic": None,
                        "extreme_to_nonextreme_ratio": None,
                        "corr_bootstrap_se": _pearson(values, bootstrap_se),
                        "corr_abs_skewness": _pearson(values, abs_skewness),
                        "corr_estimate_range": _pearson(values, estimate_range),
                        "corr_max_abs_deviation": _pearson(
                            values, max_abs_deviation
                        ),
                    }
                )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty output: {path}")
    fieldnames = list(rows[0].keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_outputs(
    output_dir: Path,
    config: Mapping[str, object],
    truth_by_tau: Mapping[float, float],
    trial_rows: Sequence[Mapping[str, object]],
    replicate_rows: Sequence[Mapping[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_trials = sorted(
        trial_rows, key=lambda row: (int(row["outer_trial"]), float(row["tau"]))
    )
    ordered_replicates = sorted(
        replicate_rows,
        key=lambda row: (
            int(row["outer_trial"]),
            int(row["bootstrap_replicate"]),
            float(row["tau"]),
        ),
    )
    summary = summarize_calibration(ordered_trials)
    weight_relationships = summarize_weight_relationships(
        ordered_trials, ordered_replicates
    )
    _write_csv(output_dir / "finance_cmdp_bootstrap_estimates.csv", ordered_replicates)
    _write_csv(output_dir / "finance_cmdp_bootstrap_trials.csv", ordered_trials)
    _write_csv(output_dir / "finance_cmdp_bootstrap_summary.csv", summary)
    _write_csv(
        output_dir / "finance_cmdp_bootstrap_weight_relationships.csv",
        weight_relationships,
    )
    payload = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "config": dict(config),
        "truth_by_tau": {str(tau): truth for tau, truth in truth_by_tau.items()},
        "trial_results": ordered_trials,
        "bootstrap_estimates": ordered_replicates,
        "summary_results": summary,
        "weight_relationships": weight_relationships,
        "definitions": {
            "resampling_unit": "Complete trajectory; all S/A/M/R time series use identical sampled indices.",
            "bootstrap_refit": "Mediator/behavior nuisance, selected pooled outcome learner, cross-fitting, and QOPE estimator are rerun in every replicate.",
            "bootstrap_se": "Across-replicate SD of QOPE point estimates with ddof=1.",
            "bootstrap_percentile_ci": "2.5% and 97.5% empirical quantiles of bootstrap point estimates.",
            "bootstrap_normal_ci": "Original point estimate +/- 1.96 * bootstrap SE.",
            "analytic_ci": "Unchanged estimator-provided analytic CI.",
            "bootstrap_estimate_skewness": "Bias-corrected Fisher-Pearson skewness within an outer trial.",
            "coverage_mc_uncertainty": "Binomial Monte Carlo SE and two-sided 95% Wilson score interval across outer trials.",
            "empirical_sd_mc_uncertainty": "Approximate SE empirical_sd/sqrt(2*(outer_trials-1)); assumes iid approximately normal outer estimates.",
            "bootstrap_extreme_5pct": "Largest 5% absolute bootstrap deviations within outer trial and tau.",
            "outer_weight_summaries": "Across bootstrap replicates: median, 5% quantile, and minimum.",
            "weight_relationships": "Descriptive Pearson relationships; replicate extremes are defined within outer trial and tau.",
        },
    }
    json_path = output_dir / "finance_cmdp_bootstrap_results.json"
    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    temporary.replace(json_path)


def _normalized_config_value(value: object) -> object:
    if isinstance(value, (tuple, list)):
        return tuple(_normalized_config_value(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def _load_checkpoint(
    output_dir: Path,
    config: Mapping[str, object],
    taus: Sequence[float],
    bootstrap_replicates: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], set[int]]:
    json_path = output_dir / "finance_cmdp_bootstrap_results.json"
    if not json_path.exists():
        return [], [], set()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != OUTPUT_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema is incompatible with current diagnostics: {json_path}"
        )
    prior_config = payload.get("config")
    if not isinstance(prior_config, dict):
        raise ValueError(f"checkpoint has no valid config: {json_path}")
    for field in RESUME_CONFIG_FIELDS:
        if _normalized_config_value(prior_config.get(field)) != _normalized_config_value(
            config.get(field)
        ):
            raise ValueError(
                f"checkpoint config mismatch for {field}: "
                f"{prior_config.get(field)!r} != {config.get(field)!r}"
            )

    trial_rows = [dict(row) for row in payload.get("trial_results", [])]
    replicate_rows = [dict(row) for row in payload.get("bootstrap_estimates", [])]
    expected_trial_rows = len(taus)
    expected_replicate_rows = len(taus) * bootstrap_replicates
    completed: set[int] = set()
    candidate_trials = {
        int(row["outer_trial"]) for row in trial_rows if "outer_trial" in row
    }
    for outer_trial in candidate_trials:
        n_trials = sum(int(row["outer_trial"]) == outer_trial for row in trial_rows)
        n_replicates = sum(
            int(row["outer_trial"]) == outer_trial for row in replicate_rows
        )
        if n_trials != expected_trial_rows or n_replicates != expected_replicate_rows:
            raise ValueError(
                f"checkpoint outer_trial={outer_trial} is incomplete: "
                f"trial rows {n_trials}/{expected_trial_rows}, "
                f"replicate rows {n_replicates}/{expected_replicate_rows}"
            )
        completed.add(outer_trial)
    return trial_rows, replicate_rows, completed


def run_bootstrap(args: argparse.Namespace) -> Dict[str, object]:
    _validate_args(args)
    config = dict(vars(args))
    config["taus"] = tuple(float(tau) for tau in args.taus)
    config["mdn_hidden_dims"] = tuple(int(value) for value in args.mdn_hidden_dims)
    output_dir = Path(args.output_dir)
    truth_by_tau = truth_for_horizon(
        args.horizon,
        args.taus,
        args.truth_trajectories,
        args.gamma,
        args.seed + 1_000_003 * args.horizon,
    )

    if args.resume:
        trial_rows, replicate_rows, completed_trials = _load_checkpoint(
            output_dir,
            config,
            args.taus,
            args.bootstrap_replicates,
        )
    else:
        trial_rows, replicate_rows, completed_trials = [], [], set()
    pending_trials = [
        trial for trial in range(args.outer_trials) if trial not in completed_trials
    ]
    if completed_trials:
        print(
            f"resuming with {len(completed_trials)}/{args.outer_trials} "
            "outer trials complete",
            flush=True,
        )
    if args.jobs == 1:
        for outer_trial in pending_trials:
            new_trials, new_replicates = _run_outer_trial(
                config, outer_trial, truth_by_tau
            )
            trial_rows.extend(new_trials)
            replicate_rows.extend(new_replicates)
            # A complete outer trial is a useful recoverable checkpoint.
            _write_outputs(
                output_dir, config, truth_by_tau, trial_rows, replicate_rows
            )
    else:
        # Spawn is safer than forking a process after TensorFlow has initialized.
        with ProcessPoolExecutor(
            max_workers=args.jobs, mp_context=get_context("spawn")
        ) as executor:
            futures = {
                executor.submit(_run_outer_trial, config, trial, truth_by_tau): trial
                for trial in pending_trials
            }
            for future in as_completed(futures):
                new_trials, new_replicates = future.result()
                trial_rows.extend(new_trials)
                replicate_rows.extend(new_replicates)
                _write_outputs(
                    output_dir, config, truth_by_tau, trial_rows, replicate_rows
                )

    _write_outputs(output_dir, config, truth_by_tau, trial_rows, replicate_rows)
    summary = summarize_calibration(trial_rows)
    weight_relationships = summarize_weight_relationships(
        trial_rows, replicate_rows
    )
    print(f"wrote {output_dir / 'finance_cmdp_bootstrap_estimates.csv'}")
    print(f"wrote {output_dir / 'finance_cmdp_bootstrap_trials.csv'}")
    print(f"wrote {output_dir / 'finance_cmdp_bootstrap_summary.csv'}")
    print(
        f"wrote {output_dir / 'finance_cmdp_bootstrap_weight_relationships.csv'}"
    )
    print(f"wrote {output_dir / 'finance_cmdp_bootstrap_results.json'}")
    return {
        "config": config,
        "truth_by_tau": truth_by_tau,
        "trial_results": trial_rows,
        "bootstrap_estimates": replicate_rows,
        "summary_results": summary,
        "weight_relationships": weight_relationships,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--horizon", type=int, default=DEFAULT_T)
    parser.add_argument("--outer-trials", type=int, default=DEFAULT_OUTER_TRIALS)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--taus", type=lambda text: _parse_csv(text, float), default=DEFAULT_TAUS)
    parser.add_argument(
        "--weight-type",
        choices=("frontdoor", "behavior", "none"),
        default="frontdoor",
    )
    parser.add_argument(
        "--outcome-backend",
        choices=("mdn", "gbdt"),
        default="gbdt",
        help="pooled conditional outcome learner used by point and all bootstrap fits",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--truth-trajectories", type=int, default=10_000)
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--n-mc", type=int, default=50)
    parser.add_argument("--mdn-inference-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--clip-ratio", type=float, default=20.0)
    parser.add_argument("--nuisance-clip", type=float, default=1e-4)
    parser.add_argument("--optimize-maxiter", type=int, default=100)
    parser.add_argument("--density-bandwidth", type=float, default=None)
    parser.add_argument("--mdn-components", type=int, default=5)
    parser.add_argument(
        "--mdn-hidden-dims",
        type=lambda text: _parse_csv(text, int),
        default=(64, 64),
    )
    parser.add_argument("--mdn-lr", type=float, default=1e-3)
    parser.add_argument("--mdn-batch-size", type=int, default=128)
    parser.add_argument("--mdn-epochs", type=int, default=50)
    parser.add_argument("--mdn-min-sigma", type=float, default=1e-3)
    parser.add_argument("--mdn-verbose", action="store_true")
    parser.add_argument("--gbdt-n-estimators", type=int, default=100)
    parser.add_argument("--gbdt-learning-rate", type=float, default=0.05)
    parser.add_argument("--gbdt-max-depth", type=int, default=3)
    parser.add_argument("--gbdt-min-samples-leaf", type=int, default=5)
    parser.add_argument("--gbdt-subsample", type=float, default=1.0)
    parser.add_argument("--gbdt-min-sigma", type=float, default=1e-3)
    parser.add_argument("--gbdt-residual-n-folds", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from complete outer-trial checkpoints in --output-dir",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bootstrap_demo"),
    )
    return parser


def main() -> None:
    run_bootstrap(build_parser().parse_args())


if __name__ == "__main__":
    main()
