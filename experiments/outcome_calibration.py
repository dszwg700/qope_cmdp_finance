"""Held-out predictive calibration for pooled MDN and GBDT outcome learners.

Training/evaluation splits are at trajectory level.  Features and conditional
targets are built by the same estimator methods used for QOPE cross-fitting.
"""
from __future__ import annotations

import argparse
import csv
import gc
import resource
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
from sklearn.model_selection import KFold

from src.finance_cmdp_sim import generate_finance_cmdp
from src.gbdtlearner_cmdp import GBDTLearnerConfig
from src.mdnlearner_cmdp import MDNLearnerConfig
from src.qope_cmdp_dr import (
    CMDPDRConfig,
    CMDPDRQuantileEstimator,
    discounted_returns,
)


QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
INTERVALS = ((0.50, 0.25, 0.75), (0.80, 0.10, 0.90), (0.90, 0.05, 0.95))
DEFAULT_SEED = 20260827


def _peak_rss_mb() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / (1024.0 * 1024.0) if sys.platform == "darwin" else rss / 1024.0


def _fit_seed(seed: int, dataset_id: int, fold: int) -> int:
    sequence = np.random.SeedSequence([seed, dataset_id, fold, 91_019])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _config(
    backend: str,
    seed: int,
    n_folds: int,
    residual_n_folds: int,
) -> CMDPDRConfig:
    return CMDPDRConfig(
        gamma=0.99,
        taus=(0.5,),
        n_folds=n_folds,
        n_mc=50,
        weight_type="none",
        random_state=seed,
        mdn_config=MDNLearnerConfig(
            n_components=5,
            hidden_dims=(64, 64),
            lr=1e-3,
            batch_size=128,
            epochs=50,
            seed=seed,
            verbose=False,
            min_sigma=1e-3,
        ),
        outcome_backend=backend,
        gbdt_config=GBDTLearnerConfig(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=5,
            subsample=1.0,
            min_sigma=1e-3,
            random_state=seed,
            residual_n_folds=residual_n_folds,
        ),
    )


def _empirical_crps(draws: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """CRPS of equally weighted empirical draws, computed without new deps."""
    sorted_draws = np.sort(np.asarray(draws, dtype=float), axis=1)
    m = sorted_draws.shape[1]
    coefficients = 2.0 * np.arange(1, m + 1) - m - 1.0
    half_pairwise = sorted_draws @ coefficients / float(m * m)
    return np.mean(np.abs(draws - observed[:, None]), axis=1) - half_pairwise


def _pinball(observed: np.ndarray, predicted: np.ndarray, q: float) -> np.ndarray:
    residual = observed - predicted
    return np.maximum(q * residual, (q - 1.0) * residual)


def _evaluate_backend(
    data: Mapping[str, np.ndarray],
    dataset_id: int,
    data_seed: int,
    backend: str,
    seed: int,
    n_folds: int,
    n_draws: int,
    residual_n_folds: int,
) -> List[Dict[str, object]]:
    n, horizon = np.asarray(data["rewards"]).shape
    returns = discounted_returns(data["rewards"], 0.99)
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows: List[Dict[str, object]] = []
    for fold, (train_idx, eval_idx) in enumerate(splitter.split(np.arange(n))):
        if np.intersect1d(train_idx, eval_idx).size:
            raise RuntimeError("trajectory-level cross-fitting split overlaps")
        fit_seed = _fit_seed(seed, dataset_id, fold)
        estimator = CMDPDRQuantileEstimator(
            [0, 1], _config(backend, fit_seed, n_folds, residual_n_folds)
        )
        try:
            model = estimator._fit_fold(
                data["states"], data["actions"], data["mediators"],
                data["rewards"], train_idx,
            )
            outcome_model = model["mdn"]
            for t in range(horizon):
                features = estimator._make_history_action_features(
                    data["states"], data["actions"], data["rewards"],
                    data["mediators"], eval_idx, t, data["actions"][eval_idx, t],
                )
                draw_seed = _fit_seed(fit_seed, t, 17)
                draws = np.asarray(
                    outcome_model.sample(features, n_samples=n_draws, random_state=draw_seed),
                    dtype=float,
                )
                if draws.shape != (len(eval_idx), n_draws) or not np.all(np.isfinite(draws)):
                    raise FloatingPointError(
                        f"invalid predictive draws: dataset={dataset_id}, backend={backend}, "
                        f"fold={fold}, t={t}, seed={fit_seed}"
                    )
                observed = returns[eval_idx, t]
                predicted_mean = np.mean(draws, axis=1)
                predictive_sd = np.std(draws, axis=1, ddof=1)
                q_values = np.quantile(
                    draws,
                    sorted(set(QUANTILES + tuple(x for _, lo, hi in INTERVALS for x in (lo, hi)))),
                    axis=1,
                )
                q_levels = sorted(set(QUANTILES + tuple(x for _, lo, hi in INTERVALS for x in (lo, hi))))
                q_map = {q: q_values[index] for index, q in enumerate(q_levels)}
                predictive_iqr = q_map[0.75] - q_map[0.25]
                crps = _empirical_crps(draws, observed)
                interval_hits = {
                    int(level * 100): (observed >= q_map[lo]) & (observed <= q_map[hi])
                    for level, lo, hi in INTERVALS
                }
                for local_index, trajectory_id in enumerate(eval_idx):
                    for q in QUANTILES:
                        prediction = float(q_map[q][local_index])
                        numeric = [
                            observed[local_index], predicted_mean[local_index],
                            predictive_sd[local_index], predictive_iqr[local_index],
                            prediction, crps[local_index],
                        ]
                        if not np.all(np.isfinite(numeric)):
                            raise FloatingPointError(
                                f"non-finite calibration metric: dataset={dataset_id}, "
                                f"backend={backend}, trajectory={trajectory_id}, t={t}"
                            )
                        rows.append(
                            {
                                "dataset_id": dataset_id,
                                "data_seed": data_seed,
                                "backend": backend,
                                "fold": fold,
                                "fit_seed": fit_seed,
                                "trajectory_id": int(trajectory_id),
                                "t": t,
                                "quantile": q,
                                "observed_return": float(observed[local_index]),
                                "predicted_mean": float(predicted_mean[local_index]),
                                "predictive_mean_error": float(predicted_mean[local_index] - observed[local_index]),
                                "predictive_sd": float(predictive_sd[local_index]),
                                "predictive_iqr": float(predictive_iqr[local_index]),
                                "predicted_quantile": prediction,
                                "quantile_hit": int(observed[local_index] <= prediction),
                                "pinball_loss": float(_pinball(observed[local_index:local_index + 1], np.asarray([prediction]), q)[0]),
                                "interval_50_coverage": int(interval_hits[50][local_index]),
                                "interval_80_coverage": int(interval_hits[80][local_index]),
                                "interval_90_coverage": int(interval_hits[90][local_index]),
                                "crps": float(crps[local_index]),
                                "n_draws": n_draws,
                            }
                        )
        finally:
            if backend == "mdn":
                import keras

                keras.backend.clear_session()
            del estimator
            gc.collect()
    expected = n * horizon * len(QUANTILES)
    if len(rows) != expected:
        raise RuntimeError(f"incomplete calibration rows: {len(rows)} != {expected}")
    return rows


def summarize(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for backend in ("mdn", "gbdt"):
        backend_rows = [row for row in rows if row["backend"] == backend]
        # Observation-level diagnostics are repeated across quantiles; select q=.5.
        observations = [row for row in backend_rows if float(row["quantile"]) == 0.5]
        mean_errors = np.asarray([float(row["predictive_mean_error"]) for row in observations])
        output.append(
            {
                "summary_level": "backend",
                "backend": backend,
                "quantile": None,
                "n_observations": len(observations),
                "predictive_mean_bias": float(np.mean(mean_errors)),
                "rmse_predictive_mean": float(np.sqrt(np.mean(np.square(mean_errors)))),
                "interval_50_coverage": float(np.mean([int(row["interval_50_coverage"]) for row in observations])),
                "interval_80_coverage": float(np.mean([int(row["interval_80_coverage"]) for row in observations])),
                "interval_90_coverage": float(np.mean([int(row["interval_90_coverage"]) for row in observations])),
                "mean_predictive_sd": float(np.mean([float(row["predictive_sd"]) for row in observations])),
                "mean_predictive_iqr": float(np.mean([float(row["predictive_iqr"]) for row in observations])),
                "mean_crps": float(np.mean([float(row["crps"]) for row in observations])),
                "empirical_quantile_coverage": None,
                "coverage_error": None,
                "mean_pinball_loss": None,
            }
        )
        for q in QUANTILES:
            group = [row for row in backend_rows if float(row["quantile"]) == q]
            empirical = float(np.mean([int(row["quantile_hit"]) for row in group]))
            output.append(
                {
                    "summary_level": "quantile",
                    "backend": backend,
                    "quantile": q,
                    "n_observations": len(group),
                    "predictive_mean_bias": None,
                    "rmse_predictive_mean": None,
                    "interval_50_coverage": None,
                    "interval_80_coverage": None,
                    "interval_90_coverage": None,
                    "mean_predictive_sd": None,
                    "mean_predictive_iqr": None,
                    "mean_crps": None,
                    "empirical_quantile_coverage": empirical,
                    "coverage_error": empirical - q,
                    "mean_pinball_loss": float(np.mean([float(row["pinball_loss"]) for row in group])),
                }
            )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(args: argparse.Namespace) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if args.n <= 0 or args.horizon <= 0 or args.datasets <= 0 or args.n_draws < 20:
        raise ValueError("N, T, datasets must be positive and n-draws at least 20")
    if args.n_folds < 2 or args.n_folds > args.n:
        raise ValueError("n-folds must lie between 2 and N")
    if args.gbdt_residual_n_folds < 2:
        raise ValueError("gbdt-residual-n-folds must be at least 2")
    started = time.perf_counter()
    rows: List[Dict[str, object]] = []
    for dataset_id in range(args.datasets):
        data_seed = args.seed + 101 * dataset_id
        data = generate_finance_cmdp(args.n, args.horizon, data_seed, "behavior", 0.99)
        for backend in ("mdn", "gbdt"):
            backend_started = time.perf_counter()
            rows.extend(
                _evaluate_backend(
                    data, dataset_id, data_seed, backend, args.seed + 31_337,
                    args.n_folds, args.n_draws,
                    args.gbdt_residual_n_folds,
                )
            )
            print(
                f"calibration dataset {dataset_id + 1}/{args.datasets}, backend={backend}, "
                f"wall={time.perf_counter() - backend_started:.3f}s",
                flush=True,
            )
    summary = summarize(rows)
    _write_csv(args.output_dir / "finance_cmdp_outcome_calibration_rows.csv", rows)
    _write_csv(args.output_dir / "finance_cmdp_outcome_calibration_summary.csv", summary)
    print(
        f"outcome-calibration runtime={time.perf_counter() - started:.3f}s, "
        f"peak_rss={_peak_rss_mb():.1f}MB"
    )
    return rows, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--datasets", type=int, default=5)
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--n-draws", type=int, default=500)
    parser.add_argument("--gbdt-residual-n-folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/outcome_calibration"),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
