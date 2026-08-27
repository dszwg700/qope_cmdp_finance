"""Small paired benchmark for MDN and GBDT pooled outcome backends.

This is intentionally fixed to N=40, T=20, taus=(0.1, 0.25, 0.5), and two
cross-fitting folds.  Each backend/refit runs in a fresh spawned process so its
peak RSS measurement is not contaminated by the preceding fit.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from queue import Empty
from typing import Dict, List, Mapping

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np

from src.finance_cmdp_policies import target_policy_probs
from src.finance_cmdp_sim import generate_finance_cmdp
from src.gbdtlearner_cmdp import GBDTLearnerConfig
from src.mdnlearner_cmdp import MDNLearnerConfig
from src.qope_cmdp_dr import (
    CMDPDRConfig,
    CMDPDRQuantileEstimator,
    discounted_returns,
)


N_TRAJECTORIES = 40
HORIZON = 20
TAUS = (0.10, 0.25, 0.50)
N_FOLDS = 2
N_MC = 50
GAMMA = 0.99
DEFAULT_REFITS = 5
DEFAULT_SEED = 20260827
DEFAULT_TRUTH_TRAJECTORIES = 10_000


class BenchmarkCMDPDRQuantileEstimator(CMDPDRQuantileEstimator):
    """Read-only benchmark instrumentation around the unchanged solver."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pooled_total_draws_diagnostics_: Dict[str, float] = {}

    def _solve_tau(self, tau, payloads):
        if not self.pooled_total_draws_diagnostics_:
            pooled_draws = np.concatenate(
                [
                    np.asarray(draws, dtype=float).reshape(-1)
                    for payload in payloads
                    for draws in payload["total_draws"]
                ]
            )
            self.pooled_total_draws_diagnostics_ = {
                "standard_deviation": float(np.std(pooled_draws, ddof=1)),
                "iqr": float(
                    np.quantile(pooled_draws, 0.75)
                    - np.quantile(pooled_draws, 0.25)
                ),
                "count": float(len(pooled_draws)),
            }
        return super()._solve_tau(tau, payloads)


def _peak_rss_mb() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / (1024.0 * 1024.0) if sys.platform == "darwin" else rss / 1024.0


def _fit_backend(
    backend: str,
    refit: int,
    data_seed: int,
    estimator_seed: int,
    queue,
) -> None:
    """Child-process target: run exactly one estimator fit."""
    try:
        data = generate_finance_cmdp(
            N_TRAJECTORIES,
            HORIZON,
            data_seed,
            "behavior",
            GAMMA,
        )
        mdn_config = MDNLearnerConfig(
            n_components=5,
            hidden_dims=(64, 64),
            lr=1e-3,
            batch_size=128,
            epochs=50,
            seed=estimator_seed,
            verbose=False,
            min_sigma=1e-3,
        )
        gbdt_config = GBDTLearnerConfig(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=5,
            subsample=1.0,
            min_sigma=1e-3,
            random_state=estimator_seed,
        )
        config = CMDPDRConfig(
            gamma=GAMMA,
            taus=TAUS,
            n_folds=N_FOLDS,
            n_mc=N_MC,
            mdn_inference_batch_size=4096,
            weight_type="frontdoor",
            clip_ratio=20.0,
            nuisance_clip=1e-4,
            optimize_maxiter=100,
            random_state=estimator_seed,
            mdn_config=mdn_config,
            outcome_backend=backend,
            gbdt_config=gbdt_config,
        )
        estimator = BenchmarkCMDPDRQuantileEstimator([0, 1], config)
        baseline_rss_mb = _peak_rss_mb()
        started = time.perf_counter()
        estimator.fit(
            states=data["states"],
            actions=data["actions"],
            rewards=data["rewards"],
            mediators=data["mediators"],
            target_policy=target_policy_probs,
        )
        wall_seconds = time.perf_counter() - started
        peak_rss_mb = _peak_rss_mb()
        queue.put(
            {
                "ok": True,
                "backend": backend,
                "refit": refit,
                "data_seed": data_seed,
                "estimator_seed": estimator_seed,
                "wall_seconds": wall_seconds,
                "baseline_rss_mb": baseline_rss_mb,
                "peak_rss_mb": peak_rss_mb,
                "incremental_peak_rss_mb": max(peak_rss_mb - baseline_rss_mb, 0.0),
                "ess_fraction": float(
                    estimator.weight_diagnostics_["ess_fraction"]
                ),
                "pooled_total_draws_sd": estimator.pooled_total_draws_diagnostics_[
                    "standard_deviation"
                ],
                "pooled_total_draws_iqr": estimator.pooled_total_draws_diagnostics_[
                    "iqr"
                ],
                "pooled_total_draws_count": int(
                    estimator.pooled_total_draws_diagnostics_["count"]
                ),
                "quantiles": {
                    str(tau): {
                        "eta": float(estimator.results_[tau].eta),
                        "analytic_se": float(estimator.results_[tau].se),
                        "j0": float(estimator.results_[tau].j0),
                        "score_sd": float(estimator.results_[tau].score_sd),
                        "bandwidth": float(estimator.results_[tau].bandwidth),
                    }
                    for tau in TAUS
                },
            }
        )
    except BaseException:
        queue.put(
            {
                "ok": False,
                "backend": backend,
                "refit": refit,
                "traceback": traceback.format_exc(),
            }
        )


def _run_isolated(
    context: mp.context.BaseContext,
    backend: str,
    refit: int,
    data_seed: int,
    estimator_seed: int,
) -> Dict[str, object]:
    queue = context.Queue()
    process = context.Process(
        target=_fit_backend,
        args=(backend, refit, data_seed, estimator_seed, queue),
    )
    process.start()
    process.join()
    try:
        result = queue.get(timeout=5.0)
    except Empty as exc:
        raise RuntimeError(
            f"benchmark child produced no result: backend={backend}, "
            f"refit={refit}, exitcode={process.exitcode}"
        ) from exc
    finally:
        queue.close()
    if process.exitcode != 0 or not result.get("ok"):
        raise RuntimeError(
            f"benchmark child failed: backend={backend}, refit={refit}\n"
            f"{result.get('traceback', '')}"
        )
    return result


def _paired_rows(
    results: Mapping[tuple[int, str], Mapping[str, object]],
    refits: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for refit in range(refits):
        mdn = results[(refit, "mdn")]
        gbdt = results[(refit, "gbdt")]
        for tau in TAUS:
            mdn_quantile = mdn["quantiles"][str(tau)]  # type: ignore[index]
            gbdt_quantile = gbdt["quantiles"][str(tau)]  # type: ignore[index]
            mdn_eta = float(mdn_quantile["eta"])
            gbdt_eta = float(gbdt_quantile["eta"])
            mdn_ess = float(mdn["ess_fraction"])
            gbdt_ess = float(gbdt["ess_fraction"])
            mdn_score_sd = float(mdn_quantile["score_sd"])
            gbdt_score_sd = float(gbdt_quantile["score_sd"])
            mdn_j0 = float(mdn_quantile["j0"])
            gbdt_j0 = float(gbdt_quantile["j0"])
            score_sd_ratio = gbdt_score_sd / mdn_score_sd
            j0_ratio = gbdt_j0 / mdn_j0
            reconstructed_se_ratio = score_sd_ratio / j0_ratio
            rows.append(
                {
                    "refit": refit,
                    "N": N_TRAJECTORIES,
                    "T": HORIZON,
                    "data_seed": int(mdn["data_seed"]),
                    "estimator_seed": int(mdn["estimator_seed"]),
                    "tau": tau,
                    "mdn_wall_seconds": float(mdn["wall_seconds"]),
                    "gbdt_wall_seconds": float(gbdt["wall_seconds"]),
                    "wall_time_ratio_mdn_over_gbdt": (
                        float(mdn["wall_seconds"]) / float(gbdt["wall_seconds"])
                    ),
                    "mdn_baseline_rss_mb": float(mdn["baseline_rss_mb"]),
                    "gbdt_baseline_rss_mb": float(gbdt["baseline_rss_mb"]),
                    "mdn_peak_rss_mb": float(mdn["peak_rss_mb"]),
                    "gbdt_peak_rss_mb": float(gbdt["peak_rss_mb"]),
                    "mdn_incremental_peak_rss_mb": float(
                        mdn["incremental_peak_rss_mb"]
                    ),
                    "gbdt_incremental_peak_rss_mb": float(
                        gbdt["incremental_peak_rss_mb"]
                    ),
                    "mdn_eta": mdn_eta,
                    "gbdt_eta": gbdt_eta,
                    "absolute_eta_difference": abs(mdn_eta - gbdt_eta),
                    "mdn_analytic_se": float(mdn_quantile["analytic_se"]),
                    "gbdt_analytic_se": float(gbdt_quantile["analytic_se"]),
                    "mdn_j0": mdn_j0,
                    "gbdt_j0": gbdt_j0,
                    "gbdt_over_mdn_j0_ratio": j0_ratio,
                    "mdn_score_sd": mdn_score_sd,
                    "gbdt_score_sd": gbdt_score_sd,
                    "gbdt_over_mdn_score_sd_ratio": score_sd_ratio,
                    "mdn_bandwidth": float(mdn_quantile["bandwidth"]),
                    "gbdt_bandwidth": float(gbdt_quantile["bandwidth"]),
                    "reconstructed_gbdt_over_mdn_analytic_se_ratio": (
                        reconstructed_se_ratio
                    ),
                    "direct_gbdt_over_mdn_analytic_se_ratio": (
                        float(gbdt_quantile["analytic_se"])
                        / float(mdn_quantile["analytic_se"])
                    ),
                    "mdn_pooled_total_draws_sd": float(
                        mdn["pooled_total_draws_sd"]
                    ),
                    "gbdt_pooled_total_draws_sd": float(
                        gbdt["pooled_total_draws_sd"]
                    ),
                    "gbdt_over_mdn_pooled_total_draws_sd_ratio": (
                        float(gbdt["pooled_total_draws_sd"])
                        / float(mdn["pooled_total_draws_sd"])
                    ),
                    "mdn_pooled_total_draws_iqr": float(
                        mdn["pooled_total_draws_iqr"]
                    ),
                    "gbdt_pooled_total_draws_iqr": float(
                        gbdt["pooled_total_draws_iqr"]
                    ),
                    "gbdt_over_mdn_pooled_total_draws_iqr_ratio": (
                        float(gbdt["pooled_total_draws_iqr"])
                        / float(mdn["pooled_total_draws_iqr"])
                    ),
                    "pooled_total_draws_count": int(
                        mdn["pooled_total_draws_count"]
                    ),
                    "mdn_ess_fraction": mdn_ess,
                    "gbdt_ess_fraction": gbdt_ess,
                    "absolute_ess_fraction_difference": abs(mdn_ess - gbdt_ess),
                }
            )
    return rows


def _truth_by_tau(seed: int, n_trajectories: int) -> Dict[float, float]:
    truth_seed = seed + 1_000_003 * HORIZON
    data = generate_finance_cmdp(
        n_trajectories,
        HORIZON,
        truth_seed,
        "target",
        GAMMA,
    )
    returns = discounted_returns(data["rewards"], GAMMA)[:, 0]
    return {tau: float(np.quantile(returns, tau)) for tau in TAUS}


def _outer_summary(
    rows: List[Dict[str, object]],
    truth_by_tau: Mapping[float, float],
    refits: int,
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for backend in ("mdn", "gbdt"):
        for tau in TAUS:
            group = [row for row in rows if float(row["tau"]) == tau]
            estimates = np.asarray(
                [float(row[f"{backend}_eta"]) for row in group], dtype=float
            )
            analytic_se = np.asarray(
                [float(row[f"{backend}_analytic_se"]) for row in group],
                dtype=float,
            )
            truth = float(truth_by_tau[tau])
            errors = estimates - truth
            empirical_sd = float(np.std(estimates, ddof=1))
            mean_se = float(np.mean(analytic_se))
            output.append(
                {
                    "N": N_TRAJECTORIES,
                    "T": HORIZON,
                    "backend": backend,
                    "tau": tau,
                    "truth": truth,
                    "outer_datasets": refits,
                    "mean_eta": float(np.mean(estimates)),
                    "empirical_sd": empirical_sd,
                    "mean_analytic_se": mean_se,
                    "se_ratio": mean_se / empirical_sd,
                    "bias": float(np.mean(errors)),
                    "mae": float(np.mean(np.abs(errors))),
                }
            )
    return output


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(
    results: Mapping[tuple[int, str], Mapping[str, object]],
    rows: List[Dict[str, object]],
    outer_summary: List[Dict[str, object]],
    refits: int,
) -> None:
    print("\nOutcome backend benchmark summary")
    print(f"configuration: N={N_TRAJECTORIES}, T={HORIZON}, refits={refits}, n_folds={N_FOLDS}")
    print("backend   mean wall(s)   mean peak RSS(MB)   mean incremental peak(MB)")
    for backend in ("mdn", "gbdt"):
        backend_results = [results[(refit, backend)] for refit in range(refits)]
        print(
            f"{backend:<8} "
            f"{np.mean([float(row['wall_seconds']) for row in backend_results]):>12.3f} "
            f"{np.mean([float(row['peak_rss_mb']) for row in backend_results]):>19.1f} "
            f"{np.mean([float(row['incremental_peak_rss_mb']) for row in backend_results]):>27.1f}"
        )
    print("tau   mean MDN eta   mean GBDT eta   mean |difference|   MDN SE   GBDT SE   ESS/N")
    for tau in TAUS:
        group = [row for row in rows if float(row["tau"]) == tau]
        print(
            f"{tau:<4g} "
            f"{np.mean([float(row['mdn_eta']) for row in group]):>13.6f} "
            f"{np.mean([float(row['gbdt_eta']) for row in group]):>15.6f} "
            f"{np.mean([float(row['absolute_eta_difference']) for row in group]):>19.6f} "
            f"{np.mean([float(row['mdn_analytic_se']) for row in group]):>8.5f} "
            f"{np.mean([float(row['gbdt_analytic_se']) for row in group]):>9.5f} "
            f"{np.mean([float(row['mdn_ess_fraction']) for row in group]):>7.4f}"
        )
    print(
        "tau   median paired ratios: scoreSD G/M   J0 G/M   "
        "reconstructed SE G/M   direct SE G/M   draw SD G/M   draw IQR G/M"
    )
    for tau in TAUS:
        group = [row for row in rows if float(row["tau"]) == tau]
        def paired_median(field: str) -> float:
            return float(np.median([float(row[field]) for row in group]))

        print(
            f"{tau:<4g} "
            f"{paired_median('gbdt_over_mdn_score_sd_ratio'):>34.4f} "
            f"{paired_median('gbdt_over_mdn_j0_ratio'):>8.4f} "
            f"{paired_median('reconstructed_gbdt_over_mdn_analytic_se_ratio'):>22.4f} "
            f"{paired_median('direct_gbdt_over_mdn_analytic_se_ratio'):>15.4f} "
            f"{paired_median('gbdt_over_mdn_pooled_total_draws_sd_ratio'):>13.4f} "
            f"{paired_median('gbdt_over_mdn_pooled_total_draws_iqr_ratio'):>14.4f}"
        )
    print("\nOuter-dataset calibration")
    print("backend tau   empirical SD   mean analytic SE   SE ratio      bias       MAE")
    for row in outer_summary:
        print(
            f"{str(row['backend']):<7} {float(row['tau']):<4g} "
            f"{float(row['empirical_sd']):>13.6f} "
            f"{float(row['mean_analytic_se']):>18.6f} "
            f"{float(row['se_ratio']):>10.4f} "
            f"{float(row['bias']):>9.6f} "
            f"{float(row['mae']):>9.6f}"
        )


def run_benchmark(
    refits: int,
    seed: int,
    output: Path,
    outer_summary_output: Path,
    truth_trajectories: int,
) -> List[Dict[str, object]]:
    if not 5 <= refits <= 10:
        raise ValueError("--refits must lie between 5 and 10")
    if truth_trajectories <= 0:
        raise ValueError("--truth-trajectories must be positive")
    context = mp.get_context("spawn")
    results: Dict[tuple[int, str], Mapping[str, object]] = {}
    for refit in range(refits):
        data_seed = seed + 101 * refit
        estimator_seed = data_seed + 31_337
        backend_order = ("mdn", "gbdt") if refit % 2 == 0 else ("gbdt", "mdn")
        for backend in backend_order:
            result = _run_isolated(
                context,
                backend,
                refit,
                data_seed,
                estimator_seed,
            )
            results[(refit, backend)] = result
            print(
                f"completed refit {refit + 1}/{refits}, backend={backend}, "
                f"wall={float(result['wall_seconds']):.3f}s, "
                f"peak_rss={float(result['peak_rss_mb']):.1f}MB",
                flush=True,
            )
    rows = _paired_rows(results, refits)
    truth_by_tau = _truth_by_tau(seed, truth_trajectories)
    outer_summary = _outer_summary(rows, truth_by_tau, refits)
    _write_csv(output, rows)
    _write_csv(outer_summary_output, outer_summary)
    _print_summary(results, rows, outer_summary, refits)
    print(f"wrote {output}")
    print(f"wrote {outer_summary_output}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refits", type=int, default=DEFAULT_REFITS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--truth-trajectories",
        type=int,
        default=DEFAULT_TRUTH_TRAJECTORIES,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/backend_benchmark/backend_benchmark_rows.csv"
        ),
    )
    parser.add_argument(
        "--outer-summary-output",
        type=Path,
        default=Path(
            "outputs/backend_benchmark/backend_outer_summary.csv"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_benchmark(
        args.refits,
        args.seed,
        args.output,
        args.outer_summary_output,
        args.truth_trajectories,
    )


if __name__ == "__main__":
    main()
