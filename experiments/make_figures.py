"""Regenerate all release figures from summary-level CSV files only."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPOSITORY_ROOT / "results"
COLORS = {
    "mdn": "#0072B2",
    "gbdt": "#D55E00",
    "analytic": "#6A3D9A",
    "bootstrap": "#009E73",
    "in_sample": "#999999",
    "oof": "#009E73",
}
TAU_COLORS = {0.10: "#0072B2", 0.25: "#D55E00", 0.50: "#009E73"}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "savefig.facecolor": "white",
        }
    )


def _finish(figure: plt.Figure, path: Path, dpi: int) -> None:
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def backend_comparison(results: Path, figures: Path, dpi: int) -> None:
    data = pd.read_csv(results / "backend_benchmark_summary.csv")
    compute = data.drop_duplicates("backend").set_index("backend")
    backends = ["MDN", "GBDT"]
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    values = compute.loc[backends, "mean_wall_seconds"]
    errors = compute.loc[backends, "sd_wall_seconds"]
    axes[0].bar(backends, values, yerr=errors, capsize=4,
                color=[COLORS["mdn"], COLORS["gbdt"]], alpha=0.9)
    axes[0].set_ylabel("Wall-clock time (seconds)")
    axes[0].set_title("(a) Estimator refit time")

    values = compute.loc[backends, "mean_incremental_peak_rss_mb"]
    errors = compute.loc[backends, "sd_incremental_peak_rss_mb"]
    axes[1].bar(backends, values, yerr=errors, capsize=4,
                color=[COLORS["mdn"], COLORS["gbdt"]], alpha=0.9)
    axes[1].set_ylabel("Incremental peak RSS (MB)")
    axes[1].set_title("(b) Incremental memory")
    figure.suptitle("Conditional outcome backend benchmark ($N=40$, $T=20$)")
    figure.tight_layout()
    _finish(figure, figures / "backend_comparison.png", dpi)


def outcome_calibration(results: Path, figures: Path, dpi: int) -> None:
    data = pd.read_csv(results / "outcome_calibration_summary.csv")
    backend = data[data.summary_level.eq("backend")].set_index("calibration_version")
    quantiles = data[data.summary_level.eq("quantile")].copy()
    intervals = [0.50, 0.80, 0.90]
    fields = [f"interval_{int(level * 100)}_coverage" for level in intervals]
    x = np.arange(len(intervals))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for offset, (version, label, color) in enumerate(
        [
            ("in_sample_residual", "In-sample residual", COLORS["in_sample"]),
            ("oof_residual", "OOF residual", COLORS["oof"]),
        ]
    ):
        values = backend.loc[version, fields].astype(float).to_numpy()
        axes[0].bar(x + (offset - 0.5) * width, values, width=width,
                    label=label, color=color, alpha=0.9)
    axes[0].plot(x, intervals, color="#333333", marker="o", linestyle="--",
                 label="Nominal")
    axes[0].set_xticks(x, ["50%", "80%", "90%"])
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title("(a) Central predictive intervals")
    axes[0].legend(frameon=False)

    for version, label, color, marker in [
        ("in_sample_residual", "In-sample residual", COLORS["in_sample"], "s"),
        ("oof_residual", "OOF residual", COLORS["oof"], "o"),
    ]:
        subset = quantiles[quantiles.calibration_version.eq(version)].sort_values("quantile")
        axes[1].plot(subset["quantile"], subset["empirical_quantile_coverage"],
                     color=color, marker=marker, linewidth=2, label=label)
    axes[1].plot([0, 1], [0, 1], color="#333333", linestyle="--", label="Ideal")
    axes[1].set_xlim(0.05, 0.95)
    axes[1].set_ylim(0.05, 0.95)
    axes[1].set_xlabel("Nominal quantile level")
    axes[1].set_ylabel("Empirical quantile coverage")
    axes[1].set_title("(b) Quantile calibration")
    axes[1].legend(frameon=False)
    figure.suptitle("GBDT conditional return distribution calibration")
    figure.tight_layout()
    _finish(figure, figures / "outcome_calibration.png", dpi)


def bootstrap_calibration(results: Path, figures: Path, dpi: int) -> None:
    data = pd.read_csv(results / "bootstrap_summary.csv")
    data = data[data.interval_method.isin(["analytic", "bootstrap_normal"])]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    for method, label, color, marker in [
        ("analytic", "Analytic Wald", COLORS["analytic"], "s"),
        ("bootstrap_normal", "Trajectory bootstrap-normal", COLORS["bootstrap"], "o"),
    ]:
        subset = data[data.interval_method.eq(method)].sort_values("tau")
        axes[0].plot(subset.tau, subset.se_ratio, color=color, marker=marker,
                     linewidth=2, label=label)
        lower = subset.coverage - subset.coverage_wilson_95_low
        upper = subset.coverage_wilson_95_high - subset.coverage
        axes[1].errorbar(subset.tau, subset.coverage, yerr=[lower, upper],
                         color=color, marker=marker, linewidth=2, capsize=4,
                         label=label)
    axes[0].axhline(1.0, color="#333333", linestyle="--", label="Ideal ratio")
    axes[0].set_xlabel("Quantile level $\\tau$")
    axes[0].set_ylabel("Mean reported SE / empirical SD")
    axes[0].set_title("(a) Standard-error calibration")
    axes[0].legend(frameon=False)
    axes[1].axhline(0.95, color="#333333", linestyle="--", label="Nominal 0.95")
    axes[1].set_ylim(0.35, 1.02)
    axes[1].set_xlabel("Quantile level $\\tau$")
    axes[1].set_ylabel("Coverage (Wilson 95% CI)")
    axes[1].set_title("(b) Interval coverage")
    axes[1].legend(frameon=False)
    figure.suptitle("Finite-sample uncertainty calibration ($N=40$, $T=20$)")
    figure.tight_layout()
    _finish(figure, figures / "bootstrap_calibration.png", dpi)


def _mean_by_configuration(data: pd.DataFrame, values: Iterable[str]) -> pd.DataFrame:
    return data.groupby(["N", "T"], as_index=False)[list(values)].mean()


def sample_size_sensitivity(results: Path, figures: Path, dpi: int) -> None:
    data = pd.read_csv(results / "sensitivity_summary.csv")
    data = _mean_by_configuration(data[data["T"].eq(20)], ["rmse", "mae"])
    figure, axis = plt.subplots(figsize=(5.8, 3.8))
    axis.plot(data.N, data.rmse, color="#0072B2", marker="o", linewidth=2,
              label="Mean RMSE across $\\tau$")
    axis.plot(data.N, data.mae, color="#D55E00", marker="s", linewidth=2,
              label="Mean MAE across $\\tau$")
    axis.set_xticks(data.N)
    axis.set_xlabel("Number of trajectories $N$")
    axis.set_ylabel("Point-estimation error")
    axis.set_title("Sample-size sensitivity at fixed $T=20$")
    axis.legend(frameon=False)
    figure.tight_layout()
    _finish(figure, figures / "sample_size_sensitivity.png", dpi)


def horizon_ess_sensitivity(results: Path, figures: Path, dpi: int) -> None:
    data = pd.read_csv(results / "sensitivity_summary.csv")
    data = _mean_by_configuration(
        data[data.N.eq(40)],
        ["analytic_ess_fraction_mean", "analytic_ess_fraction_mcse",
         "bootstrap_ess_fraction_median"],
    )
    figure, axis = plt.subplots(figsize=(5.8, 3.8))
    axis.errorbar(
        data["T"],
        data.analytic_ess_fraction_mean,
        yerr=1.96 * data.analytic_ess_fraction_mcse,
        color="#3B2E8C",
        marker="o",
        linewidth=2,
        capsize=4,
        label="Point fit: mean (95% MC interval)",
    )
    axis.plot(data["T"], data.bootstrap_ess_fraction_median, color="#777777",
              marker="s", linestyle="--", linewidth=2,
              label="Bootstrap replicates: median")
    axis.set_xticks(data["T"])
    axis.set_xlabel("Trajectory horizon $T$")
    axis.set_ylabel("Effective sample size fraction (ESS/N)")
    axis.set_title("Weight degeneration at fixed $N=40$")
    axis.legend(frameon=False)
    figure.tight_layout()
    _finish(figure, figures / "horizon_ess_sensitivity.png", dpi)


def run(results: Path, figures: Path, dpi: int) -> None:
    _configure_style()
    figures.mkdir(parents=True, exist_ok=True)
    backend_comparison(results, figures, dpi)
    outcome_calibration(results, figures, dpi)
    bootstrap_calibration(results, figures, dpi)
    sample_size_sensitivity(results, figures, dpi)
    horizon_ess_sensitivity(results, figures, dpi)
    print(f"Wrote five figures to {figures}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_RESULTS / "figures")
    parser.add_argument("--dpi", type=int, default=220)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run(arguments.results_dir, arguments.figures_dir, arguments.dpi)
