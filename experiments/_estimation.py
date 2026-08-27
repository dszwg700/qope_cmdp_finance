"""Shared estimator construction used by public experiment entry points.

This module only adapts command-line arguments to the frozen estimator API. It
does not alter the weighting, doubly robust objective, quantile optimization,
kernel density derivative, or analytic confidence interval.
"""
from __future__ import annotations

import argparse
from typing import Dict, Sequence

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


def truth_for_horizon(
    horizon: int,
    taus: Sequence[float],
    n_trajectories: int,
    gamma: float,
    seed: int,
) -> Dict[float, float]:
    """Approximate target-policy quantiles with an independent Monte Carlo rollout."""
    data = generate_finance_cmdp(n_trajectories, horizon, seed, "target", gamma)
    returns = discounted_returns(data["rewards"], gamma)[:, 0]
    return {float(tau): float(np.quantile(returns, tau)) for tau in taus}


def qope_results(
    data: Dict[str, np.ndarray],
    weight_type: str,
    taus: Sequence[float],
    args: argparse.Namespace,
    seed: int,
) -> Dict[float, Dict[str, float]]:
    """Fit the public estimator and expose point, interval, and weight diagnostics."""
    mdn_config = MDNLearnerConfig(
        n_components=args.mdn_components,
        hidden_dims=tuple(args.mdn_hidden_dims),
        lr=args.mdn_lr,
        batch_size=args.mdn_batch_size,
        epochs=args.mdn_epochs,
        seed=seed,
        verbose=args.mdn_verbose,
        min_sigma=args.mdn_min_sigma,
    )
    gbdt_config = GBDTLearnerConfig(
        n_estimators=args.gbdt_n_estimators,
        learning_rate=args.gbdt_learning_rate,
        max_depth=args.gbdt_max_depth,
        min_samples_leaf=args.gbdt_min_samples_leaf,
        subsample=args.gbdt_subsample,
        min_sigma=args.gbdt_min_sigma,
        random_state=seed,
        residual_n_folds=args.gbdt_residual_n_folds,
    )
    config = CMDPDRConfig(
        gamma=args.gamma,
        taus=tuple(float(tau) for tau in taus),
        n_folds=args.n_folds,
        n_mc=args.n_mc,
        mdn_inference_batch_size=args.mdn_inference_batch_size,
        weight_type=weight_type,
        clip_ratio=args.clip_ratio,
        nuisance_clip=args.nuisance_clip,
        optimize_maxiter=args.optimize_maxiter,
        random_state=seed,
        density_bandwidth=args.density_bandwidth,
        mdn_config=mdn_config,
        outcome_backend=args.outcome_backend,
        gbdt_config=gbdt_config,
    )
    estimator = CMDPDRQuantileEstimator(actions=[0, 1], config=config)
    estimator.fit(
        states=data["states"],
        actions=data["actions"],
        rewards=data["rewards"],
        target_policy=target_policy_probs,
        mediators=data["mediators"],
    )
    diagnostics = dict(estimator.weight_diagnostics_)
    diagnostics["clipping_fraction"] = (
        diagnostics["step_ratio_clip_low_fraction"]
        + diagnostics["step_ratio_clip_high_fraction"]
    )
    output: Dict[float, Dict[str, float]] = {}
    for tau in taus:
        result = estimator.results_[float(tau)]
        output[float(tau)] = {
            "estimate": float(result.eta),
            "se": float(result.se),
            "ci_low": float(result.ci_low),
            "ci_high": float(result.ci_high),
            "j0": float(result.j0),
            "score_sd": float(result.score_sd),
            "bandwidth": float(result.bandwidth),
            **diagnostics,
        }
    return output
