"""Multi-period synthetic financial CMDP with hidden market pressure.

Causal structure at each step:

    S_t, U_t -> A_t; S_t, A_t -> M_t;
    S_t, U_t, M_t -> R_t, S_{t+1}.

``latent_u`` is returned only for simulation diagnostics.  Experiment code must
not pass it to an off-policy estimator.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .finance_cmdp_policies import (
    behavior_policy_probability,
    target_policy_probs,
)


Array = np.ndarray
FEATURE_NAMES = (
    "past_return",
    "volatility",
    "momentum",
    "liquidity",
    "drawdown",
)
REGIME_NAMES = ("bull", "bear", "sideways")

_REGIME_TRANSITION = np.array(
    [
        [0.94, 0.02, 0.04],
        [0.03, 0.93, 0.04],
        [0.05, 0.05, 0.90],
    ],
    dtype=float,
)
_REGIME_MEAN = np.array([0.0012, -0.0014, 0.0001], dtype=float)
_REGIME_VOL = np.array([0.0080, 0.0140, 0.0065], dtype=float)
_REGIME_LATENT_SHIFT = np.array([0.18, -0.22, 0.0], dtype=float)
_REGIME_LIQUIDITY_SHIFT = np.array([0.10, -0.14, 0.02], dtype=float)


def _sigmoid(x: Array) -> Array:
    x = np.clip(np.asarray(x, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _sample_next_regime(regime: Array, rng: np.random.Generator) -> Array:
    cumulative = np.cumsum(_REGIME_TRANSITION[regime], axis=1)
    uniforms = rng.random(len(regime))
    return np.sum(uniforms[:, None] > cumulative, axis=1).astype(int)


def _mediator_probability(states_t: Array, actions_t: Array) -> Array:
    """P(M=1 | S,A); intentionally contains no latent-U term."""
    volatility_z = (states_t[:, 1] - 0.012) / 0.008
    score = (
        -1.20
        + 2.70 * actions_t
        + 0.45 * states_t[:, 3]
        - 0.35 * volatility_z
    )
    return np.clip(_sigmoid(score), 0.05, 0.95)


def generate_finance_cmdp(
    n_trajectories: int,
    t_horizon: int,
    seed: int,
    policy_mode: str = "behavior",
    gamma: float = 0.99,
) -> Dict[str, Array]:
    """Generate logged or target-policy trajectories from shared dynamics.

    Parameters
    ----------
    policy_mode:
        ``"behavior"`` samples A from P(A|S,U); ``"target"`` samples A from
        the observable-state-only callback used by the estimator.
    gamma:
        Recorded in the output metadata for experiment consistency.  Discounting
        is applied by ``discounted_returns``, not inside the market dynamics.
    """
    if not isinstance(n_trajectories, (int, np.integer)) or n_trajectories <= 0:
        raise ValueError("n_trajectories must be a positive integer")
    if not isinstance(t_horizon, (int, np.integer)) or t_horizon <= 0:
        raise ValueError("t_horizon must be a positive integer")
    if policy_mode not in {"behavior", "target"}:
        raise ValueError("policy_mode must be 'behavior' or 'target'")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")

    n = int(n_trajectories)
    horizon = int(t_horizon)
    rng = np.random.default_rng(seed)

    states = np.empty((n, horizon, len(FEATURE_NAMES)), dtype=float)
    actions = np.empty((n, horizon), dtype=int)
    mediators = np.empty((n, horizon), dtype=int)
    rewards = np.empty((n, horizon), dtype=float)
    latent_u = np.empty((n, horizon), dtype=float)
    regimes = np.empty((n, horizon), dtype=int)
    market_returns = np.empty((n, horizon), dtype=float)
    action_probabilities = np.empty((n, horizon), dtype=float)
    mediator_probabilities = np.empty((n, horizon), dtype=float)

    regime = rng.choice(3, size=n, p=np.array([0.38, 0.27, 0.35]))
    u = np.clip(rng.normal(_REGIME_LATENT_SHIFT[regime], 0.85, size=n), -2.5, 2.5)
    past_return = _REGIME_MEAN[regime] + _REGIME_VOL[regime] * rng.normal(size=n)
    volatility = np.clip(_REGIME_VOL[regime] * rng.lognormal(0.0, 0.12, size=n), 0.004, 0.035)
    momentum = 0.60 * past_return + rng.normal(0.0, 0.004, size=n)
    liquidity = np.clip(rng.normal(_REGIME_LIQUIDITY_SHIFT[regime], 0.65, size=n), -2.5, 2.5)
    drawdown = rng.uniform(0.0, 0.025, size=n)

    cumulative_pnl = -drawdown.copy()
    running_peak = np.zeros(n, dtype=float)
    previous_exposure = np.zeros(n, dtype=int)
    t_scale = np.sqrt(5.0 / 3.0)  # Unit-variance Student-t(5) shocks.

    for t in range(horizon):
        states_t = np.column_stack((past_return, volatility, momentum, liquidity, drawdown))
        states[:, t, :] = states_t
        latent_u[:, t] = u
        regimes[:, t] = regime

        if policy_mode == "behavior":
            p_long = behavior_policy_probability(states_t, u)
        else:
            p_long = target_policy_probs(states_t, t, [0, 1])[:, 1]
        action = np.asarray(rng.binomial(1, p_long), dtype=int)

        p_exposure = _mediator_probability(states_t, action)
        exposure = np.asarray(rng.binomial(1, p_exposure), dtype=int)

        innovation = rng.standard_t(df=5, size=n) / t_scale
        next_market_return = (
            _REGIME_MEAN[regime]
            + 0.16 * past_return
            + 0.0045 * u
            + 0.72 * volatility * innovation
        )
        next_market_return = np.clip(next_market_return, -0.08, 0.08)
        turnover = np.abs(exposure - previous_exposure)
        reward = exposure * next_market_return - 0.00035 * turnover

        actions[:, t] = action
        mediators[:, t] = exposure
        rewards[:, t] = reward
        market_returns[:, t] = next_market_return
        action_probabilities[:, t] = p_long
        mediator_probabilities[:, t] = p_exposure

        cumulative_pnl += reward
        running_peak = np.maximum(running_peak, cumulative_pnl)
        next_drawdown = np.clip(running_peak - cumulative_pnl, 0.0, 0.30)
        next_volatility = np.clip(
            np.sqrt(0.82 * volatility**2 + 0.18 * next_market_return**2)
            + 0.00035 * np.abs(u)
            + 0.00020 * exposure,
            0.004,
            0.040,
        )
        next_momentum = np.clip(0.74 * momentum + 0.26 * next_market_return, -0.06, 0.06)
        next_liquidity = np.clip(
            0.72 * liquidity
            + _REGIME_LIQUIDITY_SHIFT[regime]
            - 0.14 * np.abs(u)
            - 0.08 * exposure
            + rng.normal(0.0, 0.24, size=n),
            -3.0,
            3.0,
        )

        next_regime = _sample_next_regime(regime, rng)
        next_u = np.clip(
            0.64 * u
            + _REGIME_LATENT_SHIFT[next_regime]
            + rng.normal(0.0, 0.62, size=n),
            -2.5,
            2.5,
        )
        past_return = next_market_return
        volatility = next_volatility
        momentum = next_momentum
        liquidity = next_liquidity
        drawdown = next_drawdown
        regime = next_regime
        u = next_u
        previous_exposure = exposure

    return {
        "states": states,
        "actions": actions,
        "mediators": mediators,
        "rewards": rewards,
        "latent_u": latent_u,
        "regimes": regimes,
        "market_returns": market_returns,
        "action_probabilities": action_probabilities,
        "mediator_probabilities": mediator_probabilities,
        "feature_names": np.asarray(FEATURE_NAMES),
        "gamma": np.asarray(float(gamma)),
    }
