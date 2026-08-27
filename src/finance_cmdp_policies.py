"""Behavior and target policies for the multi-period finance CMDP demo.

The behavior policy deliberately uses the latent market-pressure variable.  The
target policy has the callback signature required by CMDPDRQuantileEstimator
and only uses observable state features.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np


Array = np.ndarray
RandomState = Union[int, np.random.Generator, None]

# State columns shared with finance_cmdp_sim.py.
PAST_RETURN = 0
VOLATILITY = 1
MOMENTUM = 2
LIQUIDITY = 3
DRAWDOWN = 4


def _as_state_matrix(states: Array) -> Array:
    states = np.asarray(states, dtype=float)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    if states.ndim != 2 or states.shape[1] != 5:
        raise ValueError("states must have shape (n, 5)")
    if not np.all(np.isfinite(states)):
        raise ValueError("states must contain only finite values")
    return states


def _sigmoid(x: Array) -> Array:
    x = np.clip(np.asarray(x, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _rng(random_state: RandomState) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def behavior_policy_probability(states: Array, latent_u: Array) -> Array:
    """Return P_b(A=1 | S, U) for the confounded logging policy."""
    states = _as_state_matrix(states)
    latent_u = np.asarray(latent_u, dtype=float).reshape(-1)
    if latent_u.size == 1 and len(states) != 1:
        latent_u = np.repeat(latent_u, len(states))
    if latent_u.shape != (len(states),):
        raise ValueError("latent_u must contain one value per state row")

    volatility_z = (states[:, VOLATILITY] - 0.012) / 0.008
    momentum_z = states[:, MOMENTUM] / 0.012
    drawdown_z = states[:, DRAWDOWN] / 0.05
    score = (
        -0.05
        + 0.70 * momentum_z
        - 0.35 * volatility_z
        + 0.18 * states[:, LIQUIDITY]
        - 0.20 * drawdown_z
        + 1.10 * latent_u
    )
    # Epsilon-soft probabilities retain overlap even for unusual simulated states.
    return 0.08 + 0.84 * _sigmoid(score)


def sample_behavior_actions(
    states: Array,
    latent_u: Array,
    random_state: RandomState = None,
) -> Array:
    """Sample binary actions from the behavior policy."""
    probabilities = behavior_policy_probability(states, latent_u)
    return np.asarray(_rng(random_state).binomial(1, probabilities), dtype=int)


def target_long_probability(states: Array) -> Array:
    """Return the defensive target policy's P(A=1 | S)."""
    states = _as_state_matrix(states)
    volatility_z = (states[:, VOLATILITY] - 0.012) / 0.008
    momentum_z = states[:, MOMENTUM] / 0.012
    drawdown_z = states[:, DRAWDOWN] / 0.05
    score = (
        -0.15
        + 0.80 * momentum_z
        - 0.80 * volatility_z
        + 0.25 * states[:, LIQUIDITY]
        - 0.70 * drawdown_z
    )
    return 0.08 + 0.84 * _sigmoid(score)


def target_policy_probs(states_t: Array, t: int, actions: Sequence[int]) -> Array:
    """Estimator callback returning probabilities in the supplied action order.

    ``t`` is accepted because it is part of the estimator interface.  This MVP
    uses a stationary target policy, so the probabilities do not depend on it.
    """
    del t
    actions = list(actions)
    if len(actions) != 2 or set(actions) != {0, 1}:
        raise ValueError("the finance CMDP target policy supports actions {0, 1}")

    p_long = target_long_probability(states_t)
    probability_by_action = {0: 1.0 - p_long, 1: p_long}
    return np.column_stack([probability_by_action[action] for action in actions])


def sample_target_actions(
    states: Array,
    t: int = 0,
    random_state: RandomState = None,
) -> Array:
    """Sample binary actions from the observable-state-only target policy."""
    probabilities = target_policy_probs(states, t, [0, 1])[:, 1]
    return np.asarray(_rng(random_state).binomial(1, probabilities), dtype=int)
