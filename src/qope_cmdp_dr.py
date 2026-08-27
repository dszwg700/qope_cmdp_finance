"""Pooled-outcome doubly robust quantile OPE for a finite-horizon CMDP.

One conditional-distribution nuisance learner is trained across stage-specific
history/action rows. Mixture-density and two-stage GBDT location-scale backends
share the same padded history representation. Quantile standard errors use a
KDE estimate of the local score derivative ``j0``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

from .gbdtlearner_cmdp import GBDTDistributionLearner, GBDTLearnerConfig
from .mdnlearner_cmdp import MDNLearner, MDNLearnerConfig

Array = np.ndarray
TargetPolicy = Callable[[Array, int, Sequence[int]], Array]


def pinball(u: Array, tau: float) -> Array:
    u = np.asarray(u, dtype=float)
    return u * (tau - (u < 0).astype(float))


def discounted_returns(rewards: Array, gamma: float) -> Array:
    """G[:, k] = sum_{j=k}^{T-1} gamma^(j-k) R[:, j]."""
    rewards = np.asarray(rewards, dtype=float)
    n, T = rewards.shape
    G = np.zeros_like(rewards, dtype=float)
    running = np.zeros(n, dtype=float)
    for k in range(T - 1, -1, -1):
        running = rewards[:, k] + gamma * running
        G[:, k] = running
    return G


def prefix_discounted_rewards(rewards: Array, gamma: float) -> Array:
    """pre[:, k] = sum_{j=0}^{k-1} gamma^j R[:, j], pre[:,0]=0."""
    rewards = np.asarray(rewards, dtype=float)
    n, T = rewards.shape
    pre = np.zeros((n, T), dtype=float)
    running = np.zeros(n, dtype=float)
    for k in range(T):
        pre[:, k] = running
        running = running + (gamma ** k) * rewards[:, k]
    return pre


def gaussian_kernel_density_at(x: Array, eta: float, h: float) -> Array:
    z = (np.asarray(x, dtype=float) - eta) / h
    return np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * h)


def default_bandwidth(values: Array, n: int) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 1.0
    sd = float(np.std(values, ddof=1))
    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    scale = min(sd, iqr / 1.349) if iqr > 0 else sd
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = max(sd, 1.0)
    return max(1.06 * scale * max(n, 2) ** (-1.0 / 5.0), 1e-3)


class DiscreteMediatorModel:
    """Estimate p_m(m | s, a) for discrete mediator values."""

    def __init__(self, actions: Sequence[int], clip: float = 1e-4, random_state: int = 123):
        self.actions = list(actions)
        self.clip = float(clip)
        self.action_encoder = OneHotEncoder(categories=[self.actions], sparse_output=False, handle_unknown="ignore")
        self.clf = GradientBoostingClassifier(random_state=random_state)
        self.classes_: Optional[np.ndarray] = None

    def _features(self, states: Array, actions: Array) -> Array:
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions).reshape(-1, 1)
        if not hasattr(self.action_encoder, "categories_"):
            self.action_encoder.fit(np.asarray(self.actions).reshape(-1, 1))
        return np.hstack([states, self.action_encoder.transform(actions)])

    def fit(self, states: Array, actions: Array, mediators: Array) -> "DiscreteMediatorModel":
        self.action_encoder.fit(np.asarray(self.actions).reshape(-1, 1))
        self.clf.fit(self._features(states, actions), np.asarray(mediators).reshape(-1))
        self.classes_ = self.clf.classes_
        return self

    def prob(self, states: Array, actions: Array, mediators: Array) -> Array:
        if self.classes_ is None:
            raise RuntimeError("Mediator model is not fitted.")
        proba = self.clf.predict_proba(self._features(states, actions))
        med = np.asarray(mediators).reshape(-1)
        idx = np.searchsorted(self.classes_, med)
        ok = (idx >= 0) & (idx < len(self.classes_)) & (self.classes_[np.clip(idx, 0, len(self.classes_) - 1)] == med)
        out = np.full(len(med), self.clip, dtype=float)
        out[ok] = proba[np.where(ok)[0], idx[ok]]
        return np.clip(out, self.clip, 1.0)


class BehaviorPolicyModel:
    """Estimate b(a | s) for behavior-style baseline weights."""

    def __init__(self, actions: Sequence[int], clip: float = 1e-4, random_state: int = 123):
        self.actions = list(actions)
        self.clip = float(clip)
        self.clf = GradientBoostingClassifier(random_state=random_state)
        self.classes_: Optional[np.ndarray] = None

    def fit(self, states: Array, actions: Array) -> "BehaviorPolicyModel":
        self.clf.fit(np.asarray(states, dtype=float), np.asarray(actions).reshape(-1))
        self.classes_ = self.clf.classes_
        return self

    def prob(self, states: Array, actions: Array) -> Array:
        if self.classes_ is None:
            raise RuntimeError("Behavior model is not fitted.")
        proba = self.clf.predict_proba(np.asarray(states, dtype=float))
        a = np.asarray(actions).reshape(-1)
        idx = np.searchsorted(self.classes_, a)
        ok = (idx >= 0) & (idx < len(self.classes_)) & (self.classes_[np.clip(idx, 0, len(self.classes_) - 1)] == a)
        out = np.full(len(a), self.clip, dtype=float)
        out[ok] = proba[np.where(ok)[0], idx[ok]]
        return np.clip(out, self.clip, 1.0)


@dataclass
class CMDPDRConfig:
    gamma: float = 0.95
    taus: Tuple[float, ...] = (0.25, 0.5, 0.75)
    n_folds: int = 2
    n_mc: int = 50
    weight_type: str = "frontdoor"  # frontdoor | behavior | none
    clip_ratio: float = 20.0
    nuisance_clip: float = 1e-4
    optimize_maxiter: int = 100
    random_state: int = 123
    density_bandwidth: Optional[float] = None
    min_j0: float = 1e-6
    mdn_config: MDNLearnerConfig = field(default_factory=MDNLearnerConfig)
    # Appended to preserve the positional order of all pre-existing fields.
    mdn_inference_batch_size: int = 4096
    outcome_backend: str = "mdn"  # mdn | gbdt
    gbdt_config: GBDTLearnerConfig = field(default_factory=GBDTLearnerConfig)


@dataclass
class QuantileResult:
    tau: float
    eta: float
    se: float
    ci_low: float
    ci_high: float
    objective: float
    j0: float
    score_sd: float
    bandwidth: float


class CMDPDRQuantileEstimator:
    def __init__(self, actions: Sequence[int], config: Optional[CMDPDRConfig] = None):
        self.actions = list(actions)
        self.config = config or CMDPDRConfig()
        if self.config.weight_type not in {"frontdoor", "behavior", "none"}:
            raise ValueError("weight_type must be one of {'frontdoor','behavior','none'}")
        if self.config.outcome_backend not in {"mdn", "gbdt"}:
            raise ValueError("outcome_backend must be one of {'mdn','gbdt'}")
        if self.config.n_mc <= 0:
            raise ValueError("n_mc must be positive")
        if self.config.mdn_inference_batch_size <= 0:
            raise ValueError("mdn_inference_batch_size must be positive")
        self.fold_models_: List[Dict[str, object]] = []
        self.results_: Dict[float, QuantileResult] = {}
        # Read-only post-fit diagnostics; these do not enter point or CI logic.
        self.weight_diagnostics_: Dict[str, float] = {}

    def _target_probs(self, target_policy: TargetPolicy, states: Array, t: int) -> Array:
        probs = np.asarray(target_policy(states, t, self.actions), dtype=float)
        if probs.ndim != 2 or probs.shape != (len(states), len(self.actions)):
            raise ValueError("target_policy(states, t, actions) must return shape (n, n_actions)")
        probs = np.clip(probs, 0.0, 1.0)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        return probs / row_sums

    def _pi_observed(self, probs: Array, actions_obs: Array) -> Array:
        action_to_col = {a: j for j, a in enumerate(self.actions)}
        cols = np.array([action_to_col[a] for a in actions_obs], dtype=int)
        return probs[np.arange(len(actions_obs)), cols]

    def _onehot_actions(self, actions_t: Array) -> Array:
        actions_t = np.asarray(actions_t).reshape(-1)
        enc = np.zeros((len(actions_t), len(self.actions)), dtype=float)
        action_to_col = {a: j for j, a in enumerate(self.actions)}
        for i, a in enumerate(actions_t):
            enc[i, action_to_col[a]] = 1.0
        return enc

    def _make_history_action_features(
        self,
        states: Array,
        actions: Array,
        rewards: Array,
        mediators: Optional[Array],
        idx: Array,
        t: int,
        current_actions: Array,
    ) -> Array:
        """Fixed-dimensional padded representation of (H_t, A_t)."""
        idx = np.asarray(idx, dtype=int)
        n = len(idx)
        T = states.shape[1]
        d = states.shape[2]
        time_col = np.full((n, 1), 0.0 if T <= 1 else t / (T - 1), dtype=float)

        state_pad = np.zeros((n, T, d), dtype=float)
        state_mask = np.zeros((n, T), dtype=float)
        state_pad[:, : t + 1, :] = states[idx, : t + 1, :]
        state_mask[:, : t + 1] = 1.0

        past_action_pad = np.zeros((n, T, len(self.actions)), dtype=float)
        past_reward_pad = np.zeros((n, T), dtype=float)
        past_mediator_pad = np.zeros((n, T), dtype=float)
        past_mask = np.zeros((n, T), dtype=float)
        for j in range(t):
            past_action_pad[:, j, :] = self._onehot_actions(actions[idx, j])
            past_reward_pad[:, j] = rewards[idx, j]
            if mediators is not None:
                past_mediator_pad[:, j] = mediators[idx, j]
            past_mask[:, j] = 1.0

        current_action_enc = self._onehot_actions(current_actions)
        return np.hstack([
            time_col,
            state_mask,
            state_pad.reshape(n, -1),
            past_mask,
            past_action_pad.reshape(n, -1),
            past_mediator_pad,
            past_reward_pad,
            current_action_enc,
        ])

    def _fit_fold(self, states: Array, actions: Array, mediators: Optional[Array], rewards: Array, train_idx: Array) -> Dict[str, object]:
        cfg = self.config
        n, T, d = states.shape
        flat_s = states[train_idx].reshape(-1, d)
        flat_a = actions[train_idx].reshape(-1)
        model: Dict[str, object] = {}

        if cfg.weight_type == "frontdoor":
            if mediators is None:
                raise ValueError("mediators are required for weight_type='frontdoor'")
            flat_m = mediators[train_idx].reshape(-1)
            med_model = DiscreteMediatorModel(self.actions, clip=cfg.nuisance_clip, random_state=cfg.random_state)
            med_model.fit(flat_s, flat_a, flat_m)
            model["mediator"] = med_model
        elif cfg.weight_type == "behavior":
            beh = BehaviorPolicyModel(self.actions, clip=cfg.nuisance_clip, random_state=cfg.random_state)
            beh.fit(flat_s, flat_a)
            model["behavior"] = beh

        # Pooled outcome model: all stage-specific training samples share one
        # conditional-distribution learner.  The default MDN branch is kept
        # byte-for-byte equivalent in construction and fitting order.
        G = discounted_returns(rewards, cfg.gamma)
        X_list, y_list = [], []
        for t in range(T):
            X_list.append(self._make_history_action_features(states, actions, rewards, mediators, train_idx, t, actions[train_idx, t]))
            y_list.append(G[train_idx, t])
        X_pool = np.vstack(X_list)
        y_pool = np.concatenate(y_list)
        if cfg.outcome_backend == "mdn":
            outcome_model = MDNLearner(
                input_dim=X_pool.shape[1], config=cfg.mdn_config
            )
        else:
            outcome_model = GBDTDistributionLearner(
                input_dim=X_pool.shape[1], config=cfg.gbdt_config
            )
        if cfg.outcome_backend == "gbdt":
            # Each training trajectory contributes one row per stage.  Keeping
            # those rows together makes the GBDT residual law trajectory-OOF.
            residual_groups = np.concatenate([
                np.asarray(train_idx, dtype=int) for _ in range(T)
            ])
            outcome_model.fit(X_pool, y_pool, groups=residual_groups)
        else:
            outcome_model.fit(X_pool, y_pool)
        # Preserve the legacy model key so existing MDN integrations and tests
        # continue to see exactly the same fold payload structure.
        model["mdn"] = outcome_model
        return model

    def _step_ratios(self, model: Dict[str, object], states_t: Array, actions_t: Array, mediators_t: Optional[Array], pi_probs: Array) -> Array:
        cfg = self.config
        if cfg.weight_type == "none":
            return np.ones(len(states_t), dtype=float)
        pi_obs = self._pi_observed(pi_probs, actions_t)
        if cfg.weight_type == "behavior":
            b_obs = model["behavior"].prob(states_t, actions_t)  # type: ignore[index]
            r = pi_obs / b_obs
        else:
            med_model: DiscreteMediatorModel = model["mediator"]  # type: ignore[assignment]
            denom = med_model.prob(states_t, actions_t, mediators_t)
            numer = np.zeros(len(states_t), dtype=float)
            for j, a in enumerate(self.actions):
                a_vec = np.full(len(states_t), a)
                numer += pi_probs[:, j] * med_model.prob(states_t, a_vec, mediators_t)
            r = numer / denom
        return np.clip(r, 1.0 / cfg.clip_ratio, cfg.clip_ratio)

    def _simulate_target_remaining(self, model: Dict[str, object], states: Array, actions: Array, rewards: Array, mediators: Optional[Array], eval_idx: Array, pi_probs: Array, t: int, random_state: int) -> Array:
        cfg = self.config
        n = len(eval_idx)
        # Derive independent child streams for target-action sampling and MDN
        # return draws. Each inference batch gets a separate stateless TF seed.
        root_seed = np.random.SeedSequence(int(random_state) % (2**32))
        action_seed, draw_seed = root_seed.spawn(2)
        rng = np.random.default_rng(action_seed)
        action_cols = np.empty((n, cfg.n_mc), dtype=int)
        for i in range(n):
            action_cols[i, :] = rng.choice(len(self.actions), size=cfg.n_mc, p=pi_probs[i])

        # Row-major flattening maps flat position i*n_mc+j back to draws[i,j].
        # Build history features in bounded chunks instead of materializing the
        # full (n*n_mc, history_dimension) matrix.
        total = n * cfg.n_mc
        flat_eval_idx = np.repeat(np.asarray(eval_idx, dtype=int), cfg.n_mc)
        flat_action_cols = action_cols.reshape(-1)
        flat_draws = np.empty(total, dtype=float)
        batch_size = min(cfg.mdn_inference_batch_size, max(total, 1))
        n_batches = (total + batch_size - 1) // batch_size
        batch_seeds = draw_seed.spawn(n_batches)
        outcome_model: MDNLearner | GBDTDistributionLearner = model["mdn"]  # type: ignore[assignment]
        for batch_id, start in enumerate(range(0, total, batch_size)):
            stop = min(start + batch_size, total)
            acts = np.asarray([self.actions[c] for c in flat_action_cols[start:stop]])
            X = self._make_history_action_features(
                states,
                actions,
                rewards,
                mediators,
                flat_eval_idx[start:stop],
                t,
                acts,
            )
            batch_seed = int(batch_seeds[batch_id].generate_state(1, dtype=np.uint32)[0])
            flat_draws[start:stop] = outcome_model.sample(
                X,
                n_samples=1,
                random_state=batch_seed,
            ).reshape(-1)
        return flat_draws.reshape(n, cfg.n_mc)

    def _fold_arrays(self, states: Array, actions: Array, mediators: Optional[Array], rewards: Array, target_policy: TargetPolicy, model: Dict[str, object], eval_idx: Array) -> Dict[str, object]:
        cfg = self.config
        T = states.shape[1]
        G = discounted_returns(rewards, cfg.gamma)
        pre = prefix_discounted_rewards(rewards, cfg.gamma)
        n_eval = len(eval_idx)

        step_ratio = np.ones((n_eval, T), dtype=float)
        pi_cache = []
        for t in range(T):
            s_t = states[eval_idx, t, :]
            a_t = actions[eval_idx, t]
            m_t = None if mediators is None else mediators[eval_idx, t]
            pi_probs = self._target_probs(target_policy, s_t, t)
            pi_cache.append(pi_probs)
            step_ratio[:, t] = self._step_ratios(model, s_t, a_t, m_t, pi_probs)

        prefix_w = np.ones((n_eval, T), dtype=float)
        running = np.ones(n_eval, dtype=float)
        for t in range(T):
            prefix_w[:, t] = running
            running = running * step_ratio[:, t]
        full_w = running

        total_draws = []
        for t in range(T):
            rem = self._simulate_target_remaining(model, states, actions, rewards, mediators, eval_idx, pi_cache[t], t, cfg.random_state + 17 * t)
            total_draws.append(pre[eval_idx, t].reshape(-1, 1) + (cfg.gamma ** t) * rem)

        return {
            "idx": eval_idx,
            "observed_total": G[eval_idx, 0],
            "step_ratio": step_ratio,
            "prefix_w": prefix_w,
            "full_w": full_w,
            "total_draws": total_draws,
        }

    def _weight_diagnostics(self, payloads: List[Dict[str, object]]) -> Dict[str, float]:
        """Summarize cross-fitted ratios without changing estimator calculations."""
        weights = np.concatenate([
            np.asarray(payload["full_w"], dtype=float).reshape(-1)
            for payload in payloads
        ])
        step_ratios = np.concatenate([
            np.asarray(payload["step_ratio"], dtype=float).reshape(-1)
            for payload in payloads
        ])
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(step_ratios)):
            raise FloatingPointError("non-finite importance weights encountered")

        # Scaling avoids overflow in (sum w)^2 / sum(w^2) for long horizons.
        scale = float(np.max(np.abs(weights))) if len(weights) else 0.0
        if scale <= 0.0:
            ess = 0.0
        else:
            scaled = weights / scale
            denominator = float(np.sum(np.square(scaled)))
            ess = float(np.square(np.sum(scaled)) / denominator) if denominator > 0.0 else 0.0

        return {
            "weight_mean": float(np.mean(weights)),
            "weight_sd": float(np.std(weights, ddof=1)) if len(weights) > 1 else 0.0,
            "weight_min": float(np.min(weights)),
            "weight_p50": float(np.quantile(weights, 0.50)),
            "weight_p90": float(np.quantile(weights, 0.90)),
            "weight_p95": float(np.quantile(weights, 0.95)),
            "weight_p99": float(np.quantile(weights, 0.99)),
            "weight_max": float(np.max(weights)),
            "ess": ess,
            "ess_fraction": ess / len(weights) if len(weights) else 0.0,
            "step_ratio_mean": float(np.mean(step_ratios)),
            "step_ratio_sd": float(np.std(step_ratios, ddof=1)) if len(step_ratios) > 1 else 0.0,
            "step_ratio_min": float(np.min(step_ratios)),
            "step_ratio_p95": float(np.quantile(step_ratios, 0.95)),
            "step_ratio_max": float(np.max(step_ratios)),
            "step_ratio_clip_low_fraction": float(np.mean(np.isclose(
                step_ratios, 1.0 / self.config.clip_ratio, rtol=1e-10, atol=1e-12
            ))),
            "step_ratio_clip_high_fraction": float(np.mean(np.isclose(
                step_ratios, self.config.clip_ratio, rtol=1e-10, atol=1e-12
            ))),
        }

    def fit(self, states: Array, actions: Array, rewards: Array, target_policy: TargetPolicy, mediators: Optional[Array] = None) -> "CMDPDRQuantileEstimator":
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards, dtype=float)
        if states.ndim != 3 or actions.ndim != 2 or rewards.ndim != 2:
            raise ValueError("states must be (N,T,d), actions/rewards must be (N,T)")
        if actions.shape != rewards.shape or states.shape[:2] != rewards.shape:
            raise ValueError("states, actions, rewards have incompatible shapes")
        if mediators is not None:
            mediators = np.asarray(mediators)
            if mediators.shape != rewards.shape:
                raise ValueError("mediators must have shape (N,T)")

        n = states.shape[0]
        kf = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.random_state)
        self.fold_models_.clear()
        self.weight_diagnostics_.clear()
        fold_payloads = []
        for train_idx, eval_idx in kf.split(np.arange(n)):
            model = self._fit_fold(states, actions, mediators, rewards, train_idx)
            self.fold_models_.append(model)
            fold_payloads.append(self._fold_arrays(states, actions, mediators, rewards, target_policy, model, eval_idx))
        self.weight_diagnostics_ = self._weight_diagnostics(fold_payloads)
        for tau in self.config.taus:
            self.results_[float(tau)] = self._solve_tau(float(tau), fold_payloads)
        return self

    def _psi_values(self, eta: float, tau: float, payloads: List[Dict[str, object]]) -> Array:
        vals = []
        for p in payloads:
            y = p["observed_total"]
            full_w = p["full_w"]
            step_ratio = p["step_ratio"]
            prefix_w = p["prefix_w"]
            total_draws = p["total_draws"]
            v = full_w * pinball(y - eta, tau)
            for t in range(step_ratio.shape[1]):
                L = np.mean(pinball(total_draws[t] - eta, tau), axis=1)
                v = v + prefix_w[:, t] * (1.0 - step_ratio[:, t]) * L
            vals.append(v)
        return np.concatenate(vals)

    def _score_values(self, eta: float, tau: float, payloads: List[Dict[str, object]]) -> Array:
        vals = []
        for p in payloads:
            y = p["observed_total"]
            full_w = p["full_w"]
            step_ratio = p["step_ratio"]
            prefix_w = p["prefix_w"]
            total_draws = p["total_draws"]
            v = full_w * ((y < eta).astype(float) - tau)
            for t in range(step_ratio.shape[1]):
                dL = np.mean((total_draws[t] < eta).astype(float) - tau, axis=1)
                v = v + prefix_w[:, t] * (1.0 - step_ratio[:, t]) * dL
            vals.append(v)
        return np.concatenate(vals)

    def _j0_kde_values(self, eta: float, payloads: List[Dict[str, object]], bandwidth: float) -> Array:
        vals = []
        for p in payloads:
            y = p["observed_total"]
            full_w = p["full_w"]
            step_ratio = p["step_ratio"]
            prefix_w = p["prefix_w"]
            total_draws = p["total_draws"]
            v = full_w * gaussian_kernel_density_at(y, eta, bandwidth)
            for t in range(step_ratio.shape[1]):
                dens = np.mean(gaussian_kernel_density_at(total_draws[t], eta, bandwidth), axis=1)
                v = v + prefix_w[:, t] * (1.0 - step_ratio[:, t]) * dens
            vals.append(v)
        return np.concatenate(vals)

    def _solve_tau(self, tau: float, payloads: List[Dict[str, object]]) -> QuantileResult:
        all_y = [p["observed_total"] for p in payloads]
        for p in payloads:
            all_y.extend([td.reshape(-1) for td in p["total_draws"]])
        support = np.concatenate(all_y)
        lo, hi = np.nanquantile(support, [0.001, 0.999])
        pad = 0.1 * max(1.0, hi - lo)
        lo, hi = float(lo - pad), float(hi + pad)

        def obj(x: float) -> float:
            return float(np.mean(self._psi_values(x, tau, payloads)))

        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"maxiter": self.config.optimize_maxiter})
        eta = float(res.x)
        score = self._score_values(eta, tau, payloads)
        h = self.config.density_bandwidth or default_bandwidth(support, len(score))
        j0_values = self._j0_kde_values(eta, payloads, h)
        j0 = max(abs(float(np.mean(j0_values))), self.config.min_j0)
        score_sd = float(np.std(score, ddof=1)) if len(score) > 1 else np.nan
        se = float(score_sd / np.sqrt(len(score)) / j0) if np.isfinite(score_sd) else np.nan
        return QuantileResult(
            tau=tau,
            eta=eta,
            se=se,
            ci_low=eta - 1.96 * se,
            ci_high=eta + 1.96 * se,
            objective=obj(eta),
            j0=j0,
            score_sd=score_sd,
            bandwidth=float(h),
        )

    def summary(self) -> Dict[str, object]:
        if not self.results_:
            raise RuntimeError("Call fit() first")
        rows = []
        for tau, r in sorted(self.results_.items()):
            rows.append({
                "tau": tau,
                "eta": r.eta,
                "se": r.se,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "objective": r.objective,
                "j0": r.j0,
                "score_sd": r.score_sd,
                "bandwidth": r.bandwidth,
            })
        return {"quantiles": rows, "tail_robust_mean": float(np.mean([r.eta for r in self.results_.values()]))}
