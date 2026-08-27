"""Conditional return distribution learner based on location-scale GBDTs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold


@dataclass
class GBDTLearnerConfig:
    """Hyperparameters shared by the conditional mean and variance models."""

    n_estimators: int = 100
    learning_rate: float = 0.05
    max_depth: int = 3
    min_samples_leaf: int = 5
    subsample: float = 1.0
    min_sigma: float = 1e-3
    random_state: int = 123
    residual_n_folds: int = 2


class GBDTDistributionLearner:
    """Two-stage GBDT location-scale conditional distribution model.

    The first regressor estimates ``mu(X)``.  The second estimates the squared
    residual conditional mean.  Conditional samples combine these predictions
    with resampled standardized training residuals.
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[GBDTLearnerConfig] = None,
    ):
        self.input_dim = int(input_dim)
        self.config = config or GBDTLearnerConfig()
        self._validate_config()
        self._regressor_kwargs = {
            "n_estimators": self.config.n_estimators,
            "learning_rate": self.config.learning_rate,
            "max_depth": self.config.max_depth,
            "min_samples_leaf": self.config.min_samples_leaf,
            "subsample": self.config.subsample,
        }
        self.location_model = self._make_regressor(self.config.random_state)
        self.scale_model = self._make_regressor(self.config.random_state + 1)
        self.standardized_residuals_: Optional[np.ndarray] = None
        self.sorted_standardized_residuals_: Optional[np.ndarray] = None
        self.residual_kde_bandwidth_: Optional[float] = None
        self.oof_location_residuals_: Optional[np.ndarray] = None
        self.residual_scale_: Optional[np.ndarray] = None
        self.oof_fold_id_: Optional[np.ndarray] = None

    def _make_regressor(self, random_state: int) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            **self._regressor_kwargs,
            random_state=int(random_state),
        )

    def _validate_config(self) -> None:
        cfg = self.config
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if cfg.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if cfg.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if cfg.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if cfg.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive")
        if not 0.0 < cfg.subsample <= 1.0:
            raise ValueError("subsample must lie in (0, 1]")
        if cfg.min_sigma <= 0.0:
            raise ValueError("min_sigma must be positive")
        if cfg.residual_n_folds < 2:
            raise ValueError("residual_n_folds must be at least 2")

    def _as_feature_matrix(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(f"X must have shape (n, {self.input_dim})")
        if not np.all(np.isfinite(values)):
            raise ValueError("X contains non-finite values")
        return values

    def _require_fitted(self) -> np.ndarray:
        if self.standardized_residuals_ is None:
            raise RuntimeError("Call fit() before distribution methods")
        return self.standardized_residuals_

    @staticmethod
    def _kde_bandwidth(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) < 2:
            return 1.0
        sd = float(np.std(values, ddof=1))
        iqr = float(np.subtract(*np.quantile(values, [0.75, 0.25])))
        robust_scale = min(sd, iqr / 1.349) if iqr > 0.0 else sd
        if not np.isfinite(robust_scale) or robust_scale <= 1e-12:
            robust_scale = max(sd, 1.0)
        return max(0.9 * robust_scale * len(values) ** (-1.0 / 5.0), 1e-3)

    def _oof_splits(
        self,
        n_rows: int,
        groups: Optional[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            units = np.arange(n_rows)
            row_groups = units
        else:
            row_groups = np.asarray(groups).reshape(-1)
            if len(row_groups) != n_rows:
                raise ValueError("groups must contain one value per X row")
            units = np.unique(row_groups)
        if len(units) < 2:
            raise ValueError("at least two OOF units are required")
        n_splits = min(int(self.config.residual_n_folds), len(units))
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )
        output: list[tuple[np.ndarray, np.ndarray]] = []
        for train_unit_positions, validation_unit_positions in splitter.split(units):
            train_units = units[train_unit_positions]
            validation_units = units[validation_unit_positions]
            train_rows = np.flatnonzero(np.isin(row_groups, train_units))
            validation_rows = np.flatnonzero(np.isin(row_groups, validation_units))
            output.append((train_rows, validation_rows))
        return output

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> "GBDTDistributionLearner":
        features = self._as_feature_matrix(X)
        targets = np.asarray(y, dtype=float).reshape(-1)
        if len(targets) != len(features):
            raise ValueError("X and y must contain the same number of rows")
        if not np.all(np.isfinite(targets)):
            raise ValueError("y contains non-finite values")

        splits = self._oof_splits(len(features), groups)
        oof_location = np.empty(len(features), dtype=float)
        fold_id = np.full(len(features), -1, dtype=int)
        for fold, (train_rows, validation_rows) in enumerate(splits):
            location = self._make_regressor(self.config.random_state + 101 + fold)
            location.fit(features[train_rows], targets[train_rows])
            oof_location[validation_rows] = location.predict(features[validation_rows])
            fold_id[validation_rows] = fold
        if np.any(fold_id < 0) or not np.all(np.isfinite(oof_location)):
            raise FloatingPointError("incomplete OOF location predictions")

        residuals = targets - oof_location
        squared_residuals = np.square(residuals)
        # Final prediction models use all training rows.  The scale target and
        # empirical residual law use location-OOF residuals, avoiding in-sample
        # GBDT residual shrinkage.  Scaling those residuals with the final scale
        # model avoids unstable divisions by fold-specific variance predictions
        # at the min_sigma floor in small/resampled training folds.
        self.location_model.fit(features, targets)
        self.scale_model.fit(features, squared_residuals)
        sigma = self._predict_sigma(features)
        standardized = residuals / sigma
        # Keep the resampled residual distribution centered so mu(X) remains
        # the conditional location model used by sample/cdf/pdf.
        standardized = standardized - float(np.mean(standardized))
        if not np.all(np.isfinite(standardized)):
            raise FloatingPointError("non-finite standardized residuals")
        self.standardized_residuals_ = standardized
        self.sorted_standardized_residuals_ = np.sort(standardized)
        self.residual_kde_bandwidth_ = self._kde_bandwidth(standardized)
        self.oof_location_residuals_ = residuals
        self.residual_scale_ = sigma
        self.oof_fold_id_ = fold_id
        return self

    def _predict_sigma(self, X: np.ndarray) -> np.ndarray:
        variance = np.asarray(self.scale_model.predict(X), dtype=float)
        variance = np.maximum(variance, self.config.min_sigma**2)
        sigma = np.sqrt(variance)
        if not np.all(np.isfinite(sigma)):
            raise FloatingPointError("non-finite conditional scale prediction")
        return sigma

    def predict_location_scale(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        features = self._as_feature_matrix(X)
        mu = np.asarray(self.location_model.predict(features), dtype=float)
        sigma = self._predict_sigma(features)
        return mu, sigma

    def sample(
        self,
        X: np.ndarray,
        n_samples: int = 1,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        residuals = self._require_fitted()
        if int(n_samples) <= 0:
            raise ValueError("n_samples must be positive")
        mu, sigma = self.predict_location_scale(X)
        rng = np.random.default_rng(random_state)
        draws = rng.choice(
            residuals,
            size=(len(mu), int(n_samples)),
            replace=True,
        )
        return mu[:, None] + sigma[:, None] * draws

    def _standardized_value(
        self, X: np.ndarray, value: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mu, sigma = self.predict_location_scale(X)
        values = np.asarray(value, dtype=float)
        if values.ndim == 0:
            values = np.full(len(mu), float(values), dtype=float)
        else:
            values = np.broadcast_to(values.reshape(-1), (len(mu),)).astype(float)
        if not np.all(np.isfinite(values)):
            raise ValueError("value contains non-finite entries")
        return (values - mu) / sigma, sigma

    def cdf(self, X: np.ndarray, value: float | np.ndarray) -> np.ndarray:
        self._require_fitted()
        assert self.sorted_standardized_residuals_ is not None
        standardized_value, _ = self._standardized_value(X, value)
        counts = np.searchsorted(
            self.sorted_standardized_residuals_,
            standardized_value,
            side="right",
        )
        return counts.astype(float) / len(self.sorted_standardized_residuals_)

    def pdf(self, X: np.ndarray, value: float | np.ndarray) -> np.ndarray:
        residuals = self._require_fitted()
        assert self.residual_kde_bandwidth_ is not None
        standardized_value, sigma = self._standardized_value(X, value)
        bandwidth = self.residual_kde_bandwidth_
        output = np.empty(len(standardized_value), dtype=float)
        # Bound the temporary kernel matrix for large inference calls.
        batch_size = max(1, 1_000_000 // max(len(residuals), 1))
        normalizer = np.sqrt(2.0 * np.pi) * bandwidth
        for start in range(0, len(standardized_value), batch_size):
            stop = min(start + batch_size, len(standardized_value))
            z = (
                standardized_value[start:stop, None] - residuals[None, :]
            ) / bandwidth
            standardized_density = np.mean(np.exp(-0.5 * z * z), axis=1) / normalizer
            output[start:stop] = standardized_density / sigma[start:stop]
        return output
