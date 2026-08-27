from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Optional

import numpy as np
try:
    import tensorflow as tf
    import keras
    from keras import layers
    import tensorflow_probability as tfp
except ModuleNotFoundError as exc:  # GBDT-only installs do not require TensorFlow.
    tf = None  # type: ignore[assignment]
    keras = None  # type: ignore[assignment]
    layers = None  # type: ignore[assignment]
    tfp = None  # type: ignore[assignment]
    _TENSORFLOW_IMPORT_ERROR: Optional[ModuleNotFoundError] = exc
else:
    _TENSORFLOW_IMPORT_ERROR = None

tfd = None if tfp is None else tfp.distributions


@dataclass
class MDNLearnerConfig:
    n_components: int = 5
    hidden_dims: Tuple[int, ...] = (64, 64)
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 50
    seed: int = 123
    verbose: bool = False
    min_sigma: float = 1e-3
    max_log_sigma: float = 5.0


class MDNLearner:
    """Small TensorFlow/Keras mixture density network for one-dimensional targets."""

    def __init__(self, input_dim: int, config: Optional[MDNLearnerConfig] = None):
        if _TENSORFLOW_IMPORT_ERROR is not None:
            raise ImportError(
                "The MDN backend requires the packages in requirements-full.txt"
            ) from _TENSORFLOW_IMPORT_ERROR
        self.input_dim = int(input_dim)
        self.config = config or MDNLearnerConfig()
        tf.random.set_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.model = self._build_model()
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.lr),
            loss=self._loss,
            jit_compile=False,
        )

    def _build_model(self) -> keras.Model:
        inputs = layers.Input(shape=(self.input_dim,))
        x = inputs
        for h in self.config.hidden_dims:
            x = layers.Dense(int(h), activation="relu")(x)
        params = layers.Dense(3 * self.config.n_components, activation=None)(x)
        return keras.Model(inputs, params)

    def _distribution_from_params(self, params: tf.Tensor) -> Any:
        logits, locs, raw_scales = tf.split(params, 3, axis=-1)
        # Softplus is more stable than exp(log_sigma) for small samples.
        scales = tf.nn.softplus(tf.clip_by_value(raw_scales, -20.0, self.config.max_log_sigma)) + self.config.min_sigma
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=logits),
            components_distribution=tfd.Normal(loc=locs, scale=scales),
        )

    def _loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        dist = self._distribution_from_params(y_pred)
        y = tf.reshape(y_true, [-1])
        return -tf.reduce_mean(dist.log_prob(y))

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        return self.model.fit(
            X,
            y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=int(self.config.verbose),
        )

    def distribution(self, X: np.ndarray) -> Any:
        X = np.asarray(X, dtype=np.float32)
        params = self.model(np.atleast_2d(X), training=False)
        return self._distribution_from_params(params)

    def sample(self, X: np.ndarray, n_samples: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        dist = self.distribution(X)
        # Use an operation-local stateless seed rather than resetting TensorFlow's
        # global RNG. Batched rows receive independent draws, while repeated calls
        # with the same inputs and seed remain reproducible.
        seed = None
        if random_state is not None:
            seed_value = int(random_state) % (2**32)
            seed = tf.constant(
                [
                    seed_value & 0x7FFFFFFF,
                    (seed_value ^ 0x6A09E667) & 0x7FFFFFFF,
                ],
                dtype=tf.int32,
            )
        samples = dist.sample(int(n_samples), seed=seed)  # (n_samples, n)
        return tf.transpose(samples).numpy()

    def cdf(self, X: np.ndarray, value: float) -> np.ndarray:
        dist = self.distribution(X)
        return dist.cdf(value).numpy()

    def pdf(self, X: np.ndarray, value: float) -> np.ndarray:
        dist = self.distribution(X)
        return tf.exp(dist.log_prob(value)).numpy()
