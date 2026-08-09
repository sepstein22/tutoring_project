"""Model zoo. Every entry implements clone(), fit(), predict() and info()."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone as sk_clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from . import config as C
from .protocol import folds


class Model:
    """Interface every entry implements."""

    name = "model"
    needs_images = False

    def clone(self):
        """Return an unfitted copy with the same configuration."""
        raise NotImplementedError

    def fit(self, X, y, groups=None, images=None):
        """Fit on one training fold. Returns self."""
        raise NotImplementedError

    def predict(self, X, images=None):
        """Predict for held-out rows."""
        raise NotImplementedError

    def info(self):
        """Per-fold diagnostics, such as the chosen hyper-parameter."""
        return {}


# ------------------------------------------------------------------ helpers

def _as_2d(y):
    y = np.asarray(y, dtype=np.float64)
    return y.reshape(-1, 1) if y.ndim == 1 else y


def _select(estimator_for, grid, X, y, groups, n_splits=3):
    """Pick the grid entry with the lowest MAE on inner subject-wise folds.

    Inputs
        estimator_for  callable(value) -> a fresh scikit-learn estimator
        grid           values to try
        groups         subject labels for the TRAINING rows only

    Returns
        (best_value, its mean inner MAE)

    Raises
        ValueError with fewer than two training subjects.
    """
    # TODO: (@takashi) The inner split uses only training subjects. Say what
    # would leak if it used all of them, and why the leak is invisible in the
    # reported score.
    n_group = len(np.unique(groups))
    if n_group < 2:
        raise ValueError(
            "hyper-parameter selection needs at least two training subjects, "
            "got %d" % n_group)
    n_splits = min(n_splits, n_group)
    splits = list(folds(groups, n_splits))

    best, best_error = None, np.inf
    for value in grid:
        errors = []
        for train_idx, test_idx in splits:
            est = estimator_for(value)
            est.fit(X[train_idx], y[train_idx])
            errors.append(np.abs(y[test_idx] - est.predict(X[test_idx])).mean())
        score = float(np.mean(errors))
        if score < best_error:
            best, best_error = value, score
    return best, best_error


class _SklearnModel(Model):
    """Wraps any scikit-learn regressor, with optional inner selection."""

    def __init__(self, name, build, grid=None, multioutput=False,
                 inner_splits=3):
        self.name = name
        self._build = build
        self._grid = None if grid is None else list(grid)
        self._multioutput = multioutput
        self._inner_splits = inner_splits
        self._estimator = None
        self._chosen = None

    def clone(self):
        """Return an unfitted copy with the same configuration."""
        return _SklearnModel(self.name, self._build, self._grid,
                             self._multioutput, self._inner_splits)

    def _make(self, value):
        est = make_pipeline(StandardScaler(), self._build(value))
        if self._multioutput and not getattr(self, "_single", False):
            return MultiOutputRegressor(est)
        return est

    def fit(self, X, y, groups=None, images=None):
        """Select a hyper-parameter, then fit. Returns self."""
        X = np.asarray(X, dtype=np.float64)
        y = _as_2d(y)
        # A single-column target goes in as 1-D: some estimators warn otherwise.
        self._single = y.shape[1] == 1
        if self._single:
            y = y.reshape(-1)
        selectable = (self._grid is not None and groups is not None
                      and len(self._grid) > 1)
        if selectable:
            self._chosen, _ = _select(self._make, self._grid, X, y, groups,
                                      self._inner_splits)
            at_edge = self._chosen in (self._grid[0], self._grid[-1])
            if at_edge and len(self._grid) > 2:
                self._edge = True
            else:
                self._edge = False
        else:
            self._chosen = None if self._grid is None else self._grid[0]
            self._edge = False
        self._estimator = self._make(self._chosen)
        self._estimator.fit(X, y)
        return self

    def predict(self, X, images=None):
        """Predict for held-out rows."""
        return self._estimator.predict(np.asarray(X, dtype=np.float64))

    def info(self):
        """Per-fold diagnostics, including the selected hyper-parameter."""
        out = {"model": self.name}
        if self._chosen is not None:
            out["hyperparameter"] = self._chosen
            if getattr(self, "_edge", False):
                out["warning"] = "selected value sits at the edge of the grid"
        return out


class Floor(Model):
    """Predict the training-fold mean for everybody, ignoring the features.

    Constant within a fold, but not across folds.
    """

    name = "floor"

    def clone(self):
        """Return an unfitted copy."""
        return Floor()

    def fit(self, X, y, groups=None, images=None):
        """Record the training-fold mean. Returns self."""
        y2 = _as_2d(y)
        self._single = y2.shape[1] == 1
        self._mean = y2.mean(axis=0)
        return self

    def predict(self, X, images=None):
        """Return the training-fold mean for every row."""
        out = np.tile(self._mean, (len(X), 1))
        return out.reshape(-1) if self._single else out


class CNN(Model):
    """Small convolutional net over per-band topomap grids.

    Inputs and targets are standardised using training-fold statistics only.
    """

    name = "cnn"
    needs_images = True

    def __init__(self, epochs=40, batch_size=32, learning_rate=1e-3,
                 dropout=0.3, width=16, seed=C.SEED):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.width = width
        self.seed = seed
        self._net = None

    def clone(self):
        """Return an unfitted copy with the same configuration."""
        return CNN(self.epochs, self.batch_size, self.learning_rate,
                   self.dropout, self.width, self.seed)

    def _build(self, input_shape, n_out):
        import tensorflow as tf
        from tensorflow.keras import layers, models

        tf.keras.utils.set_random_seed(self.seed)
        w = self.width
        return models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(w, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(2 * w, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(4 * w, 3, padding="same", activation="relu"),
            # TODO: (@takashi) Why GlobalAveragePooling2D rather than Flatten?
            # Count the parameters each would add and compare with the number
            # of training trials.
            layers.GlobalAveragePooling2D(),
            layers.Dropout(self.dropout),
            layers.Dense(2 * w, activation="relu"),
            layers.Dense(n_out),        # linear: tanh cannot reach a 0-7 rating
        ])

    def fit(self, X, y, groups=None, images=None):
        """Standardise on this fold and train. Returns self."""
        import logging
        import tensorflow as tf

        tf.get_logger().setLevel(logging.ERROR)
        if images is None:
            raise ValueError("CNN needs images=; pass the topomap stack")

        y2 = _as_2d(y)
        self._mu, self._sd = images.mean(), images.std() + 1e-8
        self._ymu, self._ysd = y2.mean(axis=0), y2.std(axis=0) + 1e-8

        self._net = self._build(images.shape[1:], y2.shape[1])
        self._net.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="mse", metrics=["mae"])
        self._net.fit(
            (images - self._mu) / self._sd, (y2 - self._ymu) / self._ysd,
            epochs=self.epochs, batch_size=self.batch_size, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=6, restore_best_weights=True)])
        return self

    def predict(self, X, images=None):
        """Predict for held-out rows, undoing the target standardisation."""
        if images is None:
            raise ValueError("CNN needs images=")
        z = self._net.predict((images - self._mu) / self._sd, verbose=0)
        return z * self._ysd + self._ymu

    def info(self):
        """Per-fold diagnostics, including the parameter count."""
        return {"model": self.name,
                "parameters": (None if self._net is None
                               else self._net.count_params())}


# -------------------------------------------------------------------- zoo

def ridge(alphas=None, multioutput=False):
    """Ridge regression with alpha chosen inside the training folds."""
    grid = np.logspace(-1, 6, 20) if alphas is None else np.asarray(alphas)
    return _SklearnModel("ridge", lambda a: Ridge(alpha=float(a)), grid,
                         multioutput=multioutput)


def elastic_net(alphas=None):
    """Elastic net at l1_ratio 0.5, alpha chosen inside the training folds."""
    grid = np.logspace(-4, 0, 12) if alphas is None else np.asarray(alphas)
    return _SklearnModel(
        "elastic_net",
        lambda a: ElasticNet(alpha=float(a), l1_ratio=0.5, max_iter=20000),
        grid)


def random_forest(n_estimators=300, seed=C.SEED):
    """Random forest with fixed hyper-parameters."""
    return _SklearnModel(
        "random_forest",
        lambda _: RandomForestRegressor(n_estimators=n_estimators,
                                        random_state=seed, n_jobs=-1,
                                        min_samples_leaf=3),
        grid=[None])


def svr(cs=None):
    """RBF support vector regression with C chosen inside the training folds."""
    grid = np.logspace(-2, 3, 8) if cs is None else np.asarray(cs)
    return _SklearnModel("svr", lambda c: SVR(C=float(c), kernel="rbf"),
                         grid, multioutput=True)


def zoo(include_cnn=True, fast=False):
    """The default comparison set.

    Inputs
        fast  smaller grids and fewer epochs, for a smoke test
    """
    models = [
        Floor(),
        ridge(np.logspace(-1, 6, 8 if fast else 20)),
        elastic_net(np.logspace(-4, 0, 5 if fast else 12)),
        random_forest(n_estimators=60 if fast else 300),
        svr(np.logspace(-2, 3, 4 if fast else 8)),
    ]
    if include_cnn:
        models.append(CNN(epochs=8 if fast else 40))
    return models

