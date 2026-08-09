"""Validation protocol, metrics and resampling inference."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from . import config as C


def _validate_metric_inputs(y_true, y_pred):
    """Return both arrays as float64, or raise.

    Raises
        ValueError if the shapes differ, either is empty, or either holds a
        non-finite value.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    # TODO: (@takashi) Shapes (n,) and (n, 1) hold the same numbers, so why
    # does this reject them instead of reshaping? Work out what numpy does to
    # y_true - y_pred in that case, then write one sentence.
    if y_true.shape != y_pred.shape:
        raise ValueError("shape mismatch: y_true=%s, y_pred=%s"
                         % (y_true.shape, y_pred.shape))
    if y_true.size == 0:
        raise ValueError("metric inputs must not be empty")
    if not np.isfinite(y_true).all():
        raise ValueError("y_true contains NaN or infinity")
    if not np.isfinite(y_pred).all():
        raise ValueError("y_pred contains NaN or infinity")
    return y_true, y_pred


def mae(y_true, y_pred):
    """Mean absolute error."""
    y_true, y_pred = _validate_metric_inputs(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """Root mean squared error."""
    y_true, y_pred = _validate_metric_inputs(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def ccc(y_true, y_pred):
    """Lin's concordance correlation coefficient.

        2 * cov / (var_true + var_pred + (mean_true - mean_pred)^2)

    Conventions at the edges, asserted in the tests:
        one input constant, the other not -> 0.0
        both constant and unequal         -> 0.0
        both constant and equal           -> nan
        fewer than two observations       -> nan

    Returns
        float in [-1, 1], or nan.
    """
    y_true, y_pred = _validate_metric_inputs(y_true, y_pred)
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    if y_true.size < 2:
        return float("nan")

    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    # ddof=0 throughout: mixing it between the covariance and the variances
    # shifts the result by percent-level amounts at small n.
    denominator = (y_true.var(ddof=0) + y_pred.var(ddof=0)
                   + (mean_true - mean_pred) ** 2)
    if denominator == 0:
        return float("nan")
    covariance = ((y_true - mean_true) * (y_pred - mean_pred)).mean()
    return float(2.0 * covariance / denominator)


METRICS = {"mae": mae, "rmse": rmse, "ccc": ccc}


def per_subject(y_true, y_pred, groups, metric="mae"):
    """Score each subject separately.

    Inputs
        groups  (n_trial,) subject label per row
        metric  a key of METRICS, or a callable (y_true, y_pred) -> float

    Returns
        (subjects_in_first_appearance_order, score_per_subject)
    """
    score = METRICS[metric] if isinstance(metric, str) else metric
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    groups = np.asarray(groups).reshape(-1)

    seen, order = set(), []
    for subject in groups:
        if subject not in seen:
            seen.add(subject)
            order.append(subject)

    scores = [score(y_true[groups == s].reshape(-1),
                    y_pred[groups == s].reshape(-1)) for s in order]
    return np.asarray(order), np.asarray(scores, dtype=np.float64)


def assert_no_leak(train_idx, test_idx, groups):
    """Raise if any subject appears on both sides of a split."""
    groups = np.asarray(groups)
    shared = np.intersect1d(groups[train_idx], groups[test_idx])
    if shared.size:
        raise AssertionError("SUBJECT LEAK: %d subject(s) on both sides, "
                             "e.g. %s"
                             % (shared.size, shared[:5]))


def folds(groups, n_splits=3):
    """Yield subject-wise (train_idx, test_idx) pairs.

    Raises
        ValueError if n_splits is below 2 or exceeds the number of subjects.
    """
    groups = np.asarray(groups).reshape(-1)
    n_group = len(np.unique(groups))
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2, got %d" % n_splits)
    if n_splits > n_group:
        raise ValueError("%d splits requested but only %d subjects"
                         % (n_splits, n_group))

    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(
            np.zeros(len(groups)), None, groups):
        # TODO: (@takashi) GroupKFold already guarantees disjoint groups, so
        # why check again? Describe the upstream mistake this catches — it is
        # not a bug in GroupKFold.
        assert_no_leak(train_idx, test_idx, groups)
        yield train_idx, test_idx


def cross_val_predict(model, X, y, groups, n_splits=3, images=None):
    """Out-of-fold predictions under a subject-wise split.

    Inputs
        model   object with clone(), fit(X, y, groups, images) and
                predict(X, images), as in models.py
        images  optional (n_trial, height, width, n_band) array

    Returns
        (predictions shaped like y, fold index per row, per-fold info dicts)

    Raises
        AssertionError if any row is predicted zero times or more than once.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups).reshape(-1)
    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y, groups lengths differ: %d, %d, %d"
                         % (len(X), len(y), len(groups)))
    if images is not None and len(images) != len(y):
        raise ValueError("images has %d rows but y has %d"
                         % (len(images), len(y)))

    predictions = np.full(y.shape, np.nan, dtype=np.float64)
    fold_of = np.full(len(y), -1, dtype=int)
    times_predicted = np.zeros(len(y), dtype=int)
    info = []

    for fold, (train_idx, test_idx) in enumerate(folds(groups, n_splits)):
        fitted = model.clone()
        fitted.fit(X[train_idx], y[train_idx], groups=groups[train_idx],
                   images=None if images is None else images[train_idx])
        block = np.asarray(fitted.predict(
            X[test_idx], images=None if images is None else images[test_idx]),
            dtype=np.float64)
        # Models may return (n,) or (n, 1) for a single target.
        predictions[test_idx] = block.reshape(predictions[test_idx].shape)
        fold_of[test_idx] = fold
        times_predicted[test_idx] += 1
        info.append(fitted.info())

    if not np.all(times_predicted == 1):
        raise AssertionError(
            "every row must get exactly one out-of-fold prediction; "
            "%d got none, %d got more than one"
            % (int((times_predicted == 0).sum()),
               int((times_predicted > 1).sum())))
    if not np.isfinite(predictions).all():
        raise ValueError("out-of-fold predictions contain NaN or infinity")
    return predictions, fold_of, info


def bootstrap_mean(values, n_boot=20000, alpha=0.05, seed=C.SEED):
    """Percentile bootstrap interval for the mean of a per-subject array.

    Inputs
        values  one score per subject
        alpha   1 - alpha is the nominal coverage

    Returns
        dict with mean, lo, hi, se_boot, se_formula, n. All nan when n < 2.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 2:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "se_boot": float("nan"), "se_formula": float("nan"),
                "n": int(n)}

    # TODO: (@takashi) This resamples subjects. Explain what goes wrong if you
    # resample trials instead, and say which direction the interval moves.
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, n, (n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(values.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "se_boot": float(draws.std(ddof=1)),
        "se_formula": float(values.std(ddof=1) / np.sqrt(n)),
        "n": int(n),
    }


def paired_difference(a, b, n_boot=20000, alpha=0.05, seed=C.SEED):
    """Bootstrap interval for the per-subject difference a - b.

    Returns
        the bootstrap_mean dict, plus n_a_lower, n_pairs and excludes_zero.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("paired arrays must align: %s vs %s"
                         % (a.shape, b.shape))

    usable = np.isfinite(a) & np.isfinite(b)
    difference = a[usable] - b[usable]
    out = bootstrap_mean(difference, n_boot=n_boot, alpha=alpha, seed=seed)
    out["n_a_lower"] = int((difference < 0).sum())
    out["n_pairs"] = int(usable.sum())
    out["excludes_zero"] = bool(out["lo"] > 0 or out["hi"] < 0)
    return out


def permutation(fit_predict, y, groups, n_permutation=1000, seed=C.SEED):
    """Within-subject label permutation test.

    Inputs
        fit_predict  callable(labels) -> predictions. Must refit the whole
                     pipeline on the labels it is given.

    Returns
        dict with observed, null_mean, null_sd, count_ge, p, p_floor, null.
        p = (1 + count) / (n_permutation + 1), so its smallest value is
        1 / (n_permutation + 1) and never zero.
    """
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups).reshape(-1)
    rng = np.random.default_rng(seed)

    def score(prediction):
        return -mae(y, prediction)               # higher is better

    observed = score(fit_predict(y))
    rows_of = {subject: np.flatnonzero(groups == subject)
               for subject in np.unique(groups)}

    null = np.empty(n_permutation, dtype=np.float64)
    for draw in range(n_permutation):
        shuffled = y.copy()
        # TODO: (@takashi) The shuffle stays inside each subject. Say what a
        # global shuffle would additionally destroy, and which way that moves
        # the p-value.
        for rows in rows_of.values():
            shuffled[rows] = y[rng.permutation(rows)]
        null[draw] = score(fit_predict(shuffled))

    count_ge = int((null >= observed).sum())
    return {
        "observed": float(observed),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "count_ge": count_ge,
        "n_permutation": int(n_permutation),
        "p": (1.0 + count_ge) / (n_permutation + 1.0),
        "p_floor": 1.0 / (n_permutation + 1.0),
        "null": null,
    }


def family_wise_error(n_tests, alpha=0.05):
    """P(at least one false positive) over n independent tests.

    Optimistic: configurations tried on the same data are correlated, not
    independent, so treat this as a lower bound.
    """
    return 1.0 - (1.0 - alpha) ** int(n_tests)

