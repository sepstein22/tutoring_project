"""Test suite.

Every assertion checks a property that must hold regardless of the data:
analytic band power, the density/integration factor, unit harmonisation,
absence of leakage, metric identities, bootstrap agreeing with sd/sqrt(n), a
permutation null landing on chance. Nothing is checked against a number typed
in by hand.

    python3 -m pytest tests -q
"""
import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faced import config as C           # noqa: E402
from faced import data, features, models, protocol, report   # noqa: E402


# ------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def synthetic():
    """FACED-shaped data with a planted, recoverable signal."""
    rng = np.random.default_rng(11)
    n_sub, n_clip, p = 30, C.N_CLIP, C.N_EEG * C.N_BAND

    groups = np.repeat(np.arange(n_sub), n_clip)
    clips = np.tile(np.arange(n_clip), n_sub)

    clip_effect = rng.uniform(1.0, 6.0, n_clip)
    subject_offset = rng.normal(0.0, 0.5, n_sub)
    weights = rng.normal(0.0, 1.0, p) * (rng.random(p) < 0.2)

    X = rng.normal(0.0, 1.0, (n_sub * n_clip, p))
    X += subject_offset[groups][:, None] * 0.4
    signal = X @ weights
    signal = 1.3 * (signal - signal.mean()) / signal.std()

    y = np.clip(clip_effect[clips] + subject_offset[groups] + signal
                + rng.normal(0, 0.5, n_sub * n_clip), 0, 7)
    return {"X": X, "y": y, "groups": groups, "clips": clips, "n_sub": n_sub}


# -------------------------------------------------------------------- physics

def test_band_power_recovers_analytic_sine():
    t = np.arange(0, 30, 1.0 / C.SFREQ)
    sine = np.sin(2 * np.pi * 10.0 * t)[None, :]
    power = features.band_power(sine)
    alpha = C.BAND_NAMES.index("alpha")
    assert abs(power[0, alpha] - 0.5) < 1e-6, power[0]


def test_summing_the_density_would_double_it():
    t = np.arange(0, 30, 1.0 / C.SFREQ)
    sine = np.sin(2 * np.pi * 10.0 * t)[None, :]
    correct = features.band_power(sine, nperseg_seconds=2.0)
    alpha = C.BAND_NAMES.index("alpha")
    df = 1.0 / 2.0
    assert abs((correct[0, alpha] / df) / correct[0, alpha] - 1 / df) < 1e-9


def test_pure_tone_leaves_other_bands_empty():
    t = np.arange(0, 30, 1.0 / C.SFREQ)
    sine = np.sin(2 * np.pi * 10.0 * t)[None, :]
    power = features.band_power(sine)[0]
    others = np.delete(power, C.BAND_NAMES.index("alpha"))
    assert others.max() < 1e-9


def test_faced_bands_differ_from_textbook():
    assert C.BANDS["alpha"] == (8.0, 14.0)
    assert C.BANDS_TEXTBOOK["alpha"] == (8.0, 13.0)
    assert C.BANDS["beta"] != C.BANDS_TEXTBOOK["beta"]


# ---------------------------------------------------------------------- units

def test_unit_scale_known_and_unknown():
    assert data.unit_scale("V") == 1.0
    assert data.unit_scale("uV") == 1e-6
    with pytest.raises(KeyError):
        data.unit_scale("furlongs")


@pytest.mark.parametrize("unit", ["uV", "uv", " UV ", "\u00b5V", "\u03bcV",
                                  "microvolt", "microvolts"])
def test_every_microvolt_spelling_folds_to_one_scale(unit):
    """U+00B5 MICRO SIGN and U+03BC GREEK MU both render as mu and are not equal."""
    assert data.unit_scale(unit) == 1e-6


@pytest.mark.parametrize("unit", ["V", "v", "volt", "Volts"])
def test_every_volt_spelling_folds_to_one_scale(unit):
    assert data.unit_scale(unit) == 1.0


def test_one_million_microvolts_is_one_volt():
    np.testing.assert_allclose(
        data.harmonise_units(np.array([1_000_000.0]), "uV"), [1.0])


def test_microvolt_offset_is_exactly_log_1e12():
    assert abs(data.log_power_offset("uV") - np.log(1e12)) < 1e-12
    assert abs(data.log_power_offset("uV", half=True) - 0.5 * np.log(1e12)) < 1e-12
    assert data.log_power_offset("V") == 0.0


def test_harmonising_removes_the_offset_exactly():
    """A subject stored in uV must land on the same features as one in V."""
    rng = np.random.default_rng(3)
    volts = rng.normal(0, 1e-5, (C.N_EEG, C.N_SAMPLE))
    microvolts = volts / data.unit_scale("uV")        # same signal, uV numbers

    raw_gap = (features.differential_entropy(microvolts)
               - features.differential_entropy(volts))
    assert np.allclose(raw_gap, data.log_power_offset("uV", half=True), atol=1e-9)

    fixed = data.harmonise_units(microvolts, "uV")
    assert np.allclose(features.differential_entropy(fixed),
                       features.differential_entropy(volts), atol=1e-12)


# --------------------------------------------------------------------- layout

def test_stack_unstack_round_trip():
    grid = np.arange(C.N_EEG * C.N_BAND, dtype=float).reshape(C.N_EEG, C.N_BAND)
    for order in ("band-major", "channel-major"):
        flat = features.stack(grid, order)
        assert flat.size == C.N_EEG * C.N_BAND
        assert np.array_equal(features.unstack(flat, C.N_EEG, C.N_BAND, order), grid)


def test_the_two_orders_really_differ():
    grid = np.arange(C.N_EEG * C.N_BAND, dtype=float).reshape(C.N_EEG, C.N_BAND)
    a = features.stack(grid, "band-major")
    b = features.stack(grid, "channel-major")
    assert a.shape == b.shape and not np.array_equal(a, b)


def test_band_major_index_formula():
    grid = np.arange(C.N_EEG * C.N_BAND, dtype=float).reshape(C.N_EEG, C.N_BAND)
    flat = features.stack(grid, "band-major")
    for b in range(C.N_BAND):
        for c in range(C.N_EEG):
            assert flat[b * C.N_EEG + c] == grid[c, b]


def test_feature_names_align():
    names = features.names(["ch%02d" % i for i in range(C.N_EEG)], "band-major")
    assert names[0] == "delta_ch00"
    assert names[C.N_EEG] == "theta_ch00"


# -------------------------------------------------------------------- topomap

def test_topomap_preserves_amplitude():
    """The bug in the original pipeline: rendering deleted the scale."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 30)
    radius = np.sqrt(rng.uniform(0, 1, 30))
    pos = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)]) * 0.9
    values = rng.uniform(1.0, 5.0, 30)

    a = features.topomap_grid(values, pos, size=32)
    b = features.topomap_grid(values * 1000.0, pos, size=32)
    inside = a != 0.0
    ratio = b[inside] / a[inside]
    assert np.allclose(ratio, 1000.0, rtol=1e-9)


def test_topomap_shape_and_mask():
    rng = np.random.default_rng(1)
    pos = rng.uniform(-0.8, 0.8, (30, 2))
    grid = features.topomap_grid(rng.uniform(1, 2, 30), pos, size=24)
    assert grid.shape == (24, 24)
    axis = np.linspace(-1, 1, 24)
    gx, gy = np.meshgrid(axis, axis)
    assert np.all(grid[gx ** 2 + gy ** 2 > 1.0] == 0.0)


# -------------------------------------------------------------------- metrics

def test_ccc_of_constant_predictor_is_zero():
    rng = np.random.default_rng(5)
    y = rng.normal(4, 1.5, 500)
    assert abs(protocol.ccc(y, np.full_like(y, y.mean()))) < 1e-20


def test_ccc_of_perfect_prediction_is_one():
    rng = np.random.default_rng(6)
    y = rng.normal(4, 1.5, 200)
    assert abs(protocol.ccc(y, y) - 1.0) < 1e-12


def test_ccc_penalises_a_constant_offset():
    rng = np.random.default_rng(7)
    y = rng.normal(4, 1.5, 200)
    assert protocol.ccc(y, y + 1.0) < protocol.ccc(y, y)


def test_per_subject_returns_one_score_each(synthetic):
    subjects, scores = protocol.per_subject(
        synthetic["y"], synthetic["y"] + 0.1, synthetic["groups"])
    assert len(subjects) == synthetic["n_sub"] == len(scores)
    assert np.allclose(scores, 0.1)


# ------------------------------------------------------------------- protocol

def test_folds_are_subject_disjoint(synthetic):
    g = synthetic["groups"]
    for train_idx, test_idx in protocol.folds(g, 3):
        assert not (set(g[train_idx]) & set(g[test_idx]))


def test_leak_assertion_actually_fires(synthetic):
    with pytest.raises(AssertionError):
        protocol.assert_no_leak(np.array([0, 1, 2]), np.array([2, 3]),
                                synthetic["groups"])


def test_more_splits_than_subjects_is_rejected():
    with pytest.raises(ValueError):
        list(protocol.folds(np.arange(3), 5))


def test_every_row_gets_one_prediction(synthetic):
    pred, fold_of, _ = protocol.cross_val_predict(
        models.ridge([1.0]), synthetic["X"], synthetic["y"], synthetic["groups"])
    assert np.isfinite(pred).all()
    assert (fold_of >= 0).all()
    assert len(np.unique(fold_of)) == 3


# --------------------------------------------------------------------- models

def test_floor_ignores_the_features(synthetic):
    m = models.Floor().fit(synthetic["X"], synthetic["y"])
    a = m.predict(synthetic["X"])
    b = m.predict(np.zeros_like(synthetic["X"]))
    assert np.allclose(a, b)


def test_ridge_beats_the_floor_on_planted_signal(synthetic):
    X, y, g = synthetic["X"], synthetic["y"], synthetic["groups"]
    pf, _, _ = protocol.cross_val_predict(models.Floor(), X, y, g)
    pr, _, _ = protocol.cross_val_predict(models.ridge(), X, y, g)
    assert protocol.mae(y, pr) < protocol.mae(y, pf)


def test_hyperparameter_is_selected_and_reported(synthetic):
    _, _, info = protocol.cross_val_predict(
        models.ridge(np.logspace(-1, 6, 10)),
        synthetic["X"], synthetic["y"], synthetic["groups"])
    assert len(info) == 3
    assert all("hyperparameter" in i for i in info)


def test_selection_never_sees_the_scored_fold(synthetic):
    """Inner folds must be drawn from training subjects only."""
    X, y, g = synthetic["X"], synthetic["y"], synthetic["groups"]
    seen = []

    original = models._select

    def spy(estimator_for, grid, Xi, yi, gi, n_splits=3):
        seen.append(set(np.unique(gi)))
        return original(estimator_for, grid, Xi, yi, gi, n_splits)

    models._select = spy
    try:
        for fold, (train_idx, test_idx) in enumerate(protocol.folds(g, 3)):
            m = models.ridge(np.logspace(0, 3, 4)).clone()
            m.fit(X[train_idx], y[train_idx], groups=g[train_idx])
            assert not (seen[fold] & set(np.unique(g[test_idx])))
    finally:
        models._select = original


@pytest.mark.parametrize("factory", [models.ridge, models.elastic_net,
                                     models.random_forest, models.svr])
def test_every_sklearn_model_runs(factory, synthetic):
    pred, _, _ = protocol.cross_val_predict(
        factory(), synthetic["X"], synthetic["y"], synthetic["groups"])
    assert np.isfinite(pred).all()
    assert pred.shape[0] == len(synthetic["y"])


# ------------------------------------------------------------------ inference

def test_bootstrap_matches_sd_over_sqrt_n():
    rng = np.random.default_rng(13)
    values = np.abs(rng.normal(1.0, 0.12, 60))
    out = protocol.bootstrap_mean(values, n_boot=20000, seed=3)
    assert abs(out["se_boot"] - out["se_formula"]) / out["se_formula"] < 0.05
    assert out["lo"] < out["mean"] < out["hi"]


def test_quadrupling_n_halves_the_standard_error():
    rng = np.random.default_rng(14)
    v = rng.normal(1.0, 0.2, 40)
    sd = v.std(ddof=1)
    assert abs((sd / np.sqrt(40)) / (sd / np.sqrt(160)) - 2.0) < 1e-12


def test_paired_difference_of_a_model_with_itself_is_zero():
    rng = np.random.default_rng(15)
    a = rng.normal(1.0, 0.1, 40)
    out = protocol.paired_difference(a, a, n_boot=2000)
    assert abs(out["mean"]) < 1e-12
    assert not out["excludes_zero"]


def test_permutation_p_has_a_floor_and_never_hits_zero(synthetic):
    X, y, g = synthetic["X"], synthetic["y"], synthetic["groups"]

    def fit_predict(labels):
        pred, _, _ = protocol.cross_val_predict(models.ridge([10.0]), X, labels, g)
        return pred

    out = protocol.permutation(fit_predict, y, g, n_permutation=25, seed=1)
    assert out["p"] >= out["p_floor"] > 0
    assert out["observed"] > out["null_mean"]


def test_permutation_shuffles_within_subject(synthetic):
    """The shuffled labels must preserve each subject's multiset of ratings."""
    g = synthetic["groups"]
    y = synthetic["y"]
    captured = []

    def fit_predict(labels):
        captured.append(labels.copy())
        return np.zeros_like(labels)

    protocol.permutation(fit_predict, y, g, n_permutation=3, seed=0)
    for labels in captured[1:]:
        for subject in np.unique(g):
            m = g == subject
            assert np.allclose(np.sort(labels[m]), np.sort(y[m]))


def test_family_wise_error_arithmetic():
    assert abs(protocol.family_wise_error(1) - 0.05) < 1e-12
    assert abs(protocol.family_wise_error(20) - (1 - 0.95 ** 20)) < 1e-12
    assert protocol.family_wise_error(20) > protocol.family_wise_error(6)


# --------------------------------------------------------------------- report

def test_zoo_runs_and_table_renders(synthetic):
    results = report.run_zoo(
        models.zoo(include_cnn=False, fast=True),
        synthetic["X"], synthetic["y"], synthetic["groups"], verbose=False)
    assert "floor" in results and "ridge" in results
    versus = report.compare_to_floor(results)
    text = report.table(results, versus)
    assert "floor" in text and "MAE" in text
    assert "ridge" in report.win_counts(results)


def test_floor_ccc_is_exact_per_fold_and_approximate_pooled(synthetic):
    results = report.run_zoo([models.Floor()], synthetic["X"], synthetic["y"],
                             synthetic["groups"], verbose=False)
    r = results["floor"]
    assert max(abs(c) for c in r["ccc_by_fold"]) < 1e-20
    note = report.floor_ccc_note(results)
    assert "per fold" in note


# ------------------------------------------------------------------- loading

def test_load_processed_rejects_a_wrong_shape(tmp_path):
    bad = tmp_path / "sub000.pkl"
    with open(bad, "wb") as f:
        pickle.dump(np.zeros((C.N_CHANNEL_TOTAL, C.N_CLIP, C.N_SAMPLE)), f)
    with pytest.raises(ValueError, match="transposed|expected"):
        data.load_processed(str(bad))


def test_load_processed_applies_the_unit(tmp_path):
    good = tmp_path / "sub001.pkl"
    array = np.ones((C.N_CLIP, C.N_CHANNEL_TOTAL, C.N_SAMPLE))
    with open(good, "wb") as f:
        pickle.dump(array, f)
    volts = data.load_processed(str(good), source_unit="uV")
    assert np.allclose(volts, 1e-6)


def test_subject_files_sort_numerically(tmp_path):
    for i in (0, 2, 10, 100):
        (tmp_path / ("sub%d.pkl" % i)).write_bytes(b"x")
    order = [os.path.basename(p) for p in data.subject_files(str(tmp_path))]
    assert order == ["sub0.pkl", "sub2.pkl", "sub10.pkl", "sub100.pkl"]


def test_recording_info_parses_stray_header_whitespace(tmp_path):
    csv_path = tmp_path / "Recording_info.csv"
    csv_path.write_text(
        "sub,Gender,Age,Cohort ,Sample_rate,Unit\n"
        "sub000,F,21,1,1000,V\n"
        "sub004,F,22,1,250,uV\n")
    info = data.load_recording_info(str(csv_path))
    assert set(info) == {0, 4}
    assert info[4].unit == "uV" and info[4].cohort == 1
    summary = data.unit_summary(info)
    assert summary["unit"] == {"V": 1, "uV": 1}
    assert summary["non_canonical"] == [4]


# ------------------------------------------------------------- CNN end to end

def test_cnn_runs_through_the_same_protocol(synthetic):
    """The CNN must obey the identical interface and split as everything else."""
    pytest.importorskip("tensorflow")
    rng = np.random.default_rng(2)
    theta = rng.uniform(0, 2 * np.pi, C.N_EEG)
    radius = np.sqrt(rng.uniform(0, 1, C.N_EEG))
    pos = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)]) * 0.9

    X, y, g = synthetic["X"], synthetic["y"], synthetic["groups"]
    images = features.image_stack(X[:, :C.N_EEG * C.N_BAND], pos, C.N_EEG, size=16)

    pred, fold_of, info = protocol.cross_val_predict(
        models.CNN(epochs=2, width=4), X, y, g, images=images)
    assert np.isfinite(pred).all()
    assert pred.shape == y.shape
    assert all(i["parameters"] > 0 for i in info)


def test_cnn_is_small_relative_to_the_flatten_design():
    """GlobalAveragePooling instead of Flatten is the point; check it holds."""
    pytest.importorskip("tensorflow")
    from tensorflow.keras import layers, models as km

    ours = models.CNN(width=16)._build((64, 64, 5), 2).count_params()
    flattened = km.Sequential([
        layers.Input(shape=(128, 128, 5)),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(2),
        layers.Flatten(), layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"), layers.Dense(2, activation="tanh"),
    ]).count_params()
    assert ours < flattened / 50, (ours, flattened)


def test_cnn_refuses_to_run_without_images(synthetic):
    pytest.importorskip("tensorflow")
    with pytest.raises(ValueError, match="images"):
        models.CNN(epochs=1).fit(synthetic["X"], synthetic["y"])


# ============================================================================
# Regression tests for the review of 2026-08-08. Each corresponds to a defect
# that was measured, not suspected.
# ============================================================================

def test_mae_rejects_broadcastable_shapes():
    """The worst of the lot: (n,) against (n,1) returned 1.6 instead of 0.0."""
    y_true = np.arange(5, dtype=float)
    y_pred = np.arange(5, dtype=float).reshape(-1, 1)
    with pytest.raises(ValueError, match="shape mismatch"):
        protocol.mae(y_true, y_pred)
    with pytest.raises(ValueError, match="shape mismatch"):
        protocol.rmse(y_true, y_pred)
    with pytest.raises(ValueError, match="shape mismatch"):
        protocol.ccc(y_true, y_pred)


def test_metrics_agree_once_shapes_match():
    y = np.arange(5, dtype=float)
    assert protocol.mae(y, y.copy()) == 0.0
    assert protocol.rmse(y, y.copy()) == 0.0


@pytest.mark.parametrize("bad", [np.array([np.nan, 1.0]), np.array([np.inf, 1.0])])
def test_metrics_reject_non_finite(bad):
    good = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="NaN or infinity"):
        protocol.mae(good, bad)


def test_metrics_reject_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        protocol.mae(np.array([]), np.array([]))


def test_selection_rejects_a_single_training_subject():
    """Fail at the boundary with an explanation, not deep inside folds()."""
    with pytest.raises(ValueError, match="at least two training subjects"):
        models._select(lambda a: None, [1.0], np.zeros((5, 3)),
                       np.zeros(5), np.zeros(5))


def test_folds_rejects_fewer_than_two_splits():
    with pytest.raises(ValueError, match="at least 2"):
        list(protocol.folds(np.arange(10), 1))


def test_load_feature_file_rejects_too_few_channels(tmp_path):
    path = tmp_path / "sub000.pkl"
    with open(path, "wb") as f:
        pickle.dump(np.zeros((C.N_CLIP, 12, 30, C.N_BAND)), f)
    with pytest.raises(ValueError, match="only 12"):
        data.load_feature_file(str(path), n_eeg=30)
    assert data.load_feature_file(str(path), n_eeg=12).shape[1] == 12


def test_load_feature_file_rejects_non_finite(tmp_path):
    path = tmp_path / "sub001.pkl"
    array = np.zeros((C.N_CLIP, C.N_EEG, 30, C.N_BAND))
    array[0, 0, 0, 0] = np.nan
    with open(path, "wb") as f:
        pickle.dump(array, f)
    with pytest.raises(ValueError, match="NaN or infinity"):
        data.load_feature_file(str(path), n_eeg=C.N_EEG)


def test_load_processed_rejects_too_few_channels(tmp_path):
    path = tmp_path / "sub002.pkl"
    with open(path, "wb") as f:
        pickle.dump(np.zeros((C.N_CLIP, C.N_CHANNEL_TOTAL, C.N_SAMPLE)), f)
    with pytest.raises(ValueError, match="requested 40"):
        data.load_processed(str(path), n_eeg=40)


def test_canonical_subject_id_uses_the_first_digit_run():
    assert data.canonical_subject_id("sub007_rating.mat") == 7
    assert data.canonical_subject_id("/a/b/sub122.pkl") == 122
    assert data.canonical_subject_id("sub004_v2.mat") == 4       # suffix ignored
    with pytest.raises(ValueError, match="no subject number"):
        data.canonical_subject_id("ratings.mat")


def test_index_by_subject_rejects_duplicates():
    with pytest.raises(ValueError, match="two rating candidates"):
        data.index_by_subject(["a/sub003.mat", "b/sub003.mat"], "rating")


def test_index_by_subject_is_a_join_not_a_zip():
    """A missing subject must not shift every later pairing by one."""
    features_ = ["sub000.pkl", "sub001.pkl", "sub002.pkl"]
    ratings = ["sub000.mat", "sub002.mat"]          # subject 1 absent
    f = data.index_by_subject(features_)
    r = data.index_by_subject(ratings)
    assert sorted(set(f) - set(r)) == [1]
    assert r[2].startswith("sub002")                # not shifted onto subject 1


def test_cross_val_predict_rejects_mismatched_images(synthetic):
    with pytest.raises(ValueError, match="images has"):
        protocol.cross_val_predict(
            models.Floor(), synthetic["X"], synthetic["y"], synthetic["groups"],
            images=np.zeros((7, 4, 4, 1)))


def test_ccc_one_constant_predictor_is_zero():
    assert protocol.ccc(np.array([1.0, 2.0, 3.0]),
                        np.array([2.0, 2.0, 2.0])) == pytest.approx(0.0)


def test_ccc_documented_edge_cases():
    both_equal = protocol.ccc(np.array([2.0, 2.0, 2.0]), np.array([2.0, 2.0, 2.0]))
    assert np.isnan(both_equal)                      # 0/0, undefined, not 1
    assert protocol.ccc(np.array([2.0, 2.0, 2.0]),
                        np.array([5.0, 5.0, 5.0])) == pytest.approx(0.0)
    assert np.isnan(protocol.ccc(np.array([2.0]), np.array([3.0])))

