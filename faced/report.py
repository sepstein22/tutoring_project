"""Run the zoo under one protocol and report it."""
from __future__ import annotations

import numpy as np

from . import config as C
from . import protocol as P


def run_zoo(models, X, y, groups, images=None, n_splits=3, seed=C.SEED,
            verbose=True):
    """Score every model on identical folds.

    Returns
        {model name: dict of prediction, per-subject scores, bootstrap, ...}
    """
    y = np.asarray(y, dtype=np.float64)
    results = {}

    for model in models:
        if model.needs_images and images is None:
            if verbose:
                print("  skipping %s (needs images)" % model.name)
            continue
        if verbose:
            print("  %s ..." % model.name, flush=True)

        pred, fold_of, info = P.cross_val_predict(
            model, X, y, groups, n_splits=n_splits, images=images)

        subjects, mae_by_subject = P.per_subject(y, pred, groups, "mae")
        results[model.name] = {
            "prediction": pred,
            "fold_of": fold_of,
            "fold_info": info,
            "subjects": subjects,
            "mae_by_subject": mae_by_subject,
            "mae": float(np.mean(mae_by_subject)),
            "ccc_pooled": P.ccc(y.reshape(-1), pred.reshape(-1)),
            "ccc_by_fold": [P.ccc(y[fold_of == f].reshape(-1),
                                  pred[fold_of == f].reshape(-1))
                            for f in np.unique(fold_of)],
            "bootstrap": P.bootstrap_mean(mae_by_subject, seed=seed),
        }
    return results


def compare_to_floor(results, floor_name="floor", seed=C.SEED):
    """Paired per-subject comparison of every model against the floor.

    Returns
        {model name: paired_difference dict}, empty if the floor is absent.
    """
    if floor_name not in results:
        return {}
    base = results[floor_name]["mae_by_subject"]
    return {name: P.paired_difference(r["mae_by_subject"], base, seed=seed)
            for name, r in results.items() if name != floor_name}


def table(results, versus=None):
    """Render the comparison as plain text."""
    lines = []
    head = ("%-14s %8s %8s %8s %19s %9s" %
            ("model", "MAE", "SD", "CCC", "95% CI on MAE", "vs floor"))
    lines.append(head)
    lines.append("-" * len(head))

    order = sorted(results, key=lambda n: (n != "floor", results[n]["mae"]))
    for name in order:
        r = results[name]
        b = r["bootstrap"]
        delta = ""
        if versus and name in versus:
            v = versus[name]
            delta = "%+8.4f%s" % (v["mean"], "*" if v["excludes_zero"] else " ")
        lines.append("%-14s %8.4f %8.4f %8.4f  [%7.4f, %7.4f] %9s"
                     % (name, r["mae"],
                        float(np.std(r["mae_by_subject"], ddof=1)),
                        r["ccc_pooled"], b["lo"], b["hi"], delta))

    lines.append("")
    lines.append("MAE and SD are across SUBJECTS, not folds. CI is a")
    lines.append("percentile bootstrap over subjects. * marks a paired")
    lines.append("difference whose 95% interval excludes zero. Lower MAE is")
    lines.append("better; a model that does not beat the floor has learned")
    lines.append("nothing that transfers.")
    return "\n".join(lines)


def win_counts(results, floor_name="floor"):
    """Count subjects on which each model beats the floor and each other."""
    lines = ["per-subject win counts"]
    names = [n for n in results if n != floor_name]
    if floor_name in results:
        base = results[floor_name]["mae_by_subject"]
        for name in names:
            wins = int((results[name]["mae_by_subject"] < base).sum())
            lines.append("  %-14s beats floor on %3d / %d subjects"
                         % (name, wins, len(base)))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            wins = int((results[a]["mae_by_subject"]
                        < results[b]["mae_by_subject"]).sum())
            lines.append("  %-14s beats %-14s on %3d / %d"
                         % (a, b, wins, len(results[a]["mae_by_subject"])))
    return "\n".join(lines)


def floor_ccc_note(results, floor_name="floor"):
    """Report the floor's CCC per fold and pooled."""
    if floor_name not in results:
        return ""
    r = results[floor_name]
    per_fold = ", ".join("%+.1e" % c for c in r["ccc_by_fold"])
    return ("floor CCC per fold: %s\nfloor CCC pooled  : %+.4e\n"
            "A constant predictor scores exactly zero. The cross-validated "
            "floor is constant WITHIN a fold but changes between folds, so "
            "pooled it is only approximately zero. Write 'exactly zero' only "
            "about the per-fold number." % (per_fold, r["ccc_pooled"]))


def save_per_subject(path, results, groups=None):
    """Write one error per subject per model to an .npz."""
    payload = {"%s_mae" % name: r["mae_by_subject"]
               for name, r in results.items()}
    any_result = next(iter(results.values()))
    payload["subjects"] = any_result["subjects"]
    np.savez(path, **payload)
    return path

