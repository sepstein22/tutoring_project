#!/usr/bin/env python3
"""
Build the FACED dataset, run every model under one protocol, report the result.

    python3 run_all.py --root ~/Downloads/FACED --target valence
    python3 run_all.py --root ./fake --fast --no-cnn
    python3 run_all.py --root ~/Downloads/FACED --diagnose-units

It refuses to guess. If the rating columns are unknown it stops and tells you
how to find them, rather than producing a plausible-looking wrong answer.

Outputs, into --out:
    results_<target>.txt      the table, win counts and diagnostics
    per_subject_<target>.npz  one error per subject per model
    errors_<target>.png       per-subject error distributions
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from faced import config as C          # noqa: E402
from faced import data, features, models, protocol, report   # noqa: E402


def rule(title):
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


# --------------------------------------------------------------- diagnostics

def diagnose_units(root, found, n_probe=12):
    """Is the volt/microvolt difference actually present in the released data?

    Recording_info.csv says 33 of 123 subjects are stored in microvolts. If that
    survives preprocessing, their log band powers sit log(1e12) = 27.63 above
    everyone else's (13.82 with the 0.5 factor). This measures it instead of
    assuming either way.
    """
    rule("UNIT DIAGNOSTIC")
    if not found["recording_info"]:
        print("Recording_info.csv not found; cannot check.")
        return None

    info = data.load_recording_info(found["recording_info"])
    summary = data.unit_summary(info)
    print("subjects        : %d" % summary["n_subject"])
    print("unit            : %s" % summary["unit"])
    print("cohort          : %s" % summary["cohort"])
    print("sample rate     : %s" % summary["sample_rate"])
    print("combinations    :")
    for k, v in summary["combinations"].items():
        print("    %-34s %3d" % (k, v))

    non_canonical = summary["non_canonical"]
    if not non_canonical:
        print("\nAll subjects share one unit. Nothing to harmonise.")
        return info
    print("\n%d subject(s) are not in %s. Predicted offset on log band power "
          "if unharmonised: %+.4f (%.4f with the 0.5 factor)."
          % (len(non_canonical), C.CANONICAL_UNIT,
             data.log_power_offset(info[non_canonical[0]].unit),
             data.log_power_offset(info[non_canonical[0]].unit, half=True)))

    directory = found["processed"] or found["features_de"]
    if not directory:
        print("No EEG on disk yet, so this stays a prediction.")
        return info

    print("\nMeasuring it on %d subjects ..." % n_probe)
    use_features = found["processed"] is None
    paths = data.subject_files(directory)
    canonical, other = [], []
    for path in paths:
        index = int("".join(c for c in os.path.basename(path) if c.isdigit()))
        if index not in info:
            continue
        bucket = other if index in non_canonical else canonical
        if len(bucket) >= n_probe // 2:
            continue
        if use_features:
            de = data.collapse_windows(data.load_feature_file(path))
            bucket.append(float(np.mean(de)))
        else:
            trials = data.load_processed(path)   # NOT harmonised, on purpose
            bucket.append(float(np.mean(
                features.differential_entropy(trials[0]))))
        if len(canonical) >= n_probe // 2 and len(other) >= n_probe // 2:
            break

    if not canonical or not other:
        print("Could not sample both groups.")
        return info

    gap = float(np.mean(other) - np.mean(canonical))
    expected = data.log_power_offset(info[non_canonical[0]].unit, half=True)
    print("  mean DE, %-3s subjects : %+9.4f  (n=%d)"
          % (C.CANONICAL_UNIT, np.mean(canonical), len(canonical)))
    print("  mean DE, other unit   : %+9.4f  (n=%d)" % (np.mean(other), len(other)))
    print("  observed gap          : %+9.4f" % gap)
    print("  gap if unharmonised   : %+9.4f  (or %+.4f without the 0.5 factor)"
          % (expected, 2 * expected))

    if abs(gap - expected) < 0.15 * abs(expected):
        print("\nVERDICT: the unit difference SURVIVED preprocessing. Harmonise, "
              "or a quarter of your subjects are offset by more than any real "
              "effect in the dataset. Pass --harmonise.")
    elif abs(gap) < 0.25 * abs(expected):
        print("\nVERDICT: no unit offset detectable. The release appears already "
              "harmonised; --harmonise would then INTRODUCE the error.")
    else:
        print("\nVERDICT: unclear -- the gap matches neither prediction. Do not "
              "proceed on assumption; look at a subject from each group directly.")
    return info


# ---------------------------------------------------------------- assembling

def build_dataset(args, found, info):
    rule("BUILDING THE DATASET")

    if found["features_de"] and not args.recompute:
        source, paths = "shipped DE features", data.subject_files(found["features_de"])
    elif found["processed"]:
        source, paths = "recomputed from processed EEG", data.subject_files(found["processed"])
    else:
        sys.exit("Found neither EEG_Features/DE nor Processed_data under %s" % args.root)
    paths = paths[: args.subjects]
    print("source     : %s" % source)
    print("subjects   : %d" % len(paths))
    print("channels   : first %d of %d" % (C.N_EEG, C.N_CHANNEL_TOTAL))
    print("bands      : %s" % dict(C.BANDS))

    rating_paths = sorted(
        p for p in __import__("glob").glob(
            os.path.join(args.root, "**", "*.mat"), recursive=True))
    if not rating_paths:
        sys.exit("No ratings .mat found under %s -- the targets live there."
                 % args.root)

    # Join on the parsed subject id, never on position in two sorted lists.
    # Two lists that both look sorted can still disagree the moment one source
    # is missing a subject, and every label after that point is attached to the
    # wrong person -- which does not crash and does not look wrong.
    try:
        feature_by_subject = data.index_by_subject(paths, "feature")
        rating_by_subject = data.index_by_subject(rating_paths, "rating")
    except ValueError as exc:
        sys.exit(str(exc))

    missing = sorted(set(feature_by_subject) - set(rating_by_subject))
    if missing:
        sys.exit("no ratings file for subject(s): %s" % missing[:10])
    extra = sorted(set(rating_by_subject) - set(feature_by_subject))
    if extra:
        print("note: %d subject(s) have ratings but no EEG in this run: %s"
              % (len(extra), extra[:10]))
    subject_order = sorted(feature_by_subject)
    print("joined %d subjects on parsed id" % len(subject_order))
    if args.valence_col is None or args.arousal_col is None:
        layout = (data.load_rating_layout(found["behaviour_structure"])
                  if found["behaviour_structure"] else None)
        if layout and "valence" in layout and "arousal" in layout:
            args.valence_col = layout["valence"]
            args.arousal_col = layout["arousal"]
            print("rating cols: valence=%d arousal=%d (from %s)"
                  % (args.valence_col, args.arousal_col,
                     os.path.basename(found["behaviour_structure"])))
        else:
            sys.exit(
                "Rating column indices unknown and DataStructureOfBehaviouralData.xlsx\n"
                "did not yield them. Open a ratings .mat, identify which of the 12\n"
                "columns are valence and arousal, and pass --valence-col/--arousal-col.\n"
                "Guessing here produces a result that looks fine and is wrong.")

    rows, targets, groups, clips = [], [], [], []
    started = time.time()
    for order, index in enumerate(subject_order):
        path = feature_by_subject[index]
        unit = info[index].unit if (info and index in info and args.harmonise) else None

        if source.startswith("shipped"):
            de = data.collapse_windows(data.load_feature_file(path))   # (clips, ch, band)
            if unit is not None:
                de = de - data.log_power_offset(unit, half=True)
            block = np.stack([features.stack(de[c]) for c in range(de.shape[0])])
        else:
            trials = data.load_processed(path, source_unit=unit)
            block = features.subject_matrix(trials)

        y_sub = data.load_ratings(rating_by_subject[index],
                                  args.valence_col, args.arousal_col)

        rows.append(block)
        targets.append(y_sub)
        groups.append(np.full(C.N_CLIP, index))
        clips.append(np.arange(C.N_CLIP))
        if (order + 1) % 20 == 0:
            print("   %3d / %d  (%.1fs)"
                  % (order + 1, len(subject_order), time.time() - started))

    X = np.vstack(rows)
    Y = np.vstack(targets)
    G = np.concatenate(groups)
    K = np.concatenate(clips)
    print("\nX %s   Y %s   subjects %d   clips %d"
          % (X.shape, Y.shape, len(np.unique(G)), len(np.unique(K))))
    print("target range [%.2f, %.2f]   distinct target pairs %d"
          % (Y.min(), Y.max(), len({tuple(r) for r in Y})))
    if len({tuple(r) for r in Y}) <= C.N_CLIP:
        print("WARNING: at most one distinct target per clip. If that is because "
              "the labels come from a per-category lookup table rather than "
              "per-subject ratings, this is classification, not regression.")
    return X, Y, G, K


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="./results")
    ap.add_argument("--target", default="valence", choices=["valence", "arousal", "both"])
    ap.add_argument("--subjects", type=int, default=C.N_SUBJECT)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--valence-col", type=int, default=None)
    ap.add_argument("--arousal-col", type=int, default=None)
    ap.add_argument("--harmonise", action="store_true",
                    help="convert every subject to volts (run --diagnose-units first)")
    ap.add_argument("--recompute", action="store_true",
                    help="derive features from Processed_data instead of the shipped DE")
    ap.add_argument("--no-cnn", action="store_true")
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--diagnose-units", action="store_true")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    args.root = root
    os.makedirs(args.out, exist_ok=True)

    rule("WHAT IS ON DISK")
    found = data.locate(root)
    for key, value in found.items():
        print("  %-20s %s" % (key, value or "-- not found --"))

    info = diagnose_units(root, found)
    if args.diagnose_units:
        return

    X, Y, G, K = build_dataset(args, found, info)

    columns = {"valence": [0], "arousal": [1], "both": [0, 1]}[args.target]
    y = Y[:, columns] if len(columns) > 1 else Y[:, columns[0]]

    images = None
    if not args.no_cnn:
        rule("TOPOMAP GRIDS")
        try:
            names = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2",
                     "FC5", "FC6", "Cz", "C3", "C4", "T7", "T8", "CP1", "CP2",
                     "CP5", "CP6", "Pz", "P3", "P4", "P7", "P8", "PO3", "PO4",
                     "Oz", "O1", "O2"][:C.N_EEG]
            positions = features.montage_positions(names)
            print("PLACEHOLDER montage -- replace with Electrode_Location.xlsx.")
            print("A wrong channel order means the CNN learns a scrambled map.")
            images = features.image_stack(X, positions, C.N_EEG, args.image_size)
            print("images %s" % (images.shape,))
        except Exception as exc:
            print("could not build images (%s: %s); skipping the CNN"
                  % (type(exc).__name__, exc))

    rule("RUNNING THE MODELS")
    zoo = models.zoo(include_cnn=images is not None, fast=args.fast)
    results = report.run_zoo(zoo, X, y, G, images=images,
                             n_splits=args.folds, seed=args.seed)

    rule("RESULTS -- target: %s" % args.target)
    versus = report.compare_to_floor(results, seed=args.seed)
    text = report.table(results, versus)
    counts = report.win_counts(results)
    note = report.floor_ccc_note(results)
    print(text + "\n\n" + counts + "\n\n" + note)

    print("\nselected hyper-parameters per fold")
    for name, r in results.items():
        picks = [i.get("hyperparameter") for i in r["fold_info"]]
        if any(p is not None for p in picks):
            print("  %-14s %s" % (name, picks))
        warn = [i for i in r["fold_info"] if i.get("warning")]
        if warn:
            print("      WARNING: %s" % warn[0]["warning"])

    stem = os.path.join(args.out, "%s_%s" % ("%s", args.target))
    with open(stem % "results" + ".txt", "w") as handle:
        handle.write(text + "\n\n" + counts + "\n\n" + note + "\n")
    report.save_per_subject(stem % "per_subject" + ".npz", results)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names_ = sorted(results, key=lambda n: results[n]["mae"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot([results[n]["mae_by_subject"] for n in names_], labels=names_,
                   showmeans=True)
        ax.set_ylabel("per-subject MAE")
        ax.set_title("FACED %s -- per-subject error, subject-wise CV" % args.target)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(stem % "errors" + ".png", dpi=150)
        print("\nwrote %s.png" % (stem % "errors"))
    except Exception as exc:
        print("figure skipped: %s" % exc)

    print("wrote %s.txt and %s.npz" % (stem % "results", stem % "per_subject"))


if __name__ == "__main__":
    main()

