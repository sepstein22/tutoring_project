#!/usr/bin/env python3
"""
Write a FACED-shaped dataset to disk so the pipeline runs with nothing downloaded.

    python3 make_fake_faced.py OUTDIR [--subjects 123] [--features-only] [--seed 0]

WHAT THIS IS
    A structural stand-in. Every file has the exact name, format, shape, dtype
    and value range the real FACED release uses, so code written against it runs
    unchanged against the real thing.

WHAT THIS IS NOT
    Real EEG. There is a planted relationship between band power and the
    ratings, so models will beat the floor -- that is by construction, so that
    the plumbing can be exercised. Any MAE, CCC or effect size you measure here
    is a property of this generator and says nothing whatever about FACED.

    Use it to check that code runs, that a split does not leak, that a floor is
    computed, that an array has the shape you expected. Never to estimate how
    well a model will do.

LAYOUT PRODUCED
    OUTDIR/Processed_data/sub000.pkl ...     (28, 32, 7500) float32
    OUTDIR/EEG_Features/DE/sub000.pkl ...    (28, 32, 30, 5) float64
    OUTDIR/ratings/sub000_rating.mat ...     variable 'rating', (28, 12) on 0-7
    OUTDIR/GROUND_TRUTH.txt                  the planted structure, for checking

SIZE
    Raw EEG is about 27 MB per subject at float32, so 123 subjects is roughly
    3.2 GB. --features-only skips the raw EEG and writes about 1 MB per
    subject instead, which is all the ridge / statistics path ever needs.
"""
import argparse
import os
import pickle

import numpy as np
from scipy.io import savemat

N_CLIP, N_CHAN, SFREQ, SECONDS = 28, 32, 250, 30
N_SAMPLE = SFREQ * SECONDS
N_EEG = 30                       # first 30 scalp, last 2 mastoid
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 14),
         "beta": (14, 30), "gamma": (30, 47)}
N_BAND = len(BANDS)

# 9 categories, 3 clips each except neutral with 4 -> 28
CATEGORIES = ["Anger", "Disgust", "Fear", "Sadness", "Neutral",
              "Amusement", "Inspiration", "Joy", "Tenderness"]
CLIP_CATEGORY = [c for c in CATEGORIES for _ in range(4 if c == "Neutral" else 3)]
assert len(CLIP_CATEGORY) == N_CLIP

# Rating item order. The REAL column order is undocumented in the paper HTML --
# this is a plausible guess so the probe workflow has something to find. Do not
# carry these indices over to the real files without checking.
ITEMS = CATEGORIES[:8] + ["arousal", "valence", "familiarity", "liking"]
VALENCE_COL, AROUSAL_COL = ITEMS.index("valence"), ITEMS.index("arousal")


def pink_noise(rng, n_channel, n_sample, sfreq):
    """1/f background, so band powers fall off with frequency like real EEG."""
    spectrum = np.fft.rfftfreq(n_sample, 1.0 / sfreq)
    spectrum[0] = spectrum[1]
    shape = 1.0 / spectrum
    phase = rng.uniform(0, 2 * np.pi, (n_channel, len(spectrum)))
    amp = shape[None, :] * rng.rayleigh(1.0, (n_channel, len(spectrum)))
    return np.fft.irfft(amp * np.exp(1j * phase), n=n_sample, axis=-1)


def band_power_of(trial, sfreq=SFREQ):
    from scipy.signal import welch
    f, psd = welch(trial, fs=sfreq, nperseg=2 * sfreq, axis=-1)
    df = f[1] - f[0]
    return np.stack([psd[:, (f >= lo) & (f < hi)].sum(-1) * df
                     for lo, hi in BANDS.values()], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--subjects", type=int, default=123)
    ap.add_argument("--features-only", action="store_true",
                    help="skip the raw EEG; ~50x smaller, enough for everything "
                         "except the topomap CNN")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mixed-units", action="store_true",
                    help="store a third of subjects in microvolts, as the real "
                         "release does, and write Recording_info.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = args.outdir
    wanted = ["EEG_Features/DE", "ratings"]
    if not args.features_only:
        wanted.insert(0, "Processed_data")     # never leave it empty
    for sub in wanted:
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    # ---- the planted structure -------------------------------------------
    # Clips differ in how much alpha they evoke, over a set of frontal channels.
    # Ratings depend on that, plus a per-person offset, plus noise. So a model
    # CAN win, subjects DO differ, and a subject-wise split matters.
    alpha_channels = rng.choice(N_EEG, 8, replace=False)
    clip_drive = rng.uniform(0.3, 3.0, N_CLIP)
    subject_offset = rng.normal(0.0, 0.45, args.subjects)
    subject_gain = rng.uniform(0.9, 1.15, args.subjects)

    units = []
    t = np.arange(N_SAMPLE) / SFREQ
    alpha_wave = np.sin(2 * np.pi * 10.5 * t)

    for s in range(args.subjects):
        trials = np.empty((N_CLIP, N_CHAN, N_SAMPLE), dtype=np.float32)
        for c in range(N_CLIP):
            x = pink_noise(rng, N_CHAN, N_SAMPLE, SFREQ) * 8e-6
            gain = clip_drive[c] * subject_gain[s] * 6e-6
            x[alpha_channels] += gain * alpha_wave[None, :]
            x[N_EEG:] *= 0.35                      # mastoids: quieter
            trials[c] = x.astype(np.float32)

        unit = "uV" if (args.mixed_units and s % 3 == 1) else "V"
        if unit == "uV":
            trials = (trials.astype(np.float64) * 1e6).astype(np.float32)
        units.append((s, unit))

        if not args.features_only:
            with open(os.path.join(out, "Processed_data", "sub%03d.pkl" % s), "wb") as f:
                pickle.dump(trials, f, protocol=4)

        # DE features, one value per 1-second window, matching the release shape
        de = np.empty((N_CLIP, N_CHAN, SECONDS, N_BAND))
        for c in range(N_CLIP):
            for w in range(SECONDS):
                seg = trials[c][:, w * SFREQ:(w + 1) * SFREQ].astype(np.float64)
                de[c, :, w, :] = 0.5 * np.log(band_power_of(seg, SFREQ) + 1e-30)
        with open(os.path.join(out, "EEG_Features", "DE", "sub%03d.pkl" % s), "wb") as f:
            pickle.dump(de, f, protocol=4)

        # ratings: 12 items on a continuous 0-7 scale
        table = rng.uniform(0.5, 6.5, (N_CLIP, len(ITEMS)))
        table[:, VALENCE_COL] = np.clip(
            1.10 * clip_drive + subject_offset[s] + rng.normal(0, 0.40, N_CLIP) + 1.8, 0, 7)
        table[:, AROUSAL_COL] = np.clip(
            0.80 * clip_drive + 0.5 * subject_offset[s] + rng.normal(0, 0.75, N_CLIP) + 2.6, 0, 7)
        savemat(os.path.join(out, "ratings", "sub%03d_rating.mat" % s),
                {"rating": table, "items": np.array(ITEMS, dtype=object)})

        if (s + 1) % 10 == 0 or s + 1 == args.subjects:
            print("  %3d / %d subjects" % (s + 1, args.subjects))

    with open(os.path.join(out, "Recording_info.csv"), "w") as f:
        f.write("sub,Gender,Age,Cohort ,Sample_rate,Unit\n")
        for i, u in units:
            f.write("sub%03d,F,22,%d,250,%s\n" % (i, 1 if u == "uV" else 2, u))

    with open(os.path.join(out, "GROUND_TRUTH.txt"), "w") as f:
        f.write(
            "SYNTHETIC. Shapes and formats match FACED; the numbers do not.\n\n"
            "subjects            %d\n"
            "valence column      %d   arousal column %d   (variable 'rating')\n"
            "item order          %s\n"
            "alpha channels      %s\n"
            "clip drive          %s\n\n"
            "Planted relationship:\n"
            "  valence = 1.10*clip_drive + subject_offset + N(0,0.40) + 1.8, clipped 0-7\n"
            "  arousal = 0.80*clip_drive + 0.5*subject_offset + N(0,0.75) + 2.6, clipped 0-7\n"
            "  clip_drive shows up as 10.5 Hz power on the alpha channels above,\n"
            "  scaled per subject by a gain in [0.90, 1.15].\n\n"
            "So a model SHOULD beat the floor here, by construction. That tells\n"
            "you the code works. It tells you nothing about FACED.\n"
            % (args.subjects, VALENCE_COL, AROUSAL_COL, ITEMS,
               sorted(alpha_channels.tolist()), np.round(clip_drive, 3).tolist()))

    print("\nwrote %s" % out)
    print("valence column %d, arousal column %d, variable name 'rating'"
          % (VALENCE_COL, AROUSAL_COL))
    print("GROUND_TRUTH.txt records the planted structure -- read it before")
    print("believing any number this dataset produces.")


if __name__ == "__main__":
    main()

