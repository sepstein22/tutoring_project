# FACED Analysis Pipeline

A reproducible Python pipeline for preparing, validating, modelling and
reporting results from the **FACED EEG dataset**.

> This repository contains analysis code only. Do not commit the FACED
> dataset, participant data, generated features, trained models or large
> result artefacts.

## Project status

The classical-model path is the recommended starting point. The CNN path is
experimental until the electrode order, cohort-specific channel names and
reference-channel assumptions have been checked against the release metadata.

The pipeline stops rather than guesses when a dataset assumption cannot be
verified. Two places do this deliberately: the rating column indices, and the
electrode montage.

## Why this project exists

EEG pipelines produce plausible results even when the experiment is invalid.
Six failure modes are easy to miss and each one is addressed in the code:

1. **Subject leakage.** Trials from one participant must not appear in both
   training and test.
2. **Mixed recording units.** FACED records subjects in volts or microvolts.
   An unresolved difference is a constant offset larger than any real effect.
3. **Incorrect behavioural targets.** Valence and arousal column positions must
   be read from the behavioural workbook, never assumed.
4. **Misaligned subjects.** EEG, features, ratings and metadata are joined by a
   canonical subject id, not by position in independently sorted lists.
5. **Fold leakage.** Scaling, normalisation and hyper-parameter selection use
   training subjects only.
6. **Unverified electrode mappings.** A CNN runs happily on a scrambled
   spatial arrangement.

## Repository structure

```text
tutoring_project/
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── run_all.py
├── make_fake_faced.py
├── faced/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── protocol.py
│   └── report.py
└── tests/
    └── test_faced.py
```

### Main modules

- `faced/config.py` — dataset dimensions, frequency bands, unit table,
  filenames. Values that are inferred rather than documented say so.
- `faced/data.py` — file discovery, metadata parsing, ratings, electrode
  information, subject loading, unit harmonisation.
- `faced/features.py` — band power, differential entropy, feature reshaping,
  sensor positions, topomap grids.
- `faced/models.py` — floor, ridge, elastic net, random forest, SVR, CNN.
- `faced/protocol.py` — subject-wise folds, out-of-fold prediction, metrics,
  bootstrap intervals, permutation.
- `faced/report.py` — model execution, paired comparisons, tables, saved
  per-subject outputs.
- `run_all.py` — entry point: build the dataset, run the models, write results.
- `make_fake_faced.py` — synthetic FACED-shaped generator for development.

## Installation

```bash
git clone https://github.com/sepstein22/tutoring_project.git
cd tutoring_project
```

Create a virtual environment. Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Install. Only `numpy`, `scipy` and `scikit-learn` are required — everything
else is imported inside the function that needs it, so the core install
genuinely stands alone.

```bash
python -m pip install --upgrade pip
python -m pip install -e .            # core: floor, ridge, elastic net, forest, SVR
python -m pip install -e ".[all]"     # adds the CNN, topomaps, Excel, figures
python -m pip install -r requirements.txt   # the exact tested versions
```

| Extra | Unlocks |
|---|---|
| `[cnn]` | `models.CNN` (tensorflow) |
| `[topomap]` | `features.montage_positions` (mne) |
| `[excel]` | reading the rating layout and electrode tables (openpyxl) |
| `[figures]` | the plots in `run_all.py` (matplotlib) |
| `[test]` | the test suite (pytest) |

Tested on Python 3.11 with the versions pinned in `requirements.txt`. The
ranges in `pyproject.toml` are permissive and untested at their edges.

### Verify

```bash
python -m compileall faced run_all.py make_fake_faced.py
python -c "import faced; print(faced.__version__)"
python -m pytest tests -q
```

Do not proceed to real data until the package imports and the tests pass.

## Quick start with synthetic data

No download required. The generator writes files with FACED-compatible names,
shapes and directory structure, and plants an artificial relationship between
features and ratings so the pipeline can be exercised end to end.

> **Synthetic results are not estimates of performance on FACED.** They test
> software behaviour only. `GROUND_TRUTH.txt` records exactly what was planted.

```bash
python make_fake_faced.py ./fake6 --subjects 6 --features-only --mixed-units

python run_all.py \
  --root ./fake6 --out ./results/fake6 \
  --target valence --valence-col 9 --arousal-col 8 \
  --folds 3 --harmonise --no-cnn --fast
```

The rating-column values above belong to the synthetic generator. Do not
assume they are correct for the real release.

A larger integration run:

```bash
python make_fake_faced.py ./fake24 --subjects 24 --features-only --mixed-units

python run_all.py \
  --root ./fake24 --out ./results/fake24 \
  --target both --valence-col 9 --arousal-col 8 \
  --folds 3 --harmonise --no-cnn --fast
```

## Expected FACED layout

The loader tolerates several directory-name variants. A typical release:

```text
FACED/
├── Processed_data/
│   ├── sub000.pkl
│   └── ...
├── EEG_Features/
│   ├── DE/
│   │   ├── sub000.pkl
│   │   └── ...
│   └── PSD/
├── ratings/
│   ├── sub000_rating.mat
│   └── ...
├── Recording_info.csv
├── DataStructureOfBehaviouralData.xlsx
└── Electrode_Location.xlsx
```

An empty directory does not count as present, so a partial download cannot be
mistaken for a complete one.

## Required checks before real-data modelling

### 1. Diagnose recording units

```bash
python run_all.py --root /path/to/FACED --out ./results/units --diagnose-units
```

This prints the per-subject unit breakdown, predicts the offset an unharmonised
release would carry, measures whether that offset is actually present, and
writes `unit_diagnostic.txt`. Read the verdict before trusting any score.

Accepted spellings include `V`, `uV`, `µV` (U+00B5), `μV` (U+03BC) and
`microvolts`. Conversion happens in exactly one place —
`data.load_processed(path, source_unit=...)`. Do not also call
`harmonise_units` on the result.

### 2. Verify target columns

Valence and arousal are resolved from `DataStructureOfBehaviouralData.xlsx`, or
supplied on the command line only after manual verification. If neither is
available the run stops. `run_manifest.json` records which route was used under
`target.column_source`.

`data.load_ratings` rejects any table whose values fall outside the documented
0–7 scale.

### 3. Verify channel assumptions before enabling the CNN

The configuration assumes 30 scalp channels of the 32 rows, the remaining two
being references. **This is inferred from the standard reader's default, not
from the release documentation.** Check it against `Electrode_Location.xlsx`
before the CNN is used, and note that the two cohorts name six electrodes
differently at identical positions — one montage cannot serve both.

## Running the classical-model pipeline

```bash
# small real-data validation run
python run_all.py --root /path/to/FACED --out ./results/real_small \
  --subjects 12 --target valence --folds 3 --harmonise --no-cnn --fast

# full classical run
python run_all.py --root /path/to/FACED --out ./results/full \
  --subjects 123 --target both --folds 3 --harmonise --no-cnn

# recompute features from processed EEG instead of the shipped DE
python run_all.py --root /path/to/FACED --out ./results/recomputed \
  --target valence --folds 3 --harmonise --recompute --no-cnn
```

## Models

| Model | Notes |
|---|---|
| floor | predicts the training-fold mean, ignores the EEG |
| ridge | inner subject-wise selection of alpha |
| elastic net | combined L1 and L2, inner selection of alpha |
| random forest | fixed hyper-parameters |
| SVR | RBF kernel, inner subject-wise selection of C |
| CNN | per-band topomap grids; experimental until the montage is verified |

The floor is not optional. A model that does not improve on it has not
demonstrated transferable information under this protocol.

## Validation protocol

Outer splits are by subject; no participant appears on both sides; every sample
receives exactly one out-of-fold prediction, which is asserted rather than
assumed; all models see the same outer folds.

Hyper-parameters are selected on training subjects only, using a further
subject-wise split, and at least two training subjects are required. Any
learned transformation — standardisation, image normalisation, early stopping —
is fitted inside the fold.

## Feature definitions

Frequency bands, from the release:

```text
delta  1 to 4 Hz
theta  4 to 8 Hz
alpha  8 to 14 Hz
beta   14 to 30 Hz
gamma  30 to 47 Hz
```

Alpha and beta differ from the common textbook boundaries.

Welch's method returns a power spectral density, so band power is obtained by
integrating over frequency — the sum of bins multiplied by the bin width.

The shipped DE arrays contain one value per one-second window. Averaging them
is a modelling decision, not a dataset fact; the reducer is recorded in the
manifest.

Topomap inputs are interpolated into floating-point arrays rather than rendered
as separately autoscaled images, which preserves amplitude. The spatial mapping
is only valid if the channel order is correct.

## Metrics and inference

Subject-level MAE, RMSE on request, Lin's concordance correlation coefficient,
percentile bootstrap intervals over subjects, paired subject-level differences
against the floor, and descriptive win counts.

Lower MAE is better; a negative `model − floor` difference favours the model. A
bootstrap interval excluding zero means the paired interval does not cross zero
under this resampling procedure. Win counts are descriptive and do not replace
an interval.

A constant predictor scores exactly zero CCC. The cross-validated floor is
constant *within* a fold but changes between folds, so its pooled CCC is only
approximately zero — `report.floor_ccc_note()` prints both.

## Outputs

A complete run writes, into `--out`:

```text
results/
├── results_<target>.txt        table, win counts, floor CCC note
├── per_subject_<target>.npz    one error per subject per model
├── predictions_<target>.npz    y, out-of-fold predictions, subject, clip, fold
├── folds_<target>.csv          subject to outer fold assignment
├── run_manifest.json           arguments, versions, provenance, seeds
├── unit_diagnostic.txt         the unit report, as printed
└── errors_<target>.png         per-subject error distributions
```

`run_manifest.json` records the command line, package and Python versions, the
git commit, the seed, whether the data was synthetic, the feature source and
window reducer, the target columns and where they came from, the unit policy
and the diagnostic verdict, the protocol, and every model's MAE, CCC, interval
and selected hyper-parameters. Together with `predictions_<target>.npz` this is
enough to re-derive any table entry without refitting.

## Tests

```bash
python -m pytest tests -q
```

Coverage spans feature physics, unit handling, feature layout, topomap
amplitude, metric edge cases, split integrity, model behaviour, resampling
inference, and file loading. Several tests assert properties that must hold
regardless of the data, such as recovering the analytic power of a unit sine.

## Known limitations

1. The 30-channel selection is an assumption until verified from the electrode
   workbook.
2. Cohort-specific electrode naming affects topomap construction.
3. Rating-column order must be verified from release metadata.
4. Averaging DE windows is a modelling choice.
5. Synthetic performance does not estimate real FACED performance.
6. CNN reproducibility depends on hardware and framework settings.
7. Model families receive different amounts of hyper-parameter tuning, so the
   comparison between them is not perfectly like-for-like.
8. Pickle files should be loaded only from the trusted release or locally
   generated test data.

## Recommended sequence

1. Run compilation and the unit tests.
2. Run a six-subject synthetic smoke test.
3. Run a 24-subject synthetic integration test.
4. Audit real-data subject alignment.
5. Diagnose and resolve recording units.
6. Verify the behavioural target columns.
7. Run 9 to 15 real subjects with the classical models.
8. Inspect predictions, folds and warnings.
9. Run all 123 subjects with the classical models.
10. Verify electrode mappings before enabling the CNN.

## Responsible interpretation

A trained model is not evidence that the experiment is correct. Treat results
as trustworthy only when subjects are aligned across every source, units are
resolved and recorded, targets are verified, the split is subject-wise,
preprocessing stays inside the fold, every sample has one out-of-fold
prediction, the saved artefacts are sufficient to audit the reported values,
and unresolved channel assumptions are disclosed.

## Contributing

Explain the problem, include or update tests, show the commands used to
validate the change, avoid unrelated formatting churn, confirm no FACED data or
generated artefacts are included, and document any new scientific assumption.

Short branches with one purpose:

```text
fix/subject-alignment
fix/unit-normalization
feature/run-manifest
feature/cnn-topomap
```

Before committing:

```bash
git status
git diff --cached --name-status
find . -type f -size +25M -print
```

## Licence and data access

Released under the MIT licence; see `LICENSE`.

The FACED dataset has its own access and usage terms. Obtain it through the
authorised source and follow the dataset's licence, citation, privacy and
redistribution requirements. Project SynID `syn50614194`; see Chen et al. 2023,
*Scientific Data* 10:740.

## Acknowledgments

An educational and research-preparation project. Its purpose is careful
implementation, validation, version control and reproducible analysis using EEG
data.
