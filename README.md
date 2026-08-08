# FACED Analysis Pipeline

A reproducible Python pipeline for preparing, validating, modeling, and reporting results from the **FACED EEG dataset**.

The project supports:

- dataset discovery and validation;
- recording-unit diagnosis and harmonization;
- shipped differential-entropy features;
- feature recomputation from processed EEG;
- subject-wise nested cross-validation;
- classical regression models and an optional topomap CNN;
- subject-level bootstrap comparisons;
- synthetic FACED-shaped data for development and testing.

> **Important:** This repository contains analysis code only. Do not commit or redistribute the FACED dataset, participant data, generated feature files, trained models, or large result artifacts.

## Project status

The classical-model path is the recommended starting point. The CNN path should be considered experimental until the electrode order, cohort-specific channel names, and reference-channel assumptions have been verified against the release metadata.

The pipeline is designed to stop rather than guess when a dataset assumption cannot be verified.

## Why this project exists

EEG pipelines can produce plausible results even when the experiment is invalid. This project focuses on several failure modes that are easy to miss:

1. **Subject leakage:** Trials from one participant must not appear in both the training and test sets.
2. **Mixed recording units:** FACED metadata may describe subjects in volts or microvolts. An unresolved unit difference can dominate power-derived features.
3. **Incorrect behavioral targets:** Valence and arousal column positions must be verified from the behavioral-data workbook.
4. **Misaligned subjects:** EEG, features, ratings, and metadata must be joined by a canonical subject identifier, not by independent list positions.
5. **Fold leakage:** Scaling, feature normalization, and hyperparameter selection must use training subjects only.
6. **Unverified electrode mappings:** A CNN can run successfully while using an incorrect spatial channel arrangement.

## Repository structure

```text
Research_Prep_2026/
├── README.md
├── pyproject.toml
├── requirements.txt
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
├── tests/
│   └── test_faced.py
├── notebooks/
│   └── CNNproject_legacy.ipynb
└── docs/
    ├── IMPLEMENTATION_PLAN.md
    └── DATA_ASSUMPTIONS.md
```

### Main modules

- `faced/config.py`: dataset dimensions, frequency bands, unit conversions, filenames, and documented assumptions.
- `faced/data.py`: file discovery, metadata parsing, ratings, electrode information, subject loading, and unit harmonization.
- `faced/features.py`: band power, differential-entropy features, feature reshaping, sensor positions, and topomap grids.
- `faced/models.py`: floor baseline, ridge, elastic net, random forest, SVR, and CNN wrappers.
- `faced/protocol.py`: subject-wise folds, out-of-fold prediction, metrics, bootstrap intervals, and permutation procedures.
- `faced/report.py`: model execution, paired comparisons, result tables, and saved subject-level outputs.
- `run_all.py`: command-line entry point for building the dataset, running models, and writing results.
- `make_fake_faced.py`: synthetic FACED-shaped dataset generator for tests and development.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/takashihnaito1-cyber/Research_Prep_2026.git
cd Research_Prep_2026
```

### 2. Create a virtual environment

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the project is configured as an installable package:

```bash
python -m pip install -e .
```

### 4. Verify the installation

```bash
python -m compileall faced run_all.py make_fake_faced.py
python -c "import faced; print(faced.__version__)"
python -m pytest tests -q
```

Do not proceed to real-data analysis until the package imports successfully and the relevant tests pass.

## Quick start with synthetic data

No FACED download is required for this step.

The synthetic generator creates files with FACED-compatible names, shapes, and directory structure. It also plants an artificial relationship between EEG features and ratings so the pipeline can be exercised end to end.

> **Synthetic results are not estimates of performance on FACED.** They test software behavior only.

### Small feature-only smoke test

```bash
python make_fake_faced.py ./fake6 \
  --subjects 6 \
  --features-only \
  --mixed-units
```

Then run the classical models:

```bash
python run_all.py \
  --root ./fake6 \
  --out ./results/fake6 \
  --target valence \
  --valence-col 9 \
  --arousal-col 8 \
  --folds 3 \
  --harmonise \
  --no-cnn \
  --fast
```

The explicit rating-column values above belong to the synthetic generator. Do not assume that they are correct for the real FACED release.

### Larger synthetic integration test

```bash
python make_fake_faced.py ./fake24 \
  --subjects 24 \
  --features-only \
  --mixed-units

python run_all.py \
  --root ./fake24 \
  --out ./results/fake24 \
  --target both \
  --valence-col 9 \
  --arousal-col 8 \
  --folds 3 \
  --harmonise \
  --no-cnn \
  --fast
```

## Expected FACED layout

The loader tolerates several known directory-name variants, but a typical release layout resembles:

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

Do not rename release files unless the loader and tests are updated together.

## Required checks before real-data modeling

### 1. Verify subject alignment

Every source must be associated by canonical subject ID:

- recording metadata;
- behavioral ratings;
- shipped features;
- processed EEG, when used;
- cohort and electrode metadata.

The dataset audit should report:

```text
Subjects in metadata: 123
Subjects with ratings: 123
Subjects with DE features: 123
Duplicate IDs: 0
Missing IDs: 0
Clips per subject: 28
Alignment status: PASS
```

If subject sets differ, stop and resolve the discrepancy. Do not pair independently sorted file lists with `zip()`.

### 2. Diagnose recording units

Run the unit diagnostic before trusting any model result:

```bash
python run_all.py \
  --root /path/to/FACED \
  --out ./results/unit_diagnostic \
  --diagnose-units
```

The pipeline should record:

- each subject's source unit;
- the accepted spelling or normalized representation;
- whether a unit-related feature offset is present;
- whether conversion was applied;
- the final canonical unit.

Supported unit forms should include common representations of volts and microvolts, including `V`, `uV`, `µV`, and `μV`.

Unit conversion must occur exactly once.

### 3. Verify target columns

Valence and arousal must be resolved from `DataStructureOfBehaviouralData.xlsx`, or supplied explicitly only after manual verification.

The program should print and save information such as:

```text
Target: valence
Workbook label: Valence
Python column index: 9
Observed range: 0.0 to 7.0
Finite values: yes
Nonzero variance: yes
Status: PASS
```

Required validations:

- the index is within the 12-item rating structure;
- valence and arousal are different columns;
- all values are finite;
- all values lie between 0 and 7;
- each included subject has 28 target values;
- the selected target has nonzero variance.

### 4. Verify channel assumptions before CNN use

The current configuration assumes 30 scalp EEG channels from 32 total rows, with the remaining rows treated as references. This assumption must be checked against `Electrode_Location.xlsx`.

Before enabling the CNN:

- verify channel order;
- verify which channels are references;
- handle cohort-specific electrode names;
- reject unknown or duplicate channel names;
- save the final channel-to-position mapping;
- inspect diagnostic sensor plots for each cohort;
- verify that image preprocessing is trained on outer-training subjects only.

## Running the classical-model pipeline

The recommended first real-data experiment uses shipped DE features and excludes the CNN.

### Small real-data validation run

```bash
python run_all.py \
  --root /path/to/FACED \
  --out ./results/real_small \
  --subjects 12 \
  --target valence \
  --folds 3 \
  --harmonise \
  --no-cnn \
  --fast
```

Inspect all warnings and outputs before increasing the subject count.

### Full classical run

```bash
python run_all.py \
  --root /path/to/FACED \
  --out ./results/full_classical \
  --subjects 123 \
  --target both \
  --folds 3 \
  --harmonise \
  --no-cnn
```

### Recompute features from processed EEG

Use this only after unit handling and processed-data channel assumptions have been verified:

```bash
python run_all.py \
  --root /path/to/FACED \
  --out ./results/recomputed \
  --target valence \
  --folds 3 \
  --harmonise \
  --recompute \
  --no-cnn
```

## Models

The default model set includes:

1. **Floor baseline:** Predicts the training-fold mean and ignores EEG.
2. **Ridge regression:** Linear model with inner subject-wise selection of the regularization value.
3. **Elastic net:** Linear model with combined L1 and L2 regularization.
4. **Random forest:** Prespecified nonlinear ensemble model.
5. **SVR:** Radial-basis-function support vector regression with inner subject-wise selection of `C`.
6. **CNN:** Convolutional model operating on per-band topomap grids. Experimental until electrode mapping is verified.

The floor is essential. A model that does not improve on the training-mean baseline has not demonstrated transferable information from EEG under the selected protocol.

## Validation protocol

### Outer validation

- Splits are made by subject.
- A participant cannot appear in both training and test data.
- Every sample receives exactly one out-of-fold prediction.
- All models use the same outer folds.

### Inner validation

- Hyperparameters are selected using training subjects only.
- Inner splits are also grouped by subject.
- At least two unique training subjects are required for hyperparameter selection.
- Inner-selection scores should be averaged over validation subjects.

### Fold-local preprocessing

Any learned transformation must be fitted inside the fold, including:

- feature standardization;
- image normalization;
- learned imputation;
- hyperparameter selection;
- CNN early stopping or validation.

No outer-test subject may influence these operations.

## Feature definitions

### Frequency bands

The release-specific boundaries are:

```text
delta: 1 to 4 Hz
theta: 4 to 8 Hz
alpha: 8 to 14 Hz
beta: 14 to 30 Hz
gamma: 30 to 47 Hz
```

These differ from some textbook conventions, especially for alpha and beta.

### Band power

Welch's method returns power spectral density. Band power is obtained by integrating over frequency, including multiplication by the frequency-bin width.

### Differential-entropy feature convention

The implementation uses a log-band-power or release-compatible differential-entropy convention. Any one-half scaling, additive constants, and epsilon floor should be documented in the run configuration.

### Window reduction

The shipped DE feature arrays contain multiple temporal windows. Averaging those windows is a modeling decision, not a dataset fact. Every result should record which reducer was used.

### Topomap grids

Topomap inputs are interpolated directly into floating-point arrays rather than rendered as separately autoscaled image files. This preserves feature amplitude.

The spatial mapping is valid only if channel names, channel order, and sensor positions are correct.

## Metrics and inference

The pipeline reports:

- subject-level mean absolute error;
- root mean squared error where requested;
- Lin's concordance correlation coefficient;
- percentile bootstrap intervals over subjects;
- paired subject-level differences against the floor;
- descriptive per-subject win counts.

### Interpretation

- Lower MAE is better.
- For `model MAE - floor MAE`, a negative difference favors the model.
- A bootstrap interval excluding zero indicates that the paired interval does not cross zero under the implemented resampling procedure.
- Win counts are descriptive and are not a substitute for an uncertainty interval.
- Pooled CCC and fold-level CCC answer slightly different questions for a fold-specific constant baseline.

## Expected outputs

A complete run should save enough information to reproduce every table entry without retraining:

```text
results/
├── results_valence.txt
├── per_subject_valence.npz
├── predictions_valence.npz
├── folds_valence.csv
├── run_manifest.json
├── errors_valence.png
└── unit_diagnostic.txt
```

Recommended saved fields include:

- target values;
- out-of-fold predictions for every model;
- sample-level subject IDs;
- fold assignments;
- per-subject errors;
- selected hyperparameters by outer fold;
- target-column provenance;
- unit policy;
- feature source and reducer;
- command-line arguments;
- random seeds;
- Python and package versions;
- synthetic or real-data status.

## Tests

Run the full test suite with:

```bash
python -m pytest tests -q
```

High-priority tests should cover:

- canonical subject-ID parsing;
- duplicate and missing subjects;
- unit aliases and conversion;
- rating-column validation;
- feature shapes and finite values;
- stack and unstack round trips;
- invalid feature-order strings;
- subject leakage;
- infeasible fold counts;
- metric shape mismatches;
- exactly one prediction per sample;
- CCC constant-input cases;
- repeated-run reproducibility;
- duplicate model names;
- empty result handling;
- paired model subject alignment;
- ambiguous metadata-file discovery;
- topomap amplitude preservation.

## Development workflow

Use short branches with one purpose:

```text
fix/subject-alignment
fix/unit-normalization
fix/protocol-validation
feature/run-manifest
feature/cnn-topomap
```

Before committing:

```bash
git status
git diff --cached --name-status
find . -type f -size +25M -print
```

Suggested commit messages:

```text
Add canonical subject ID parser
Validate subject coverage across data sources
Normalize recording-unit labels
Reject metric shape mismatches
Persist folds and out-of-fold predictions
Verify cohort-specific electrode mappings
```

Avoid committing directly identifiable data, release archives, local paths, virtual environments, synthetic datasets, or generated model outputs.

## Known limitations

1. The 30-channel EEG selection remains an assumption until verified from the electrode workbook.
2. Cohort-specific electrode naming may affect topomap construction.
3. Rating-column order must be verified from release metadata.
4. Averaging DE windows is a modeling choice.
5. Synthetic performance does not estimate real FACED performance.
6. CNN reproducibility may depend on hardware and deep-learning framework settings.
7. Model families currently receive different levels of hyperparameter tuning.
8. Pickle files should be loaded only from the trusted FACED release or locally generated test data.

## Recommended analysis sequence

1. Run compilation and unit tests.
2. Run a six-subject synthetic smoke test.
3. Run a 24-subject synthetic integration test.
4. audit real-data subject alignment;
5. diagnose and resolve recording units;
6. verify behavioral target columns;
7. run 9 to 15 real subjects with classical models;
8. inspect predictions, folds, and warnings;
9. run all 123 subjects with classical models;
10. verify electrode mappings before enabling the CNN.

## Responsible interpretation

A trained model is not evidence that the experiment is correct. Results should be considered trustworthy only when:

- subjects are correctly aligned across all data sources;
- units are resolved and recorded;
- targets are verified;
- the validation split is subject-wise;
- preprocessing remains inside each fold;
- every sample receives one out-of-fold prediction;
- saved artifacts are sufficient to audit the reported values;
- unresolved channel assumptions are disclosed.

## Contributing

When opening a pull request:

1. explain the problem being solved;
2. include or update tests;
3. show the commands used to validate the change;
4. avoid unrelated formatting changes;
5. confirm that no FACED data or generated artifacts are included;
6. document any new scientific assumption.

## License and data access

Add the repository's software license here.

The FACED dataset has its own access and usage terms. Users are responsible for obtaining the data through the authorized source and following the dataset's license, citation, privacy, and redistribution requirements.

## Acknowledgments

This repository is an educational and research-preparation project. Its primary goal is to teach careful implementation, validation, version control, and reproducible analysis practices using EEG data.

