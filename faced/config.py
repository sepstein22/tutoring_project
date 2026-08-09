"""Dataset constants for FACED.

Sourced from Readme.md and Dataset_description.md in the release, and from
Chen et al. 2023, Sci Data 10:740. Values that are inferred rather than
documented say so.
"""
from __future__ import annotations

# ---------------------------------------------------------------- structure

N_SUBJECT = 123          # indexed S000..S122
N_CLIP = 28              # 9 categories, 3 clips each, 4 for neutral
N_CHANNEL_TOTAL = 32
SFREQ = 250              # sampling rate of the PRE-PROCESSED data
TRIAL_SECONDS = 30
N_SAMPLE = SFREQ * TRIAL_SECONDS      # 7500

# INFERRED from the standard reader's num_channel=30 default, not documented:
# the scalp channels come first and the two mastoids last. Verify against
# Electrode_Location.xlsx.
N_EEG = 30

# ---------------------------------------------------------------- frequency

# From the release Readme. Note alpha and beta are not the textbook
# 8-13 / 13-30 boundaries.
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 14.0),
    "beta": (14.0, 30.0),
    "gamma": (30.0, 47.0),
}
BAND_NAMES = tuple(BANDS)
N_BAND = len(BANDS)

BANDS_TEXTBOOK = {
    "delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 47.0),
}

# ------------------------------------------------------------------ emotion

CATEGORIES = ("anger", "fear", "disgust", "sadness", "neutral",
              "amusement", "inspiration", "joy", "tenderness")

# 12 items on a continuous 0-7 scale. Column order is defined by
# DataStructureOfBehaviouralData.xlsx and read by data.load_rating_layout();
# do not hard-code indices here.
N_RATING_ITEM = 12
RATING_MIN, RATING_MAX = 0.0, 7.0

# --------------------------------------------------------------------- units

# Recording_info.csv gives a per-subject unit: 90 subjects in volts, 33 in
# microvolts, all 33 in cohort 1. Spelling is handled by data.normalize_unit()
# rather than by enumerating variants here.
CANONICAL_UNIT = "V"

UNIT_SCALE = {"V": 1.0, "uV": 1e-6}
UNIT_ALIASES = {
    "v": "V", "volt": "V", "volts": "V",
    "uv": "uV", "microvolt": "uV", "microvolts": "uV", "micro volt": "uV",
}

# ------------------------------------------------------------------- layout

FEATURE_ORDER = "band-major"      # column = band_index * n_channels + channel

# --------------------------------------------------------------- file names

FILES = {
    "recording_info": "Recording_info.csv",
    "electrodes": "Electrode_Location.xlsx",
    "behaviour_structure": "DataStructureOfBehaviouralData.xlsx",
    "stimuli": "Stimuli_info.xlsx",
    "task_event": "Task_event.xlsx",
    "readme": "Readme.md",
}
DIRS = {
    "processed": ("Processed_data", "Processed_Data", "PKLs"),
    "features_de": ("EEG_Features/DE", "EEG_Features/de", "DE"),
    "features_psd": ("EEG_Features/PSD", "EEG_Features/psd", "PSD"),
    "raw": ("Data",),
}

SEED = 0

