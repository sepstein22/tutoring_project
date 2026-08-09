"""Spectral features and topomap grids for FACED."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata
from scipy.signal import welch

from . import config as C


def band_power(trial, sfreq=C.SFREQ, bands=None, nperseg_seconds=2.0):
    """Integrated power in each frequency band.

    Inputs
        trial            (n_channel, n_sample) array, in volts
        sfreq            sampling rate [Hz]
        bands            {name: (low, high)} in Hz; defaults to config.BANDS
        nperseg_seconds  Welch segment length [s]

    Returns
        (n_channel, n_band) array of band power [V^2]

    Raises
        ValueError if trial is not 2-D, or a band contains no frequency bins.
    """
    bands = C.BANDS if bands is None else bands
    trial = np.asarray(trial, dtype=np.float64)
    if trial.ndim != 2:
        raise ValueError("expected (channels, samples), got %s"
                         % (trial.shape,))

    nperseg = min(int(round(nperseg_seconds * sfreq)), trial.shape[-1])
    freqs, psd = welch(trial, fs=sfreq, nperseg=nperseg, axis=-1)
    bin_width = freqs[1] - freqs[0]          # [Hz]

    power = np.empty((trial.shape[0], len(bands)), dtype=np.float64)
    for index, (low, high) in enumerate(bands.values()):
        in_band = (freqs >= low) & (freqs < high)
        if not in_band.any():
            raise ValueError("band %g-%g Hz has no bins at %g Hz resolution"
                             % (low, high, bin_width))
        # TODO: (@takashi) Explain why this multiplies by bin_width. Say what
        # welch() returns and what its units are. Two sentences, no more.
        power[:, index] = psd[:, in_band].sum(axis=-1) * bin_width
    return power


def differential_entropy(trial, sfreq=C.SFREQ, bands=None,
                         nperseg_seconds=2.0, eps=1e-30, half=True):
    """Log band power per channel and band.

    Inputs
        trial   (n_channel, n_sample) array, in volts
        half    apply the 0.5 factor, matching the SEED convention FACED follows
        eps     floor added before the log, guarding log(0)

    Returns
        (n_channel, n_band) array
    """
    value = np.log(band_power(trial, sfreq, bands, nperseg_seconds) + eps)
    return 0.5 * value if half else value


def stack(per_channel_band, order=C.FEATURE_ORDER):
    """Flatten (n_channel, n_band) into one feature vector.

    Inputs
        order   "band-major"    column = band * n_channel + channel
                "channel-major" column = channel * n_band + band

    Returns
        (n_channel * n_band,) array

    Raises
        ValueError on a 1-D or 3-D input, or an unrecognised order.
    """
    array = np.asarray(per_channel_band)
    if array.ndim != 2:
        raise ValueError("expected (channels, bands), got %s" % (array.shape,))
    # TODO: (@takashi) Both orders are valid, so why does the choice matter?
    # Name the downstream operation that breaks silently under the wrong one.
    if order == "band-major":
        return array.T.reshape(-1)
    if order == "channel-major":
        return array.reshape(-1)
    raise ValueError("order must be band-major or channel-major")


def unstack(vector, n_channel, n_band=C.N_BAND, order=C.FEATURE_ORDER):
    """Inverse of stack().

    Inputs
        vector      (n_channel * n_band,) array
        n_channel   channels the vector was built from
        n_band      bands the vector was built from
        order       must match the order used by stack()

    Returns
        (n_channel, n_band) array
    """
    vector = np.asarray(vector)
    if vector.size != n_channel * n_band:
        raise ValueError("length %d does not match %d channels x %d bands"
                         % (vector.size, n_channel, n_band))
    if order == "band-major":
        return vector.reshape(n_band, n_channel).T
    return vector.reshape(n_channel, n_band)


def names(channel_names, order=C.FEATURE_ORDER, band_names=C.BAND_NAMES):
    """Column labels matching stack(), for reading coefficients back."""
    if order == "band-major":
        return ["%s_%s" % (b, c) for b in band_names for c in channel_names]
    return ["%s_%s" % (b, c) for c in channel_names for b in band_names]


def subject_matrix(trials, sfreq=C.SFREQ, bands=None, order=C.FEATURE_ORDER,
                   nperseg_seconds=2.0, half=True):
    """Feature matrix for one subject.

    Inputs
        trials  (n_clip, n_channel, n_sample) array, in volts

    Returns
        (n_clip, n_channel * n_band) array
    """
    trials = np.asarray(trials, dtype=np.float64)
    if trials.ndim != 3:
        raise ValueError("expected (clips, channels, samples), got %s"
                         % (trials.shape,))
    return np.asarray([
        stack(differential_entropy(trial, sfreq, bands, nperseg_seconds,
                                   half=half), order=order)
        for trial in trials
    ], dtype=np.float64)


def montage_positions(channel_names, montage="standard_1020"):
    """Projected 2-D sensor positions, scaled into [-1, 1].

    Inputs
        channel_names   names in the same order as the data's channel axis

    Returns
        (n_channel, 2) array

    Raises
        KeyError if any name is absent from the montage. FACED's two cohorts
        name six electrodes differently, so check which cohort the data is from.
    """
    import mne

    standard = mne.channels.make_standard_montage(montage)
    known = standard.get_positions()["ch_pos"]
    missing = [name for name in channel_names if name not in known]
    if missing:
        raise KeyError("not in the %s montage: %s" % (montage, missing))
    positions = np.array([known[name][:2] for name in channel_names],
                         dtype=np.float64)
    extent = np.abs(positions).max()
    return positions / extent if extent else positions


def topomap_grid(values, positions, size=64, fill=0.0):
    """Interpolate per-channel values onto a square grid.

    Linear inside the sensor hull, nearest-neighbour outside it, `fill` beyond
    the unit disc.

    Inputs
        values      (n_channel,) array
        positions   (n_channel, 2) array in [-1, 1]
        size        grid edge length in pixels

    Returns
        (size, size) array
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    positions = np.asarray(positions, dtype=np.float64)
    if values.shape[0] != positions.shape[0]:
        raise ValueError("%d values but %d positions"
                         % (values.shape[0], positions.shape[0]))

    axis = np.linspace(-1.0, 1.0, size)
    grid_x, grid_y = np.meshgrid(axis, axis)
    grid = griddata(positions, values, (grid_x, grid_y), method="linear")
    outside_hull = ~np.isfinite(grid)
    if outside_hull.any():
        grid[outside_hull] = griddata(
            positions, values, (grid_x[outside_hull], grid_y[outside_hull]),
            method="nearest")
    grid[grid_x ** 2 + grid_y ** 2 > 1.0] = fill
    # TODO: (@takashi) This deliberately does not normalise the grid. Explain
    # what would be lost if it did, and name the library function that does
    # normalise and therefore must not be used to build model input.
    return grid


def trial_image(per_channel_band, positions, size=64):
    """One band per image plane.

    Returns
        (size, size, n_band) float32 array
    """
    array = np.asarray(per_channel_band, dtype=np.float64)
    planes = [topomap_grid(array[:, band], positions, size)
              for band in range(array.shape[1])]
    return np.stack(planes, axis=-1).astype(np.float32)


def image_stack(feature_matrix, positions, n_channel, size=64,
                n_band=C.N_BAND, order=C.FEATURE_ORDER):
    """Images for every trial.

    Inputs
        feature_matrix  (n_trial, n_channel * n_band) array

    Returns
        (n_trial, size, size, n_band) float32 array
    """
    return np.stack([
        trial_image(unstack(row, n_channel, n_band, order), positions, size)
        for row in np.asarray(feature_matrix)
    ]).astype(np.float32)

