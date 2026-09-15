"""R-peak detection and the handcrafted feature families of Experiment 5.

Three feature sets, all computed from the same R/L/Q signals the learned model sees:

    A  statistical aggregation   order-insensitive distributional summaries
    B  Cruces-2016-inspired      linear / angular / spatial velocity summaries
    C  Cruces-2020-inspired      loop areas, phase maxima, SVD roundness, QT_omega

B and C are inspired by their papers, not replications: the task, the dataset and the
available annotation all differ, so every definition below is stated explicitly and
labelled as a reimplementation on PTB-XL.

Everything here is deterministic and fitted to no data, so a feature computed for a
validation or test record cannot depend on any other record. Only the classifier that
consumes these features is fitted, and only on the training folds.
"""

import numpy as np
from scipy.signal import find_peaks

from .geometry import KORS, INDEPENDENT_LEADS

FEATURE_SETS = ("stat", "velocity", "biomarker")
# Quality control for the detector, fixed in advance (plan section 9.2 asks for one
# detector and one pre-specified rule). A 10 s record must hold a plausible number of
# beats at a plausible regularity, or its phase-dependent features fall back to
# whole-record windows rather than being silently wrong.
MIN_BEATS, MAX_BEATS = 4, 25
MAX_RR_VARIATION = 0.5
# A candidate must reach this fraction of the record's own typical beat to be kept.
DETECTION_FRACTION = 0.35
# R-peak-relative windows, in milliseconds. QRS is the depolarisation complex around the
# peak; the T window covers repolarisation at a typical heart rate.
QRS_WINDOW = (-50, 50)
T_WINDOW = (150, 450)


def cardiac_vector(ecg):
    """(12, T) physical-mV ECG -> (3, T) VCG, the same Kors transform the models use."""
    return np.asarray(KORS, dtype=np.float32) @ ecg[list(INDEPENDENT_LEADS)]


def detect_r_peaks(vcg, sampling_rate):
    """Indices of the R peaks, from the spatial magnitude of the cardiac vector.

    Two passes, because no single fixed threshold works on both a sparse record and one
    with uneven beat amplitudes. A percentile alone sits in the baseline when the
    complexes occupy only a few percent of the record; a fraction of the largest complex
    alone drops the smaller beats of a record whose amplitude varies. So candidates are
    taken first with only a refractory constraint, the record's own typical beat height
    is estimated from the tallest dozen of them, and candidates below a fraction of that
    are dropped. The peak is then refined to the local magnitude maximum.

    One detector, one rule, applied everywhere; it is not claimed to be state of the art.
    """
    magnitude = np.linalg.norm(vcg, axis=0)
    window = max(1, int(0.1 * sampling_rate))
    smoothed = np.convolve(magnitude**2, np.ones(window) / window, mode="same")
    candidates, _ = find_peaks(smoothed, distance=int(0.2 * sampling_rate))
    if not len(candidates):
        return np.array([], dtype=int)
    heights = smoothed[candidates]
    typical = float(np.median(np.sort(heights)[-min(len(heights), MAX_BEATS // 2) :]))
    kept = candidates[heights >= DETECTION_FRACTION * typical]
    refined = []
    half = window // 2
    for peak in kept:
        low, high = max(0, peak - half), min(len(magnitude), peak + half + 1)
        refined.append(low + int(np.argmax(magnitude[low:high])))
    return np.unique(refined)


def beat_quality(peaks, sampling_rate, length):
    """Whether the detection is usable, under the pre-specified rule."""
    if not (MIN_BEATS <= len(peaks) <= MAX_BEATS):
        return False
    intervals = np.diff(peaks) / sampling_rate
    if len(intervals) < 2:
        return False
    return bool(intervals.std() / max(intervals.mean(), 1e-6) < MAX_RR_VARIATION)


def _summaries(values):
    """Seven order-insensitive statistics, the set named in plan section 6.1."""
    values = np.asarray(values, dtype=np.float64)
    quartiles = np.quantile(values, [0.25, 0.75])
    return [
        values.mean(),
        values.std(),
        np.median(values),
        quartiles[1] - quartiles[0],
        values.min(),
        values.max(),
        np.sqrt(np.mean(values**2)),
    ]


def rlq_signals(vcg, sampling_rate, lag_ms=20, eps=1e-8):
    """The three scalar streams the handcrafted sets summarise.

    radius      r_t = ||V_t||
    linear      ||V_{t+1} - V_t|| / dt, the spatial speed of the cardiac vector
    angle       theta_t, the rotation angle carried by q_t over the same 20 ms lag

    theta is used rather than raw quaternion components because a component-wise mean of
    quaternions is not a physiological quantity, which plan section 6.1 warns against.
    """
    lag = int(round(lag_ms * sampling_rate / 1000))
    radius = np.linalg.norm(vcg, axis=0)
    direction = vcg / (radius + eps)
    speed = np.linalg.norm(np.diff(vcg, axis=1), axis=0) * sampling_rate
    cosine = np.clip((direction[:, :-lag] * direction[:, lag:]).sum(0), -1.0, 1.0)
    angle = np.arccos(cosine)
    return radius, speed, angle


def statistical_features(vcg, sampling_rate):
    """Set A: distributional summaries of r, |linear velocity| and theta."""
    radius, speed, angle = rlq_signals(vcg, sampling_rate)
    return np.array(_summaries(radius) + _summaries(speed) + _summaries(angle), dtype=np.float32)


def velocity_features(vcg, sampling_rate):
    """Set B, Cruces-2016-inspired: linear, angular and spatial velocity summaries.

    The 2016 method derives a velocity index from the cardiac vector's motion. Here the
    three velocities are defined as

        linear    per-axis |dV_i/dt|, summarised per axis
        spatial   ||dV/dt||, the total speed
        angular   dtheta/dt over the fixed 20 ms lag

    and the index ICVV is the time integral of the spatial velocity, i.e. the path
    length the cardiac vector travels in the 10 s. Mean, max and the integral of each
    velocity are kept. This is a reimplementation on PTB-XL, not a replication: the
    original works on a different task with beat-level annotation.
    """
    radius, speed, angle = rlq_signals(vcg, sampling_rate)
    step = 1.0 / sampling_rate
    axis_speed = np.abs(np.diff(vcg, axis=1)) * sampling_rate
    angular_speed = np.diff(angle) * sampling_rate if len(angle) > 1 else np.zeros(1)
    features = []
    for series in (*axis_speed, speed, np.abs(angular_speed)):
        features += [series.mean(), series.max(), series.sum() * step]
    # ICVV: path length of the cardiac vector, and the same normalised by mean radius,
    # which removes the overall amplitude and leaves the shape of the trajectory.
    path = speed.sum() * step
    features += [path, path / max(radius.mean(), 1e-6)]
    return np.array(features, dtype=np.float32)


def _loop_areas(vcg):
    """Signed area of the VCG loop projected on each plane, by the shoelace formula."""
    areas = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        x, y = vcg[first], vcg[second]
        areas.append(0.5 * float(np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))))
    return areas


def _roundness(vcg):
    """SVD of the centred loop: singular values and the ratios between them.

    A planar loop has a small third singular value; a circular one has s2/s1 near 1.
    These are the 'roundness/SVD features' of the 2020 paper in their simplest form.
    """
    centred = vcg - vcg.mean(axis=1, keepdims=True)
    values = np.linalg.svd(centred, compute_uv=False)
    total = max(values.sum(), 1e-12)
    return [*(values / total), values[1] / max(values[0], 1e-12), values[2] / max(values[0], 1e-12)]


def biomarker_features(vcg, sampling_rate):
    """Set C, Cruces-2020-inspired: areas, phase maxima, roundness and QT_omega.

    Phase maxima need R peaks, so a record that fails the pre-specified quality rule
    falls back to whole-record windows; the fallback is recorded as a feature of its own
    so the classifier can tell the two regimes apart rather than being misled.

    QT_omega is approximated as the time from the R peak to the point where the spatial
    magnitude of the repolarisation wave has decayed to 10% of its own peak, averaged
    over beats. The original is defined on a delineated T wave, which PTB-XL does not
    annotate, so this is an approximation and labelled as one.
    """
    radius = np.linalg.norm(vcg, axis=0)
    peaks = detect_r_peaks(vcg, sampling_rate)
    usable = beat_quality(peaks, sampling_rate, vcg.shape[1])
    features = [*_loop_areas(vcg), *_roundness(vcg), float(not usable)]
    if not usable:
        # Whole-record fallback: the same quantities over the entire 10 s.
        features += [
            radius.max(),
            radius.max(),
            0.0,
            float(len(peaks)) / (vcg.shape[1] / sampling_rate),
        ]
        return np.array(features, dtype=np.float32)
    qrs_max, t_max, qt = [], [], []
    for peak in peaks:
        for window, store in ((QRS_WINDOW, qrs_max), (T_WINDOW, t_max)):
            low = peak + int(window[0] * sampling_rate / 1000)
            high = peak + int(window[1] * sampling_rate / 1000)
            low, high = max(0, low), min(len(radius), high)
            if high > low:
                store.append(float(radius[low:high].max()))
        low = peak + int(T_WINDOW[0] * sampling_rate / 1000)
        high = min(len(radius), peak + int(T_WINDOW[1] * sampling_rate / 1000))
        if high > low + 1:
            segment = radius[low:high]
            threshold = 0.1 * segment.max()
            below = np.flatnonzero(segment < threshold)
            offset = below[0] if len(below) else len(segment) - 1
            qt.append(1000 * (low + offset - peak) / sampling_rate)
    features += [
        float(np.mean(qrs_max)) if qrs_max else radius.max(),
        float(np.mean(t_max)) if t_max else radius.max(),
        float(np.mean(qt)) if qt else 0.0,
        60 * sampling_rate / float(np.mean(np.diff(peaks))),
    ]
    return np.array(features, dtype=np.float32)


BUILDERS = {
    "stat": statistical_features,
    "velocity": velocity_features,
    "biomarker": biomarker_features,
}


def features_for(ecg, sampling_rate, kinds):
    """Concatenate the named feature sets for one record."""
    vcg = cardiac_vector(np.asarray(ecg, dtype=np.float32))
    return np.concatenate([BUILDERS[kind](vcg, sampling_rate) for kind in kinds])
