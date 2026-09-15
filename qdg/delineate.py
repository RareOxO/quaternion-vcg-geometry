"""Per-beat cardiac delineation: QRS onset, R peak, QRS offset, T peak, T end.

The phase-resolved interpretability analysis needs each patient's own depolarisation and
repolarisation intervals, because a fixed R-relative window such as [-60, +60] ms is not
that patient's QRS -- least of all in conduction disease, where the complex is widened.
This module produces those boundaries; nothing here touches the model.

Where the boundaries come from
------------------------------
Delineation runs on the cardiac vector, not on the quaternion sequence: the magnitude
||V|| and the spatial velocity ||dV/dt||. Both are functions of all eight independent
leads through the fixed Kors transform, so this is a multilead consensus rather than one
chosen lead, and it is the same front end the R-peak detector already uses.

* QRS onset and offset. Walking outward from the velocity maxima that flank the R peak,
  the first sample on each side whose spatial velocity falls below a fixed fraction of
  that beat's own velocity maximum. The search starts at those maxima and not at the
  peak itself, because the vector is momentarily slowest at the apex of the complex. Spatial velocity is the standard multilead criterion for the complex: it is
  large exactly while the vector is moving quickly, which is what depolarisation is.
* T peak. The largest magnitude excursion above baseline in a search interval that
  starts after the QRS and ends at the earlier of a fraction of the RR interval, a fixed
  cap, and the next beat.
* T end. The tangent (Lepeschkin) construction: the steepest descending slope after the
  T peak, extrapolated to the record's baseline.

Every threshold below is fixed in advance and applied to every record identically.

Quality control flags, never repairs
------------------------------------
A boundary that violates the temporal ordering or leaves a physiological duration range
is *flagged*, not moved. A beat with a reliable QRS but an unreliable T wave still enters
the depolarisation analysis and is excluded from the repolarisation analysis; that is the
whole reason the two flags are separate.
"""

import numpy as np

from .handcrafted import cardiac_vector, detect_r_peaks

# Search limits and thresholds, in milliseconds unless named otherwise. Pre-specified.
QRS_ON_SEARCH_MS = 140
QRS_OFF_SEARCH_MS = 220
# A sample belongs to the complex while the spatial velocity is above this fraction of
# the beat's own velocity maximum. Beat-local rather than record-wide, so a small beat
# is delineated by its own scale.
VELOCITY_FRACTION = 0.08
T_SEARCH_START_MS = 60
T_SEARCH_RR_FRACTION = 0.7
T_SEARCH_CAP_MS = 600
NEXT_BEAT_MARGIN_MS = 60
# The isoelectric level of ||V||, taken as a low quantile of the whole record: the
# vector spends most of a cardiac cycle near rest.
BASELINE_QUANTILE = 0.05
# Physiological ranges. These are QC flags. Deliberately wide at the upper end of the
# QRS so that a genuinely widened complex is kept and reported rather than discarded.
QRS_MS = (40, 200)
QT_MS = (250, 600)
R_TO_TPEAK_MS = (100, 450)
R_TO_TEND_MS = (150, 700)

FIDUCIALS = ("qrs_on", "r_peak", "qrs_off", "t_peak", "t_end")
DURATIONS = ("qrs_duration_ms", "qt_duration_ms", "r_to_tpeak_ms", "r_to_tend_ms")
FLAGS = ("valid_qrs", "valid_twave", "valid_full_beat")


def spatial_velocity(vcg, sampling_rate):
    """||dV/dt|| by central difference, in mV/s, same length as the record."""
    return np.linalg.norm(np.gradient(np.asarray(vcg, dtype=np.float64), axis=1), axis=0) * (
        sampling_rate
    )


def _first_below(values, start, stop, threshold, backwards):
    """Index where `values` first drops below `threshold`, walking away from the peak.

    Returns None when the whole search interval stays above it, which is a delineation
    failure and is reported as one rather than silently clamped to the search limit.
    """
    span = range(start, stop - 1, -1) if backwards else range(start, stop + 1)
    for index in span:
        if values[index] < threshold:
            return index
    return None


def delineate_beats(vcg, peaks, sampling_rate):
    """Fiducial points for every detected beat of one record.

    Returns a dict of arrays, one entry per beat, with every fiducial as a sample index
    (-1 where delineation failed) and the derived durations in milliseconds.
    """
    peaks = np.asarray(peaks, dtype=int)
    magnitude = np.linalg.norm(np.asarray(vcg, dtype=np.float64), axis=0)
    velocity = spatial_velocity(vcg, sampling_rate)
    steps = len(magnitude)
    per_ms = sampling_rate / 1000.0
    baseline = float(np.quantile(magnitude, BASELINE_QUANTILE))
    intervals = np.diff(peaks)
    typical_rr = float(np.median(intervals)) if len(intervals) else float(steps)
    out = {name: np.full(len(peaks), -1, dtype=int) for name in FIDUCIALS}
    out["r_peak"] = peaks.copy()
    for beat, peak in enumerate(peaks):
        back = max(0, peak - int(QRS_ON_SEARCH_MS * per_ms))
        forward = min(steps - 1, peak + int(QRS_OFF_SEARCH_MS * per_ms))
        threshold = VELOCITY_FRACTION * float(velocity[back : forward + 1].max())
        # Walk outward from the velocity maxima on either side of the peak, not from the
        # peak itself: the vector is momentarily slowest at the apex of the complex, so a
        # search starting there stops immediately and reports a QRS of zero width.
        upstroke = back + int(np.argmax(velocity[back : peak + 1]))
        downstroke = peak + int(np.argmax(velocity[peak : forward + 1]))
        onset = _first_below(velocity, upstroke, back, threshold, backwards=True)
        offset = _first_below(velocity, downstroke, forward, threshold, backwards=False)
        if onset is None or offset is None:
            continue
        out["qrs_on"][beat] = onset
        out["qrs_off"][beat] = offset
        # The T search ends at whichever comes first: a fraction of the RR interval, the
        # fixed cap, or safely before the next detected beat.
        limit = min(
            peak + int(T_SEARCH_RR_FRACTION * typical_rr),
            peak + int(T_SEARCH_CAP_MS * per_ms),
            steps - 1,
        )
        if beat + 1 < len(peaks):
            limit = min(limit, peaks[beat + 1] - int(NEXT_BEAT_MARGIN_MS * per_ms))
        start = offset + int(T_SEARCH_START_MS * per_ms)
        if start >= limit:
            continue
        apex = start + int(np.argmax(np.abs(magnitude[start : limit + 1] - baseline)))
        out["t_peak"][beat] = apex
        tail = magnitude[apex : limit + 1]
        if len(tail) < 3:
            continue
        slope = np.gradient(tail)
        step = int(np.argmin(slope))
        if slope[step] >= 0:
            continue
        # Tangent at the steepest descent, extrapolated to the isoelectric level.
        crossing = step + (baseline - tail[step]) / slope[step]
        end = apex + int(round(crossing))
        if apex < end <= limit:
            out["t_end"][beat] = end
    for name, (first, second) in zip(
        DURATIONS,
        (("qrs_on", "qrs_off"), ("qrs_on", "t_end"), ("r_peak", "t_peak"), ("r_peak", "t_end")),
    ):
        span = (out[second] - out[first]) / per_ms
        out[name] = np.where((out[first] >= 0) & (out[second] >= 0), span, np.nan)
    return out


def _within(values, bounds):
    low, high = bounds
    return (values >= low) & (values <= high)


def quality_flags(beats):
    """The pre-specified QC rule, as three independent flags.

    `valid_qrs` gates the depolarisation analysis, `valid_twave` the repolarisation one.
    A beat can pass the first and fail the second; that is the point of separating them.
    """
    ordered_qrs = (beats["qrs_on"] >= 0) & (beats["qrs_on"] < beats["r_peak"])
    ordered_qrs &= beats["r_peak"] < beats["qrs_off"]
    valid_qrs = ordered_qrs & _within(beats["qrs_duration_ms"], QRS_MS)
    ordered_t = (beats["t_peak"] > beats["qrs_off"]) & (beats["t_end"] > beats["t_peak"])
    ordered_t &= beats["t_peak"] >= 0
    valid_twave = valid_qrs & ordered_t
    valid_twave &= _within(beats["qt_duration_ms"], QT_MS)
    valid_twave &= _within(beats["r_to_tpeak_ms"], R_TO_TPEAK_MS)
    valid_twave &= _within(beats["r_to_tend_ms"], R_TO_TEND_MS)
    return {
        "valid_qrs": valid_qrs,
        "valid_twave": valid_twave,
        "valid_full_beat": valid_qrs & valid_twave,
    }


def delineate_record(signal, sampling_rate, peaks=None):
    """Fiducials and QC flags for one 12-lead record."""
    vcg = cardiac_vector(np.asarray(signal))
    if peaks is None:
        peaks = detect_r_peaks(vcg, sampling_rate)
    beats = delineate_beats(vcg, peaks, sampling_rate)
    beats.update(quality_flags(beats))
    return beats
