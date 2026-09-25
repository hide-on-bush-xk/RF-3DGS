"""Communication metrics of a set of predicted spectra, for the live tab and its reference lines.

Only physically grounded quantities (Ke, 2026-09-24: "物理依据的指标"), per view against the truth spectrum and
the true channel:
  angle        degrees between the predicted and the true main-peak direction (where a beam would be steered)
  <= 1 deg     the share of views within 1 deg, over all views and over the distinct ones (the true main peak
               >= 1 dB above its best rival >= 3 deg away; mvdr_peaks.py)
  beam loss    dB below the best beam when the array's beam is steered at the predicted main peak, on the true
               channel: max(B) - B(predicted peak), B the matched-beam gain map a^H R a / |a|^2 of the view's tap
               covariance R (sionna_port/beam_maps.py, from t6_incoherent_mvdr.py's re-solve)
  top-3        the share of the truth's top-3 peaks (>= 3 deg apart, within 20 dB) with a predicted local maximum
               within 1.5 deg (multipath: the beams a second or third link would use)
The prediction may be in any monotonic unit of power (dB, normalised dB): only its argmax and local maxima matter.
Pure numpy (mvdr_peaks.py's helpers), so the trainer and the sionna environment compute the same thing.
"""

from __future__ import annotations

import numpy as np

from mvdr_peaks import angle, local_maxima, top_peaks


def view_metrics(pred, truth_db, beam_db, dirs):
    """One view -> dict, or None for an all-floor truth (no paths: nothing to steer at). beam_db may be None (a
    training view has no tap covariance): the beam-gain loss is then left out."""
    t = np.asarray(truth_db, dtype=np.float64)
    if t.max() < -250 or t.max() - t.min() < 1e-3:
        return None
    p = np.asarray(pred, dtype=np.float64)
    flat = dirs.reshape(-1, 3)
    kt, kp = int(t.argmax()), int(p.argmax())
    kept = top_peaks(t, dirs, 3)
    pm = np.flatnonzero(local_maxima(p))
    det = [bool(len(pm)) and float(np.degrees(np.arccos(np.clip(flat[pm] @ flat[c], -1, 1))).min()) <= 1.5 for c in kept]
    out = {"angle": angle(dirs, kt, kp), "distinct": len(kept) < 2 or t.flat[kept[0]] - t.flat[kept[1]] >= 1.0,
           "beam_loss": None, "beam_loss_truth_peak": None, "top3": det}
    if beam_db is not None:
        b = np.asarray(beam_db, dtype=np.float64)
        out["beam_loss"], out["beam_loss_truth_peak"] = float(b.max() - b.flat[kp]), float(b.max() - b.flat[kt])
    return out


def summarise(rows):
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    a = np.array([r["angle"] for r in rows]); d = np.array([r["distinct"] for r in rows], dtype=bool)
    t3 = [x for r in rows for x in r["top3"]]
    has_beam = all(r["beam_loss"] is not None for r in rows)
    bl = np.array([r["beam_loss"] for r in rows]) if has_beam else None
    return {"views": len(rows), "angle_median": float(np.median(a)), "within_1deg": float((a <= 1).mean()),
            "distinct_views": int(d.sum()),
            "distinct_within_1deg": float((a[d] <= 1).mean()) if d.any() else None,
            "beam_loss_median": float(np.median(bl)) if has_beam else None,
            "beam_loss_p90": float(np.percentile(bl, 90)) if has_beam else None,
            "beam_loss_truth_peak_median": float(np.median([r["beam_loss_truth_peak"] for r in rows])) if has_beam else None,
            "top3_detected": float(np.mean(t3)) if t3 else None}
