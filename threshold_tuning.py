"""
Threshold selection for the Capuchinbird classifier.
Paste these cells into the notebook after training (they use model, val_ds, test_ds,
y_val, y_test, recording_files, recording_windows, make_spectrogram, load_audio, BATCH_SIZE).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score


# ---------------------------------------------------------------------------
# 1) Clip-level: choose the threshold on VALIDATION, report on TEST
# ---------------------------------------------------------------------------
def pick_threshold(y_true, p, method="fbeta", beta=1.0,
                   min_recall=None, min_precision=None, prior=None):
    """
    method:
      'fbeta'    - maximise F-beta (beta>1 favours recall, beta<1 favours precision)
      'youden'   - maximise TPR - FPR (balanced errors, ignores class ratio)
      'recall'   - highest threshold that still gives recall >= min_recall
      'precision'- lowest threshold that gives precision >= min_precision
    prior: expected fraction of positive windows in deployment (e.g. 0.03).
           If given, precision is re-computed for that prior (see note in the reply).
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).reshape(-1)

    prec, rec, thr = precision_recall_curve(y_true, p)
    prec, rec = prec[:-1], rec[:-1]          # align with thr

    if prior is not None:
        fpr_all, tpr_all, thr_roc = roc_curve(y_true, p)
        # FPR at each PR threshold
        idx = np.searchsorted(-thr_roc, -thr, side="left").clip(0, len(thr_roc) - 1)
        fpr = fpr_all[idx]
        tpr = rec
        prec = tpr * prior / np.maximum(tpr * prior + fpr * (1 - prior), 1e-12)

    if method == "fbeta":
        f = (1 + beta**2) * prec * rec / np.maximum(beta**2 * prec + rec, 1e-12)
        i = int(np.argmax(f))
    elif method == "youden":
        fpr, tpr, t = roc_curve(y_true, p)
        j = int(np.argmax(tpr - fpr))
        return float(t[j])
    elif method == "recall":
        ok = np.where(rec >= min_recall)[0]
        i = ok[-1] if len(ok) else 0
    elif method == "precision":
        ok = np.where(prec >= min_precision)[0]
        i = ok[0] if len(ok) else len(thr) - 1
    else:
        raise ValueError(method)
    return float(thr[i])


def bootstrap_threshold(y_true, p, n=1000, seed=0, **kw):
    """Stability check: how much does the 'best' threshold move across resamples?"""
    rng = np.random.default_rng(seed)
    y_true, p = np.asarray(y_true), np.asarray(p).reshape(-1)
    ts = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        ts.append(pick_threshold(y_true[idx], p[idx], **kw))
    ts = np.array(ts)
    return np.median(ts), np.percentile(ts, [5, 95])


def plot_threshold_curves(y_true, p):
    prec, rec, thr = precision_recall_curve(y_true, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    plt.figure(figsize=(8, 3.5))
    plt.plot(thr, prec[:-1], label="precision")
    plt.plot(thr, rec[:-1], label="recall")
    plt.plot(thr, f1, label="F1", ls="--")
    plt.xlabel("threshold"); plt.legend(); plt.grid(alpha=.3)
    plt.title(f"AP = {average_precision_score(y_true, p):.3f}")
    plt.tight_layout(); plt.show()


# Example (clip level):
# p_val  = model.predict(val_ds,  verbose=0).reshape(-1)
# p_test = model.predict(test_ds, verbose=0).reshape(-1)
# plot_threshold_curves(y_val, p_val)
# t = pick_threshold(y_val, p_val, method="fbeta", beta=1.0, prior=0.05)
# print("chosen threshold:", t, bootstrap_threshold(y_val, p_val, method="fbeta", prior=0.05))
# pred = (p_test >= t).astype(int)       # report test metrics ONCE, with this t


# ---------------------------------------------------------------------------
# 2) Recording-level: tune threshold + event rule on forest recordings
#    (this is the setting the model is actually used in)
# ---------------------------------------------------------------------------
def overlapping_windows(audio, win, stride, min_len):
    out = []
    for s in range(0, len(audio), stride):
        chunk = audio[s:s + win]
        if len(chunk) < min_len:
            break
        out.append(chunk)
    return out


def cache_recording_probs(model, paths, load_audio, make_spectrogram, SR, SAMPLES,
                          stride_s=1.0, batch_size=16):
    """Run the model ONCE per recording; reuse the probabilities for every threshold."""
    cache = {}
    for p in paths:
        audio = load_audio(p)
        wins = overlapping_windows(audio, SAMPLES, int(stride_s * SR), SR)
        specs = np.stack([make_spectrogram(w) for w in wins])
        cache[p.name] = model.predict(specs, batch_size=batch_size, verbose=0).reshape(-1)
    return cache


def count_events(probs, threshold, min_run=1, merge_gap=0):
    """Number of events = runs of >= min_run consecutive positive windows.
    merge_gap lets a run bridge up to N negative windows (a call split by one weak window)."""
    det = (probs >= threshold).astype(int)
    if merge_gap:
        runs = [(v, len(list(g))) for v, g in groupby(det)]
        for k in range(1, len(runs) - 1):
            if runs[k][0] == 0 and runs[k][1] <= merge_gap and runs[k-1][0] == 1 and runs[k+1][0] == 1:
                runs[k] = (1, runs[k][1])
        det = np.concatenate([np.full(n, v) for v, n in runs])
    return sum(1 for v, g in groupby(det) if v == 1 and len(list(g)) >= min_run)


def sweep_recording_params(cache, true_counts, thresholds=np.arange(0.5, 0.995, 0.01),
                           min_runs=(1, 2, 3), merge_gaps=(0, 1)):
    """
    true_counts: {recording_name: number of calls you labelled by listening}
                 (even 10-15 recordings gives a far better threshold than guessing)
    Returns a table sorted by mean absolute error of the call count.
    """
    rows = []
    for t in thresholds:
        for mr in min_runs:
            for mg in merge_gaps:
                err = [abs(count_events(cache[k], t, mr, mg) - v) for k, v in true_counts.items()]
                rows.append(dict(threshold=round(float(t), 3), min_run=mr, merge_gap=mg,
                                 MAE=np.mean(err), total_err=int(np.sum(err))))
    return pd.DataFrame(rows).sort_values(["MAE", "min_run"]).reset_index(drop=True)


# Example:
# cache = cache_recording_probs(model, recording_files, load_audio, make_spectrogram, SR, SAMPLES)
# true_counts = {"recording_00.mp3": 4, "recording_01.mp3": 0, ...}   # your own labels
# table = sweep_recording_params(cache, true_counts)
# table.head(10)
