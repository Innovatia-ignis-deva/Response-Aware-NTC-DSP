"""
Response-Aware Thermal Transient Detection Pipeline (v2.1)
Ideated by Devashrit Panda

Usage:
1. Synthetic benchmark (with optional tau calibration mismatch):
     python ntcfinal.py --mode benchmark --synthetic --tau 2.2 --tau-true 2.6

2. Step-response calibration from CSV (columns: timestamp_ms, temp_c or raw_adc):
     python ntcfinal.py --mode calibrate --step-csv path/to/step_test.csv

3. Real session evaluation (CSV files matching *.csv with columns: timestamp_ms, temp_c, label):
     python ntcfinal.py --mode benchmark --data-dir path/to/sessions --tau 2.4 --out-json report.json

4. Train & save a model bundle for reuse:
     python ntcfinal.py --mode train --data-dir path/to/sessions --tau 2.4 --save-model thermal_model.joblib

5. Batch predict on an unlabelled session CSV:
     python ntcfinal.py --mode predict --load-model thermal_model.joblib --input-csv path/to/new_session.csv

6. Live/streaming inference (tailing a growing CSV or reading lines from stdin):
     python ntcfinal.py --mode stream --load-model thermal_model.joblib --stream-csv path/to/live.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


LABELS: List[str] = ["baseline", "drift", "heating", "cooling"]
LABEL_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(LABELS)}
ID_TO_LABEL: Dict[int, str] = {idx: name for idx, name in enumerate(LABELS)}

RESP_FEATURE_NAMES: List[str] = [
    "temp_mean",
    "peak_dT_dt",
    "peak_d2T_dt2",
    "peak_comp_dT_dt",
    "comp_peak_excursion",
    "peak_drift_residual",
]

DIFF_FEATURE_NAMES: List[str] = [
    "diff_temp_mean",
    "peak_diff_dT_dt",
]

LOGGER = logging.getLogger("thermal_transient_pipeline")


def _odd_window_le(desired: int, max_len: int) -> int:
    """Returns a valid odd Savitzky-Golay window length not exceeding max_len."""
    if max_len <= 1:
        return 1
    cap = max_len if max_len % 2 == 1 else max_len - 1
    win = min(desired, cap)
    if win % 2 == 0:
        win = max(1, win - 1)
    return max(1, win)


def configure_logging(verbose: bool = False) -> None:
    """Configures structured console logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def adc_to_temperature_steinhart_hart(
    raw_adc: np.ndarray,
    adc_max: float = 4095.0,
    v_ref: float = 3.3,
    r_fixed: float = 10000.0,
    r0: float = 10000.0,
    beta: float = 3950.0,
    t0_celsius: float = 25.0,
) -> np.ndarray:
    """Converts ADC readings on a voltage divider to Celsius using Beta / Steinhart-Hart model."""
    raw = raw_adc.astype(np.float64)
    adc_clipped = np.clip(raw, 1.0, max(adc_max - 1.0, 2.0))
    v_out = (adc_clipped / max(adc_max, 1.0)) * v_ref
    # NTC connected to V_ref and fixed resistor to GND:
    # V_out = V_ref * R_fixed / (R_ntc + R_fixed) => R_ntc = R_fixed * (V_ref / V_out - 1.0)
    r_ntc = r_fixed * ((v_ref / np.maximum(v_out, 1e-6)) - 1.0)
    r_ntc = np.maximum(r_ntc, 1.0)

    t0_kelvin = t0_celsius + 273.15
    inv_t = (1.0 / t0_kelvin) + (1.0 / max(beta, 1.0)) * np.log(r_ntc / max(r0, 1e-6))
    temp_kelvin = 1.0 / np.maximum(inv_t, 1e-6)
    return temp_kelvin - 273.15


def inspect_hardware_diagnostics(
    df: pd.DataFrame,
    adc_max: float = 4095.0,
    v_ref: float = 3.3,
    r_fixed: float = 10000.0,
    duty_cycle: float = 1.0,
) -> Dict[str, Any]:
    """
    Checks for ADC rail clipping, timestamp non-monotonicity/jitter, and estimates
    worst-case NTC self-heating power dissipation (I^2 * R).
    """
    diag: Dict[str, Any] = {
        "n_samples": int(len(df)),
        "adc_clipped_low": 0,
        "adc_clipped_high": 0,
        "non_monotonic_ts": 0,
        "duplicate_ts": 0,
        "dt_median_sec": None,
        "dt_jitter_std_sec": None,
        "max_self_heating_mw": None,
    }

    if "timestamp_ms" in df.columns and len(df) >= 2:
        ts = df["timestamp_ms"].to_numpy(dtype=np.int64)
        diffs = np.diff(ts)
        diag["non_monotonic_ts"] = int(np.sum(diffs < 0))
        diag["duplicate_ts"] = int(np.sum(diffs == 0))
        dt_sec = np.diff(np.sort(ts)) / 1000.0
        diag["dt_median_sec"] = float(np.median(dt_sec))
        diag["dt_jitter_std_sec"] = float(np.std(dt_sec))

    if "raw_adc" in df.columns and len(df) > 0:
        raw = df["raw_adc"].to_numpy(dtype=np.float64)
        diag["adc_clipped_low"] = int(np.sum(raw <= 0.0))
        diag["adc_clipped_high"] = int(np.sum(raw >= adc_max))

        adc_clipped = np.clip(raw, 1.0, max(adc_max - 1.0, 2.0))
        v_out = (adc_clipped / max(adc_max, 1.0)) * v_ref
        r_ntc = np.maximum(r_fixed * ((v_ref / np.maximum(v_out, 1e-6)) - 1.0), 1.0)
        current = v_ref / (r_ntc + r_fixed)
        power_mw = (current ** 2) * r_ntc * 1000.0 * float(np.clip(duty_cycle, 0.0, 1.0))
        diag["max_self_heating_mw"] = float(np.max(power_mw))

    return diag


def _log_hardware_warnings(diag: Dict[str, Any], source_name: str) -> None:
    """Emits clear warnings when hardware or timing anomalies are detected."""
    if diag["adc_clipped_low"] or diag["adc_clipped_high"]:
        LOGGER.warning(
            "%s: ADC rail clipping detected (low=%d, high=%d).",
            source_name,
            diag["adc_clipped_low"],
            diag["adc_clipped_high"],
        )
    if diag["non_monotonic_ts"] or diag["duplicate_ts"]:
        LOGGER.warning(
            "%s: timestamp anomalies detected (non_monotonic=%d, duplicates=%d).",
            source_name,
            diag["non_monotonic_ts"],
            diag["duplicate_ts"],
        )


def _clean_and_regularize_series(
    t_sec: np.ndarray,
    values: np.ndarray,
    target_dt: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Sorts samples by timestamp, averages exact duplicate timestamps, and
    linearly interpolates onto a uniform time grid when jitter or gaps are present.
    """
    if len(t_sec) == 0:
        return t_sec, values, 0.05

    order = np.argsort(t_sec, kind="mergesort")
    t_sorted = t_sec[order]
    v_sorted = values[order]

    valid = np.isfinite(t_sorted) & np.isfinite(v_sorted)
    t_sorted = t_sorted[valid]
    v_sorted = v_sorted[valid]
    if len(t_sorted) < 2:
        return t_sorted, v_sorted, 0.05

    unique_t, inv_idx = np.unique(t_sorted, return_inverse=True)
    if len(unique_t) < len(t_sorted):
        v_acc = np.zeros_like(unique_t, dtype=np.float64)
        counts = np.zeros_like(unique_t, dtype=np.float64)
        np.add.at(v_acc, inv_idx, v_sorted)
        np.add.at(counts, inv_idx, 1.0)
        v_unique = v_acc / np.maximum(counts, 1.0)
    else:
        v_unique = v_sorted

    if len(unique_t) < 2:
        return unique_t, v_unique, 0.05

    diffs = np.diff(unique_t)
    pos_diffs = diffs[diffs > 1e-6]
    dt_med = float(np.median(pos_diffs)) if len(pos_diffs) > 0 else 0.05
    dt_use = float(target_dt) if target_dt is not None and target_dt > 1e-6 else dt_med
    dt_use = max(dt_use, 1e-3)

    has_jitter = len(pos_diffs) > 0 and (np.max(np.abs(pos_diffs - dt_med)) > 0.05 * dt_med)
    duration = float(unique_t[-1] - unique_t[0])
    if has_jitter and duration > dt_use and duration / dt_use <= 200_000:
        n_grid = max(int(round(duration / dt_use)) + 1, 2)
        t_grid = unique_t[0] + np.arange(n_grid, dtype=np.float64) * dt_use
        v_grid = np.interp(t_grid, unique_t, v_unique)
        return t_grid, v_grid, dt_use

    return unique_t, v_unique, dt_use


def _align_nearest_labels(t_sorted_raw: np.ndarray, lbl_sorted_raw: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    """Aligns labels from raw timestamps to target timestamps via nearest neighbor."""
    if len(t_sorted_raw) == 0:
        return np.array([""] * len(t_target), dtype=object)
    if len(t_sorted_raw) == 1:
        return np.repeat(lbl_sorted_raw[0], len(t_target))

    idx_right = np.searchsorted(t_sorted_raw, t_target, side="left")
    idx_left = np.clip(idx_right - 1, 0, len(t_sorted_raw) - 1)
    idx_right_c = np.clip(idx_right, 0, len(t_sorted_raw) - 1)

    dist_left = np.abs(t_target - t_sorted_raw[idx_left])
    dist_right = np.abs(t_sorted_raw[idx_right_c] - t_target)
    choose_right = dist_right < dist_left
    nearest_idx = np.where(choose_right, idx_right_c, idx_left)
    return lbl_sorted_raw[nearest_idx]


def _select_window_label(win_labels: np.ndarray) -> int:
    """
    Selects a representative label for a window:
    - If a transient ('heating' or 'cooling') occupies at least 25% of valid samples,
      pick the dominant transient so short pulses aren't drowned out by background.
    - Otherwise fall back to majority vote across valid labels.
    """
    valid_lbls = [str(lbl) for lbl in win_labels if str(lbl) in LABEL_TO_ID]
    if not valid_lbls:
        return -1

    total = len(valid_lbls)
    for trans in ("heating", "cooling"):
        cnt = sum(1 for lbl in valid_lbls if lbl == trans)
        if cnt >= 0.25 * total and cnt >= 2:
            return LABEL_TO_ID[trans]

    unique_lbls, counts = np.unique(valid_lbls, return_counts=True)
    maj_lbl = str(unique_lbls[int(np.argmax(counts))])
    return LABEL_TO_ID[maj_lbl]


def _step_model(t: np.ndarray, t_start: float, t_step: float, tau: float, t_base: float) -> np.ndarray:
    """First-order thermal step response model starting at t_start."""
    response = np.zeros_like(t, dtype=np.float64)
    mask = t >= t_start
    response[mask] = t_step * (1.0 - np.exp(-(t[mask] - t_start) / max(tau, 1e-3)))
    return t_base + response


def calibrate_tau(
    step_df: pd.DataFrame,
    adc_max: float = 4095.0,
    v_ref: float = 3.3,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Estimates thermal time constant tau (seconds), baseline noise std (deg C),
    and step-fit diagnostics from a step-response CSV dataframe.
    """
    if "timestamp_ms" not in step_df.columns:
        raise ValueError("Step CSV must include a 'timestamp_ms' column.")

    if "temp_c" in step_df.columns:
        temp_raw = step_df["temp_c"].to_numpy(dtype=np.float64)
    elif "raw_adc" in step_df.columns:
        temp_raw = adc_to_temperature_steinhart_hart(
            step_df["raw_adc"].to_numpy(dtype=np.float64), adc_max=adc_max, v_ref=v_ref
        )
    else:
        raise ValueError("Step CSV must include either 'temp_c' or 'raw_adc'.")

    t_raw = (step_df["timestamp_ms"].to_numpy(dtype=np.float64) - float(step_df["timestamp_ms"].iloc[0])) / 1000.0
    t_sec, temp, _ = _clean_and_regularize_series(t_raw, temp_raw)
    if len(t_sec) < 15:
        raise ValueError("Step CSV must contain at least 15 valid samples.")

    n_init = max(5, int(0.15 * len(temp)))
    t_base_init = float(np.median(temp[:n_init]))
    noise_std = float(np.std(temp[:n_init], ddof=1)) if n_init > 1 else 0.0

    thresh = max(3.0 * noise_std, 0.25)
    deviations = np.abs(temp[n_init:] - t_base_init)
    exceed_mask = deviations > thresh
    if np.any(exceed_mask):
        onset_idx = n_init + int(np.argmax(exceed_mask))
    else:
        onset_idx = n_init

    # Use post-onset peak/tail plateau rather than blindly assuming final sample is asymptotic
    tail_start = max(onset_idx + 1, int(0.80 * len(temp)))
    t_final_est = float(np.median(temp[tail_start:])) if tail_start < len(temp) else float(temp[-1])
    t_step_init = t_final_est - t_base_init
    if abs(t_step_init) < 1e-3:
        t_step_init = 2.0

    t_start_init = float(t_sec[onset_idx])
    p0 = [t_start_init, t_step_init, 2.0, t_base_init]
    bounds = (
        [float(t_sec[0]), -50.0, 0.05, -20.0],
        [float(t_sec[-1]), 50.0, 60.0, 120.0],
    )

    try:
        popt, _ = curve_fit(_step_model, t_sec, temp, p0=p0, bounds=bounds, maxfev=10000)
        tau_est = float(popt[2])
        fitted = _step_model(t_sec, *popt)
        rmse = float(np.sqrt(np.mean((temp - fitted) ** 2)))
        step_amp = float(popt[1])
    except Exception as exc:
        LOGGER.warning("Nonlinear step fit did not converge (%s); using 63.2%% rise-time fallback.", exc)
        step_amp = t_step_init
        target = t_base_init + 0.632 * step_amp
        post_t = t_sec[onset_idx:]
        post_temp = temp[onset_idx:]
        if abs(step_amp) > 1e-3 and len(post_temp) > 0:
            idx_63 = int(np.argmin(np.abs(post_temp - target)))
            tau_est = max(0.1, float(post_t[idx_63] - t_start_init))
        else:
            tau_est = 2.0
        fitted = _step_model(t_sec, t_start_init, step_amp, tau_est, t_base_init)
        rmse = float(np.sqrt(np.mean((temp - fitted) ** 2)))

    diag = {
        "fit_rmse_c": rmse,
        "step_amplitude_c": step_amp,
        "n_samples_clean": float(len(t_sec)),
    }
    return tau_est, noise_std, diag


def _simulate_first_order_response(driver_signal: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Simulates a first-order thermal lag filter on a driving temperature signal."""
    measured = np.empty_like(driver_signal, dtype=np.float64)
    measured[0] = driver_signal[0]
    alpha = 1.0 - np.exp(-dt / max(tau, 1e-4))
    for i in range(1, len(driver_signal)):
        measured[i] = measured[i - 1] + alpha * (driver_signal[i] - measured[i - 1])
    return measured


def generate_synthetic_sessions(
    n_sessions: int = 5,
    events_per_class: int = 12,
    fs_hz: float = 20.0,
    tau_true: float = 2.2,
    include_ref_sensor: bool = False,
    seed: int = 42,
) -> List[pd.DataFrame]:
    """
    Generates realistic synthetic recording sessions containing labeled windows of:
    'baseline', 'drift', 'heating', and 'cooling'.
    """
    rng = np.random.default_rng(seed)
    sessions: List[pd.DataFrame] = []
    dt = 1.0 / fs_hz
    win_samples = int(4.0 * fs_hz)

    for s_idx in range(n_sessions):
        t_parts: List[np.ndarray] = []
        true_driver_parts: List[np.ndarray] = []
        ambient_parts: List[np.ndarray] = []
        label_parts: List[str] = []

        current_t = 0.0
        ambient_base = float(rng.uniform(22.0, 29.0))
        drift_rate = float(rng.uniform(-0.015, 0.015))

        schedule: List[str] = []
        for lbl in LABELS:
            schedule.extend([lbl] * events_per_class)
        rng.shuffle(schedule)

        for lbl in schedule:
            t_local = np.arange(win_samples, dtype=np.float64) * dt
            t_win = current_t + t_local
            ambient_curve = ambient_base + drift_rate * t_win

            if lbl == "baseline":
                driver = ambient_curve + rng.normal(0.0, 0.02, size=win_samples)
            elif lbl == "drift":
                step_drift = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.4, 1.2))
                ramp = step_drift * np.clip((t_local - 0.5) / 2.5, 0.0, 1.0)
                driver = ambient_curve + ramp + rng.normal(0.0, 0.02, size=win_samples)
            elif lbl == "heating":
                amp = float(rng.uniform(1.8, 5.5))
                start_t = float(rng.uniform(0.6, 1.4))
                dur_t = float(rng.uniform(1.0, 2.0))
                pulse = np.where((t_local >= start_t) & (t_local <= start_t + dur_t), amp, 0.0)
                driver = ambient_curve + pulse + rng.normal(0.0, 0.02, size=win_samples)
            elif lbl == "cooling":
                amp = -float(rng.uniform(1.8, 5.5))
                start_t = float(rng.uniform(0.6, 1.4))
                dur_t = float(rng.uniform(1.0, 2.0))
                pulse = np.where((t_local >= start_t) & (t_local <= start_t + dur_t), amp, 0.0)
                driver = ambient_curve + pulse + rng.normal(0.0, 0.02, size=win_samples)
            else:
                raise ValueError(f"Unknown label: {lbl}")

            t_parts.append(t_win)
            true_driver_parts.append(driver)
            ambient_parts.append(ambient_curve)
            label_parts.extend([lbl] * win_samples)
            current_t = float(t_win[-1] + dt)
            ambient_base = float(driver[-1])

        t_all = np.concatenate(t_parts)
        driver_all = np.concatenate(true_driver_parts)
        ambient_all = np.concatenate(ambient_parts)
        lagged_temp = _simulate_first_order_response(driver_all, dt=dt, tau=tau_true)
        noisy_temp = lagged_temp + rng.normal(0.0, 0.06, size=len(lagged_temp))

        data: Dict[str, Any] = {
            "timestamp_ms": np.round(t_all * 1000.0).astype(np.int64),
            "temp_c": noisy_temp,
            "label": label_parts,
        }
        if include_ref_sensor:
            ref_lagged = _simulate_first_order_response(ambient_all, dt=dt, tau=tau_true)
            data["ref_temp_c"] = ref_lagged + rng.normal(0.0, 0.04, size=len(ref_lagged))

        df = pd.DataFrame(data)
        df.attrs["session_id"] = f"synthetic_session_{s_idx + 1}"
        sessions.append(df)

    return sessions


def _ensure_temperature_columns(
    df: pd.DataFrame,
    adc_max: float = 4095.0,
    v_ref: float = 3.3,
    source_name: str = "dataframe",
) -> pd.DataFrame:
    """Normalizes temp_c and optional ref_temp_c from raw columns if needed."""
    out = df.copy()
    if "timestamp_ms" not in out.columns:
        raise ValueError(f"'{source_name}' must contain a 'timestamp_ms' column.")

    if "temp_c" not in out.columns:
        if "raw_adc" in out.columns:
            out["temp_c"] = adc_to_temperature_steinhart_hart(
                out["raw_adc"].to_numpy(dtype=np.float64), adc_max=adc_max, v_ref=v_ref
            )
        else:
            raise ValueError(f"'{source_name}' must contain 'temp_c' or 'raw_adc'.")

    if "ref_temp_c" not in out.columns and "ref_raw_adc" in out.columns:
        out["ref_temp_c"] = adc_to_temperature_steinhart_hart(
            out["ref_raw_adc"].to_numpy(dtype=np.float64), adc_max=adc_max, v_ref=v_ref
        )

    out = out.sort_values("timestamp_ms", kind="mergesort").reset_index(drop=True)
    return out


def load_session_csvs(
    data_dir: str,
    adc_max: float = 4095.0,
    v_ref: float = 3.3,
    duty_cycle: float = 1.0,
) -> List[pd.DataFrame]:
    """Loads and validates session CSV files from data_dir."""
    pattern = os.path.join(data_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'.")

    sessions: List[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path)
        if "label" not in df.columns:
            raise ValueError(f"File '{path}' must contain a 'label' column.")
        df = _ensure_temperature_columns(df, adc_max=adc_max, v_ref=v_ref, source_name=path)

        diag = inspect_hardware_diagnostics(df, adc_max=adc_max, v_ref=v_ref, duty_cycle=duty_cycle)
        _log_hardware_warnings(diag, os.path.basename(path))

        raw_len = len(df)
        valid_mask = df["label"].isin(LABELS)
        dropped = int(raw_len - np.sum(valid_mask))
        if dropped > 0:
            unknown_vals = sorted(set(df.loc[~valid_mask, "label"].astype(str)))
            LOGGER.warning(
                "%s: dropped %d row(s) with unrecognized labels %s.",
                os.path.basename(path),
                dropped,
                unknown_vals,
            )

        df = df.loc[valid_mask].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"File '{path}' contains zero rows with valid labels {LABELS}.")

        df.attrs["session_id"] = os.path.basename(path)
        sessions.append(df)

    return sessions


def _signed_peak_excursion(series: np.ndarray) -> float:
    """
    Returns the signed peak excursion relative to the window baseline (median),
    preserving polarity without distortion from a near-zero mean slope.
    """
    if len(series) == 0:
        return 0.0
    baseline = float(np.median(series))
    deltas = series - baseline
    idx_max_abs = int(np.argmax(np.abs(deltas)))
    return float(deltas[idx_max_abs])


def _compute_signal_derivatives_and_comp(
    temp: np.ndarray,
    dt: float,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes smoothed first/second derivatives, response-lag compensated temperature
    T_hat(t) = T_meas(t) + tau * dT/dt, smoothed compensated derivative,
    moving-average temperature, and slow drift residual.
    """
    n_samples = len(temp)
    if n_samples < 3:
        zeros = np.zeros(n_samples, dtype=np.float64)
        return zeros, zeros, temp.copy(), zeros, temp.copy(), zeros

    fs = 1.0 / max(dt, 1e-3)
    sg_win = _odd_window_le(max(5, int(round(1.1 * fs))), n_samples)
    polyorder = min(3, sg_win - 1)

    d1 = savgol_filter(temp, window_length=sg_win, polyorder=polyorder, deriv=1, delta=dt)
    d2 = savgol_filter(temp, window_length=sg_win, polyorder=polyorder, deriv=2, delta=dt)

    # Response-lag compensation with post-compensation smoothing to curb derivative noise
    t_comp_raw = temp + tau * d1
    comp_win = _odd_window_le(min(sg_win, max(5, int(round(0.8 * fs)) | 1)), n_samples)
    t_comp = savgol_filter(t_comp_raw, window_length=comp_win, polyorder=min(2, max(1, comp_win - 1)))
    d1_comp = savgol_filter(
        t_comp,
        window_length=comp_win,
        polyorder=min(2, max(1, comp_win - 1)),
        deriv=1,
        delta=dt,
    )

    ma_win = max(1, min(n_samples, max(3, int(round(0.8 * fs)))))
    ma_temp = pd.Series(temp).rolling(window=ma_win, min_periods=1, center=True).mean().to_numpy()

    slow_win = max(1, min(n_samples, max(11, int(round(3.0 * fs)))))
    slow_base = pd.Series(temp).rolling(window=slow_win, min_periods=1, center=True).median().to_numpy()
    drift_residual = t_comp - slow_base

    return d1, d2, t_comp, d1_comp, ma_temp, drift_residual


def extract_window_features(
    session_df: pd.DataFrame,
    tau: float,
    window_sec: float = 4.0,
    step_sec: float = 2.0,
    require_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts sliding-window features from a session dataframe.
    Returns:
      X_raw:       shape (N, 2) -> [temp_mean, temp_delta]
      X_ma:        shape (N, 3) -> [ma_temp, ma_slope, ma_range]
      X_resp:      shape (N, 6 or 8) -> RESP_FEATURE_NAMES (+ DIFF_FEATURE_NAMES if ref_temp_c present)
      y:           shape (N,) integer class labels (-1 when require_labels=False and label absent)
      win_end_sec: shape (N,) window end timestamps in seconds relative to session start
    """
    t_raw = (session_df["timestamp_ms"].to_numpy(dtype=np.float64) - float(session_df["timestamp_ms"].iloc[0])) / 1000.0
    temp_raw = session_df["temp_c"].to_numpy(dtype=np.float64)
    t_sec, temp, dt = _clean_and_regularize_series(t_raw, temp_raw)

    has_ref = "ref_temp_c" in session_df.columns
    if has_ref:
        ref_raw = session_df["ref_temp_c"].to_numpy(dtype=np.float64)
        _, ref_temp, _ = _clean_and_regularize_series(t_raw, ref_raw, target_dt=dt)
        if len(ref_temp) == len(temp):
            diff_temp = temp - ref_temp
        else:
            has_ref = False
            diff_temp = np.zeros_like(temp)
    else:
        diff_temp = np.zeros_like(temp)

    n_resp_cols = len(RESP_FEATURE_NAMES) + (len(DIFF_FEATURE_NAMES) if has_ref else 0)
    if len(t_sec) < 10:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, n_resp_cols), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
        )

    has_labels = "label" in session_df.columns
    if has_labels:
        raw_labels = session_df["label"].astype(str).to_numpy()
        order = np.argsort(t_raw, kind="mergesort")
        t_sorted_raw = t_raw[order]
        lbl_sorted_raw = raw_labels[order]
        labels_aligned = _align_nearest_labels(t_sorted_raw, lbl_sorted_raw, t_sec)
    else:
        labels_aligned = np.array([""] * len(t_sec), dtype=object)

    d1, d2, t_comp, d1_comp, _, drift_residual = _compute_signal_derivatives_and_comp(temp, dt=dt, tau=tau)

    if has_ref:
        diff_d1, _, _, _, _, _ = _compute_signal_derivatives_and_comp(diff_temp, dt=dt, tau=tau)
    else:
        diff_d1 = np.zeros_like(temp)

    win_samples = max(4, int(round(window_sec / dt)))
    step_samples = max(1, int(round(step_sec / dt)))

    raw_rows: List[List[float]] = []
    ma_rows: List[List[float]] = []
    resp_rows: List[List[float]] = []
    y_rows: List[int] = []
    end_times: List[float] = []

    for start in range(0, len(temp) - win_samples + 1, step_samples):
        end = start + win_samples
        if has_labels:
            y_val = _select_window_label(labels_aligned[start:end])
            if y_val < 0 and require_labels:
                continue
        else:
            if require_labels:
                continue
            y_val = -1

        t_w = temp[start:end]
        d1_w = d1[start:end]
        d2_w = d2[start:end]
        d1_c_w = d1_comp[start:end]
        t_c_w = t_comp[start:end]
        res_w = drift_residual[start:end]

        ma_w_local = pd.Series(t_w).rolling(window=max(3, min(len(t_w), 5)), min_periods=1, center=True).mean().to_numpy()

        raw_rows.append([float(np.mean(t_w)), float(t_w[-1] - t_w[0])])
        ma_rows.append([
            float(np.mean(ma_w_local)),
            float((ma_w_local[-1] - ma_w_local[0]) / max(window_sec, 1e-3)),
            float(np.ptp(ma_w_local)),
        ])

        resp_vec = [
            float(np.mean(t_w)),
            float(d1_w[int(np.argmax(np.abs(d1_w)))]),
            float(d2_w[int(np.argmax(np.abs(d2_w)))]),
            float(d1_c_w[int(np.argmax(np.abs(d1_c_w)))]),
            _signed_peak_excursion(t_c_w),
            float(res_w[int(np.argmax(np.abs(res_w)))]),
        ]
        if has_ref:
            diff_w = diff_temp[start:end]
            diff_d1_w = diff_d1[start:end]
            resp_vec.extend([
                float(np.mean(diff_w)),
                float(diff_d1_w[int(np.argmax(np.abs(diff_d1_w)))]),
            ])

        resp_rows.append(resp_vec)
        y_rows.append(y_val)
        end_times.append(float(t_sec[end - 1]))

    return (
        np.asarray(raw_rows, dtype=np.float64),
        np.asarray(ma_rows, dtype=np.float64),
        np.asarray(resp_rows, dtype=np.float64),
        np.asarray(y_rows, dtype=np.int64),
        np.asarray(end_times, dtype=np.float64),
    )


def _compute_false_alarms_per_hour(y_true: np.ndarray, y_pred: np.ndarray, step_sec: float) -> float:
    """
    Computes false alarms per hour where a non-transient window ('baseline' or 'drift')
    is predicted as a transient ('heating' or 'cooling').
    """
    if len(y_true) == 0:
        return 0.0
    transient_ids = {LABEL_TO_ID["heating"], LABEL_TO_ID["cooling"]}
    non_transient_mask = ~np.isin(y_true, list(transient_ids))
    false_positives = int(np.sum(non_transient_mask & np.isin(y_pred, list(transient_ids))))
    total_hours = max((len(y_true) * step_sec) / 3600.0, 1e-6)
    return false_positives / total_hours


def _compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Computes precision, recall, and F1 per class label."""
    label_ids = list(range(len(LABELS)))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, average=None, zero_division=0
    )
    per_class: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(LABELS):
        per_class[name] = {
            "precision": float(prec[idx]),
            "recall": float(rec[idx]),
            "f1": float(f1[idx]),
        }
    return per_class


def evaluate_loso(
    sessions: List[pd.DataFrame],
    tau: float,
    window_sec: float = 4.0,
    step_sec: float = 2.0,
    n_estimators: int = 60,
    max_depth: Optional[int] = 6,
) -> Dict[str, Dict[str, Any]]:
    """
    Runs Leave-One-Session-Out (LOSO) cross-validation across three pipelines:
      1. Raw signal classifier
      2. Moving-average smoothing classifier
      3. Response-aware compensated derivative classifier
    """
    if len(sessions) < 2:
        raise ValueError(
            f"Leave-One-Session-Out evaluation requires at least 2 sessions, got {len(sessions)}."
        )

    per_session = [
        extract_window_features(s, tau=tau, window_sec=window_sec, step_sec=step_sec, require_labels=True)
        for s in sessions
    ]

    pipelines = {
        "1_Raw_Signal": 0,
        "2_Moving_Average": 1,
        "3_Response_Aware": 2,
    }

    results: Dict[str, Dict[str, Any]] = {}

    for pipe_name, feat_idx in pipelines.items():
        all_true: List[np.ndarray] = []
        all_pred: List[np.ndarray] = []
        session_f1s: List[float] = []
        session_fah: List[float] = []

        for val_i in range(len(per_session)):
            train_X_list: List[np.ndarray] = []
            train_y_list: List[np.ndarray] = []
            for tr_i in range(len(per_session)):
                if tr_i == val_i:
                    continue
                feats = per_session[tr_i][feat_idx]
                y_tr = per_session[tr_i][3]
                if len(y_tr) > 0:
                    train_X_list.append(feats)
                    train_y_list.append(y_tr)

            val_X = per_session[val_i][feat_idx]
            val_y = per_session[val_i][3]
            if len(val_y) == 0 or not train_X_list:
                continue

            X_train = np.vstack(train_X_list)
            y_train = np.concatenate(train_y_list)

            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42 + val_i,
                class_weight="balanced",
            )
            clf.fit(X_train, y_train)
            preds = clf.predict(val_X)

            all_true.append(val_y)
            all_pred.append(preds)
            session_f1s.append(float(f1_score(val_y, preds, average="macro", zero_division=0)))
            session_fah.append(_compute_false_alarms_per_hour(val_y, preds, step_sec=step_sec))

        if not all_true:
            raise ValueError(f"No valid train/validation folds could be evaluated for '{pipe_name}'.")

        y_true_cat = np.concatenate(all_true)
        y_pred_cat = np.concatenate(all_pred)

        results[pipe_name] = {
            "accuracy": float(accuracy_score(y_true_cat, y_pred_cat)),
            "macro_f1": float(f1_score(y_true_cat, y_pred_cat, average="macro", zero_division=0)),
            "mean_session_f1": float(np.mean(session_f1s)) if session_f1s else 0.0,
            "false_alarms_per_hr": float(np.mean(session_fah)) if session_fah else 0.0,
            "per_class": _compute_per_class_metrics(y_true_cat, y_pred_cat),
            "_y_true": y_true_cat,
            "_y_pred": y_pred_cat,
        }

    return results


def fit_and_save_model(
    sessions: List[pd.DataFrame],
    model_path: str,
    tau: float,
    window_sec: float = 4.0,
    step_sec: float = 2.0,
    n_estimators: int = 80,
    max_depth: Optional[int] = 7,
    seed: int = 42,
) -> Dict[str, Any]:
    """Trains a Response-Aware classifier across all provided sessions and saves a joblib bundle."""
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    has_ref_any = False

    for s in sessions:
        _, _, x_resp, y, _ = extract_window_features(
            s, tau=tau, window_sec=window_sec, step_sec=step_sec, require_labels=True
        )
        if len(y) > 0:
            X_list.append(x_resp)
            y_list.append(y)
            if x_resp.shape[1] > len(RESP_FEATURE_NAMES):
                has_ref_any = True

    if not X_list:
        raise ValueError("No labeled training windows found across provided sessions.")

    expected_cols = max(arr.shape[1] for arr in X_list)
    aligned_X = [
        arr if arr.shape[1] == expected_cols else np.hstack([arr, np.zeros((len(arr), expected_cols - arr.shape[1]))])
        for arr in X_list
    ]
    X_train = np.vstack(aligned_X)
    y_train = np.concatenate(y_list)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    feature_names = list(RESP_FEATURE_NAMES) + (
        list(DIFF_FEATURE_NAMES) if expected_cols > len(RESP_FEATURE_NAMES) else []
    )
    bundle: Dict[str, Any] = {
        "model": clf,
        "tau": float(tau),
        "window_sec": float(window_sec),
        "step_sec": float(step_sec),
        "labels": LABELS,
        "feature_names": feature_names,
        "has_ref_sensor": has_ref_any,
    }

    out_dir = os.path.dirname(os.path.abspath(model_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    joblib.dump(bundle, model_path)

    return {
        "model_path": model_path,
        "n_training_windows": int(len(y_train)),
        "n_features": int(X_train.shape[1]),
        "tau": float(tau),
    }


def predict_session_dataframe(
    df: pd.DataFrame,
    bundle: Dict[str, Any],
    adc_max: float = 4095.0,
    v_ref: float = 3.3,
) -> pd.DataFrame:
    """Runs sliding-window inference on a session DataFrame using a saved model bundle."""
    norm_df = _ensure_temperature_columns(df, adc_max=adc_max, v_ref=v_ref, source_name="input_csv")
    _log_hardware_warnings(
        inspect_hardware_diagnostics(norm_df, adc_max=adc_max, v_ref=v_ref),
        "input_csv",
    )

    tau = float(bundle.get("tau", 2.2))
    window_sec = float(bundle.get("window_sec", 4.0))
    step_sec = float(bundle.get("step_sec", 2.0))
    clf: RandomForestClassifier = bundle["model"]

    _, _, x_resp, y_true, end_times = extract_window_features(
        norm_df, tau=tau, window_sec=window_sec, step_sec=step_sec, require_labels=False
    )
    if len(x_resp) == 0:
        return pd.DataFrame(columns=["window_end_sec", "pred_label", "confidence"])

    expected_n = getattr(clf, "n_features_in_", x_resp.shape[1])
    if x_resp.shape[1] < expected_n:
        x_resp = np.hstack([x_resp, np.zeros((len(x_resp), expected_n - x_resp.shape[1]))])
    elif x_resp.shape[1] > expected_n:
        x_resp = x_resp[:, :expected_n]

    pred_ids = clf.predict(x_resp)
    probs = clf.predict_proba(x_resp)
    confidences = np.max(probs, axis=1)

    rows: List[Dict[str, Any]] = []
    for i in range(len(pred_ids)):
        row: Dict[str, Any] = {
            "window_end_sec": round(float(end_times[i]), 3),
            "pred_label": ID_TO_LABEL[int(pred_ids[i])],
            "confidence": round(float(confidences[i]), 4),
        }
        if i < len(y_true) and int(y_true[i]) in ID_TO_LABEL:
            row["true_label"] = ID_TO_LABEL[int(y_true[i])]
        rows.append(row)

    return pd.DataFrame(rows)


class StreamingThermalDetector:
    """
    Stateful rolling-buffer detector for live telemetry streams with debounce/hysteresis
    so isolated single-window flickers do not trigger false state transitions.
    """

    def __init__(
        self,
        bundle: Dict[str, Any],
        debounce_windows: int = 2,
        min_confidence: float = 0.55,
        adc_max: float = 4095.0,
        v_ref: float = 3.3,
    ) -> None:
        self.bundle = bundle
        self.clf: RandomForestClassifier = bundle["model"]
        self.tau = float(bundle.get("tau", 2.2))
        self.window_sec = float(bundle.get("window_sec", 4.0))
        self.step_sec = float(bundle.get("step_sec", 2.0))
        self.debounce_windows = max(1, int(debounce_windows))
        self.min_confidence = float(min_confidence)
        self.adc_max = float(adc_max)
        self.v_ref = float(v_ref)

        self._t0_sec: Optional[float] = None
        self._rows: List[Dict[str, float]] = []
        self._last_eval_t: Optional[float] = None
        self._candidate_label: Optional[str] = None
        self._candidate_count: int = 0
        self._confirmed_state: str = "baseline"

    def push_sample(
        self,
        timestamp_ms: float,
        temp_c: Optional[float] = None,
        raw_adc: Optional[float] = None,
        ref_temp_c: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Appends a telemetry sample and emits a state event dict whenever a new step
        window is evaluated and debounce criteria are met.
        """
        if temp_c is None:
            if raw_adc is None:
                raise ValueError("Either temp_c or raw_adc must be provided.")
            if raw_adc <= 0.0 or raw_adc >= self.adc_max:
                LOGGER.warning("Streaming raw_adc=%.1f near/at rail (max=%.1f).", raw_adc, self.adc_max)
            temp_c = float(
                adc_to_temperature_steinhart_hart(
                    np.array([raw_adc], dtype=np.float64), adc_max=self.adc_max, v_ref=self.v_ref
                )[0]
            )

        t_abs_sec = float(timestamp_ms) / 1000.0
        if self._t0_sec is None:
            self._t0_sec = t_abs_sec
        t_rel_sec = t_abs_sec - self._t0_sec

        entry: Dict[str, float] = {
            "timestamp_ms": float(timestamp_ms),
            "t_sec": t_rel_sec,
            "temp_c": float(temp_c),
        }
        if ref_temp_c is not None:
            entry["ref_temp_c"] = float(ref_temp_c)
        self._rows.append(entry)

        keep_after = t_rel_sec - max(self.window_sec * 2.5, 10.0)
        self._rows = [r for r in self._rows if r["t_sec"] >= keep_after]

        if self._last_eval_t is not None and (t_rel_sec - self._last_eval_t) < self.step_sec:
            return None

        win_rows = [r for r in self._rows if r["t_sec"] >= (t_rel_sec - self.window_sec)]
        if len(win_rows) < 10 or (win_rows[-1]["t_sec"] - win_rows[0]["t_sec"]) < 0.5 * self.window_sec:
            return None

        self._last_eval_t = t_rel_sec
        win_df = pd.DataFrame(win_rows)
        _, _, x_resp, _, _ = extract_window_features(
            win_df,
            tau=self.tau,
            window_sec=max(self.window_sec, 0.5),
            step_sec=max(self.window_sec, 0.5),
            require_labels=False,
        )
        if len(x_resp) == 0:
            return None

        feat_vec = x_resp[-1:].copy()
        expected_n = getattr(self.clf, "n_features_in_", feat_vec.shape[1])
        if feat_vec.shape[1] < expected_n:
            feat_vec = np.hstack([feat_vec, np.zeros((1, expected_n - feat_vec.shape[1]))])
        elif feat_vec.shape[1] > expected_n:
            feat_vec = feat_vec[:, :expected_n]

        pred_id = int(self.clf.predict(feat_vec)[0])
        probs = self.clf.predict_proba(feat_vec)[0]
        conf = float(np.max(probs))
        raw_label = ID_TO_LABEL[pred_id] if conf >= self.min_confidence else self._confirmed_state

        if raw_label == self._candidate_label:
            self._candidate_count += 1
        else:
            self._candidate_label = raw_label
            self._candidate_count = 1

        transitioned = False
        if self._candidate_count >= self.debounce_windows and raw_label != self._confirmed_state:
            prev_state = self._confirmed_state
            self._confirmed_state = raw_label
            transitioned = True
        else:
            prev_state = self._confirmed_state

        return {
            "timestamp_ms": int(timestamp_ms),
            "temp_c": round(float(temp_c), 3),
            "raw_pred": raw_label,
            "state": self._confirmed_state,
            "prev_state": prev_state,
            "transitioned": transitioned,
            "confidence": round(conf, 4),
        }


def run_stream_mode(
    bundle: Dict[str, Any],
    stream_csv: str = "",
    poll_interval: float = 0.25,
    debounce_windows: int = 2,
    min_confidence: float = 0.55,
    adc_max: float = 4095.0,
    v_ref: float = 3.3,
) -> None:
    """
    Runs live/streaming inference by tailing a CSV file (columns: timestamp_ms, temp_c or raw_adc)
    or reading CSV-formatted lines (`timestamp_ms,temp_c`) from stdin.
    """
    detector = StreamingThermalDetector(
        bundle=bundle,
        debounce_windows=debounce_windows,
        min_confidence=min_confidence,
        adc_max=adc_max,
        v_ref=v_ref,
    )

    if stream_csv:
        LOGGER.info("Streaming from CSV file '%s' (Ctrl+C to stop)...", stream_csv)
        header_parsed = False
        columns: List[str] = []
        file_pos = 0

        try:
            while True:
                if os.path.exists(stream_csv):
                    file_size = os.path.getsize(stream_csv)
                    if file_size < file_pos:
                        file_pos = 0
                        header_parsed = False
                        columns = []

                    with open(stream_csv, "r", encoding="utf-8") as fh:
                        fh.seek(file_pos)
                        lines = fh.readlines()
                        file_pos = fh.tell()

                    for line in lines:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        parts = [p.strip() for p in stripped.split(",")]
                        if not header_parsed:
                            columns = parts
                            header_parsed = True
                            continue
                        row_map = (
                            dict(zip(columns, parts))
                            if columns
                            else {"timestamp_ms": parts[0], "temp_c": parts[1]}
                        )
                        ts_ms = float(row_map["timestamp_ms"])
                        t_c = (
                            float(row_map["temp_c"])
                            if "temp_c" in row_map and row_map["temp_c"] != ""
                            else None
                        )
                        adc_v = (
                            float(row_map["raw_adc"])
                            if "raw_adc" in row_map and row_map["raw_adc"] != ""
                            else None
                        )
                        ref_c = (
                            float(row_map["ref_temp_c"])
                            if "ref_temp_c" in row_map and row_map["ref_temp_c"] != ""
                            else None
                        )
                        event = detector.push_sample(ts_ms, temp_c=t_c, raw_adc=adc_v, ref_temp_c=ref_c)
                        if event and event["transitioned"]:
                            print(
                                f"[ALERT @ {event['timestamp_ms']} ms] "
                                f"{event['prev_state']} -> {event['state']} "
                                f"(T={event['temp_c']} C, conf={event['confidence']:.2f})",
                                flush=True,
                            )
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            LOGGER.info("Stopped streaming session.")
    else:
        LOGGER.info("Reading stream lines '<timestamp_ms>,<temp_c>' from stdin (Ctrl+D/Ctrl+C to stop)...")
        try:
            for line in sys.stdin:
                stripped = line.strip()
                if not stripped or stripped.lower().startswith("timestamp_ms"):
                    continue
                parts = [p.strip() for p in stripped.split(",")]
                if len(parts) < 2:
                    continue
                ts_ms = float(parts[0])
                t_c = float(parts[1])
                event = detector.push_sample(ts_ms, temp_c=t_c)
                if event:
                    tag = "TRANSITION" if event["transitioned"] else "STATUS"
                    print(
                        f"[{tag} @ {event['timestamp_ms']} ms] state={event['state']} "
                        f"(raw={event['raw_pred']}, T={event['temp_c']} C, conf={event['confidence']:.2f})",
                        flush=True,
                    )
        except KeyboardInterrupt:
            LOGGER.info("Stopped stdin stream.")


def maybe_plot_benchmark(results: Dict[str, Dict[str, Any]], plot_out: str) -> None:
    """Generates a diagnostic bar chart + confusion matrix if matplotlib is installed."""
    if not plot_out:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib not installed; skipping plot generation for '%s'.", plot_out)
        return

    pipe_names = list(results.keys())
    macro_f1s = [results[p]["macro_f1"] for p in pipe_names]
    fahs = [results[p]["false_alarms_per_hr"] for p in pipe_names]
    best_cm = confusion_matrix(
        results["3_Response_Aware"]["_y_true"],
        results["3_Response_Aware"]["_y_pred"],
        labels=list(range(len(LABELS))),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(pipe_names, macro_f1s)
    axes[0].set_title("Macro-F1 by Pipeline")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(pipe_names, fahs)
    axes[1].set_title("False Alarms / Hour")
    axes[1].tick_params(axis="x", rotation=20)

    im = axes[2].imshow(best_cm, cmap="Blues")
    axes[2].set_title("3_Response_Aware Confusion Matrix")
    axes[2].set_xticks(range(len(LABELS)))
    axes[2].set_yticks(range(len(LABELS)))
    axes[2].set_xticklabels(LABELS, rotation=25)
    axes[2].set_yticklabels(LABELS)
    for r in range(best_cm.shape[0]):
        for c in range(best_cm.shape[1]):
            axes[2].text(c, r, str(best_cm[r, c]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(plot_out))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(plot_out, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved diagnostic plot to '%s'.", plot_out)


def _serialize_results_for_json(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Removes non-JSON numpy array keys before exporting benchmark metrics."""
    clean: Dict[str, Any] = {}
    for pipe, metrics in results.items():
        clean[pipe] = {k: v for k, v in metrics.items() if not k.startswith("_")}
        if "_y_true" in metrics and "_y_pred" in metrics:
            clean[pipe]["confusion_matrix"] = confusion_matrix(
                metrics["_y_true"], metrics["_y_pred"], labels=list(range(len(LABELS)))
            ).tolist()
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description="Response-Aware NTC Thermal Transient Detection Pipeline (v2.1)")
    parser.add_argument(
        "--mode",
        choices=["benchmark", "calibrate", "train", "predict", "stream"],
        default="benchmark",
    )
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic recording sessions")
    parser.add_argument("--ref-sensor", action="store_true", help="Include reference ambient NTC channel in synthetic data")
    parser.add_argument("--data-dir", type=str, default="", help="Directory containing session CSV files")
    parser.add_argument("--step-csv", type=str, default="", help="Path to step-response calibration CSV")
    parser.add_argument("--input-csv", type=str, default="", help="Path to session CSV for batch --mode predict")
    parser.add_argument("--stream-csv", type=str, default="", help="Path to live CSV file for --mode stream")
    parser.add_argument("--save-model", type=str, default="", help="Path to save trained joblib model bundle")
    parser.add_argument("--load-model", type=str, default="", help="Path to load trained joblib model bundle")
    parser.add_argument("--out-json", type=str, default="", help="Optional path to export benchmark/prediction JSON")
    parser.add_argument("--plot-out", type=str, default="", help="Optional path to save diagnostic PNG chart")

    parser.add_argument("--tau", type=float, default=2.2, help="Nominal/estimated thermal time constant tau (s)")
    parser.add_argument(
        "--tau-true",
        type=float,
        default=None,
        help="True synthetic tau (s) for testing calibration mismatch (defaults to --tau)",
    )
    parser.add_argument("--adc-max", type=float, default=4095.0, help="ADC full-scale count (e.g., 4095 for 12-bit)")
    parser.add_argument("--v-ref", type=float, default=3.3, help="ADC reference voltage in volts")
    parser.add_argument("--window-sec", type=float, default=4.0, help="Sliding window duration in seconds")
    parser.add_argument("--step-sec", type=float, default=2.0, help="Sliding window step size in seconds")
    parser.add_argument("--n-estimators", type=int, default=60, help="Number of Random Forest trees")
    parser.add_argument("--max-depth", type=int, default=6, help="Max depth of Random Forest trees")
    parser.add_argument("--debounce-windows", type=int, default=2, help="Consecutive windows required for stream state change")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Minimum class probability in stream mode")
    parser.add_argument("--poll-interval", type=float, default=0.25, help="File polling interval (s) in stream mode")
    parser.add_argument("--duty-cycle", type=float, default=1.0, help="Sensor excitation duty cycle (0.0-1.0) for self-heating check")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)
    max_depth: Optional[int] = None if args.max_depth <= 0 else args.max_depth

    if args.mode == "calibrate":
        if not args.step_csv:
            raise ValueError("Please provide --step-csv when running --mode calibrate.")
        step_df = pd.read_csv(args.step_csv)
        hw_diag = inspect_hardware_diagnostics(
            step_df, adc_max=args.adc_max, v_ref=args.v_ref, duty_cycle=args.duty_cycle
        )
        _log_hardware_warnings(hw_diag, os.path.basename(args.step_csv))
        tau_est, noise_std, fit_diag = calibrate_tau(step_df, adc_max=args.adc_max, v_ref=args.v_ref)
        print("=== Step-Response Calibration Results ===")
        print(f"Estimated thermal time constant (tau): {tau_est:.3f} s")
        print(f"Baseline noise standard deviation (sigma): {noise_std:.4f} deg C")
        print(f"Step fit RMSE: {fit_diag['fit_rmse_c']:.4f} deg C | Step delta: {fit_diag['step_amplitude_c']:.3f} deg C")
        if hw_diag["max_self_heating_mw"] is not None:
            print(f"Max estimated NTC self-heating power: {hw_diag['max_self_heating_mw']:.4f} mW")
        if args.out_json:
            with open(args.out_json, "w", encoding="utf-8") as fh:
                json.dump(
                    {"tau_est_s": tau_est, "noise_std_c": noise_std, "fit_diag": fit_diag, "hw_diag": hw_diag},
                    fh,
                    indent=2,
                )
        return

    if args.mode == "train":
        if args.synthetic or not args.data_dir:
            tau_sim = args.tau if args.tau_true is None else args.tau_true
            sessions = generate_synthetic_sessions(
                n_sessions=5,
                events_per_class=12,
                tau_true=tau_sim,
                include_ref_sensor=args.ref_sensor,
                seed=args.seed,
            )
        else:
            sessions = load_session_csvs(
                args.data_dir, adc_max=args.adc_max, v_ref=args.v_ref, duty_cycle=args.duty_cycle
            )
        out_path = args.save_model or "thermal_transient_model.joblib"
        summary = fit_and_save_model(
            sessions=sessions,
            model_path=out_path,
            tau=args.tau,
            window_sec=args.window_sec,
            step_sec=args.step_sec,
            n_estimators=max(args.n_estimators, 80),
            max_depth=max_depth,
            seed=args.seed,
        )
        print("=== Model Training Complete ===")
        for k, v in summary.items():
            print(f"{k}: {v}")
        return

    if args.mode == "predict":
        if not args.load_model:
            raise ValueError("Please provide --load-model when running --mode predict.")
        if not args.input_csv:
            raise ValueError("Please provide --input-csv when running --mode predict.")
        bundle = joblib.load(args.load_model)
        in_df = pd.read_csv(args.input_csv)
        preds_df = predict_session_dataframe(in_df, bundle=bundle, adc_max=args.adc_max, v_ref=args.v_ref)
        print(preds_df.to_string(index=False))
        if args.out_json:
            preds_df.to_json(args.out_json, orient="records", indent=2)
        return

    if args.mode == "stream":
        if not args.load_model:
            raise ValueError("Please provide --load-model when running --mode stream.")
        bundle = joblib.load(args.load_model)
        run_stream_mode(
            bundle=bundle,
            stream_csv=args.stream_csv,
            poll_interval=args.poll_interval,
            debounce_windows=args.debounce_windows,
            min_confidence=args.min_confidence,
            adc_max=args.adc_max,
            v_ref=args.v_ref,
        )
        return

    # Default: benchmark mode
    if args.synthetic or not args.data_dir:
        tau_sim = args.tau if args.tau_true is None else args.tau_true
        sessions = generate_synthetic_sessions(
            n_sessions=5,
            events_per_class=12,
            tau_true=tau_sim,
            include_ref_sensor=args.ref_sensor,
            seed=args.seed,
        )
        source_desc = f"Synthetic sessions (tau_true={tau_sim:.2f}s, tau_model={args.tau:.2f}s)"
    else:
        sessions = load_session_csvs(
            args.data_dir, adc_max=args.adc_max, v_ref=args.v_ref, duty_cycle=args.duty_cycle
        )
        source_desc = f"Real CSV sessions from '{args.data_dir}' (tau={args.tau:.2f}s)"

    results = evaluate_loso(
        sessions,
        tau=args.tau,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        n_estimators=args.n_estimators,
        max_depth=max_depth,
    )

    print(f"=== Leave-One-Session-Out Benchmark: {source_desc} ===")
    print(f"{'Pipeline':<22} | {'Accuracy':<8} | {'Macro-F1':<8} | {'False Alarms/Hr':<15}")
    print("-" * 64)
    for name, metrics in results.items():
        print(
            f"{name:<22} | {metrics['accuracy']:<8.3f} | {metrics['macro_f1']:<8.3f} | "
            f"{metrics['false_alarms_per_hr']:<15.2f}"
        )

    best_metrics = results["3_Response_Aware"]
    print("\nPer-Class Metrics (3_Response_Aware):")
    for lbl, cls_m in best_metrics["per_class"].items():
        print(
            f"  {lbl:<10} -> Precision: {cls_m['precision']:.3f} | "
            f"Recall: {cls_m['recall']:.3f} | F1: {cls_m['f1']:.3f}"
        )

    cm = confusion_matrix(best_metrics["_y_true"], best_metrics["_y_pred"], labels=list(range(len(LABELS))))
    print("\nConfusion Matrix (3_Response_Aware) [Rows=True, Cols=Pred]:")
    print("Classes:", LABELS)
    print(cm)

    if args.plot_out:
        maybe_plot_benchmark(results, args.plot_out)

    if args.out_json:
        payload = {
            "source": source_desc,
            "tau_model": args.tau,
            "tau_true": args.tau_true if args.tau_true is not None else args.tau,
            "results": _serialize_results_for_json(results),
        }
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        LOGGER.info("Exported benchmark JSON report to '%s'.", args.out_json)


if __name__ == "__main__":
    main()
