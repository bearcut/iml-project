"""
Feature extraction pipeline for wearable sensor multivariate time series.
Extracts statistical, spectral (FFT), and kinematic domain features from
250-timestep windows across 23 sensor channels.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats, signal

from iml_assignment.config import (
    SENSOR_CHANNELS,
    BODY_LOCATIONS,
    SAMPLING_RATE_HZ,
    WINDOW_SIZE,
    NUM_CHANNELS,
    SENTINEL_VALUE,
)
from iml_assignment.data import (
    get_sensor_columns,
    clean_sample_waveform,
    clean_dataset,
    ChannelScaler,
)


def extract_window_features(waveform_df: pd.DataFrame) -> pd.Series:
    """
    Extract a comprehensive feature vector from a single (250, 23) cleaned waveform.

    Extracted features per channel:
    - Statistical / Time-Domain:
      - Mean, Standard Deviation, Variance, Min, Max, Range, Median, IQR
      - Skewness, Kurtosis
      - Root Mean Square (RMS)
      - Zero-Crossing Rate (ZCR)
    - Frequency / Spectral-Domain:
      - Spectral Energy (sum of squared FFT magnitudes)
      - Dominant Frequency (peak frequency in spectrum)
      - Spectral Entropy
    - Kinematic Vector Magnitudes (Resultants):
      - Accelerometer magnitudes for Chest, Ankle, Arm: sqrt(x^2 + y^2 + z^2)
      - Gyroscope magnitudes for Ankle, Arm: sqrt(x^2 + y^2 + z^2)
      - Magnetometer magnitudes for Ankle, Arm: sqrt(x^2 + y^2 + z^2)

    Returns:
    --------
    pd.Series
        Named feature vector.
    """
    feats = {}
    time_series = waveform_df.to_numpy(dtype=float)
    n_steps = len(time_series)

    # 1. Per-Channel Features
    for c_idx, ch_name in enumerate(waveform_df.columns):
        col_vals = time_series[:, c_idx]

        # Time-domain statistics
        mean_v = float(np.mean(col_vals))
        std_v = float(np.std(col_vals))
        min_v = float(np.min(col_vals))
        max_v = float(np.max(col_vals))
        q75, q25 = np.percentile(col_vals, [75, 25])
        iqr_v = float(q75 - q25)
        med_v = float(np.median(col_vals))

        feats[f"{ch_name}__mean"] = mean_v
        feats[f"{ch_name}__std"] = std_v
        feats[f"{ch_name}__min"] = min_v
        feats[f"{ch_name}__max"] = max_v
        feats[f"{ch_name}__range"] = max_v - min_v
        feats[f"{ch_name}__median"] = med_v
        feats[f"{ch_name}__iqr"] = iqr_v
        feats[f"{ch_name}__rms"] = float(np.sqrt(np.mean(col_vals ** 2)))

        # Skewness & Kurtosis
        feats[f"{ch_name}__skew"] = float(stats.skew(col_vals)) if std_v > 1e-6 else 0.0
        feats[f"{ch_name}__kurt"] = float(stats.kurtosis(col_vals)) if std_v > 1e-6 else 0.0

        # Zero-crossing rate (around channel mean)
        zero_cross = np.sum(np.diff(col_vals - mean_v > 0) != 0) / (n_steps - 1)
        feats[f"{ch_name}__zcr"] = float(zero_cross)

        # Spectral features (FFT)
        fft_vals = np.fft.rfft(col_vals - mean_v)
        fft_mag = np.abs(fft_vals)
        fft_power = fft_mag ** 2
        freqs = np.fft.rfftfreq(n_steps, d=1.0 / SAMPLING_RATE_HZ)

        feats[f"{ch_name}__spec_energy"] = float(np.sum(fft_power) / len(fft_power))
        dom_idx = int(np.argmax(fft_mag[1:])) + 1 if len(fft_mag) > 1 else 0
        feats[f"{ch_name}__dom_freq"] = float(freqs[dom_idx])

        # Spectral entropy
        norm_power = fft_power / (np.sum(fft_power) + 1e-12)
        spec_entropy = -np.sum(norm_power * np.log2(norm_power + 1e-12))
        feats[f"{ch_name}__spec_entropy"] = float(spec_entropy)

    # 2. Kinematic 3D Vector Magnitudes
    magnitude_triplets = [
        ("chest_acc_mag", ["chest_acc_x", "chest_acc_y", "chest_acc_z"]),
        ("ankle_acc_mag", ["ankle_acc_x", "ankle_acc_y", "ankle_acc_z"]),
        ("ankle_gyro_mag", ["ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z"]),
        ("ankle_mag_mag", ["ankle_mag_x", "ankle_mag_y", "ankle_mag_z"]),
        ("arm_acc_mag", ["arm_acc_x", "arm_acc_y", "arm_acc_z"]),
        ("arm_gyro_mag", ["arm_gyro_x", "arm_gyro_y", "arm_gyro_z"]),
        ("arm_mag_mag", ["arm_mag_x", "arm_mag_y", "arm_mag_z"]),
    ]

    for mag_name, trip_cols in magnitude_triplets:
        if all(c in waveform_df.columns for c in trip_cols):
            coords = waveform_df[trip_cols].to_numpy(dtype=float)
            mag_signal = np.sqrt(np.sum(coords ** 2, axis=1))

            feats[f"{mag_name}__mean"] = float(np.mean(mag_signal))
            feats[f"{mag_name}__std"] = float(np.std(mag_signal))
            feats[f"{mag_name}__max"] = float(np.max(mag_signal))
            feats[f"{mag_name}__min"] = float(np.min(mag_signal))
            feats[f"{mag_name}__iqr"] = float(np.percentile(mag_signal, 75) - np.percentile(mag_signal, 25))

    return pd.Series(feats)


def extract_features_dataset(
    df: pd.DataFrame,
    clean: bool = True,
    impute_method: str = "cubic_spline",
    scaler: Optional[ChannelScaler] = None,
    sentinel: float = SENTINEL_VALUE,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Extract the tabular feature matrix from a full dataset DataFrame (train or test).

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing 5,750 sensor columns, and optionally 'id' and 'target'.
    clean : bool
        If True, cleans missing values and sentinels via `clean_dataset`.
    impute_method : str
        'cubic_spline' or 'linear'.
    scaler : Optional[ChannelScaler]
        Fitted ChannelScaler to apply channel normalization.
    sentinel : float
        Noise sentinel value (default 9999.0).
    verbose : bool
        Whether to print progress.

    Returns:
    --------
    X_features : pd.DataFrame
        DataFrame of extracted features (N rows, ~350 engineered features).
    y : Optional[pd.Series]
        Target labels series if 'target' column is present in df, else None.
    """
    if verbose:
        print(f"Extracting features for {len(df)} samples...")

    # Step 1: Cleaning / Imputation
    if clean:
        if verbose:
            print(f"  - Imputing corrupted readings using {impute_method}...")
        working_df = clean_dataset(df, method=impute_method, sentinel=sentinel)
    else:
        working_df = df.copy()

    # Step 2: Channel Scaling
    if scaler is not None:
        if verbose:
            print(f"  - Normalizing channels using fitted {scaler.method} scaler...")
        working_df = scaler.transform(working_df)

    # Step 3: Extract features row-by-row
    sensor_cols = get_sensor_columns(working_df)
    raw_vals = working_df[sensor_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n_samples = len(raw_vals)
    data_3d = raw_vals.reshape(n_samples, WINDOW_SIZE, NUM_CHANNELS)
    time_seconds = np.arange(WINDOW_SIZE) / SAMPLING_RATE_HZ

    feature_rows = []
    for i in range(n_samples):
        wave_df = pd.DataFrame(data_3d[i], index=time_seconds, columns=SENSOR_CHANNELS)
        row_feats = extract_window_features(wave_df)
        feature_rows.append(row_feats)

    X_features = pd.DataFrame(feature_rows)
    if "id" in df.columns:
        X_features.index = df["id"].values

    y = df["target"].copy() if "target" in df.columns else None

    if verbose:
        print(f"Feature extraction complete! Shape: {X_features.shape}")

    return X_features, y
