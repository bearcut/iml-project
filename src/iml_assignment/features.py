"""
Feature extraction pipeline for wearable sensor multivariate time series.

Extracts statistical, spectral (FFT), and kinematic domain features from
250-timestep windows across 23 sensor channels, transforming noisy high-dimensional
signals into structured tabular representations for gradient boosted trees and classical ML.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
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

    Computes 357 domain-specific features per 5-second window across three categories:

    1. Time-Domain / Statistical Features (per channel):
       - Central tendency & dispersion: Mean, Standard Deviation, Variance, Min, Max, Range, Median, IQR.
       - Shape & Distribution: Skewness (asymmetry), Kurtosis (peakedness / heavy tails).
       - Power & Dynamics: Root Mean Square (RMS), Zero-Crossing Rate (ZCR) around the channel mean.

    2. Frequency-Domain / Spectral Features (per channel via Welch/FFT):
       - Spectral Energy: Total power across discrete Fourier bins.
       - Dominant Frequency: Frequency bin with maximum power (identifies cadence/step frequency).
       - Spectral Entropy: Quantifies disorder / randomness of power distribution across frequency bins.

    3. Kinematic 3D Vector Magnitudes (Resultants):
       - Euclidean norm: sqrt(x² + y² + z²) for 3-axis sensors:
         - Chest Accelerometer Magnitude
         - Ankle Accelerometer Magnitude
         - Ankle Gyroscope Magnitude
         - Ankle Magnetometer Magnitude
         - Arm Accelerometer Magnitude
         - Arm Gyroscope Magnitude
         - Arm Magnetometer Magnitude
       - Summary statistics computed for each resultant vector: Mean, Std, Max, Min, IQR.

    Parameters
    ----------
    waveform_df : pd.DataFrame
        Cleaned (250, 23) DataFrame with time index in seconds and the 23 sensor channels as columns.

    Returns
    -------
    pd.Series
        Named feature vector containing 357 numeric scalar features.

    Examples
    --------
    >>> raw_wave, _ = extract_sample_waveform(train_df, 0)
    >>> clean_wave = clean_sample_waveform(raw_wave, method="cubic_spline")
    >>> feats = extract_window_features(clean_wave)
    >>> print(len(feats))
    357
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

    Pipeline:
    1. Independent sample imputation (zero leakage): Replaces sentinels and missing points
       via `clean_dataset` using the selected method ('cubic_spline' or 'linear').
    2. Optional channel scaling: Applies a fitted `ChannelScaler` (e.g. RobustScaler).
    3. Row-by-row feature calculation: Computes all 357 time, spectral, and kinematic
       features for each sample window.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing 5,750 sensor columns, and optionally 'id' and 'target'.
    clean : bool, default=True
        If True, cleans missing values and sentinels before feature extraction.
    impute_method : str, default="cubic_spline"
        'cubic_spline' or 'linear'.
    scaler : Optional[ChannelScaler], default=None
        Fitted ChannelScaler to apply channel normalization prior to feature computation.
    sentinel : float, default=9999.0
        Noise sentinel value.
    verbose : bool, default=True
        Whether to print extraction progress.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.Series]]
        - X_features: DataFrame of extracted features (N rows, 357 columns), indexed by sample ID.
        - y: Target label Series if 'target' column is present in df, else None.

    Examples
    --------
    >>> X_tr, y_tr = extract_features_dataset(train_df, impute_method="cubic_spline")
    >>> print(X_tr.shape)
    (679, 357)
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


# ─────────────────── Advanced Engineered Features ───────────────────


def _to_3d_tensor(
    data: Union[pd.DataFrame, np.ndarray],
    clean: bool = False,
    impute_method: str = "cubic_spline",
    sentinel: float = SENTINEL_VALUE,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Internal utility to convert arbitrary dataset formats into a standardized (N, 250, 23) tensor.

    Supports:
    - Full DataFrame (N, 5750+) with sensor columns t000_...
    - Single window DataFrame (250, 23)
    - 3D numpy array (N, 250, 23)
    - 2D numpy array (250, 23)
    """
    if isinstance(data, pd.DataFrame):
        sensor_cols = [c for c in data.columns if c.startswith("t") and c != "target"]
        if len(sensor_cols) == WINDOW_SIZE * NUM_CHANNELS:
            df_work = clean_dataset(data, method=impute_method, sentinel=sentinel) if clean else data
            raw_vals = df_work[sensor_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            sample_ids = df_work["id"].astype(str).tolist() if "id" in df_work.columns else [str(i) for i in range(len(df_work))]
            data_3d = raw_vals.reshape(len(df_work), WINDOW_SIZE, NUM_CHANNELS)
            return data_3d, sample_ids, list(SENSOR_CHANNELS)
        elif len(data) == WINDOW_SIZE and len(data.columns) == NUM_CHANNELS:
            raw_vals = data.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            return raw_vals.reshape(1, WINDOW_SIZE, NUM_CHANNELS), ["0"], list(data.columns)
        elif len(data) == WINDOW_SIZE and all(c in data.columns for c in SENSOR_CHANNELS):
            raw_vals = data[SENSOR_CHANNELS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            return raw_vals.reshape(1, WINDOW_SIZE, NUM_CHANNELS), ["0"], list(SENSOR_CHANNELS)
        else:
            raise ValueError(
                f"Unrecognized DataFrame shape {data.shape}. Expected either ({WINDOW_SIZE}, {NUM_CHANNELS}) "
                f"or full dataset with {WINDOW_SIZE * NUM_CHANNELS} sensor columns."
            )
    elif isinstance(data, np.ndarray):
        if data.ndim == 3 and data.shape[1:] == (WINDOW_SIZE, NUM_CHANNELS):
            return data.astype(float), [str(i) for i in range(len(data))], list(SENSOR_CHANNELS)
        elif data.ndim == 2 and data.shape == (WINDOW_SIZE, NUM_CHANNELS):
            return data.reshape(1, WINDOW_SIZE, NUM_CHANNELS).astype(float), ["0"], list(SENSOR_CHANNELS)
        else:
            raise ValueError(f"Expected array of shape (N, {WINDOW_SIZE}, {NUM_CHANNELS}) or ({WINDOW_SIZE}, {NUM_CHANNELS}), got {data.shape}")
    else:
        raise TypeError(f"Expected pd.DataFrame or np.ndarray, got {type(data)}")


def extract_jerk_features(
    data: Union[pd.DataFrame, np.ndarray],
    sensors: Sequence[str] = ("ankle_acc", "chest_acc"),
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
    clean: bool = False,
    impute_method: str = "cubic_spline",
) -> pd.DataFrame:
    """
    Extract L2 Jerk Magnitude features (impact sharpness) on specified 3D accelerometers.

    Physics & Domain Context:
    -------------------------
    Running involves violent, rapid foot-strike collisions, whereas jogging has a cushioned landing
    transition. The rate of change of acceleration (jerk) captures this impulse far better than raw
    acceleration alone:
        j_k(t) = (a_k(t) - a_k(t-1)) / dt = 50 * (a_k(t) - a_k(t-1)),  k in {x, y, z}
        J_res(t) = sqrt(j_x(t)^2 + j_y(t)^2 + j_z(t)^2)

    Features Extracted:
    -------------------
    - {sensor}__jerk_rms: Root Mean Square of resultant jerk magnitude.
    - {sensor}__jerk_p95: 95th percentile of resultant jerk (captures peak strike impacts).
    - {sensor}__jerk_kurt: Excess Kurtosis of jerk (captures sharpness and impulse extremity).

    Parameters
    ----------
    data : Union[pd.DataFrame, np.ndarray]
        Input dataset (train_df, test_df, clean DataFrame) or single window waveform.
    sensors : Sequence[str], default=("ankle_acc", "chest_acc")
        Tri-axial accelerometer sensor prefixes to extract jerk from.
    sampling_rate_hz : float, default=50.0
        Sampling frequency in Hertz (dt = 1 / sampling_rate_hz).
    clean : bool, default=False
        If True, cleans missing values and sentinels before extraction.
    impute_method : str, default="cubic_spline"
        Imputation method if clean=True ('cubic_spline' or 'linear').

    Returns
    -------
    pd.DataFrame
        DataFrame of jerk features indexed by sample ID.

    Examples
    --------
    >>> df_jerk = extract_jerk_features(train_df)
    >>> print(df_jerk.columns.tolist())
    ['ankle_acc__jerk_rms', 'ankle_acc__jerk_p95', 'ankle_acc__jerk_kurt',
     'chest_acc__jerk_rms', 'chest_acc__jerk_p95', 'chest_acc__jerk_kurt']
    """
    data_3d, sample_ids, ch_names = _to_3d_tensor(data, clean=clean, impute_method=impute_method)
    ch2idx = {name: i for i, name in enumerate(ch_names)}
    feats_dict = {}

    for sensor in sensors:
        axis_names = [f"{sensor}_x", f"{sensor}_y", f"{sensor}_z"]
        if not all(ax in ch2idx for ax in axis_names):
            raise KeyError(f"Sensor '{sensor}' requires channels {axis_names}, but available channels are: {ch_names}")

        axis_indices = [ch2idx[ax] for ax in axis_names]
        coords = data_3d[:, :, axis_indices]  # (N, 250, 3)

        # Jerk: (N, 249, 3)
        jerk = np.diff(coords, axis=1) * sampling_rate_hz
        # Resultant L2 norm: (N, 249)
        j_res = np.sqrt(np.sum(jerk ** 2, axis=-1))

        # Statistical features
        rms = np.sqrt(np.mean(j_res ** 2, axis=1))
        p95 = np.percentile(j_res, 95, axis=1)
        kurt = stats.kurtosis(j_res, axis=1, fisher=True)

        feats_dict[f"{sensor}__jerk_rms"] = rms
        feats_dict[f"{sensor}__jerk_p95"] = p95
        feats_dict[f"{sensor}__jerk_kurt"] = kurt

    return pd.DataFrame(feats_dict, index=sample_ids)


def extract_flight_crest_features(
    data: Union[pd.DataFrame, np.ndarray],
    sensors: Sequence[str] = ("ankle_acc", "chest_acc"),
    flight_threshold_ms2: float = 4.9,
    impact_threshold_ms2: float = 24.5,
    clean: bool = False,
    impute_method: str = "cubic_spline",
) -> pd.DataFrame:
    """
    Extract Free-Fall / Flight-Phase Fraction and Crest Factor on specified accelerometers.

    Physics & Domain Context:
    -------------------------
    Running exhibits a pronounced airborne flight phase where all three accelerometer axes
    simultaneously approach 0g (free fall, < 0.5g = 4.9 m/s^2), immediately followed by a
    high-impact landing spike (> 2.5g = 24.5 m/s^2). In contrast, jogging has a continuous
    cushioned foot contact with significantly shorter or nonexistent flight phases.

    Features Extracted:
    -------------------
    - {sensor}__flight_fraction: Fraction of timesteps where resultant acceleration drops below 0.5g (< 4.9 m/s^2).
    - {sensor}__impact_fraction: Fraction of timesteps where resultant acceleration exceeds 2.5g (> 24.5 m/s^2).
    - {sensor}__crest_factor: Ratio of peak-to-RMS magnitude (scale-invariant dynamic metric).

    Parameters
    ----------
    data : Union[pd.DataFrame, np.ndarray]
        Input dataset or waveform tensor.
    sensors : Sequence[str], default=("ankle_acc", "chest_acc")
        Tri-axial accelerometer sensor prefixes.
    flight_threshold_ms2 : float, default=4.9
        Free-fall acceleration threshold in m/s^2 (0.5g).
    impact_threshold_ms2 : float, default=24.5
        High-impact acceleration threshold in m/s^2 (2.5g).
    clean : bool, default=False
        Whether to impute sentinels before feature extraction.
    impute_method : str, default="cubic_spline"
        Imputation method if clean=True.

    Returns
    -------
    pd.DataFrame
        DataFrame of flight fraction and crest factor features indexed by sample ID.

    Examples
    --------
    >>> df_flight = extract_flight_crest_features(train_df)
    >>> print(df_flight.columns.tolist())
    ['ankle_acc__flight_fraction', 'ankle_acc__impact_fraction', 'ankle_acc__crest_factor',
     'chest_acc__flight_fraction', 'chest_acc__impact_fraction', 'chest_acc__crest_factor']
    """
    data_3d, sample_ids, ch_names = _to_3d_tensor(data, clean=clean, impute_method=impute_method)
    ch2idx = {name: i for i, name in enumerate(ch_names)}
    feats_dict = {}

    for sensor in sensors:
        axis_names = [f"{sensor}_x", f"{sensor}_y", f"{sensor}_z"]
        if not all(ax in ch2idx for ax in axis_names):
            raise KeyError(f"Sensor '{sensor}' requires channels {axis_names}, but available channels are: {ch_names}")

        axis_indices = [ch2idx[ax] for ax in axis_names]
        coords = data_3d[:, :, axis_indices]  # (N, 250, 3)
        a_res = np.sqrt(np.sum(coords ** 2, axis=-1))  # (N, 250)

        # Metrics
        f_flight = np.mean(a_res < flight_threshold_ms2, axis=1)
        f_impact = np.mean(a_res > impact_threshold_ms2, axis=1)
        rms = np.sqrt(np.mean(a_res ** 2, axis=1))
        crest_factor = np.max(a_res, axis=1) / (rms + 1e-9)

        feats_dict[f"{sensor}__flight_fraction"] = f_flight
        feats_dict[f"{sensor}__impact_fraction"] = f_impact
        feats_dict[f"{sensor}__crest_factor"] = crest_factor

    return pd.DataFrame(feats_dict, index=sample_ids)


def extract_spectral_band_ratios(
    data: Union[pd.DataFrame, np.ndarray],
    channels: Sequence[str] = ("ankle_acc", "chest_acc"),
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
    include_energies: bool = True,
    clean: bool = False,
    impute_method: str = "cubic_spline",
) -> pd.DataFrame:
    """
    Extract Multi-Band Spectral Power Ratios (FFT Band Energy) for gait and impact analysis.

    Physics & Domain Context:
    -------------------------
    While the dominant stride cadence frequency overlaps between jogging and running (2.4–3.0 Hz),
    running generates substantially stronger higher-order harmonics due to heel-strike shock waves
    propagating up the tibia and spine.

    Default Physiological Bands:
    ----------------------------
    - B1 (Locomotion Fundamental Band): 0.5 – 3.0 Hz
    - B2 (Fast Cadence & First Harmonic Band): 3.0 – 8.0 Hz
    - B3 (Impact Shockwave Band): 8.0 – 20.0 Hz

    Features Extracted:
    -------------------
    - {channel}__band_ratio_b2_b1: Energy ratio E_B2 / E_B1 (cadence harmonic strength).
    - {channel}__band_ratio_b3_b1: Energy ratio E_B3 / E_B1 (impact shockwave intensity).
    - {channel}__energy_b1, b2, b3: Raw band power integrals (if include_energies=True).

    Parameters
    ----------
    data : Union[pd.DataFrame, np.ndarray]
        Input dataset or waveform tensor.
    channels : Sequence[str], default=("ankle_acc", "chest_acc")
        Channels or 3-axis sensor prefixes to analyze. If a 3-axis prefix is given, computes
        spectral power on its resultant 3D magnitude.
    bands : Optional[Dict[str, Tuple[float, float]]], default=None
        Custom frequency bands mapping band name to (f_min, f_max) in Hz.
    sampling_rate_hz : float, default=50.0
        Sampling frequency in Hertz.
    include_energies : bool, default=True
        Whether to include raw band energy features alongside the ratios.
    clean : bool, default=False
        Whether to impute sentinels before extraction.
    impute_method : str, default="cubic_spline"
        Imputation method if clean=True.

    Returns
    -------
    pd.DataFrame
        DataFrame of spectral band ratio features indexed by sample ID.

    Examples
    --------
    >>> df_bands = extract_spectral_band_ratios(train_df)
    >>> print(df_bands.columns.tolist()[:4])
    ['ankle_acc__band_ratio_b2_b1', 'ankle_acc__band_ratio_b3_b1',
     'ankle_acc__energy_b1', 'ankle_acc__energy_b2']
    """
    data_3d, sample_ids, ch_names = _to_3d_tensor(data, clean=clean, impute_method=impute_method)
    ch2idx = {name: i for i, name in enumerate(ch_names)}
    n_samples, n_steps, _ = data_3d.shape

    if bands is None:
        bands = {
            "b1": (0.5, 3.0),
            "b2": (3.0, 8.0),
            "b3": (8.0, 20.0),
        }

    freqs = np.fft.rfftfreq(n_steps, d=1.0 / sampling_rate_hz)
    band_masks = {
        b_name: (freqs >= f_range[0]) & (freqs <= f_range[1])
        for b_name, f_range in bands.items()
    }

    feats_dict = {}

    for ch in channels:
        # Check if ch is a 3-axis sensor prefix
        axis_names = [f"{ch}_x", f"{ch}_y", f"{ch}_z"]
        if all(ax in ch2idx for ax in axis_names):
            coords = data_3d[:, :, [ch2idx[ax] for ax in axis_names]]
            signal_1d = np.sqrt(np.sum(coords ** 2, axis=-1))  # 3D resultant magnitude
            sig_name = ch
        elif ch in ch2idx:
            signal_1d = data_3d[:, :, ch2idx[ch]]
            sig_name = ch
        else:
            raise KeyError(f"Channel or sensor prefix '{ch}' not found in available channels: {ch_names}")

        # Remove DC mean
        centered = signal_1d - np.mean(signal_1d, axis=1, keepdims=True)
        fft_vals = np.fft.rfft(centered, axis=1)
        power_spec = np.abs(fft_vals) ** 2  # (N, n_freqs)

        band_energies = {}
        for b_name, mask in band_masks.items():
            band_energies[b_name] = np.sum(power_spec[:, mask], axis=1)

        # Ratios relative to b1 (fundamental)
        e_b1 = band_energies.get("b1", np.ones(n_samples))
        feats_dict[f"{sig_name}__band_ratio_b2_b1"] = band_energies.get("b2", np.zeros(n_samples)) / (e_b1 + 1e-9)
        feats_dict[f"{sig_name}__band_ratio_b3_b1"] = band_energies.get("b3", np.zeros(n_samples)) / (e_b1 + 1e-9)

        if include_energies:
            for b_name, e_val in band_energies.items():
                feats_dict[f"{sig_name}__energy_{b_name}"] = e_val

    return pd.DataFrame(feats_dict, index=sample_ids)


def extract_arm_dynamics_features(
    data: Union[pd.DataFrame, np.ndarray],
    sensors: Sequence[str] = ("arm_gyro",),
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
    clean: bool = False,
    impute_method: str = "cubic_spline",
) -> pd.DataFrame:
    """
    Extract Right-Arm Gyroscope Angular Momentum / Dynamics features.

    Physics & Domain Context:
    -------------------------
    When transitioning from a jog to a run, the upper body naturally drives the elbows with
    substantially higher angular speed and broader dynamic range. This module computes the
    resultant angular velocity:
        omega_res(t) = sqrt(omega_x(t)^2 + omega_y(t)^2 + omega_z(t)^2)

    Features Extracted:
    -------------------
    - {sensor}__omega_iqr: Interquartile Range (IQR) of resultant angular velocity.
    - {sensor}__omega_spec_centroid: Spectral Centroid (center of mass of angular velocity frequencies).
    - {sensor}__omega_rms: RMS angular speed.
    - {sensor}__omega_max: Peak rotational velocity during arm drive.

    Parameters
    ----------
    data : Union[pd.DataFrame, np.ndarray]
        Input dataset or waveform tensor.
    sensors : Sequence[str], default=("arm_gyro",)
        3-axis gyroscope sensor prefixes.
    sampling_rate_hz : float, default=50.0
        Sampling frequency in Hertz.
    clean : bool, default=False
        Whether to impute sentinels before feature extraction.
    impute_method : str, default="cubic_spline"
        Imputation method if clean=True.

    Returns
    -------
    pd.DataFrame
        DataFrame of arm rotational dynamics features indexed by sample ID.

    Examples
    --------
    >>> df_arm = extract_arm_dynamics_features(train_df)
    >>> print(df_arm.columns.tolist())
    ['arm_gyro__omega_iqr', 'arm_gyro__omega_spec_centroid',
     'arm_gyro__omega_rms', 'arm_gyro__omega_max']
    """
    data_3d, sample_ids, ch_names = _to_3d_tensor(data, clean=clean, impute_method=impute_method)
    ch2idx = {name: i for i, name in enumerate(ch_names)}
    n_samples, n_steps, _ = data_3d.shape
    freqs = np.fft.rfftfreq(n_steps, d=1.0 / sampling_rate_hz)
    feats_dict = {}

    for sensor in sensors:
        axis_names = [f"{sensor}_x", f"{sensor}_y", f"{sensor}_z"]
        if not all(ax in ch2idx for ax in axis_names):
            raise KeyError(f"Sensor '{sensor}' requires channels {axis_names}, but available channels are: {ch_names}")

        axis_indices = [ch2idx[ax] for ax in axis_names]
        coords = data_3d[:, :, axis_indices]  # (N, 250, 3)
        omega_res = np.sqrt(np.sum(coords ** 2, axis=-1))  # (N, 250)

        # 1. IQR
        q75, q25 = np.percentile(omega_res, [75, 25], axis=1)
        omega_iqr = q75 - q25

        # 2. Spectral Centroid
        centered = omega_res - np.mean(omega_res, axis=1, keepdims=True)
        fft_power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
        total_p = np.sum(fft_power, axis=1)
        spec_centroid = np.sum(fft_power * freqs, axis=1) / (total_p + 1e-9)

        # 3. RMS & Max
        rms = np.sqrt(np.mean(omega_res ** 2, axis=1))
        peak_v = np.max(omega_res, axis=1)

        feats_dict[f"{sensor}__omega_iqr"] = omega_iqr
        feats_dict[f"{sensor}__omega_spec_centroid"] = spec_centroid
        feats_dict[f"{sensor}__omega_rms"] = rms
        feats_dict[f"{sensor}__omega_max"] = peak_v

    return pd.DataFrame(feats_dict, index=sample_ids)


def extract_all_engineered_features(
    data: Union[pd.DataFrame, np.ndarray],
    clean: bool = False,
    impute_method: str = "cubic_spline",
) -> pd.DataFrame:
    """
    Extract all 4 domain-specific engineered feature sets into a unified tabular DataFrame.

    Combines:
    1. Jerk features: Impact sharpness on ankle and chest.
    2. Flight & Crest features: Free-fall proxy, high impact proxy, and crest factor.
    3. Spectral Band Ratios: Fundamental (0.5-3Hz), Cadence (3-8Hz), and Shockwave (8-20Hz) energy ratios.
    4. Arm Gyroscope Dynamics: Upper body arm-drive IQR, RMS, Peak, and Spectral Centroid.

    Parameters
    ----------
    data : Union[pd.DataFrame, np.ndarray]
        Input dataset or waveform tensor.
    clean : bool, default=False
        Whether to clean sentinels and impute missing readings.
    impute_method : str, default="cubic_spline"
        Imputation method if clean=True.

    Returns
    -------
    pd.DataFrame
        Consolidated DataFrame of all 26 advanced engineered features indexed by sample ID.

    Examples
    --------
    >>> df_all = extract_all_engineered_features(train_df, clean=True)
    >>> print(df_all.shape)
    (679, 26)
    """
    # Convert/clean once
    data_3d, sample_ids, ch_names = _to_3d_tensor(data, clean=clean, impute_method=impute_method)

    df_jerk = extract_jerk_features(data_3d)
    df_flight = extract_flight_crest_features(data_3d)
    df_bands = extract_spectral_band_ratios(data_3d)
    df_arm = extract_arm_dynamics_features(data_3d)

    df_all = pd.concat([df_jerk, df_flight, df_bands, df_arm], axis=1)
    df_all.index = sample_ids
    return df_all
