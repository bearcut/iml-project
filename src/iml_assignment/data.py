"""
Data loading, inspection, missingness analysis, reshaping, and cleaning utilities
for wearable sensor multivariate time series.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from iml_assignment.config import (
    TRAIN_PATH,
    TEST_PATH,
    TRAIN_AUGMENTED_PATH,
    FULL_PATH,
    WINDOW_SIZE,
    NUM_CHANNELS,
    TOTAL_SIGNAL_FEATURES,
    SAMPLING_RATE_HZ,
    SENSOR_CHANNELS,
    SENTINEL_VALUE,
    ACTIVITY_CLASSES,
)


# ─────────────────── 1. Loading & Column Inspection ───────────────────

def load_data(
    name_or_path: Union[str, pd.DataFrame] = "train",
    low_memory: bool = False
) -> pd.DataFrame:
    """
    Load a dataset by keyword ('train', 'test', 'augmented', 'full') or file path.

    Parameters
    ----------
    name_or_path : Union[str, pd.DataFrame], default="train"
        One of the canonical dataset keywords:
        - 'train': Primary competition training data (679 samples, corrupted).
        - 'test': Competition evaluation test data (403 samples, clean).
        - 'augmented': Full augmented dataset (3,048 samples; train + clean non-test MHEALTH).
        - 'full': Full 10-subject MHEALTH reference set (2,555 samples).
        Or a custom filesystem path / existing DataFrame (which will be copied).
    low_memory : bool, default=False
        Passed directly to `pd.read_csv`. Set False to avoid mixed-type warnings on large CSVs.

    Returns
    -------
    pd.DataFrame
        Loaded dataset with all sensor signal columns and available metadata.

    Examples
    --------
    >>> train_df = load_data("train")
    >>> print(train_df.shape)
    (679, 5752)
    """
    if isinstance(name_or_path, pd.DataFrame):
        return name_or_path.copy()

    path_map = {
        "train": TRAIN_PATH,
        "test": TEST_PATH,
        "augmented": TRAIN_AUGMENTED_PATH,
        "full": FULL_PATH,
    }
    file_path = path_map.get(str(name_or_path).lower(), name_or_path)
    df = pd.read_csv(file_path, low_memory=low_memory)
    return df


def get_sensor_columns(df: pd.DataFrame) -> List[str]:
    """
    Return all 5,750 sensor signal column names in exact time-major order.

    Filters out metadata columns such as 'id' and 'target', returning only
    columns beginning with 't' (e.g., 't000_chest_acc_x' through 't249_arm_mag_z').

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame containing sensor columns.

    Returns
    -------
    List[str]
        List of 5,750 column names in ascending time and channel order.

    Examples
    --------
    >>> cols = get_sensor_columns(train_df)
    >>> len(cols)
    5750
    """
    return [c for c in df.columns if c.startswith("t") and c != "target"]


def verify_column_ordering(df: pd.DataFrame) -> bool:
    """
    Verify that sensor columns are strictly arranged in time-major order.

    Checks that the columns match the exact sequence:
    t000_chest_acc_x ... t000_arm_mag_z, t001_..., up to t249_arm_mag_z
    (250 timesteps * 23 channels = 5,750 features).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame to inspect.

    Returns
    -------
    bool
        True if columns exactly follow canonical time-major order, False otherwise.

    Examples
    --------
    >>> verify_column_ordering(train_df)
    True
    """
    cols = get_sensor_columns(df)
    if len(cols) != TOTAL_SIGNAL_FEATURES:
        return False

    expected_cols = [
        f"t{t:03d}_{sensor}"
        for t in range(WINDOW_SIZE)
        for sensor in SENSOR_CHANNELS
    ]
    return cols == expected_cols


def get_dataset_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a side-by-side dimensional summary of training and test sets.

    Summarizes total rows, columns, signal features, timesteps, channels,
    window duration, sampling frequency, and target class counts.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset DataFrame.
    test_df : pd.DataFrame
        Test dataset DataFrame.

    Returns
    -------
    pd.DataFrame
        Side-by-side comparison table.
    """
    train_sensor_cols = get_sensor_columns(train_df)
    test_sensor_cols = get_sensor_columns(test_df)

    summary = {
        "Property": [
            "Total Samples (Rows)",
            "Total Columns",
            "Sensor Signal Columns",
            "Timesteps per Window",
            "Sensor Channels",
            "Window Duration (sec)",
            "Sampling Rate (Hz)",
            "Target Column Present",
            "Target Classes Count",
        ],
        "Training Set": [
            len(train_df),
            train_df.shape[1],
            len(train_sensor_cols),
            WINDOW_SIZE,
            NUM_CHANNELS,
            f"{WINDOW_SIZE / SAMPLING_RATE_HZ:.1f}s",
            f"{SAMPLING_RATE_HZ} Hz",
            "target" in train_df.columns,
            train_df["target"].nunique() if "target" in train_df.columns else "N/A",
        ],
        "Test Set (Kaggle)": [
            len(test_df),
            test_df.shape[1],
            len(test_sensor_cols),
            WINDOW_SIZE,
            NUM_CHANNELS,
            f"{WINDOW_SIZE / SAMPLING_RATE_HZ:.1f}s",
            f"{SAMPLING_RATE_HZ} Hz",
            "target" in test_df.columns,
            "Hidden (Evaluation)",
        ],
    }
    return pd.DataFrame(summary)


# ─────────────────── 2. Class Distribution ───────────────────

def get_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute frequency counts and percentage proportions for each activity class.

    Preserves the canonical activity ordering defined in `ACTIVITY_CLASSES`.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing a 'target' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Activity', 'Count', 'Percentage (%)'].

    Raises
    ------
    ValueError
        If 'target' column is not found in the input DataFrame.

    Examples
    --------
    >>> dist_df = get_class_distribution(train_df)
    >>> dist_df.head(3)
       Activity  Count  Percentage (%)
    0  standing     54            7.95
    1   sitting     56            8.25
    2     lying     55            8.10
    """
    if "target" not in df.columns:
        raise ValueError("DataFrame does not contain a 'target' column.")

    counts = df["target"].value_counts()
    dist = []
    total = len(df)

    for cls in ACTIVITY_CLASSES:
        cnt = counts.get(cls, 0)
        dist.append({
            "Activity": cls,
            "Count": cnt,
            "Percentage (%)": round((cnt / total) * 100, 2),
        })

    return pd.DataFrame(dist)


# ─────────────────── 3. Missingness & Sentinel Analysis ───────────────────

def analyze_missing_and_sentinels(
    df: pd.DataFrame,
    sentinel: float = SENTINEL_VALUE
) -> Dict[str, Union[int, float, pd.DataFrame]]:
    """
    Analyze standard NaNs, sentinel values (e.g. 9999.0), and combined corruption.

    Computes overall corruption rates, per-sensor channel corruption breakdowns,
    and row-level corruption statistics across the 5,750 numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame with sensor columns.
    sentinel : float, default=9999.0
        Sentinel value representing noise/corruption.

    Returns
    -------
    Dict[str, Union[int, float, pd.DataFrame]]
        Dictionary containing:
        - 'total_cells': Total signal values inspected (rows * 5,750).
        - 'nan_count': Number of standard NaN cells.
        - 'nan_percentage': Percentage of NaN cells.
        - 'sentinel_count': Number of sentinel (9999.0) cells.
        - 'sentinel_percentage': Percentage of sentinel cells.
        - 'total_corrupt_count': Combined corrupted count (NaN + sentinel).
        - 'total_corrupt_percentage': Combined corrupted percentage (~17.61% in train).
        - 'channel_summary': DataFrame detailing corruption per sensor channel.
        - 'row_min_corrupt': Minimum corrupted entries in any single row.
        - 'row_max_corrupt': Maximum corrupted entries in any single row.
        - 'row_mean_corrupt': Average corrupted entries per row.

    Examples
    --------
    >>> res = analyze_missing_and_sentinels(train_df)
    >>> print(f"Corrupt: {res['total_corrupt_percentage']}%")
    Corrupt: 17.61%
    """
    sensor_cols = get_sensor_columns(df)
    raw_vals = df[sensor_cols].apply(pd.to_numeric, errors="coerce").values
    total_cells = raw_vals.size

    nan_mask = np.isnan(raw_vals)
    sentinel_mask = (raw_vals == sentinel)
    corrupt_mask = nan_mask | sentinel_mask

    nan_count = int(np.sum(nan_mask))
    sentinel_count = int(np.sum(sentinel_mask))
    corrupt_count = int(np.sum(corrupt_mask))

    # Per-channel analysis across all 250 timesteps
    channel_stats = []
    for s_idx, sensor_name in enumerate(SENSOR_CHANNELS):
        ch_cols_idx = [t * NUM_CHANNELS + s_idx for t in range(WINDOW_SIZE)]
        ch_vals = raw_vals[:, ch_cols_idx]
        ch_total = ch_vals.size

        ch_nan = int(np.sum(np.isnan(ch_vals)))
        ch_sent = int(np.sum(ch_vals == sentinel))
        ch_corr = ch_nan + ch_sent

        channel_stats.append({
            "Sensor Channel": sensor_name,
            "NaN Count": ch_nan,
            "Sentinel (9999) Count": ch_sent,
            "Total Corrupt": ch_corr,
            "Corruption Rate (%)": round((ch_corr / ch_total) * 100, 2),
        })

    # Per-row summary
    row_corrupt = np.sum(corrupt_mask, axis=1)

    return {
        "total_cells": total_cells,
        "nan_count": nan_count,
        "nan_percentage": round((nan_count / total_cells) * 100, 2),
        "sentinel_count": sentinel_count,
        "sentinel_percentage": round((sentinel_count / total_cells) * 100, 2),
        "total_corrupt_count": corrupt_count,
        "total_corrupt_percentage": round((corrupt_count / total_cells) * 100, 2),
        "channel_summary": pd.DataFrame(channel_stats),
        "row_min_corrupt": int(np.min(row_corrupt)),
        "row_max_corrupt": int(np.max(row_corrupt)),
        "row_mean_corrupt": round(float(np.mean(row_corrupt)), 1),
    }


# ─────────────────── 4. Waveform Extraction & Cleaning ───────────────────

def extract_sample_waveform(
    df: pd.DataFrame,
    sample_idx_or_id: Union[int, str]
) -> Tuple[pd.DataFrame, str]:
    """
    Extract a single sample's time series as a (250, 23) DataFrame.

    Converts the 5,750 flattened columns into a continuous 2D matrix where:
    - Rows represent time index in seconds: [0.00, 0.02, 0.04, ..., 4.98].
    - Columns represent the 23 sensor channels.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame containing sensor columns.
    sample_idx_or_id : Union[int, str]
        Integer row index (e.g. 0) or string sample ID (e.g. 'TR000001').

    Returns
    -------
    Tuple[pd.DataFrame, str]
        - waveform_df: (250, 23) DataFrame with time in seconds as the index.
        - label_str: Formatted description string, e.g. "TR000001 (cycling)".

    Examples
    --------
    >>> wave_df, label = extract_sample_waveform(train_df, 0)
    >>> print(wave_df.shape, label)
    (250, 23) TR000001 (cycling)
    """
    if isinstance(sample_idx_or_id, str):
        row = df[df["id"] == sample_idx_or_id].iloc[0]
    else:
        row = df.iloc[sample_idx_or_id]

    sample_id = row.get("id", f"Row_{sample_idx_or_id}")
    sample_target = row.get("target", "Unlabeled")

    sensor_cols = get_sensor_columns(df)
    raw_array = pd.to_numeric(row[sensor_cols], errors="coerce").values.reshape(WINDOW_SIZE, NUM_CHANNELS)

    time_seconds = np.arange(WINDOW_SIZE) / SAMPLING_RATE_HZ
    waveform_df = pd.DataFrame(raw_array, index=time_seconds, columns=SENSOR_CHANNELS)
    waveform_df.index.name = "Time (seconds)"

    label_str = f"{sample_id} ({sample_target})"
    return waveform_df, label_str


def clean_sample_waveform(
    waveform_df: pd.DataFrame,
    method: str = "linear",
    sentinel: float = SENTINEL_VALUE
) -> pd.DataFrame:
    """
    Clean a single sample's (250, 23) waveform DataFrame.

    Imputation Pipeline:
    1. Replaces sentinel values (9999.0) with NaN.
    2. Interpolates along the temporal axis (axis 0) for each sensor channel:
       - 'linear': 1D linear interpolation across time steps.
       - 'cubic_spline' / 'cubic': Natural cubic spline interpolation using
         `scipy.interpolate.CubicSpline(bc_type='natural')`, with boundary fill
         and safety clamping to eliminate Runge overshoot on boundary gaps.
    3. Forward-fills and backward-fills boundary edges.
    4. Fills any remaining unresolvable points with 0.0.

    Parameters
    ----------
    waveform_df : pd.DataFrame
        (250, 23) DataFrame extracted via `extract_sample_waveform`.
    method : str, default="linear"
        Interpolation method: 'linear' or 'cubic_spline'.
    sentinel : float, default=9999.0
        Sentinel value representing noise/corruption.

    Returns
    -------
    pd.DataFrame
        Fully imputed, clean (250, 23) waveform DataFrame.

    Raises
    ------
    ValueError
        If an unsupported interpolation method is specified.

    Examples
    --------
    >>> raw_wave, _ = extract_sample_waveform(train_df, 0)
    >>> clean_wave = clean_sample_waveform(raw_wave, method="cubic_spline")
    >>> print("Remaining NaNs:", clean_wave.isna().sum().sum())
    Remaining NaNs: 0
    """
    cleaned = waveform_df.copy().replace(sentinel, np.nan)
    method_normalized = method.lower()

    if method_normalized == "linear":
        cleaned = cleaned.interpolate(method="linear", axis=0).bfill().ffill().fillna(0.0)
    elif method_normalized in ("cubic", "cubic_spline", "spline"):
        t = np.arange(len(cleaned))
        for col in cleaned.columns:
            s = cleaned[col].values
            valid_idx = np.where(~np.isnan(s))[0]
            if len(valid_idx) >= 4:
                # Natural cubic spline within the valid domain
                cs = CubicSpline(t[valid_idx], s[valid_idx], bc_type="natural", extrapolate=False)
                interpolated = cs(t)
                # Boundary NaNs get filled by boundary valid values
                interpolated = pd.Series(interpolated).bfill().ffill().values
                # Prevent Runge overshoot on boundary regions
                v_min, v_max = float(s[valid_idx].min()), float(s[valid_idx].max())
                rng = max(v_max - v_min, 1e-5)
                cleaned[col] = np.clip(interpolated, v_min - 0.2 * rng, v_max + 0.2 * rng)
            elif len(valid_idx) > 0:
                cleaned[col] = pd.Series(s).interpolate(method="linear").bfill().ffill().values
            else:
                cleaned[col] = 0.0
    else:
        raise ValueError(f"Unsupported interpolation method: '{method}'. Choose 'linear' or 'cubic_spline'.")

    return cleaned


def clean_dataset(
    df: pd.DataFrame,
    method: str = "linear",
    sentinel: float = SENTINEL_VALUE
) -> pd.DataFrame:
    """
    Clean all samples in a full dataset DataFrame (e.g. train_df or test_df).

    Imputes all corrupted points (NaNs and 9999.0 sentinels) along the temporal axis
    for each sample independently. Because each sample window is processed in isolation,
    this operation is strictly leak-free.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the 5,750 sensor columns.
    method : str, default="linear"
        'linear' (fast 1D linear interpolation) or 'cubic_spline'.
    sentinel : float, default=9999.0
        Sentinel value representing noise/corruption.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with clean, fully imputed sensor signals and identical schema.

    Raises
    ------
    ValueError
        If an unsupported interpolation method is specified.

    Examples
    --------
    >>> clean_train = clean_dataset(train_df, method="cubic_spline")
    >>> print("Corrupted values left:", (clean_train[get_sensor_columns(clean_train)] == 9999.0).sum().sum())
    Corrupted values left: 0
    """
    out_df = df.copy()
    sensor_cols = get_sensor_columns(out_df)
    raw_matrix = out_df[sensor_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    raw_matrix[raw_matrix == sentinel] = np.nan

    n_samples = len(raw_matrix)
    data_3d = raw_matrix.reshape(n_samples, WINDOW_SIZE, NUM_CHANNELS)
    t = np.arange(WINDOW_SIZE)
    method_normalized = method.lower()

    if method_normalized == "linear":
        for i in range(n_samples):
            for c in range(NUM_CHANNELS):
                s = data_3d[i, :, c]
                nans = np.isnan(s)
                if np.any(nans):
                    valid = ~nans
                    if np.sum(valid) > 0:
                        s[nans] = np.interp(t[nans], t[valid], s[valid])
                    else:
                        s[:] = 0.0
    elif method_normalized in ("cubic", "cubic_spline", "spline"):
        for i in range(n_samples):
            for c in range(NUM_CHANNELS):
                s = data_3d[i, :, c]
                nans = np.isnan(s)
                if np.any(nans):
                    valid_idx = np.where(~nans)[0]
                    if len(valid_idx) >= 4:
                        cs = CubicSpline(t[valid_idx], s[valid_idx], bc_type="natural", extrapolate=False)
                        interp = cs(t)
                        interp = pd.Series(interp).bfill().ffill().values
                        v_min, v_max = float(s[valid_idx].min()), float(s[valid_idx].max())
                        rng = max(v_max - v_min, 1e-5)
                        s[:] = np.clip(interp, v_min - 0.2 * rng, v_max + 0.2 * rng)
                    elif len(valid_idx) > 0:
                        s[nans] = np.interp(t[nans], t[valid_idx], s[valid_idx])
                    else:
                        s[:] = 0.0
    else:
        raise ValueError(f"Unsupported interpolation method: '{method}'. Choose 'linear' or 'cubic_spline'.")

    out_df[sensor_cols] = data_3d.reshape(n_samples, -1)
    return out_df


# ─────────────────── 5. Channel Statistics ───────────────────

def compute_channel_statistics(
    df: pd.DataFrame,
    clean: bool = True,
    sentinel: float = SENTINEL_VALUE
) -> pd.DataFrame:
    """
    Compute summary statistics for all 23 sensor channels across the entire dataset.

    Calculates Min, Max, Mean, Standard Deviation, and Median for each channel,
    pooling readings across all 250 timesteps and all samples.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame.
    clean : bool, default=True
        If True, excludes sentinel values (9999.0) and NaNs before calculating statistics.
    sentinel : float, default=9999.0
        Noise sentinel value to mask out when clean=True.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Channel', 'Min', 'Max', 'Mean', 'Std', 'Median'].

    Examples
    --------
    >>> stats = compute_channel_statistics(train_df)
    >>> print(stats.head(2))
            Channel      Min     Max    Mean     Std  Median
    0  chest_acc_x -66.5970 50.4570 -7.4128 13.8260 -8.7324
    1  chest_acc_y -45.7820 44.5960 -0.1311  7.3248 -0.5550
    """
    sensor_cols = get_sensor_columns(df)
    raw_vals = df[sensor_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)

    if clean:
        raw_vals[raw_vals == sentinel] = np.nan

    stats = []
    for s_idx, sensor_name in enumerate(SENSOR_CHANNELS):
        ch_cols_idx = [t * NUM_CHANNELS + s_idx for t in range(WINDOW_SIZE)]
        ch_vals = raw_vals[:, ch_cols_idx].flatten()
        valid = ch_vals[~np.isnan(ch_vals)]

        stats.append({
            "Channel": sensor_name,
            "Min": round(float(np.min(valid)), 4) if len(valid) else np.nan,
            "Max": round(float(np.max(valid)), 4) if len(valid) else np.nan,
            "Mean": round(float(np.mean(valid)), 4) if len(valid) else np.nan,
            "Std": round(float(np.std(valid)), 4) if len(valid) else np.nan,
            "Median": round(float(np.median(valid)), 4) if len(valid) else np.nan,
        })

    return pd.DataFrame(stats)


# ─────────────────── 6. Channel-Wise Scaling ───────────────────

class ChannelScaler:
    """
    Channel-wise feature scaler for wearable sensor multivariate time series.

    Fits normalization statistics across all samples and timesteps for each of
    the 23 physical sensor channels independently. This preserves the physical
    kinematic unity of each sensor while scaling across different modalities
    (accelerometers, gyroscopes, magnetometers, and ECG).

    Supported Scaling Methods
    -------------------------
    'robust' :
        Subtracts median and divides by IQR (75th percentile - 25th percentile).
        Highly recommended for noisy time-series data with extreme movement spikes.
        Formula: x_scaled = (x - median) / IQR
    'standard' :
        Standard z-score normalization: subtracts mean and divides by standard deviation.
        Formula: x_scaled = (x - mean) / std

    Parameters
    ----------
    method : str, default='robust'
        'robust' or 'standard'.
    clip_range : Optional[Tuple[float, float]], default=None
        Optional min/max bounds to clip scaled values (e.g. (-10.0, 10.0)).

    Attributes
    ----------
    center_ : np.ndarray of shape (23,)
        Fitted channel centers (medians for 'robust', means for 'standard').
    scale_ : np.ndarray of shape (23,)
        Fitted channel scales (IQRs for 'robust', standard deviations for 'standard').
    channel_names_ : List[str]
        List of the 23 sensor channel names in order.

    Examples
    --------
    >>> scaler = ChannelScaler(method="robust")
    >>> scaler.fit(train_df)
    >>> scaled_wave = scaler.transform(clean_wave)
    >>> print(scaled_wave.shape)
    (250, 23)
    """

    def __init__(
        self,
        method: str = "robust",
        clip_range: Optional[Tuple[float, float]] = None
    ) -> None:
        method_norm = method.lower()
        if method_norm not in ("robust", "standard"):
            raise ValueError(f"Unknown scaling method '{method}'. Choose 'robust' or 'standard'.")

        self.method = method_norm
        self.clip_range = clip_range
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.channel_names_: List[str] = list(SENSOR_CHANNELS)

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "ChannelScaler":
        """
        Fit channel-wise center and scale parameters on training data.

        Ignores NaNs and 9999.0 sentinels during statistic computation.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Training data. Can be:
            - Full dataset DataFrame with 5,750 columns (N, 5750+)
            - Single waveform DataFrame (250, 23)
            - 2D or 3D numpy array

        Returns
        -------
        ChannelScaler
            Fitted scaler instance (self).
        """
        if isinstance(X, pd.DataFrame):
            if X.shape == (WINDOW_SIZE, NUM_CHANNELS) and list(X.columns) == SENSOR_CHANNELS:
                arr = X.to_numpy(dtype=float).reshape(-1, NUM_CHANNELS)
            else:
                sensor_cols = get_sensor_columns(X)
                raw_mat = X[sensor_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                arr = raw_mat.reshape(-1, NUM_CHANNELS)
        elif isinstance(X, np.ndarray):
            arr = X.reshape(-1, NUM_CHANNELS).astype(float)
        else:
            raise TypeError("Input must be a pandas DataFrame or numpy ndarray.")

        valid_mask = ~np.isnan(arr) & (arr != SENTINEL_VALUE)
        self.center_ = np.zeros(NUM_CHANNELS, dtype=float)
        self.scale_ = np.zeros(NUM_CHANNELS, dtype=float)

        for c in range(NUM_CHANNELS):
            col_valid = arr[:, c][valid_mask[:, c]]
            if len(col_valid) == 0:
                self.center_[c] = 0.0
                self.scale_[c] = 1.0
                continue

            if self.method == "robust":
                med = float(np.median(col_valid))
                q75, q25 = np.percentile(col_valid, [75, 25])
                iqr = float(q75 - q25)
                self.center_[c] = med
                self.scale_[c] = iqr if iqr > 1e-6 else 1.0
            else:  # standard
                mean = float(np.mean(col_valid))
                std = float(np.std(col_valid))
                self.center_[c] = mean
                self.scale_[c] = std if std > 1e-6 else 1.0

        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply channel-wise scaling to an input DataFrame or numpy array.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Data to scale. Supports:
            - Single sample waveform DataFrame (250, 23)
            - Full dataset DataFrame (N, 5750+)
            - 2D or 3D numpy array

        Returns
        -------
        Union[pd.DataFrame, np.ndarray]
            Scaled data matching the input type and dimensions.

        Raises
        ------
        RuntimeError
            If called before `fit()`.
        """
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("ChannelScaler must be fitted before calling transform().")

        if isinstance(X, pd.DataFrame):
            # Case 1: Single sample waveform (250, 23)
            if X.shape == (WINDOW_SIZE, NUM_CHANNELS) and list(X.columns) == SENSOR_CHANNELS:
                vals = X.to_numpy(dtype=float)
                scaled = (vals - self.center_) / self.scale_
                if self.clip_range is not None:
                    scaled = np.clip(scaled, self.clip_range[0], self.clip_range[1])
                return pd.DataFrame(scaled, index=X.index, columns=X.columns)

            # Case 2: Full dataset DataFrame (N, 5750+)
            out_df = X.copy()
            sensor_cols = get_sensor_columns(out_df)
            raw_mat = out_df[sensor_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            n_samples = len(raw_mat)
            data_3d = raw_mat.reshape(n_samples, WINDOW_SIZE, NUM_CHANNELS)
            scaled_3d = (data_3d - self.center_) / self.scale_
            if self.clip_range is not None:
                scaled_3d = np.clip(scaled_3d, self.clip_range[0], self.clip_range[1])
            out_df[sensor_cols] = scaled_3d.reshape(n_samples, -1)
            return out_df

        elif isinstance(X, np.ndarray):
            orig_shape = X.shape
            reshaped = X.reshape(-1, NUM_CHANNELS).astype(float)
            scaled = (reshaped - self.center_) / self.scale_
            if self.clip_range is not None:
                scaled = np.clip(scaled, self.clip_range[0], self.clip_range[1])
            return scaled.reshape(orig_shape)

        else:
            raise TypeError("Input must be a pandas DataFrame or numpy ndarray.")

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Data to fit and transform.

        Returns
        -------
        Union[pd.DataFrame, np.ndarray]
            Scaled data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Reverse the scaling transformation back to original physical units.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Scaled data to invert.

        Returns
        -------
        Union[pd.DataFrame, np.ndarray]
            Unscaled data in original physical units (m/s², deg/s, µT, mV).
        """
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("ChannelScaler must be fitted before calling inverse_transform().")

        if isinstance(X, pd.DataFrame):
            if X.shape == (WINDOW_SIZE, NUM_CHANNELS) and list(X.columns) == SENSOR_CHANNELS:
                vals = X.to_numpy(dtype=float)
                unscaled = (vals * self.scale_) + self.center_
                return pd.DataFrame(unscaled, index=X.index, columns=X.columns)

            out_df = X.copy()
            sensor_cols = get_sensor_columns(out_df)
            raw_mat = out_df[sensor_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            n_samples = len(raw_mat)
            data_3d = raw_mat.reshape(n_samples, WINDOW_SIZE, NUM_CHANNELS)
            unscaled_3d = (data_3d * self.scale_) + self.center_
            out_df[sensor_cols] = unscaled_3d.reshape(n_samples, -1)
            return out_df

        elif isinstance(X, np.ndarray):
            orig_shape = X.shape
            reshaped = X.reshape(-1, NUM_CHANNELS).astype(float)
            unscaled = (reshaped * self.scale_) + self.center_
            return unscaled.reshape(orig_shape)

        else:
            raise TypeError("Input must be a pandas DataFrame or numpy ndarray.")

    def get_params_df(self) -> pd.DataFrame:
        """
        Return a readable DataFrame of fitted center and scale parameters per channel.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['Channel', 'Median'/'Mean', 'IQR'/'Std'].

        Raises
        ------
        RuntimeError
            If called before `fit()`.
        """
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted yet.")
        center_label = "Median" if self.method == "robust" else "Mean"
        scale_label = "IQR" if self.method == "robust" else "Std"
        return pd.DataFrame({
            "Channel": self.channel_names_,
            center_label: np.round(self.center_, 4),
            scale_label: np.round(self.scale_, 4),
        })
