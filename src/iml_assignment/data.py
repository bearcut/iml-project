"""
Data loading, inspection, missingness analysis, reshaping, and cleaning utilities
for wearable sensor multivariate time series.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

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
    """Return all 5,750 sensor column names in the exact time-major order."""
    return [c for c in df.columns if c.startswith("t") and c != "target"]


def verify_column_ordering(df: pd.DataFrame) -> bool:
    """
    Verify that sensor columns are strictly arranged in time-major order:
    t000_chest_acc_x ... t000_arm_mag_z, t001_..., up to t249_...
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
    """Return a side-by-side dimensional summary of training and test sets."""
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
    Compute frequency count and percentage for each activity class.
    Sorted in canonical activity order.
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
    Analyze standard NaNs, sentinel values (e.g. 9999.0), and combined corruption
    both overall, per sensor channel, and per row.
    """
    sensor_cols = get_sensor_columns(df)
    
    # Coerce to numeric matrix for fast numpy computation
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
        # Every 23rd column belongs to this sensor channel
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
    The index is elapsed time in seconds [0.00, 0.02, ..., 4.98],
    and columns are the 23 sensor names.
    Returns: (waveform_df, sample_label_or_id)
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
    sentinel: float = SENTINEL_VALUE
) -> pd.DataFrame:
    """
    Clean a (250, 23) waveform DataFrame:
    1. Replace sentinel values (9999.0) with NaN.
    2. Linear interpolation across the temporal axis for each sensor channel.
    3. Forward-fill and backward-fill boundary edges.
    4. Fill any remaining unresolvable points with 0.0.
    """
    cleaned = waveform_df.copy()
    cleaned = cleaned.replace(sentinel, np.nan)
    cleaned = cleaned.interpolate(method="linear", axis=0)
    cleaned = cleaned.bfill().ffill().fillna(0.0)
    return cleaned


# ─────────────────── 5. Channel Statistics ───────────────────

def compute_channel_statistics(
    df: pd.DataFrame,
    clean: bool = True,
    sentinel: float = SENTINEL_VALUE
) -> pd.DataFrame:
    """
    Compute basic summary statistics (min, max, mean, std, median)
    for all 23 sensor channels across the entire dataset.
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
