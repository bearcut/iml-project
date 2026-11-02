"""
Visualization library for exploratory data analysis, signal inspection,
corruption analysis, preprocessing comparisons, and publication-ready figures.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal

from iml_assignment.config import (
    FIGURES_DIR,
    SENSOR_CHANNELS,
    BODY_LOCATIONS,
    SAMPLING_RATE_HZ,
    WINDOW_SIZE,
    NUM_CHANNELS,
    SENTINEL_VALUE,
    get_channel_unit,
)
from iml_assignment.data import (
    get_class_distribution,
    extract_sample_waveform,
    clean_sample_waveform,
)

# ─────────────────── Global Style Settings ───────────────────

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.5,
})


def _maybe_save(fig: plt.Figure, save_name: Optional[str]) -> None:
    """
    Save the matplotlib figure to `reports/figures/` if save_name is provided.

    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure object to save.
    save_name : Optional[str]
        Desired output filename (e.g. 'fig1_class_distribution.png').
        If no extension is provided, defaults to '.png'.
    """
    if save_name:
        out_path = FIGURES_DIR / (save_name if save_name.endswith((".png", ".pdf", ".svg")) else f"{save_name}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {out_path}")


# ─────────────────── 1. Class Distribution Plot ───────────────────

def plot_class_distribution(
    data: Union[pd.DataFrame, pd.Series],
    title: str = "Activity Class Distribution",
    save_name: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a horizontal bar chart of the 12 activity classes with sample counts and percentages.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Dataset DataFrame containing a 'target' column, or a pre-computed
        distribution DataFrame with 'Activity' and 'Count' columns.
    title : str, default="Activity Class Distribution"
        Chart title.
    save_name : Optional[str], default=None
        Filename to save the figure inside reports/figures/.
    figsize : Tuple[int, int], default=(10, 5)
        Dimensions of the figure in inches.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes objects.

    Raises
    ------
    ValueError
        If input is not a DataFrame with 'target' or 'Activity' column.

    Examples
    --------
    >>> fig, ax = plot_class_distribution(train_df, save_name="fig1_class_dist.png")
    >>> plt.show()
    """
    if isinstance(data, pd.DataFrame) and "target" in data.columns:
        dist_df = get_class_distribution(data)
    elif isinstance(data, pd.DataFrame) and "Activity" in data.columns:
        dist_df = data
    else:
        raise ValueError("Input must be a DataFrame with 'target' or 'Activity' column.")

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("mako", n_colors=len(dist_df))

    bars = ax.barh(dist_df["Activity"], dist_df["Count"], color=colors, edgecolor="none", height=0.65)
    ax.invert_yaxis()  # Top-down order

    # Add count and percentage labels on bars
    max_count = dist_df["Count"].max()
    for bar, pct in zip(bars, dist_df["Percentage (%)"]):
        width = bar.get_width()
        ax.text(
            width + max_count * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)} ({pct:.1f}%)",
            va="center",
            fontsize=9,
            color="#333333"
        )

    ax.set_xlim(0, max_count * 1.18)
    ax.set_xlabel("Number of Windows (5-second)")
    ax.set_title(title, pad=15, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _maybe_save(fig, save_name)
    return fig, ax


# ─────────────────── 2. Missingness & Sentinel Plot ───────────────────

def plot_missingness_by_channel(
    missing_summary: pd.DataFrame,
    title: str = "Corruption Rate per Sensor Channel (NaN + 9999 Sentinel)",
    save_name: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a stacked bar chart displaying the breakdown of NaNs vs 9999 Sentinels per channel.

    Parameters
    ----------
    missing_summary : pd.DataFrame
        Channel summary DataFrame produced by `analyze_missing_and_sentinels()['channel_summary']`.
    title : str, default="Corruption Rate per Sensor Channel (NaN + 9999 Sentinel)"
        Chart title.
    save_name : Optional[str], default=None
        Filename to save the figure inside reports/figures/.
    figsize : Tuple[int, int], default=(12, 6)
        Dimensions of the figure in inches.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes objects.

    Examples
    --------
    >>> missing_info = analyze_missing_and_sentinels(train_df)
    >>> fig, ax = plot_missingness_by_channel(missing_info["channel_summary"])
    >>> plt.show()
    """
    df = missing_summary.copy()
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df))
    width = 0.65

    ax.bar(x, df["NaN Count"], width, label="NaN (Missing)", color="#4575b4", alpha=0.85)
    ax.bar(x, df["Sentinel (9999) Count"], width, bottom=df["NaN Count"], label="Sentinel (9999.0)", color="#d73027", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df["Sensor Channel"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Total Corrupted Data Points")
    ax.set_title(title, pad=15, fontweight="bold")
    ax.legend(frameon=True, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _maybe_save(fig, save_name)
    return fig, ax


# ─────────────────── 3. Multi-Channel Waveform Plot ───────────────────

def plot_sample_waveforms(
    waveform_df: pd.DataFrame,
    title: Optional[str] = None,
    body_location: Optional[str] = None,
    save_name: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot time-series signals grouped by anatomical body placement (Chest, Ankle, Arm).

    Parameters
    ----------
    waveform_df : pd.DataFrame
        (250, 23) waveform DataFrame produced by `extract_sample_waveform`.
    title : Optional[str], default=None
        Overall figure super-title.
    body_location : Optional[str], default=None
        Specific body location to plot ('Chest', 'Ankle', or 'Arm').
        If None, creates 3 stacked subplots for all locations.
    save_name : Optional[str], default=None
        Filename to save figure inside reports/figures/.
    figsize : Tuple[int, int], default=(12, 8)
        Dimensions of the figure.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and array of axes subplots.

    Examples
    --------
    >>> raw_wave, label = extract_sample_waveform(train_df, 0)
    >>> fig, axes = plot_sample_waveforms(raw_wave, title=f"Sample: {label}")
    >>> plt.show()
    """
    locations = [body_location] if body_location else list(BODY_LOCATIONS.keys())
    fig, axes = plt.subplots(len(locations), 1, figsize=figsize, sharex=True)
    if len(locations) == 1:
        axes = np.array([axes])

    palette = sns.color_palette("tab10")

    for ax, loc in zip(axes, locations):
        channels = BODY_LOCATIONS.get(loc, [])
        for idx, ch in enumerate(channels):
            if ch in waveform_df.columns:
                unit = get_channel_unit(ch)
                lbl = f"{ch} ({unit})" if unit else ch
                ax.plot(waveform_df.index, waveform_df[ch], label=lbl, color=palette[idx % 10])

        ax.set_title(f"{loc} Sensors", loc="left", fontweight="bold", fontsize=11)
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", ncol=min(len(channels), 3), fontsize=8, framealpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (seconds)")
    if title:
        fig.suptitle(title, y=0.99, fontweight="bold")

    plt.tight_layout()
    _maybe_save(fig, save_name)
    return fig, axes


# ─────────────────── 4. Raw vs Cleaned Comparison ───────────────────

def plot_raw_vs_cleaned(
    raw_waveform: pd.DataFrame,
    cleaned_waveform: pd.DataFrame,
    channel: str = "chest_acc_x",
    title: Optional[str] = None,
    save_name: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Compare raw corrupted signal against linearly interpolated cleaned signal.

    Parameters
    ----------
    raw_waveform : pd.DataFrame
        Raw (250, 23) waveform DataFrame containing NaNs and 9999.0 sentinels.
    cleaned_waveform : pd.DataFrame
        Cleaned (250, 23) waveform DataFrame.
    channel : str, default="chest_acc_x"
        Sensor channel to display.
    title : Optional[str], default=None
        Plot title.
    save_name : Optional[str], default=None
        Filename to save figure inside reports/figures/.
    figsize : Tuple[int, int], default=(12, 5)
        Dimensions of figure.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=figsize)
    unit = get_channel_unit(channel)
    time_idx = raw_waveform.index

    raw_s = raw_waveform[channel].copy()
    sentinel_idx = raw_s[raw_s == 9999.0].index

    # Plot cleaned signal first
    ax.plot(time_idx, cleaned_waveform[channel], label="Cleaned (Interpolated)", color="#2b83ba", linewidth=2)

    # Plot raw valid points
    raw_masked = raw_s.replace(9999.0, np.nan)
    ax.plot(time_idx, raw_masked, label="Original Valid Readings", color="#1a9641", linestyle="--", alpha=0.7)

    # Mark sentinels if within view
    if len(sentinel_idx):
        ax.scatter(sentinel_idx, [cleaned_waveform[channel].mean()] * len(sentinel_idx),
                   color="#d7191c", s=30, zorder=5, label=f"Corrupted Points ({len(sentinel_idx)})")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel(f"{channel} ({unit})" if unit else channel)
    ax.set_title(title or f"Signal Cleaning: {channel} (Raw vs. Interpolated)", fontweight="bold", pad=12)
    ax.legend(loc="best", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _maybe_save(fig, save_name)
    return fig, ax


# ─────────────────── 5. Activity Waveform Comparison ───────────────────

def plot_activity_comparison(
    df: pd.DataFrame,
    activity_a: str = "walking",
    activity_b: str = "running",
    channel: str = "ankle_acc_y",
    save_name: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Compare waveforms of two activities for a given sensor channel side-by-side.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame containing 'target' column and sensor features.
    activity_a : str, default="walking"
        First activity class name.
    activity_b : str, default="running"
        Second activity class name.
    channel : str, default="ankle_acc_y"
        Sensor channel to compare.
    save_name : Optional[str], default=None
        Filename to save figure inside reports/figures/.
    figsize : Tuple[int, int], default=(12, 5)
        Dimensions of figure.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and array of 2 axes subplots.
    """
    row_a = df[df["target"] == activity_a].index[0]
    row_b = df[df["target"] == activity_b].index[0]

    wave_a, _ = extract_sample_waveform(df, row_a)
    wave_b, _ = extract_sample_waveform(df, row_b)

    clean_a = clean_sample_waveform(wave_a)
    clean_b = clean_sample_waveform(wave_b)

    unit = get_channel_unit(channel)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    axes[0].plot(clean_a.index, clean_a[channel], color="#2c7bb6")
    axes[0].set_title(f"Activity: {activity_a.upper()}", fontweight="bold")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel(f"{channel} ({unit})" if unit else channel)

    axes[1].plot(clean_b.index, clean_b[channel], color="#d7191c")
    axes[1].set_title(f"Activity: {activity_b.upper()}", fontweight="bold")
    axes[1].set_xlabel("Time (seconds)")

    fig.suptitle(f"Waveform Comparison: {activity_a} vs. {activity_b} [{channel}]", y=1.02, fontweight="bold")
    plt.tight_layout()
    _maybe_save(fig, save_name)
    return fig, axes


# ─────────────────── 6. Sensor Correlation Matrix ───────────────────

def plot_sensor_correlations(
    waveform_df: pd.DataFrame,
    title: str = "Cross-Sensor Correlation Matrix",
    save_name: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a heatmap of Pearson correlation coefficients across all 23 sensor channels.

    Parameters
    ----------
    waveform_df : pd.DataFrame
        (250, 23) waveform DataFrame.
    title : str, default="Cross-Sensor Correlation Matrix"
        Plot title.
    save_name : Optional[str], default=None
        Filename to save figure inside reports/figures/.
    figsize : Tuple[int, int], default=(10, 8)
        Dimensions of figure.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects.
    """
    cleaned = clean_sample_waveform(waveform_df)
    corr = cleaned.corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        ax=ax
    )

    ax.set_title(title, pad=15, fontweight="bold")
    plt.tight_layout()
    _maybe_save(fig, save_name)
    return fig, ax


# ─────────────────── 7. Frequency Spectrum (FFT) ───────────────────

def plot_frequency_spectrum(
    waveform_df: pd.DataFrame,
    channel: str = "ankle_acc_y",
    fs: int = SAMPLING_RATE_HZ,
    title: Optional[str] = None,
    save_name: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 4)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot Power Spectral Density (PSD) using Welch's method for frequency domain analysis.

    Parameters
    ----------
    waveform_df : pd.DataFrame
        (250, 23) waveform DataFrame.
    channel : str, default="ankle_acc_y"
        Sensor channel to analyze.
    fs : int, default=50
        Sampling frequency in Hertz.
    title : Optional[str], default=None
        Plot title.
    save_name : Optional[str], default=None
        Filename to save figure inside reports/figures/.
    figsize : Tuple[int, int], default=(9, 4)
        Dimensions of figure.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects.
    """
    cleaned = clean_sample_waveform(waveform_df)
    sig = cleaned[channel].values

    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig), 128))

    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogy(freqs, psd, color="#756bb1", linewidth=2)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title(title or f"Power Spectrum: {channel} (Dominant Frequencies)", pad=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _maybe_save(fig, save_name)
    return fig, ax


# ─────────────────── 8. Imputation Comparison (Linear vs Cubic Spline) ───────────────────

def plot_imputation_comparison(
    raw_waveform: pd.DataFrame,
    linear_waveform: pd.DataFrame,
    spline_waveform: Optional[pd.DataFrame] = None,
    channel: str = "chest_acc_x",
    title: Optional[str] = None,
    save_name: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 7)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Detailed visual comparison of imputation methods:
    - Top panel: Full 5-second window showing raw points (valid in green, sentinels in red),
      Linear interpolation (blue line), and Cubic Spline (orange dashed line).
    - Bottom panel: Zoomed-in window focused on the largest missing/corrupted block,
      clearly demonstrating how linear makes straight segments vs cubic spline curves smoothly.

    Parameters
    ----------
    raw_waveform : pd.DataFrame
        Raw (250, 23) waveform with corrupted values and sentinels.
    linear_waveform : pd.DataFrame
        (250, 23) waveform cleaned via 1D linear interpolation.
    spline_waveform : Optional[pd.DataFrame], default=None
        (250, 23) waveform cleaned via natural cubic spline interpolation.
    channel : str, default="chest_acc_x"
        Sensor channel to compare.
    title : Optional[str], default=None
        Custom title for the top panel.
    save_name : Optional[str], default=None
        Filename to save figure inside reports/figures/.
    figsize : Tuple[int, int], default=(12, 7)
        Dimensions of figure.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and array of 2 axes subplots.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=False)
    time_idx = raw_waveform.index.to_numpy()
    unit = get_channel_unit(channel)
    ylabel = f"{channel} ({unit})" if unit else channel

    raw_series = raw_waveform[channel].copy()
    is_sentinel = (raw_series == 9999.0)
    is_nan = raw_series.isna()
    is_corrupt = is_sentinel | is_nan
    valid_mask = ~is_corrupt

    # --- Top Panel: Full 5-Second Window ---
    ax_top = axes[0]
    ax_top.plot(time_idx, linear_waveform[channel], label="1D Linear Interpolation", color="#1f77b4", linewidth=2.0)
    if spline_waveform is not None:
        ax_top.plot(time_idx, spline_waveform[channel], label="Cubic Spline Interpolation", color="#ff7f0e", linewidth=1.8, linestyle="--")

    # Mark valid raw readings
    ax_top.plot(time_idx[valid_mask], raw_series.values[valid_mask], label="Original Valid Observations", color="#2ca02c", linestyle="none", marker="o", markersize=2.5, alpha=0.6)

    # Highlight corrupted points
    if np.any(is_corrupt):
        corrupt_times = time_idx[is_corrupt]
        corrupt_vals = linear_waveform[channel].values[is_corrupt]
        ax_top.scatter(corrupt_times, corrupt_vals, color="#d62728", s=25, zorder=5, label=f"Corrupted / Imputed Points ({np.sum(is_corrupt)})")

    ax_top.set_title(title or f"Imputation Comparison: {channel} (Full 5.0s Window)", fontweight="bold", pad=10)
    ax_top.set_ylabel(ylabel)
    ax_top.legend(loc="upper right", frameon=True, fontsize=9)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # --- Bottom Panel: Zoomed-In on Largest Missing Gap ---
    ax_bot = axes[1]
    corrupt_indices = np.where(is_corrupt)[0]
    if len(corrupt_indices) > 0:
        diffs = np.diff(corrupt_indices)
        split_points = np.where(diffs > 1)[0] + 1
        runs = np.split(corrupt_indices, split_points)
        longest_run = max(runs, key=len)
        start_idx = max(0, longest_run[0] - 8)
        end_idx = min(len(time_idx) - 1, longest_run[-1] + 8)
        zoom_t0 = float(time_idx[start_idx])
        zoom_t1 = float(time_idx[end_idx])
    else:
        zoom_t0 = 1.75
        zoom_t1 = 3.25

    zoom_mask = (time_idx >= zoom_t0) & (time_idx <= zoom_t1)
    ax_bot.plot(time_idx[zoom_mask], linear_waveform[channel].values[zoom_mask], label="Linear (Piecewise Straight)", color="#1f77b4", linewidth=2.2, marker="s", markersize=3)
    if spline_waveform is not None:
        ax_bot.plot(time_idx[zoom_mask], spline_waveform[channel].values[zoom_mask], label="Cubic Spline (Smooth Curve)", color="#ff7f0e", linewidth=2.2, linestyle="--")

    zoom_valid = valid_mask & zoom_mask
    if np.any(zoom_valid):
        ax_bot.scatter(time_idx[zoom_valid], raw_series.values[zoom_valid], color="#2ca02c", s=40, zorder=4, label="Valid Ground Truth")

    zoom_corrupt = is_corrupt & zoom_mask
    if np.any(zoom_corrupt):
        ax_bot.scatter(time_idx[zoom_corrupt], linear_waveform[channel].values[zoom_corrupt], color="#d62728", s=50, marker="x", zorder=5, label="Reconstructed Points")

    ax_bot.set_xlim(zoom_t0, zoom_t1)
    ax_bot.set_title(f"Detailed Zoom: Window [{zoom_t0:.2f}s to {zoom_t1:.2f}s] Showing Interpolation Dynamics", fontweight="bold", pad=10)
    ax_bot.set_xlabel("Time (seconds)")
    ax_bot.set_ylabel(ylabel)
    ax_bot.legend(loc="upper right", frameon=True, fontsize=9)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    plt.tight_layout()
    _maybe_save(fig, save_name)
    return fig, axes


# ─────────────────── 9. Feature Scaling Comparison ───────────────────

def plot_scaling_comparison(
    unscaled_waveform: pd.DataFrame,
    robust_waveform: pd.DataFrame,
    standard_waveform: pd.DataFrame,
    channel: str = "chest_acc_x",
    title: Optional[str] = None,
    save_name: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    3-panel stacked comparison demonstrating the effect of ChannelScaler:
    - Panel 1: Original Clean Signal (physical units: m/s^2, deg/s, uT, mV)
    - Panel 2: Robust Scaler (Centered at Median=0, Scaled by IQR=1)
    - Panel 3: Standard Scaler (Centered at Mean=0, Scaled by Std=1)

    Parameters
    ----------
    unscaled_waveform : pd.DataFrame
        Cleaned, unscaled (250, 23) waveform.
    robust_waveform : pd.DataFrame
        Waveform transformed by `ChannelScaler(method='robust')`.
    standard_waveform : pd.DataFrame
        Waveform transformed by `ChannelScaler(method='standard')`.
    channel : str, default="chest_acc_x"
        Sensor channel to compare.
    title : Optional[str], default=None
        Overall figure title.
    save_name : Optional[str], default=None
        Filename to save figure inside reports/figures/.
    figsize : Tuple[int, int], default=(12, 8)
        Dimensions of figure.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and array of 3 axes subplots.
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    time_idx = unscaled_waveform.index.to_numpy()
    unit = get_channel_unit(channel)

    # 1. Unscaled
    ax0 = axes[0]
    s0 = unscaled_waveform[channel].values
    ax0.plot(time_idx, s0, color="#333333", linewidth=1.8, label=f"Original ({unit or 'raw units'})")
    ax0.axhline(0, color="#888888", linestyle=":", linewidth=0.8)
    ax0.set_ylabel(f"Original ({unit})" if unit else "Original")
    ax0.set_title(title or f"Feature Scaling Comparison: {channel}", fontweight="bold", pad=10)
    ax0.text(0.01, 0.85, f"Range: [{s0.min():.2f}, {s0.max():.2f}] | Mean: {s0.mean():.2f} | Median: {np.median(s0):.2f}",
             transform=ax0.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="#cccccc"))
    ax0.legend(loc="upper right", fontsize=9)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # 2. Robust Scaled
    ax1 = axes[1]
    s1 = robust_waveform[channel].values
    ax1.plot(time_idx, s1, color="#2b83ba", linewidth=1.8, label="RobustScaler (Median / IQR)")
    ax1.axhline(0, color="#d7191c", linestyle="--", linewidth=1.0, alpha=0.7, label="Median = 0")
    ax1.set_ylabel("Robust Normalized")
    ax1.text(0.01, 0.85, f"Range: [{s1.min():.2f}, {s1.max():.2f}] | Median: {np.median(s1):.2f} | IQR: 1.0",
             transform=ax1.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="#cccccc"))
    ax1.legend(loc="upper right", fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # 3. Standard Scaled
    ax2 = axes[2]
    s2 = standard_waveform[channel].values
    ax2.plot(time_idx, s2, color="#fdae61", linewidth=1.8, label="StandardScaler (Mean / Std)")
    ax2.axhline(0, color="#2b83ba", linestyle="--", linewidth=1.0, alpha=0.7, label="Mean = 0")
    ax2.set_ylabel("Standard Normalized (Z)")
    ax2.set_xlabel("Time (seconds)")
    ax2.text(0.01, 0.85, f"Range: [{s2.min():.2f}, {s2.max():.2f}] | Mean: {s2.mean():.2f} | Std: 1.0",
             transform=ax2.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="#cccccc"))
    ax2.legend(loc="upper right", fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    _maybe_save(fig, save_name)
    return fig, axes


# ─────────────────── 10. General-Purpose Signal Plotter ───────────────────

def plot_signals(
    signals: Union[pd.DataFrame, np.ndarray, Dict[str, Union[pd.DataFrame, np.ndarray]], List[Union[pd.DataFrame, np.ndarray]]],
    channels: Union[str, List[str]] = "chest_acc_x",
    sample_idx: Optional[int] = None,
    time_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    highlight_corrupt: bool = True,
    sentinel: float = SENTINEL_VALUE,
    figsize: Optional[Tuple[int, int]] = None,
    save_name: Optional[str] = None,
    palette: Optional[List[str]] = None,
    alpha: float = 0.85,
    lw: float = 1.8,
) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """
    General-purpose, versatile plotting function for wearable sensor signals.

    Features:
    ---------
    1. Flexible Inputs:
       - Single waveform DataFrame (250, 23)
       - Dataset DataFrame (e.g. train_df) with sample_idx=0
       - Dictionary of waveforms to overlay: {'Raw': raw_wave, 'Clean': clean_wave, 'Spline': spl_wave}
       - List of waveforms: [clean_wave, robust_wave]
       - 2D or 3D NumPy arrays (250, 23)
    2. Flexible Channels:
       - Single channel name: 'chest_acc_x'
       - List of channels: ['chest_acc_x', 'ankle_acc_y', 'arm_mag_z']
       - Body location alias: 'chest', 'ankle', 'arm' (plots all channels for that body part)
       - 'all': plots all 23 sensor channels
    3. Time Zoom:
       - time_range=(start_sec, end_sec) to zoom into specific time intervals.
    4. Auto-Denoising & Corruption Highlighting:
       - Detects 9999.0 sentinels and marks them with red points without blowing up the Y-axis scale.

    Parameters
    ----------
    signals : Union[pd.DataFrame, np.ndarray, Dict[str, Union[pd.DataFrame, np.ndarray]], List[Union[pd.DataFrame, np.ndarray]]]
        The signal(s) to plot.
    channels : Union[str, List[str]], default="chest_acc_x"
        Sensor channel(s) or body location ('chest', 'ankle', 'arm', 'all').
    sample_idx : Optional[int], default=None
        If passing a dataset DataFrame with 5,750 columns, row index to extract.
    time_range : Optional[Tuple[float, float]], default=None
        (t_min, t_max) in seconds to zoom into.
    title : Optional[str], default=None
        Figure super-title.
    highlight_corrupt : bool, default=True
        Whether to highlight 9999.0 sentinels with red dots.
    sentinel : float, default=9999.0
        Sentinel value representing corrupted readings.
    figsize : Optional[Tuple[int, int]], default=None
        Figure dimensions. Automatically sized if None.
    save_name : Optional[str], default=None
        Optional filename to auto-save into reports/figures/.
    palette : Optional[List[str]], default=None
        Custom colors for overlaid curves.
    alpha : float, default=0.85
        Line transparency.
    lw : float, default=1.8
        Line width.

    Returns
    -------
    Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]
        Figure and axes (single Axes if 1 channel, ndarray of Axes if multiple channels).

    Examples
    --------
    >>> fig, ax = plot_signals(clean_wave, channels="chest_acc_x")
    >>> fig, axes = plot_signals({"Raw": raw_wave, "Clean": clean_wave}, channels="chest")
    """
    def _coerce_to_waveform(item: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(item, pd.DataFrame):
            if item.shape[1] > NUM_CHANNELS and "t000_chest_acc_x" in item.columns:
                row_idx = sample_idx if sample_idx is not None else 0
                wave_df, _ = extract_sample_waveform(item, row_idx)
                return wave_df
            elif item.shape == (WINDOW_SIZE, NUM_CHANNELS) or item.index.name == "Time (seconds)":
                return item.copy()
            else:
                return item.copy()
        elif isinstance(item, np.ndarray):
            arr = item.squeeze()
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            time_idx = np.arange(len(arr)) / SAMPLING_RATE_HZ
            cols = SENSOR_CHANNELS[:arr.shape[1]] if arr.shape[1] <= NUM_CHANNELS else [f"ch_{i}" for i in range(arr.shape[1])]
            df = pd.DataFrame(arr, index=time_idx, columns=cols)
            df.index.name = "Time (seconds)"
            return df
        raise TypeError(f"Cannot coerce object of type {type(item)} to waveform DataFrame.")

    # 1. Normalize input to dictionary of labeled waveforms
    is_single_unnamed = False
    if isinstance(signals, (pd.DataFrame, np.ndarray)):
        signals_dict = {"Signal": _coerce_to_waveform(signals)}
        is_single_unnamed = True
    elif isinstance(signals, list):
        signals_dict = {f"Signal_{i+1}": _coerce_to_waveform(s) for i, s in enumerate(signals)}
    elif isinstance(signals, dict):
        signals_dict = {k: _coerce_to_waveform(v) for k, v in signals.items()}
    else:
        raise TypeError(f"Unsupported signals format: {type(signals)}")

    # 2. Normalize target channels
    if isinstance(channels, str):
        ch_lower = channels.lower()
        if ch_lower in BODY_LOCATIONS:
            target_channels = BODY_LOCATIONS[ch_lower]
        elif ch_lower in ("all", "all_channels"):
            target_channels = list(SENSOR_CHANNELS)
        else:
            target_channels = [channels]
    elif isinstance(channels, (list, tuple)):
        target_channels = list(channels)
    else:
        target_channels = list(SENSOR_CHANNELS)

    # 3. Setup subplots
    n_ch = len(target_channels)
    if figsize is None:
        figsize = (12, max(3.5, 2.5 * n_ch))

    fig, axes = plt.subplots(n_ch, 1, figsize=figsize, sharex=True, squeeze=False)
    axes_flat = axes.flatten()

    # Color palette
    if palette is None:
        if len(signals_dict) <= 4:
            palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
        else:
            palette = sns.color_palette("tab10", n_colors=len(signals_dict))

    # 4. Plot each channel
    for ax, ch in zip(axes_flat, target_channels):
        unit = get_channel_unit(ch)
        ylabel = f"{ch} ({unit})" if unit else ch

        for s_idx, (sig_name, sig_df) in enumerate(signals_dict.items()):
            if ch not in sig_df.columns:
                continue

            color = palette[s_idx % len(palette)]
            t = sig_df.index.to_numpy()
            s_raw = sig_df[ch].copy()

            if time_range is not None:
                mask = (t >= time_range[0]) & (t <= time_range[1])
                t_plot = t[mask]
                s_plot = s_raw.iloc[mask] if isinstance(s_raw, pd.Series) else s_raw[mask]
            else:
                t_plot = t
                s_plot = s_raw

            s_vals = s_plot.to_numpy(dtype=float)
            is_sentinel = (s_vals == sentinel)
            is_nan = np.isnan(s_vals)
            has_corrupt = np.any(is_sentinel | is_nan)

            label_to_use = None if is_single_unnamed else sig_name

            if highlight_corrupt and has_corrupt:
                s_clean = s_vals.copy()
                s_clean[is_sentinel] = np.nan
                ax.plot(t_plot, s_clean, label=label_to_use, color=color, linewidth=lw, alpha=alpha)

                if np.any(is_sentinel):
                    t_corrupt = t_plot[is_sentinel]
                    valid_vals = s_clean[~np.isnan(s_clean)]
                    base_h = float(np.median(valid_vals)) if len(valid_vals) else 0.0
                    sentinel_label = f"{sig_name} Sentinels ({len(t_corrupt)})" if not is_single_unnamed else f"Sentinels ({len(t_corrupt)})"
                    ax.scatter(t_corrupt, [base_h] * len(t_corrupt), color="#d62728", s=30, zorder=5, label=sentinel_label)
            else:
                ax.plot(t_plot, s_vals, label=label_to_use, color=color, linewidth=lw, alpha=alpha)

        ax.set_ylabel(ylabel, fontweight="bold")
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(loc="upper right", frameon=True, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes_flat[-1].set_xlabel("Time (seconds)", fontweight="bold")

    if title:
        fig.suptitle(title, y=1.002, fontweight="bold", fontsize=13)

    plt.tight_layout()
    _maybe_save(fig, save_name)

    ret_axes = axes_flat[0] if n_ch == 1 else axes_flat
    return fig, ret_axes
