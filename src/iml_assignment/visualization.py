# Library for visualization of data in the notebooks 

from pathlib import Path
from typing import List, Optional, Tuple, Union
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
    """Save the figure to reports/figures/ if save_name is provided."""
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
    Plot horizontal bar chart of the 12 activity classes with sample counts and percentages.
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
    Plot stacked bar chart showing breakdown of NaNs vs 9999 Sentinels per channel.
    """
    df = missing_summary.copy()
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df))
    width = 0.65

    # Compute percentage rates
    total_samples = df["NaN Count"] + df["Sentinel (9999) Count"]
    # We display counts or rates
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
    Plot waveforms grouped into 3 subplots: Chest, Ankle, and Arm (or a specific body location).
    """
    locations = [body_location] if body_location else ["Chest", "Ankle", "Arm"]
    n_plots = len(locations)

    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
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
    """
    fig, ax = plt.subplots(figsize=figsize)
    unit = get_channel_unit(channel)
    time_idx = raw_waveform.index

    # Raw signal (sentinels shown as red dots if present)
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
    Plot heatmap of correlation coefficients across all 23 sensor channels.
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
    Plot Power Spectral Density (PSD) using Welch's method for frequency analysis.
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
