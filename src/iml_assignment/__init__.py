"""
IML Assignment: Wearable Sensor Human Activity Recognition
"""

from iml_assignment.config import (
    TRAIN_PATH,
    TEST_PATH,
    TRAIN_AUGMENTED_PATH,
    FULL_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    SUBMISSIONS_DIR,
    SENSOR_CHANNELS,
    BODY_LOCATIONS,
    SENSOR_MODALITIES,
    ACTIVITY_CLASSES,
    ACTIVITY_CATEGORIES,
    SAMPLING_RATE_HZ,
    WINDOW_SIZE,
    NUM_CHANNELS,
    TOTAL_SIGNAL_FEATURES,
    SENTINEL_VALUE,
    LABEL2ID,
    ID2LABEL,
    get_channel_unit,
)

from iml_assignment.data import (
    load_data,
    get_sensor_columns,
    verify_column_ordering,
    get_dataset_summary,
    get_class_distribution,
    analyze_missing_and_sentinels,
    extract_sample_waveform,
    clean_sample_waveform,
    clean_dataset,
    compute_channel_statistics,
    ChannelScaler,
)

from iml_assignment.visualization import (
    plot_class_distribution,
    plot_missingness_by_channel,
    plot_sample_waveforms,
    plot_raw_vs_cleaned,
    plot_activity_comparison,
    plot_sensor_correlations,
    plot_frequency_spectrum,
    plot_imputation_comparison,
    plot_scaling_comparison,
    plot_signals,
)

from iml_assignment.features import (
    extract_window_features,
    extract_features_dataset,
)

from iml_assignment.models.baseline import (
    get_baseline_model,
)

from iml_assignment.evaluate import (
    compute_classification_metrics,
    get_classification_report_df,
    plot_confusion_matrix,
    save_metrics,
)

from iml_assignment.train import (
    cross_validate_model,
    create_submission_file,
)

__all__ = [
    # Config
    "TRAIN_PATH",
    "TEST_PATH",
    "TRAIN_AUGMENTED_PATH",
    "FULL_PATH",
    "FIGURES_DIR",
    "METRICS_DIR",
    "SUBMISSIONS_DIR",
    "SENSOR_CHANNELS",
    "BODY_LOCATIONS",
    "SENSOR_MODALITIES",
    "ACTIVITY_CLASSES",
    "ACTIVITY_CATEGORIES",
    "SAMPLING_RATE_HZ",
    "WINDOW_SIZE",
    "NUM_CHANNELS",
    "TOTAL_SIGNAL_FEATURES",
    "SENTINEL_VALUE",
    "LABEL2ID",
    "ID2LABEL",
    "get_channel_unit",
    # Data
    "load_data",
    "get_sensor_columns",
    "verify_column_ordering",
    "get_dataset_summary",
    "get_class_distribution",
    "analyze_missing_and_sentinels",
    "extract_sample_waveform",
    "clean_sample_waveform",
    "clean_dataset",
    "compute_channel_statistics",
    "ChannelScaler",
    # Visualization
    "plot_class_distribution",
    "plot_missingness_by_channel",
    "plot_sample_waveforms",
    "plot_raw_vs_cleaned",
    "plot_activity_comparison",
    "plot_sensor_correlations",
    "plot_frequency_spectrum",
    "plot_imputation_comparison",
    "plot_scaling_comparison",
    "plot_signals",
    # Features
    "extract_window_features",
    "extract_features_dataset",
    # Models
    "get_baseline_model",
    # Evaluation
    "compute_classification_metrics",
    "get_classification_report_df",
    "plot_confusion_matrix",
    "save_metrics",
    # Training & Cross Validation
    "cross_validate_model",
    "create_submission_file",
]
