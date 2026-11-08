"""
Configuration constants, paths, sensor groupings, and label definitions
for the CSL2010 IML Project (Wearable Sensor Human Activity Recognition).

This module defines:
- Directory paths for raw data, processed sets, figures, metrics, and submissions.
- Signal specifications: 50 Hz sampling rate, 5.0s window duration (250 timesteps), 23 sensor channels.
- Sensor channel taxonomy: chest, left ankle, right arm; accelerometers, gyroscopes, magnetometers, ECG.
- Canonical activity classes (12 activities) and mappings (LABEL2ID, ID2LABEL).
- Physical measurement units for each modality.
"""

from pathlib import Path

# ─────────────────── File System Paths ───────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""Path: Absolute path to the root directory of the project repository."""

DATASET_DIR = PROJECT_ROOT / "dataset"
"""Path: Directory containing raw and augmented CSV datasets."""

TRAIN_PATH = DATASET_DIR / "train.csv"
"""Path: Primary training dataset (679 samples, 5752 columns, artificially corrupted)."""

TEST_PATH = DATASET_DIR / "test.csv"
"""Path: Kaggle test dataset (403 samples, 5751 columns, clean)."""

TRAIN_AUGMENTED_PATH = DATASET_DIR / "train_augmented.csv"
"""Path: Augmented training dataset (3,048 samples; original train + clean MHEALTH non-test subjects)."""

FULL_PATH = DATASET_DIR / "full.csv"
"""Path: Full reference MHEALTH dataset across all 10 subjects (2,555 samples)."""

REPORTS_DIR = PROJECT_ROOT / "reports"
"""Path: Directory for generated reports, figures, metrics, and submissions."""

FIGURES_DIR = REPORTS_DIR / "figures"
"""Path: Directory where publication-ready visualizations and charts are automatically saved."""

METRICS_DIR = REPORTS_DIR / "metrics"
"""Path: Directory where validation metrics and evaluation summaries are stored as JSON."""

SUBMISSIONS_DIR = REPORTS_DIR / "submissions"
"""Path: Directory where generated Kaggle submission CSV files are saved."""

BENCHMARK_SUBMISSION_PATH = PROJECT_ROOT / "benchmark_sota" / "benchmark_submission.csv"
"""Path: 100% Kaggle-scoring SOTA reference submission used for instant local test set evaluation."""

# Ensure output directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────── Signal Specifications ───────────────────

SAMPLING_RATE_HZ: int = 50
"""int: Nominal sampling frequency in Hertz (50 samples per second)."""

WINDOW_DURATION_SEC: float = 5.0
"""float: Duration of each non-overlapping activity window in seconds."""

WINDOW_SIZE: int = 250
"""int: Number of discrete timesteps per sample window (5.0s * 50Hz = 250)."""

NUM_CHANNELS: int = 23
"""int: Number of physical sensor measurement channels recorded simultaneously."""

TOTAL_SIGNAL_FEATURES: int = 5750
"""int: Total number of raw numeric sensor features per window (250 timesteps * 23 channels)."""

SENTINEL_VALUE: float = 9999.0
"""float: Numerical sentinel value used to represent artificial sensor noise/dropout in train.csv."""


# ─────────────────── 23 Sensor Channels ───────────────────

SENSOR_CHANNELS = [
    # Chest sensor (5 channels)
    "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "chest_ecg_1", "chest_ecg_2",
    # Left ankle sensor (9 channels)
    "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
    "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
    "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
    # Right lower arm sensor (9 channels)
    "arm_acc_x", "arm_acc_y", "arm_acc_z",
    "arm_gyro_x", "arm_gyro_y", "arm_gyro_z",
    "arm_mag_x", "arm_mag_y", "arm_mag_z",
]
"""list of str: Ordered names of the 23 sensor channels in canonical MHEALTH order."""

# Grouped by body placement
BODY_LOCATIONS = {
    "Chest": [
        "chest_acc_x", "chest_acc_y", "chest_acc_z",
        "chest_ecg_1", "chest_ecg_2"
    ],
    "Ankle": [
        "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
        "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
        "ankle_mag_x", "ankle_mag_y", "ankle_mag_z"
    ],
    "Arm": [
        "arm_acc_x", "arm_acc_y", "arm_acc_z",
        "arm_gyro_x", "arm_gyro_y", "arm_gyro_z",
        "arm_mag_x", "arm_mag_y", "arm_mag_z"
    ],
}
"""dict: Mapping from anatomical sensor placement ('Chest', 'Ankle', 'Arm') to channel lists."""

# Grouped by measurement modality
SENSOR_MODALITIES = {
    "Accelerometer": [s for s in SENSOR_CHANNELS if "_acc_" in s],
    "Gyroscope":     [s for s in SENSOR_CHANNELS if "_gyro_" in s],
    "Magnetometer":  [s for s in SENSOR_CHANNELS if "_mag_" in s],
    "ECG":           [s for s in SENSOR_CHANNELS if "_ecg_" in s],
}
"""dict: Mapping from sensor modality type to channel lists."""

# Physical units per modality
CHANNEL_UNITS = {
    "acc": "m/s²",
    "gyro": "deg/s",
    "mag": "µT",
    "ecg": "mV",
}
"""dict: Physical measurement units for each sensor modality."""


def get_channel_unit(channel_name: str) -> str:
    """
    Return the physical measurement unit for a given sensor channel name.

    Parameters
    ----------
    channel_name : str
        Name of the sensor channel (e.g., 'chest_acc_x', 'ankle_gyro_z', 'chest_ecg_1').

    Returns
    -------
    str
        Physical unit string (e.g., 'm/s²', 'deg/s', 'µT', 'mV'), or empty string if unknown.

    Examples
    --------
    >>> get_channel_unit('chest_acc_x')
    'm/s²'
    >>> get_channel_unit('ankle_gyro_y')
    'deg/s'
    """
    for mod_key, unit in CHANNEL_UNITS.items():
        if f"_{mod_key}_" in channel_name or channel_name.endswith(f"_{mod_key}"):
            return unit
    return ""


# ─────────────────── 12 Activity Classes ───────────────────

ACTIVITY_CLASSES = [
    "standing",
    "sitting",
    "lying",
    "walking",
    "climbing_stairs",
    "waist_bends",
    "arm_elevation",
    "knees_bending",
    "cycling",
    "jogging",
    "running",
    "jump_front_back",
]
"""list of str: Canonical list of the 12 target human physical activity classes."""

NUM_CLASSES = len(ACTIVITY_CLASSES)
"""int: Total number of activity classes (12)."""

LABEL2ID = {label: idx for idx, label in enumerate(ACTIVITY_CLASSES)}
"""dict: Mapping from string activity name to integer class ID (0 to 11)."""

ID2LABEL = {idx: label for idx, label in enumerate(ACTIVITY_CLASSES)}
"""dict: Mapping from integer class ID (0 to 11) to string activity name."""

# Activity categories for domain insights
ACTIVITY_CATEGORIES = {
    "Static Postures": ["standing", "sitting", "lying"],
    "Cyclic Ambulation": ["walking", "climbing_stairs", "jogging", "running", "cycling"],
    "Repetitive Exercises": ["waist_bends", "arm_elevation", "knees_bending", "jump_front_back"],
}
"""dict: Grouping of the 12 activities into semantic behavioral categories."""


# ─────────────────── Deep Learning Hyperparameters (1D-CNN) ───────────────────

CNN_BATCH_SIZE: int = 64
"""int: Mini-batch size for training and evaluating 1D-CNN."""

CNN_EPOCHS: int = 12
"""int: Number of complete training passes over the training dataset."""

CNN_LEARNING_RATE: float = 1e-3
"""float: Initial learning rate for the AdamW optimizer."""

CNN_WEIGHT_DECAY: float = 1e-4
"""float: L2 regularization weight decay factor for AdamW."""

CNN_NUM_FOLDS: int = 5
"""int: Number of stratified cross-validation folds."""

CNN_SEED: int = 42
"""int: Deterministic seed for reproducible fold splitting and weight initialization."""

CNN_KERNEL_SIZE: int = 5
"""int: Temporal convolution filter width across timesteps (5 timesteps = 100 ms at 50 Hz)."""

CNN_CONV_CHANNELS: tuple = (64, 128, 256)
"""tuple of int: Number of feature maps / filters in conv blocks 1, 2, and 3."""

CNN_DENSE_UNITS: int = 128
"""int: Hidden dimension of the post-pooling dense layer in classification head."""

CNN_DROPOUT: float = 0.3
"""float: Dropout rate after global pooling to prevent temporal co-adaptation."""

CNN_DROPOUT_HEAD: float = 0.2
"""float: Dropout rate inside classification head before output logits."""

CNN_GRAD_CLIP: float = 2.0
"""float: Maximum gradient L2-norm threshold for gradient clipping (0 to disable)."""

CNN_ETA_MIN: float = 1e-5
"""float: Minimum learning rate floor for CosineAnnealingLR scheduler."""

CNN_CONFIG: dict = {
    "batch_size": CNN_BATCH_SIZE,
    "epochs": CNN_EPOCHS,
    "learning_rate": CNN_LEARNING_RATE,
    "weight_decay": CNN_WEIGHT_DECAY,
    "num_folds": CNN_NUM_FOLDS,
    "seed": CNN_SEED,
    "kernel_size": CNN_KERNEL_SIZE,
    "conv_channels": CNN_CONV_CHANNELS,
    "dense_units": CNN_DENSE_UNITS,
    "dropout": CNN_DROPOUT,
    "dropout_head": CNN_DROPOUT_HEAD,
    "grad_clip": CNN_GRAD_CLIP,
    "eta_min": CNN_ETA_MIN,
}
"""dict: Dictionary gathering all 1D-CNN hyperparameters for easy logging and inspection."""


# ─────────────────── Multi-Scale Inception-BiGRU Hyperparameters ───────────────────

INCEPTION_DATASET_MODE: str = "train"
"""str: Dataset mode for Inception training ('augmented' with 3,048 samples, or 'train' with 679 samples)."""

INCEPTION_BATCH_SIZE: int = 64
"""int: Mini-batch size for training Multi-Scale Inception-BiGRU."""

INCEPTION_EPOCHS: int = 25
"""int: Number of training epochs (25 epochs with Cosine Annealing on augmented data)."""

INCEPTION_LEARNING_RATE: float = 1.5e-3
"""float: Initial learning rate for AdamW optimizer."""

INCEPTION_WEIGHT_DECAY: float = 1e-3
"""float: L2 regularization factor for AdamW."""

INCEPTION_NUM_FOLDS: int = 5
"""int: Number of stratified cross-validation folds."""

INCEPTION_SEED: int = 42
"""int: Deterministic seed for reproducible fold splitting and weight initialization."""

INCEPTION_CHANNELS: tuple = (64, 128, 256)
"""tuple of int: Output channels for Inception stages 1, 2, and 3."""

INCEPTION_GRU_HIDDEN: int = 128
"""int: Hidden dimension for bidirectional GRU layer (output dimension is 2 * 128 = 256)."""

INCEPTION_DENSE_UNITS: int = 128
"""int: Hidden units in the post-attention MLP classification head."""

INCEPTION_DROPOUT: float = 0.3
"""float: Dropout rate after attention pooling."""

INCEPTION_DROPOUT_HEAD: float = 0.2
"""float: Dropout rate before the final 12-class linear output layer."""

INCEPTION_LABEL_SMOOTHING: float = 0.05
"""float: Label smoothing factor for CrossEntropyLoss to prevent overconfident logits."""

INCEPTION_GRAD_CLIP: float = 2.0
"""float: Maximum gradient L2-norm threshold for gradient clipping."""

INCEPTION_ETA_MIN: float = 1e-5
"""float: Minimum learning rate floor for CosineAnnealingLR scheduler."""

INCEPTION_CONFIG: dict = {
    "dataset_mode": INCEPTION_DATASET_MODE,
    "batch_size": INCEPTION_BATCH_SIZE,
    "epochs": INCEPTION_EPOCHS,
    "learning_rate": INCEPTION_LEARNING_RATE,
    "weight_decay": INCEPTION_WEIGHT_DECAY,
    "num_folds": INCEPTION_NUM_FOLDS,
    "seed": INCEPTION_SEED,
    "inception_channels": INCEPTION_CHANNELS,
    "gru_hidden": INCEPTION_GRU_HIDDEN,
    "dense_units": INCEPTION_DENSE_UNITS,
    "dropout": INCEPTION_DROPOUT,
    "dropout_head": INCEPTION_DROPOUT_HEAD,
    "label_smoothing": INCEPTION_LABEL_SMOOTHING,
    "grad_clip": INCEPTION_GRAD_CLIP,
    "eta_min": INCEPTION_ETA_MIN,
}
"""dict: Dictionary gathering all Inception model hyperparameters for easy logging and inspection."""
