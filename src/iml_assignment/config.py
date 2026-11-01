"""
Configuration constants, paths, sensor groupings, and label definitions
for the CSL2010 IML Project (Wearable Sensor HAR).
"""

from pathlib import Path

# ─────────────────── File System Paths ───────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"

TRAIN_PATH = DATASET_DIR / "train.csv"
TEST_PATH = DATASET_DIR / "test.csv"
TRAIN_AUGMENTED_PATH = DATASET_DIR / "train_augmented.csv"
FULL_PATH = DATASET_DIR / "full.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
SUBMISSIONS_DIR = REPORTS_DIR / "submissions"

# Ensure output directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────── Signal Specifications ───────────────────

SAMPLING_RATE_HZ = 50           # Nominal sampling rate
WINDOW_DURATION_SEC = 5.0       # 5-second non-overlapping windows
WINDOW_SIZE = 250               # 250 timesteps per window (5.0s * 50Hz)
NUM_CHANNELS = 23               # 23 sensor channels
TOTAL_SIGNAL_FEATURES = 5750    # 250 * 23

SENTINEL_VALUE = 9999.0         # Artificial corruption value injected in train.csv


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

# Grouped by measurement modality
SENSOR_MODALITIES = {
    "Accelerometer": [s for s in SENSOR_CHANNELS if "_acc_" in s],
    "Gyroscope":     [s for s in SENSOR_CHANNELS if "_gyro_" in s],
    "Magnetometer":  [s for s in SENSOR_CHANNELS if "_mag_" in s],
    "ECG":           [s for s in SENSOR_CHANNELS if "_ecg_" in s],
}

# Physical units per modality
CHANNEL_UNITS = {
    "acc": "m/s²",
    "gyro": "deg/s",
    "mag": "µT",
    "ecg": "mV",
}


def get_channel_unit(channel_name: str) -> str:
    """Return the physical unit for a given sensor channel name."""
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

NUM_CLASSES = len(ACTIVITY_CLASSES)

LABEL2ID = {label: idx for idx, label in enumerate(ACTIVITY_CLASSES)}
ID2LABEL = {idx: label for idx, label in enumerate(ACTIVITY_CLASSES)}

# Activity categories for domain insights
ACTIVITY_CATEGORIES = {
    "Static Postures": ["standing", "sitting", "lying"],
    "Cyclic Ambulation": ["walking", "climbing_stairs", "jogging", "running", "cycling"],
    "Repetitive Exercises": ["waist_bends", "arm_elevation", "knees_bending", "jump_front_back"],
}
