# Wearable-Sensor Activity Data Analysis Report

## 1. Executive summary

The dataset is suitable for 12-class human-activity recognition, but it should not be
modelled directly in its raw tabular form. Each example is a five-second multivariate
time-series window with 250 time steps and 23 channels. The most important issue is
data quality: **17.61% of training signal cells are corrupted**. This consists of
13.24% standard missing values and 4.37% `9999.0` sentinel values.

The class distribution is reasonably balanced for 11 activities, but
`jump_front_back` has only 19 examples (2.80%). The channels also have materially
different scales: accelerometer and magnetometer ranges are much larger than ECG and
gyroscope ranges. Cleaning, channel-wise scaling, and time-series-aware validation
should therefore be treated as mandatory.

## 2. Dataset structure

The notebook reports:

| Property | Training | Test |
|---|---:|---:|
| Rows | 679 | 403 |
| Total columns | 5,752 | 5,751 |
| Sensor signal columns | 5,750 | 5,750 |
| Time steps per window | 250 | 250 |
| Sensor channels | 23 | 23 |
| Window duration | 5.0 seconds | 5.0 seconds |
| Sampling rate | 50 Hz | 50 Hz |
| Target | Present | Hidden |
| Number of classes | 12 | Hidden |

The notebook verifies that the columns are in the expected time-major ordering. This
means the signals can safely be reshaped into `(samples, 250, 23)` for time-series
analysis and neural models.

## 3. Target distribution

| Activity | Count | Share |
|---|---:|---:|
| standing | 59 | 8.69% |
| sitting | 59 | 8.69% |
| lying | 66 | 9.72% |
| walking | 63 | 9.28% |
| climbing_stairs | 56 | 8.25% |
| waist_bends | 53 | 7.81% |
| arm_elevation | 55 | 8.10% |
| knees_bending | 55 | 8.10% |
| cycling | 67 | 9.87% |
| jogging | 66 | 9.72% |
| running | 61 | 8.98% |
| jump_front_back | 19 | 2.80% |

### Interpretation

The first 11 classes are relatively close in size, with counts from 53 to 67.
`jump_front_back` is the clear minority class and has only about one-third as many
examples as the better-represented classes. Overall accuracy alone may therefore hide
poor performance on this activity.

Use class-wise precision, recall, F1, and a confusion matrix during model selection.
For training, use stratified splits and consider class-weighted loss or carefully
controlled minority oversampling.

## 4. Signal quality and preprocessing

The notebook reports:

- Total corrupted cells: **687,583 (17.61%)**
- Standard NaNs: **516,835 (13.24%)**
- `9999.0` sentinel values: **170,748 (4.37%)**

The existing notebook cleaning workflow replaces sentinel values with missing values,
linearly interpolates along time for each channel, forward-fills and backward-fills
edge gaps, and uses zero only for any remaining unresolved values. This is a sensible
baseline because it preserves temporal continuity better than replacing every bad
value with zero.

### Required safeguards

1. Apply the same cleaning procedure to train and test.
2. Record the corruption rate per window and per channel.
3. Add a missingness/corruption indicator as an optional input feature; corruption
   patterns may contain useful information, but the model should not depend on the
   artificial value `9999`.
4. Fit any learned imputation or scaling parameters on the training fold only.
5. Verify after cleaning that no NaNs or sentinel values reach the model.

## 5. Channel statistics and implications

The cleaned channel statistics show substantial scale differences. Representative
values from the notebook include:

| Channel | Min | Max | Mean | Std |
|---|---:|---:|---:|---:|
| chest_acc_x | -67.0101 | 52.0094 | -7.4131 | 13.8261 |
| chest_acc_y | -33.6870 | 33.3260 | -0.1311 | 7.3245 |
| chest_acc_z | -48.1111 | 46.0819 | -1.0462 | 11.0455 |
| chest_ecg_1 | -10.5677 | 10.5758 | -0.0001 | 2.1545 |
| ankle_acc_z | -70.2701 | 69.3823 | -0.3815 | 15.5039 |
| ankle_gyro_x | -4.0772 | 4.3980 | 0.1584 | 1.0143 |
| ankle_mag_x | -853.8707 | 865.8636 | 1.0747 | 188.5684 |
| arm_mag_z | -962.4104 | 956.4319 | -0.9784 | 188.1327 |

Magnetometer channels have ranges near +/-900, while gyroscope channels are near
single-digit values. Without scaling, a model can overemphasize high-magnitude
channels for numerical rather than predictive reasons.

Recommended normalization:

- Start with per-channel training-fold standardization.
- Compare this with robust scaling because the ranges suggest outliers.
- For deep models, normalize each channel consistently across the full window.
- Do not normalize using the combined train and test data.

## 6. What the visual analysis suggests

The notebook creates visualizations for:

- class distribution;
- corruption by channel;
- raw versus cleaned waveform;
- cleaned multichannel waveform;
- walking versus running;
- sensor correlations;
- ankle-channel frequency spectrum.

These are appropriate diagnostics. In particular, the raw-versus-cleaned plot should
be included in the final submission/report because it demonstrates why imputation is
needed. The walking-versus-running and frequency-spectrum plots support using
temporal and frequency-domain features rather than relying only on flattened values.

The activities are time-dependent: walking, jogging, running, cycling, stair
climbing, and jumping should be distinguished partly by periodicity and energy.
Static activities should benefit more from posture, channel means, variance, and
orientation-related features.

## 7. Recommended modelling strategy

### Stage 1: establish a reliable baseline

1. Reshape each row to `(250, 23)`.
2. Replace `9999.0` with NaN and interpolate within each channel.
3. Standardize channels using training-fold statistics.
4. Train a simple baseline using engineered features:
   - mean, standard deviation, min, max, range, median;
   - signal energy and zero-crossing counts;
   - first-difference statistics;
   - axis magnitude for accelerometer and gyroscope groups;
   - dominant frequency and band power for selected accelerometer/gyro channels.
5. Compare logistic regression, random forest/extra trees, and a tree boosting model.

This baseline is valuable because it is easier to debug and gives interpretable
feature importance.

### Stage 2: model the waveform

Compare the baseline with a compact 1D CNN or CNN plus recurrent/attention block.
Use early stopping and dropout because there are only 679 labelled windows.

A practical model comparison is:

1. engineered-feature classifier;
2. 1D CNN on the cleaned `(250, 23)` sequence;
3. ensemble of the strongest feature model and neural model if their errors differ.

Do not treat the existing 100% benchmark result in `benchmark_sota/` as evidence of
competition performance: it is evaluated on `full.csv`, which contains 2,555 samples
across 10 subjects, not the 679-row competition training set, and its validation
design may not represent the held-out competition subjects. It should be treated as a
technical benchmark requiring independent leakage checks, not as the expected Kaggle
score.

## 8. Validation plan

The project instructions state that training and test subjects are separated, but the
provided model input does not include subject identifiers. Consequently, true
subject-wise validation cannot be reproduced from `train.csv` alone.

Use the following fallback:

- repeated stratified 5-fold cross-validation;
- identical preprocessing fitted inside each fold;
- fixed random seeds;
- report mean and standard deviation of accuracy and macro F1;
- inspect per-class recall, especially `jump_front_back`;
- reserve one final model-selection decision before fitting on all training data.

If subject metadata can be obtained legitimately from the instructor, replace the
fallback with leave-one-subject-out validation. Do not infer hidden test labels or use
the Kaggle test set for model selection.

## 9. How to proceed now

### Priority 1 — data pipeline

- Implement one reusable cleaning function for train and test.
- Add assertions for shape `(n, 250, 23)`, no remaining NaNs/sentinels, and correct
  class names.
- Add channel-wise scaling inside the validation pipeline.

### Priority 2 — trustworthy experiments

- Run the engineered-feature baseline with repeated stratified cross-validation.
- Run a compact 1D CNN under the same folds.
- Save accuracy, macro F1, per-class recall, and confusion matrices under
  `reports/metrics/`.

### Priority 3 — improve weak classes

- Use class-weighted loss and compare against minority oversampling.
- Review errors involving `jump_front_back`, and likely motion-confusion pairs such
  as walking/jogging/running and stair climbing.
- Add frequency and motion-energy features for these classes.

### Priority 4 — final submission

- Select the model using only training-fold results.
- Retrain it on all 679 labelled windows.
- Predict all 403 test rows.
- Verify that the submission has exactly the test IDs and valid activity labels.
- Submit only after checking that preprocessing and feature ordering match training.

## 10. Conclusion

The data has a clear and workable structure, but the raw values are not ready for
modelling. The decisive actions are to clean the 17.61% corrupted signal cells,
normalize the 23 channels, preserve temporal information, and evaluate with metrics
that expose minority-class failures. A small, carefully validated CNN should be
compared against a strong engineered-feature baseline; the best choice should be
based on cross-validation stability rather than a single score or the existing
perfect benchmark result.

**Evidence source:** the executed outputs and figures in
[`notebooks/01_eda_and_reporting.ipynb`](../notebooks/01_eda_and_reporting.ipynb).
