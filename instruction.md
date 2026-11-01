# CSL2010 IML Project

---

## 1. Overview

This is an individual course project in which students will build a multi-class classifier that takes five-second windows of wearable sensor signals and predicts one of 12 physical activities.

The training dataset contains labeled windows collected from five subjects, while the competition test set contains windows from three subjects held out from training. Students must predict one activity label for every test ID and submit their predictions directly on Kaggle using the prescribed sample submission format.

The evaluation metric is classification accuracy. The Public leaderboard provides interim feedback, while final rankings are determined exclusively by the Private leaderboard score.

---

## 2. Task

Build a multi-class classifier for physical activities using multivariate wearable-sensor time-series data. Each example represents a five-second window, and its target is one of the 12 activity classes listed below.

### 2.1 Sensor Signals

Each window contains 250 samples collected at a nominal sampling rate of 50 Hz and 23 sensor channels from the chest, left ankle, and right lower arm.

The 5,750 signal columns are arranged in time-major order, with feature names ranging from `t000_<channel>` through `t249_<channel>`. All signal values are numeric. Timestamps and subject/session identifiers are not provided as model input.

The windows are non-overlapping. A window receives an activity label only when at least 80% of its samples share that activity. Windows dominated by background activity or containing insufficiently pure transitions are excluded.

### 2.2 Activity Classes

The classifier must distinguish the following 12 activities:

1. `standing`
2. `sitting`
3. `lying`
4. `walking`
5. `climbing_stairs`
6. `waist_bends`
7. `arm_elevation`
8. `knees_bending`
9. `cycling`
10. `jogging`
11. `running`
12. `jump_front_back`

---

## 3. Dataset Description

The project dataset consists of three CSV files:

* **`train.csv`**: Contains labeled sensor windows. Each row includes an `id`, 5,750 numeric signal columns, and a `target` column containing the activity class.
* **`test.csv`**: Contains 403 unlabeled sensor windows. Each row includes a synthetic `id` and the same 5,750 signal columns as the training data. Subject and session metadata are unavailable.
* **`sample_submission.csv`**: Provides the required submission structure, with one row for every test ID and two columns: `id` and `target`. The starter file assigns `walking` to every test row; students must replace these values with their own predictions.

The dataset description specifies 679 labeled training windows. Students should verify the actual number of rows in the supplied `train.csv` before beginning their experiments.

### 3.1 Dataset Characteristics

* Each example represents a five-second, non-overlapping sensor window.
* Each window has 250 time samples and 23 sensor channels, giving 5,750 signal features.
* The data originates from subject recordings.
* Training and test subjects are separated: five subjects contribute to training, while three different subjects contribute to the test set.
* Test labels are hidden and must not be used during training or model selection.
* Background-majority and insufficiently pure transition windows have been excluded.

Since the test subjects are unseen during training, students should consider generalization across subjects when designing their features, models, and validation strategy.

---

## 4. Data Exploration and Preprocessing

Students are expected to explore the dataset before training their models. The exploration should include:

1. Checking the dimensions of the training and test datasets.
2. Verifying the number of signal features and inspecting the feature naming and ordering conventions.
3. Examining the distribution of the 12 target classes.
4. Computing basic statistics of sensor channels, including means, standard deviations, minima, and maxima.
5. Checking for missing values, constant features, and unusual sensor readings.
6. Considering suitable preprocessing, normalization, and feature extraction methods.

Any preprocessing operation that learns parameters from data must be fitted using the training portion of each validation split. Validation data must not influence fitted preprocessing parameters.

---

## 5. Validation and Evaluation

Students must evaluate their models using training data only. A suitable validation strategy should be selected and justified.

Because windows originate from multiple subjects and the competition test subjects are held out, students should consider subject-wise validation when subject identifiers are available. If such identifiers are unavailable in the supplied files, this limitation must be acknowledged, and an appropriate alternative validation strategy must be explained.

The evaluation metric is multiclass accuracy:

$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of evaluated examples}}$$

The competition leaderboard is divided into two parts:
* **Public leaderboard:** Interim feedback based on 202 test rows.
* **Private leaderboard:** Final evaluation based on the remaining 201 test rows.

The final ranking is determined by the Private leaderboard score. Up to two submissions may be selected for final Private scoring.

Students must not attempt to obtain or infer hidden test labels through unauthorized means.

---

## 6. Kaggle Account and Submission Requirements

The following requirements are mandatory. Failure to satisfy them will render the project submission invalid:

1. **Create a Kaggle account using your institute email.**
   Register on Kaggle using your official institute-provided email address. Complete any required email verification.
2. **Set your profile name appropriately.**
   Use the prescribed format `Name (Roll No.)` for your Kaggle username/profile identification. If Kaggle does not permit the required characters in the username field, use the closest permitted username and set the display name to `Name (Roll No.)`, ensuring that your identity and roll number are clearly identifiable.
3. **Enable email visibility to the host.**
   In the Kaggle competition interface, open the **Team** section and enable **Share your email address with the host**. Your institute email address must be visible to the host.
4. **Submit predictions through Kaggle.**
   Upload your final prediction CSV directly to the designated competition. Submitting code or a CSV file only to the course instructor does not replace the required Kaggle submission.

> **Important:** If you do not use your institute email, do not follow the prescribed name/roll-number format, or do not enable email sharing with the host, your project submission will be considered invalid.

---

## 7. Procedure for Submitting on Kaggle

Follow the steps below to complete the competition submission:

1. **Create and verify your account:** Visit [https://www.kaggle.com/](https://www.kaggle.com/), create an account using your institute email address, and complete the verification process if required.
2. **Set your name and roll number:** Update your Kaggle username/profile identification to follow the required `Name (Roll No.)` format, subject to Kaggle's username restrictions.
3. **Open the designated competition:** Use the competition link provided by the course instructor. Read the competition rules and accept them if prompted.
4. **Enable host email sharing:** Open the competition's Team section and enable **Share your email address with the host**. Verify that the option is enabled before submitting.
5. **Generate predictions:** Train and validate your chosen model using the permitted training data. Use the selected model to predict the activity label for every row in `test.csv`.
6. **Prepare the submission CSV:** Create a CSV file containing exactly two columns: `id` and `target`. Include one prediction for every test ID, preserve the required IDs, and use only valid activity class names as target values.
7. **Check the submission file:** Compare your file with `sample_submission.csv`. Ensure that all test IDs appear exactly once, the number of prediction rows matches the test dataset, and no target values are missing or invalid.
8. **Upload the predictions:** Open the competition's submission page, select **Submit Predictions** or the equivalent submission control, upload your CSV file, and confirm the submission.
9. **Verify the result:** Wait for Kaggle to process the submission. Confirm that it has been accepted and that a leaderboard score is displayed when available. Resolve any file-format or submission errors.

---

## 8. Submission File Format

The submitted CSV must contain the columns `id` and `target`, with one row for every test ID. An example is shown below:

```csv
id,target
TE000001,walking
TE000002,standing
TE000003,climbing_stairs
```

The example labels are illustrative only. Actual predictions must be generated by the student's trained model. Do not include additional columns, an index column, or explanatory text in the CSV file.

---

## 9. Project Deliverables

Each student must complete the following deliverables:

1. Explore the training data, including its shape, class distribution, and basic sensor-channel statistics.
2. Design and implement a suitable feature-extraction or sequence-modeling approach for multivariate time-series windows.
3. Train and evaluate candidate models using an appropriate validation strategy based exclusively on training data.
4. Generate predictions for all test examples and submit them directly to Kaggle.
5. Add clear and meaningful comments throughout the code explaining the approach, key design decisions, and important implementation details.
6. Ensure that all mandatory Kaggle account and host-contact requirements have been satisfied.

---

## 10. Evaluation (Total: 10%)

The project contributes 10% to the course evaluation:

* **5 marks:** Performance in the Kaggle Private leaderboard ranking.
* **5 marks:** **Viva**, covering the student's understanding of the dataset, preprocessing, feature engineering, model selection, validation, results, and implementation.

Students should be prepared to explain and defend their own implementation and design choices during the viva.

---

## 11. Academic Integrity and AI Policy

* Do not use AI tools to directly write the project code.
* AI tools may be used for debugging, clarifying concepts, understanding errors, and learning how standard methods work.
* Plagiarism checks may be performed. Work found to have been produced by AI in violation of this policy, or to contain plagiarism, may be considered invalid.
* All project work must be the student's own. The code, feature design, model choices, and experimental reasoning must be produced by the student.
* Standard libraries such as scikit-learn, PyTorch, and TensorFlow are allowed and encouraged.
* Students must be able to explain their code, experiments, and results during the viva.

---

## 12. Final Checklist

Before the project deadline, verify that every item below is complete:

- [x] Kaggle account created using the institute email.
- [x] Username/profile identification follows `Name (Roll No.)`.
- [x] Share your email address with the host is enabled.
- [ ] Data exploration and preprocessing completed.
- [ ] Models trained and evaluated using training data only.
- [ ] Test predictions generated for every test ID.
- [ ] Submission CSV checked and uploaded successfully to Kaggle.
- [ ] Project code and design decisions understood for the viva.
