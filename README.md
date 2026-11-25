# Dyskinetic CP Reaching Benchmark

Benchmark of classical ML, recurrent networks (Simple RNN, LSTM-lite, GRU-lite, 1D Flip-Flop RNN) and a DONN (ResHopf) model on upper-limb reaching movements, distinguishing children with **dyskinetic cerebral palsy (DCP)** from **typically developing (TD)** children.

## Data source

The underlying pose/joint-angle data come from the AVI-DYS project
(<https://researchportal.be/en/project/automated-video-based-assessment-dyskinesia-cerebral-palsy-using-markerless-pose-estimation>),
analysed retrospectively from IDCA
(<https://researchportal.be/en/project/instrumented-dystonia-and-choreoathetosis-assessment-protocol-idca-upper-limb-movements>).
See `README.txt` and `README revision.txt` for the original dataset description. All credit for the data belongs to the original authors.

## Contents

| Path | Description |
|---|---|
| `data/`, `data.zip`, `data_summarized*.zip`, `code_revised.zip` | Original dataset and code |
| `Python scripts/` | Original Jupyter notebooks (zipped) |
| `outputs/` | Classification / regression results, confusion matrices, ROC curves, model comparisons |
| `output/` | Trial-level features, group comparison, feature importance |
| `feature_plots/` | Movement-feature plots (ROM, jerk, hand speed, path efficiency, ...) |
| `waveform_plots/` | Joint-angle waveform plots |
| `pose_videos/` | Pose animation GIFs per trial |
| `FFRNN_DONN_UpperLimb_CP.pptx` | Presentation of results |

## Classification results (DCP vs TD)

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| Random Forest | 1.000 | 1.000 | 1.000 |
| SVM (RBF) | 0.976 | 0.976 | 1.000 |
| MLP Neural Net | 0.976 | 0.976 | 0.999 |
| Logistic Regression | 0.963 | 0.963 | 0.991 |
| Gradient Boosting | 0.963 | 0.963 | 0.996 |
| GRU-lite | 0.927 | 0.927 | 0.944 |
| DONN (ResHopf) | 0.919 | 0.919 | 0.981 |
| Simple RNN | 0.915 | 0.915 | 0.962 |
| LSTM-lite | 0.854 | 0.854 | 0.941 |
| 1D FlipFlop RNN | 0.561 | 0.416 | 0.811 |

## Regression results

| Model | MSE | R² | MAE |
|---|---|---|---|
| Random Forest | 195.2 | 0.918 | 3.42 |
| MLP Neural Net | 185.7 | 0.916 | 5.43 |
| Gradient Boosting | 247.0 | 0.882 | 7.60 |
| Ridge Regression | 783.3 | 0.683 | 15.92 |
