# Speech-Based Cognitive Impairment Detection

Master's thesis — DFKI Saarbrücken & Saarland University, 2025
Author: Hema Devi Nachimuthu

## Thesis Overview

This thesis is a **comparison study**: the same speech-based machine-learning
pipeline is applied to two very different datasets, and the results are
compared to understand how dataset characteristics shape model performance.

- **ADReSS** (standardized benchmark) — a balanced, age- and gender-matched
  dementia speech corpus recorded under controlled conditions.
- **KI@Home** (real-world, in-home recordings) — heterogeneous speech
  collected in participants' homes with consumer devices.

The research question: *how do dataset quality, balance, and real-world
recording conditions influence the performance of speech-based ML models?*
The pipeline covers (1) classifying speech as cognitively impaired vs.
healthy and (2) predicting cognitive screening-test scores directly from
speech.

## Data

| | ADReSS | KI@Home |
|---|---|---|
| Participants | 156 | 79 households (49 used for modeling) |
| Speech segments | 4,077 | 4,793 |
| Recording conditions | controlled, balanced | real-world, in-home, heterogeneous |

Both datasets are restricted and cannot be redistributed. This repository
contains code only. All dataset-specific details (server credentials,
participant IDs, questionnaire filenames and texts, file paths) live in
gitignored config files or environment variables — see **Setup** below.

## Approach

- **Audio preprocessing** — MP3 to WAV, voice activity detection (WebRTC VAD),
  stationary noise removal, resampling to 16 kHz, peak normalization.
- **Feature extraction** — five feature-set combinations:
  - Librosa + emobase
  - Librosa + ComParE_2016
  - Librosa + eGeMAPSv02
  - MFCC means
  - Librosa (MFCC, spectral centroid/rolloff, ZCR, chroma; mean and variance)
  - plus a minimal acoustic set and simplified MRCG (see `src/feature_extraction.py`)
- **Dimensionality reduction** — PCA retaining 95% of variance.
- **Classification** — eight models (LDA, Decision Tree, KNN, Random Forest,
  SVM, MLP, Gradient Boosting, Logistic Regression) with GridSearchCV tuning,
  5-fold cross-validation, and subject-level majority voting.
- **Score prediction** — Linear Regression, Decision Tree, Gaussian Process,
  Gradient Boosting, SVR, plus a stacked linear ensemble (average of base
  models); evaluated with RMSE under an 80/20 train/test split and
  Leave-One-Subject-Out cross-validation, with Pearson correlation between
  aggregated true and predicted scores.

## Key results

| Feature set | ADReSS (before/after PCA) | KI@Home (before/after PCA) |
|---|---|---|
| Librosa + emobase | 1044 / 286 | 1743 / 312 |
| Librosa + ComParE | 6429 / 1032 | 7651 / 1163 |
| Librosa + eGeMAPS | 144 / 84 | 169 / 98 |
| MFCC | 26 / 22 | 37 / 30 |
| Librosa | 56 / 38 | 72 / 47 |

- **KI@Home classification:** best model was the MLP neural network with a
  subject-level (majority vote) accuracy of 0.89.
- **ADReSS classification:** SVM achieved the best mean accuracy across feature
  sets — structured, clean data favors margin-based methods.
- **Score prediction (KI@Home):** Mini-Cog RMSE between 1 and 2 (stable
  predictions); 6CIT RMSE between 4 and 8 with near-zero correlation,
  reflecting the difficulty of capturing multi-domain cognition from
  real-world speech.

**Main takeaway:** model choice matters less than data quality and cohort
design — controlled data (ADReSS) and in-the-wild data (KI@Home) reward
different models, and real-world heterogeneity limits score regression.

## Setup

```bash
pip install -r requirements.txt

# 1. Create your local config files (both are gitignored)
cp configs/settings.example.yaml configs/settings.yaml
cp configs/subject_ids.example.json configs/subject_ids.json
# Fill in your own paths, questionnaire filename mapping, and subject IDs.

# 2. Provide remote-server credentials as environment variables (never in files)
export DATA_SSH_HOST=...      # server hostname
export DATA_SSH_USER=...      # SSH username
export DATA_SSH_PASSWORD=...  # SSH password (or DATA_SSH_KEY for key-based auth)
```

## Usage

```bash
# 1. Load questionnaire (ground-truth) data from the remote server
python src/load_questionnaires.py

# 2. Preprocess audio (MP3 -> WAV, VAD segmentation, denoising, resampling)
python src/audio_preprocessing.py

# 3. Classification task (cognitively impaired vs. control)
python src/classify.py

# 4. Curate cognitive scores and demographics
python src/curate_scores.py

# 5. Score prediction (regression, LOSOCV + stacked ensemble)
python src/regress.py
```

The feature set for classification is set via `classification.feature_type`
in `configs/settings.yaml` (`librosa`, `emobase`, `compare`, `egemaps`,
`mrcg`, `minimal`, or `all`).

## Repository structure

```
├── configs/
│   ├── settings.example.yaml      # config template (real file gitignored)
│   └── subject_ids.example.json   # subject ID template (real file gitignored)
├── src/
│   ├── config.py                  # shared config loading
│   ├── load_questionnaires.py     # remote questionnaire loading (SFTP, env-var credentials)
│   ├── audio_preprocessing.py     # VAD segmentation, denoising, resampling
│   ├── feature_extraction.py      # librosa + openSMILE feature extraction
│   ├── classify.py                # 8-model classification with majority voting
│   ├── curate_scores.py           # demographics + screening score curation
│   └── regress.py                 # score prediction, LOSOCV, stacked ensemble
├── requirements.txt
└── README.md
```

## Report

A 4-page visual summary of the thesis is included
([MSc-Thesis-Summary.pdf](MSc-Thesis-Summary.pdf)). The full thesis report
("Comparison Model for Cognitive Detection and Speech", Saarland University,
2025) is available from the author on request.

## License

Code released for application/portfolio purposes. The underlying data
(ADReSS, KI@Home) is restricted and not included.
