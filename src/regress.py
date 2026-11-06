"""
Regression task (predicting screening-test scores from speech).

Aggregates acoustic features per subject (averaged over audio segments),
merges them with the curated cognitive scores, and compares several
regression models. Evaluation follows the thesis:

- train/test split (80/20) with RMSE per model,
- Leave-One-Subject-Out (LOSO) cross-validation,
- a stacked linear ensemble (average of the base models' predictions),
- Pearson correlation between aggregated true and predicted scores.
"""

import os

import librosa
import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from config import load_settings

# --- Feature extraction --------------------------------------------------------


def extract_features(file_path):
    """Extract a compact acoustic feature vector from a WAV file."""
    try:
        y, sr = librosa.load(file_path, sr=None)

        # MFCC (mean of 13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfcc, axis=1)

        # Spectral centroid mean
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        # Spectral bandwidth mean
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

        # Zero-crossing rate mean
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        return np.concatenate([mfcc_means, [spec_cent, spec_bw, zcr]])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# --- Model definitions ----------------------------------------------------------

MODELS = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "GaussianProcess": GaussianProcessRegressor(),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "SupportVectorMachine": SVR(),
}


def stacked_predictions(preds_per_model):
    """Stacked linear ensemble: average of the base models' predictions."""
    return np.mean(np.vstack(preds_per_model), axis=0)


# --- Evaluation -----------------------------------------------------------------


def evaluate_rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))


def print_report(title, y_true, y_pred_per_model):
    print(f"\n{title}")
    for name, preds in y_pred_per_model.items():
        rmse = evaluate_rmse(y_true, preds)
        if np.std(preds) > 0:
            r, _ = pearsonr(y_true, preds)
            print(f"  {name}: RMSE = {rmse:.4f}, Pearson r = {r:.4f}")
        else:
            print(f"  {name}: RMSE = {rmse:.4f}")


# --- Main pipeline ------------------------------------------------------------

if __name__ == "__main__":
    settings = load_settings()
    score_cfg = settings["scores"]
    audio_cfg = settings["audio"]

    base_path = audio_cfg["output_dir"]
    folders = audio_cfg.get("folders", ["AD", "NonAD"])
    scores_file = score_cfg["file"]
    orientation_col = score_cfg["orientation_col"]
    word_recall_clock_col = score_cfg["word_recall_clock_col"]

    # Load cognitive score data
    score_data = pd.read_csv(scores_file)
    score_data[orientation_col] = pd.to_numeric(
        score_data[orientation_col], errors="coerce"
    )
    score_data[word_recall_clock_col] = pd.to_numeric(
        score_data[word_recall_clock_col], errors="coerce"
    )

    # Aggregate features per subject by averaging over all segment files
    features_dict = {}

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue

        for subject in sorted(os.listdir(folder_path)):
            subject_path = os.path.join(folder_path, subject)
            if not os.path.isdir(subject_path):
                continue

            file_features = []
            for f in os.listdir(subject_path):
                if f.endswith(".wav"):
                    feats = extract_features(os.path.join(subject_path, f))
                    if feats is not None:
                        file_features.append(feats)

            if file_features:
                features_dict[subject] = np.mean(file_features, axis=0)

    # Convert features dict to DataFrame
    features_df = pd.DataFrame.from_dict(features_dict, orient="index")
    features_df.reset_index(inplace=True)
    features_df.rename(columns={"index": "subject_id"}, inplace=True)

    # Merge with scores on subject_id
    score_cols = [orientation_col, word_recall_clock_col]
    data_merged = pd.merge(
        features_df, score_data[["subject_id"] + score_cols],
        on="subject_id", how="inner",
    ).dropna(subset=score_cols)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(
        data_merged.drop(columns=["subject_id"] + score_cols)
    )

    # Apply PCA to retain 95% variance
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    groups = data_merged["subject_id"].to_numpy()

    for target_col, target_name in [
        (orientation_col, "orientation test (6CIT-style)"),
        (word_recall_clock_col, "word-recall / clock-drawing (Mini-Cog-style)"),
    ]:
        y = data_merged[target_col].to_numpy()

        # --- Train/test split (80/20) ---
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.2, random_state=42
        )

        preds_per_model = {}
        for name, model in MODELS.items():
            fitted = clone(model).fit(X_train, y_train)
            preds_per_model[name] = fitted.predict(X_test)

        # Stacked linear ensemble (average of base models)
        preds_per_model["Stacked (avg)"] = stacked_predictions(
            list(preds_per_model.values())
        )
        print_report(f"Train/test split - {target_name}", y_test, preds_per_model)

        # --- Leave-One-Subject-Out cross-validation ---
        logo = LeaveOneGroupOut()
        loso_preds = {name: [] for name in MODELS}
        loso_true = []

        for train_idx, test_idx in logo.split(X_pca, y, groups):
            X_tr, X_te = X_pca[train_idx], X_pca[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            loso_true.extend(y_te)

            for name, model in MODELS.items():
                fitted = clone(model).fit(X_tr, y_tr)
                loso_preds[name].extend(fitted.predict(X_te))

        loso_preds["Stacked (avg)"] = stacked_predictions(
            [loso_preds[name] for name in MODELS]
        )
        print_report(
            f"Leave-One-Subject-Out CV - {target_name}",
            np.array(loso_true), loso_preds,
        )
