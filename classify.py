"""
Classification task (cognitive impairment vs. control).

Extracts acoustic features from preprocessed audio segments, then trains and
compares several classifiers with hyperparameter tuning, 5-fold
cross-validation, and both segment-level and subject-level (majority vote)
evaluation.

Expects segmented WAV files under:
    <audio.output_dir>/<GROUP>/<SUBJECT_ID>/*.wav
with GROUP in {"AD", "NonAD"}.
"""

import os

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

from config import load_settings
from feature_extraction import extract_features

# --- Data loading -------------------------------------------------------------


def load_data(base_path, folders, feature_type):
    """Load features and labels for all subjects.

    Returns
    -------
    X : np.ndarray, shape (n_segments, n_features)
    y : np.ndarray, shape (n_segments,)   (1 = AD group, 0 = control)
    subject_ids : np.ndarray, shape (n_segments,)
    """
    X, y, subject_ids = [], [], []

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        for subject in sorted(os.listdir(folder_path)):
            subject_path = os.path.join(folder_path, subject)
            if not os.path.isdir(subject_path):
                continue
            for file in os.listdir(subject_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(subject_path, file)
                    features = extract_features(file_path, feature_type)
                    X.append(features)
                    y.append(0 if folder == "NonAD" else 1)
                    subject_ids.append(subject)

    return np.array(X), np.array(y), np.array(subject_ids)


# --- Evaluation and tuning -----------------------------------------------------


def evaluate_model(y_true, y_pred):
    """Evaluate model performance using multiple metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_pred),
    }


def tune_hyperparameters(model, param_grid, X, y):
    """Perform hyperparameter tuning using GridSearchCV."""
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(X, y)
    return grid_search.best_estimator_


# --- Classifiers and parameter grids ------------------------------------------

CLASSIFIERS = {
    "LDA": (LinearDiscriminantAnalysis(), {}),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [3, 5, 7, 10]},
    ),
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9]}),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
    ),
    "SVM": (
        SVC(random_state=42),
        {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    ),
    "Neural Network": (
        MLPClassifier(random_state=42),
        {
            "hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "alpha": [0.0001, 0.001, 0.01],
        },
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
    ),
    "Logistic Regression": (
        LogisticRegression(random_state=42),
        {"C": [0.1, 1, 10], "penalty": ["l2"]},
    ),
}

# --- Main pipeline ---------------------------------------------------------------

if __name__ == "__main__":
    settings = load_settings()
    audio_cfg = settings["audio"]
    cls_cfg = settings.get("classification", {})

    base_path = audio_cfg["output_dir"]
    folders = audio_cfg.get("folders", ["AD", "NonAD"])
    feature_type = cls_cfg.get("feature_type", "librosa")

    # Load and prepare the data
    X, y, subject_ids = load_data(base_path, folders, feature_type)

    print(f"Feature type: {feature_type}")
    print(f"Feature shape: {X.shape}")
    if len(X.shape) != 2:
        raise ValueError(f"Expected 2D array for features, got shape {X.shape}")

    # Print information about NaN values
    print("Number of NaN values per feature:")
    print(np.isnan(X).sum(axis=0))
    print("Number of samples with NaN values:")
    print(np.sum(np.isnan(X).any(axis=1)))

    # Split the data into training and testing sets
    (X_train, X_test, y_train, y_test,
     subject_ids_train, subject_ids_test) = train_test_split(
        X, y, subject_ids, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing pipeline: impute -> scale -> PCA (95% variance)
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Original number of features: {X_train.shape[1]}")
    print(f"Number of features after PCA: {X_train_processed.shape[1]}")

    # Train and evaluate models
    results = {}

    for name, (model, param_grid) in CLASSIFIERS.items():
        print(f"Training and evaluating {name}...")

        # Tune hyperparameters
        best_model = tune_hyperparameters(
            model, param_grid, X_train_processed, y_train
        )

        # Perform cross-validation
        cv_scores = cross_val_score(
            best_model, X_train_processed, y_train, cv=5, scoring="accuracy"
        )

        # Train on the entire training set
        best_model.fit(X_train_processed, y_train)

        # Segment-level (SL) classification
        y_pred_segments = best_model.predict(X_test_processed)
        sl_metrics = evaluate_model(y_test, y_pred_segments)

        # Majority vote (MV) classification at subject level
        unique_subjects = np.unique(subject_ids_test)
        y_pred_mv, y_true_mv = [], []

        for subject in unique_subjects:
            subject_mask = subject_ids_test == subject
            subject_predictions = y_pred_segments[subject_mask]
            subject_true = y_test[subject_mask]

            mv_prediction = np.round(np.mean(subject_predictions))
            y_pred_mv.append(mv_prediction)
            y_true_mv.append(subject_true[0])

        y_pred_mv = np.array(y_pred_mv)
        y_true_mv = np.array(y_true_mv)
        mv_metrics = evaluate_model(y_true_mv, y_pred_mv)

        results[name] = {
            "cv_score": np.mean(cv_scores),
            "sl_metrics": sl_metrics,
            "mv_metrics": mv_metrics,
            "best_params": best_model.get_params(),
        }

    # Print results
    for name, result in results.items():
        print(f"\n{name} Results:")
        print(f"Cross-validation Accuracy: {result['cv_score']:.4f}")
        print("Segment-level Classification Metrics:")
        for metric, value in result["sl_metrics"].items():
            print(f"  {metric}: {value:.4f}")
        print("Majority Vote Classification Metrics:")
        for metric, value in result["mv_metrics"].items():
            print(f"  {metric}: {value:.4f}")
        print("Best Parameters:")
        print(result["best_params"])
