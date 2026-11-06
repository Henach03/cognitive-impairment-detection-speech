"""
Shared module - Acoustic feature extraction.

Provides a single entry point, extract_features(file_path, feature_type),
supporting several feature sets:

  - 'librosa' : MFCCs, spectral centroid/rolloff, zero-crossing rate, chroma
               (means and variances)
  - 'emobase' : openSMILE emobase functionals
  - 'compare' : openSMILE ComParE 2016 functionals
  - 'egemaps' : openSMILE eGeMAPSv02 functionals
  - 'mrcg'    : simplified MRCG (mean of log-mel spectrogram)
  - 'minimal' : zero-crossing rate, spectral centroid, spectral rolloff
  - 'all'     : emobase + ComParE + eGeMAPS + MRCG + minimal
"""

import librosa
import numpy as np
import opensmile

MAX_DURATION_S = 30

# --- openSMILE extractors -----------------------------------------------------

SMILE_EXTRACTORS = {
    "emobase": opensmile.Smile(
        feature_set=opensmile.FeatureSet.emobase,
        feature_level=opensmile.FeatureLevel.Functionals,
    ),
    "compare": opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    ),
    "egemaps": opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    ),
}

# --- Handcrafted feature extractors -------------------------------------------


def extract_mrcg(y, sr):
    """Extract a simplified version of MRCG features."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec)
    mrcg = np.mean(log_mel_spec, axis=1)
    return mrcg


def extract_minimal_features(y, sr):
    """Extract a minimal set of features."""
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    return np.array([zcr, spectral_centroid, spectral_rolloff])


def extract_librosa_features(y, sr):
    """Extract MFCC, spectral, and chroma statistics with librosa."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

    return np.concatenate([
        np.mean(mfccs.T, axis=0), np.var(mfccs.T, axis=0),
        [np.mean(spectral_centroids), np.var(spectral_centroids)],
        [np.mean(spectral_rolloff), np.var(spectral_rolloff)],
        [np.mean(zero_crossing_rate), np.var(zero_crossing_rate)],
        np.mean(chroma_stft.T, axis=0), np.var(chroma_stft.T, axis=0),
    ])


# --- Main entry point ----------------------------------------------------------


def extract_features(file_path, feature_type="all", max_duration=MAX_DURATION_S):
    """Extract audio features from a WAV file.

    Parameters
    ----------
    file_path : str
        Path to a WAV file.
    feature_type : str
        One of 'librosa', 'emobase', 'compare', 'egemaps', 'mrcg',
        'minimal', or 'all'.
    max_duration : float
        Maximum number of seconds of audio to load.

    Returns
    -------
    np.ndarray
        Flattened feature vector.
    """
    y, sr = librosa.load(file_path, duration=max_duration)

    if feature_type == "emobase":
        return SMILE_EXTRACTORS["emobase"].process_file(file_path).values.flatten()
    if feature_type == "compare":
        return SMILE_EXTRACTORS["compare"].process_file(file_path).values.flatten()
    if feature_type == "egemaps":
        return SMILE_EXTRACTORS["egemaps"].process_file(file_path).values.flatten()
    if feature_type == "mrcg":
        return extract_mrcg(y, sr).flatten()
    if feature_type == "minimal":
        return extract_minimal_features(y, sr)
    if feature_type == "librosa":
        return extract_librosa_features(y, sr)

    # 'all': combine all feature sets
    emobase_features = SMILE_EXTRACTORS["emobase"].process_file(file_path).values.flatten()
    compare_features = SMILE_EXTRACTORS["compare"].process_file(file_path).values.flatten()
    egemaps_features = SMILE_EXTRACTORS["egemaps"].process_file(file_path).values.flatten()
    mrcg_features = extract_mrcg(y, sr).flatten()
    minimal_features = extract_minimal_features(y, sr)
    return np.concatenate([
        emobase_features, compare_features, egemaps_features,
        mrcg_features, minimal_features,
    ])
