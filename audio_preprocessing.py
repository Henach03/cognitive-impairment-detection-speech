"""
Audio preprocessing.

Pipeline: MP3 -> WAV conversion, voice activity detection (VAD) segmentation,
stationary-noise removal, resampling, and amplitude normalization.

Input audio is expected as MP3 files named by subject ID under:
    <audio.input_dir>/<GROUP>/<SUBJECT_ID>*.mp3      (GROUP in {"AD", "NonAD"})
Segmented WAV files are written to:
    <audio.output_dir>/<GROUP>/<SUBJECT>_seg<i>.wav

Subject IDs per group are loaded from configs/subject_ids.json (gitignored);
paths and folders come from configs/settings.yaml (gitignored).
"""

import glob
import os

import librosa
import numpy as np
import soundfile as sf
import webrtcvad
from pydub import AudioSegment
from scipy.signal import medfilt

from config import load_settings, load_subject_ids

# --- Functions ----------------------------------------------------------------


def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert an MP3 file to WAV."""
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")


def vad_segment(audio_np, sample_rate, frame_duration_ms=30):
    """Segment audio into voiced regions using WebRTC VAD.

    Parameters
    ----------
    audio_np : np.ndarray
        Audio samples (int16 scale, mono).
    sample_rate : int
    frame_duration_ms : int
        Frame length in milliseconds (10, 20, or 30).

    Returns
    -------
    list of np.ndarray
        Voiced segments as int16 sample arrays.
    """
    vad = webrtcvad.Vad(2)  # moderately aggressive
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    audio_bytes = audio_np.tobytes()

    frames = [
        audio_bytes[i:i + frame_size * 2]
        for i in range(0, len(audio_bytes), frame_size * 2)
    ]

    segments = []
    voiced_frames = []
    for frame in frames:
        if vad.is_speech(frame, sample_rate):
            voiced_frames.append(frame)
        elif voiced_frames:
            segments.append(b"".join(voiced_frames))
            voiced_frames = []
    if voiced_frames:
        segments.append(b"".join(voiced_frames))

    return [np.frombuffer(seg, dtype=np.int16) for seg in segments]


def remove_stationary_noise(audio_np, kernel_size=5):
    """Suppress stationary noise with a median filter."""
    return medfilt(audio_np, kernel_size)


def resample_audio(audio_np, orig_sr, target_sr):
    """Resample audio to the target sample rate."""
    return librosa.resample(
        audio_np.astype(float), orig_sr=orig_sr, target_sr=target_sr
    )


def normalize_audio(audio_np):
    """Peak-normalize audio to the range [-1, 1]."""
    max_val = np.max(np.abs(audio_np))
    if max_val == 0:
        return audio_np
    return audio_np / max_val


# --- Main pipeline ------------------------------------------------------------

if __name__ == "__main__":
    audio_cfg = load_settings()["audio"]
    subject_ids = load_subject_ids()

    input_base_path = audio_cfg["input_dir"]
    output_base_path = audio_cfg["output_dir"]
    folders = audio_cfg.get("folders", ["AD", "NonAD"])
    target_sr = audio_cfg.get("target_sample_rate", 16000)

    for folder in folders:
        # Subject IDs per group come from the (gitignored) config file
        subject_group = "ad" if folder == "AD" else "non_ad"
        ids = subject_ids.get(subject_group, [])

        folder_path = os.path.join(input_base_path, folder)
        output_folder = os.path.join(output_base_path, folder)
        os.makedirs(output_folder, exist_ok=True)

        for subject_id in ids:
            for mp3_file in sorted(
                glob.glob(os.path.join(folder_path, f"{subject_id}*.mp3"))
            ):
                wav_temp = mp3_file.replace(".mp3", "_temp.wav")

                # Convert MP3 to WAV
                convert_mp3_to_wav(mp3_file, wav_temp)

                # Load WAV for processing (mono, native sample rate)
                audio_np, sr = librosa.load(wav_temp, sr=None, mono=True)

                # WebRTC VAD expects 16-bit PCM
                audio_int16 = (audio_np * 32768).astype(np.int16)

                # VAD segmentation
                segments = vad_segment(audio_int16, sr)

                # Process and save segments
                for i, segment in enumerate(segments):
                    segment_denoised = remove_stationary_noise(segment)
                    segment_resampled = resample_audio(segment_denoised, sr, target_sr)
                    segment_normalized = normalize_audio(segment_resampled)

                    stem = os.path.basename(mp3_file).replace(".mp3", "")
                    segment_filename = os.path.join(
                        output_folder, f"{stem}_seg{i}.wav"
                    )
                    sf.write(segment_filename, segment_normalized, target_sr)

                os.remove(wav_temp)

    print("Processing completed.")
