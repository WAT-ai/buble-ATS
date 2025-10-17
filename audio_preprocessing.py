import librosa
import soundfile as sf
import numpy as np

def load_audio_mono(path: str, sr: int = 22050) -> np.ndarray:
    y, orig_sr = sf.read(path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return y

def compute_logmel(y: np.ndarray, sr: int = 22050, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S.T  # (frames, n_mels)

def normalize_features(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mean) / std