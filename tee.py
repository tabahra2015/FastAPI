import time
import joblib
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pandas as pd

# Load model and preprocessing
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
X_train_mean = np.load("x_train_mean.npy")
X_train_std = np.load("x_train_std.npy")
avg_features_df = pd.read_csv("average_features_per_letter.csv", index_col=0)

# === Feature Extraction ===
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=256, hop_length=128, fmax=sr/2)
    if mfcc.shape[1] < 9:
        mfcc = np.pad(mfcc, ((0, 0), (0, 9 - mfcc.shape[1])), mode='edge')

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    def normalize_block(block):
        return (block - np.min(block)) / (np.max(block) - np.min(block) + 1e-8)

    def agg(matrix):
        return np.concatenate([
            np.mean(matrix, axis=1),
            np.std(matrix, axis=1),
            np.min(matrix, axis=1),
            np.max(matrix, axis=1)
        ])

    features = np.concatenate([
        normalize_block(agg(mfcc)),
        normalize_block(agg(delta)),
        normalize_block(agg(delta2)),
        normalize_block(agg(spectral_centroid)),
        normalize_block(agg(spectral_bandwidth)),
        normalize_block(agg(spectral_rolloff)),
        normalize_block(agg(spectral_flatness)),
        normalize_block(agg(zcr)),
        normalize_block(agg(rms))
    ])

    expected_feature_size = model.n_features_in_
    if features.shape[0] > expected_feature_size:
        features = features[:expected_feature_size]
    elif features.shape[0] < expected_feature_size:
        features = np.pad(features, (0, expected_feature_size - features.shape[0]), mode='constant')

    return features

# === Main Prediction Function ===
def predict_local(audio_file_path):
    try:
        overall_start = time.time()
        print(f"\n🕒 Prediction started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # === Load audio ===
        try:
            y, sr = sf.read(audio_file_path)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000
        except Exception as e:
            print("⚠️ soundfile.read() failed, falling back to librosa.load()")
            y, sr = librosa.load(audio_file_path, sr=16000, duration=2.0)

        # === Normalize and truncate
        y = y[:sr * 2]
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        print(f"🎧 Audio loaded. sr={sr}, samples={len(y)}")

        # === Waveform plot (using plt.plot to avoid errors)
        try:
            plt.figure(figsize=(10, 3))
            plt.plot(np.linspace(0, len(y) / sr, num=len(y)), y)
            plt.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            buf.seek(0)
            waveform_base64 = base64.b64encode(buf.read()).decode('utf-8')
            waveform_uri = f"data:image/png;base64,{waveform_base64}"
        except Exception as e:
            print("⚠️ Could not generate waveform image:", e)
            waveform_uri = None

        # === Feature extraction
        features = extract_features(y, sr)
        features_norm = (features - X_train_mean) / X_train_std
        features_input = features_norm.reshape(1, -1)

        # === Predict label
        pred_index = model.predict(features_input)[0]
        pred_label = label_encoder.inverse_transform([pred_index])[0]

        # === DTW Distance
        if pred_label in avg_features_df.index:
            avg_vector = avg_features_df.loc[pred_label].values.astype(np.float32).flatten()
            features_flat = features.astype(np.float32).flatten()

            print("🧪 DTW Debug Info")
            print("features_flat[:5]:", features_flat[:5])
            print("avg_vector[:5]:", avg_vector[:5])
            print("any NaN in features?", np.isnan(features_flat).any())
            print("any NaN in avg_vector?", np.isnan(avg_vector).any())

            dtw_distance, _ = fastdtw(features_flat, avg_vector, dist=lambda x, y: np.linalg.norm(x - y))
            print(f"📏 DTW Distance to average '{pred_label}': {dtw_distance:.4f}")
        else:
            dtw_distance = None
            print(f"⚠️ No average vector found for label {pred_label}")

        print(f"🎯 Prediction: {pred_label}")
        if dtw_distance is not None:
            print(f"📏 DTW Distance to average '{pred_label}': {dtw_distance:.4f}")
        print(f"✅ Done in {time.time() - overall_start:.2f} seconds")

        return {
            "prediction": pred_label,
            "dtw_distance": dtw_distance,
            "waveform_image_base64": waveform_uri
        }

    except Exception as e:
        print("❌ Prediction failed:", e)
        return {"error": "Prediction failed", "details": str(e)}

# === Run example ===
audio_file_path = "A1_converted.wav"  # Replace with your actual file path
result = predict_local(audio_file_path)
print(result)
