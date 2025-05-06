import time
import joblib
import numpy as np
import librosa
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime

# Load model and preprocessing
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
X_train_mean = np.load("x_train_mean.npy")
X_train_std = np.load("x_train_std.npy")

# Function to extract features from audio
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

# Main function to run the prediction locally
def predict_local(audio_file_path):
    try:
        overall_start = time.time()
        print(f"\n🕒 Prediction started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # === Load audio (try soundfile, fallback to librosa)
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

        # === Waveform plot
        try:
            plt.figure(figsize=(10, 3))
            librosa.display.waveshow(y, sr=sr)
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
        features = (features - X_train_mean) / X_train_std
        features = features.reshape(1, -1)

        # === Prediction
        pred_index = model.predict(features)[0]
        pred_label = label_encoder.inverse_transform([pred_index])[0]

        print(f"🎯 Prediction: {pred_label}")
        print(f"✅ Done in {time.time() - overall_start:.2f} seconds")

        return {
            "prediction": pred_label,
            "waveform_image_base64": waveform_uri
        }

    except Exception as e:
        print("❌ Prediction failed:", e)
        return {"error": "Prediction failed", "details": str(e)}

# Example of how to call the function locally
audio_file_path = "B23.wav"  # Update with the path to your local audio file
result = predict_local(audio_file_path)
print(result)
