import time
import os
import numpy as np
import librosa
import joblib
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from tempfile import NamedTemporaryFile
import librosa.display
import matplotlib.pyplot as plt
import base64
from io import BytesIO
# === Initialize App ===
app = FastAPI()

# === Define Request Schema ===
class PredictRequest(BaseModel):
    audio_url: str

# === Load Trained Components ===
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
X_train_mean = np.load("x_train_mean.npy")
X_train_std = np.load("x_train_std.npy")

# === Feature Extraction Function ===
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


# === Prediction Endpoint ===
@app.post("/predict")
def predict(req: PredictRequest):
    overall_start = time.time()
    print(f"\n🕒 Prediction started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Download audio
    step1 = time.time()
    response = requests.get(req.audio_url)
    print(f"⏱️ Download time: {time.time() - step1:.2f}s")

    if response.status_code != 200:
        return {"error": "Failed to download file."}

    # Step 2: Save to file
    step2 = time.time()
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name
    print(f"⏱️ Save file time: {time.time() - step2:.2f}s")

    # Step 3: Load audio
    step3 = time.time()
    y, sr = librosa.load(tmp_path, sr=16000, duration=2.0)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    print(f"⏱️ Load audio time: {time.time() - step3:.2f}s")

    # === Plot waveform and convert to base64 ===
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    waveform_base64 = base64.b64encode(buf.read()).decode('utf-8')
    waveform_uri = f"data:image/png;base64,{waveform_base64}"

    # Step 4: Feature extraction
    step4 = time.time()
    features = extract_features(y, sr)
    features = (features - X_train_mean) / X_train_std
    features = features.reshape(1, -1)
    print(f"⏱️ Feature extraction time: {time.time() - step4:.2f}s")

    # Step 5: Prediction
    step5 = time.time()
    pred_index = model.predict(features)[0]
    pred_label = label_encoder.inverse_transform([pred_index])[0]
    print(f"⏱️ Model prediction time: {time.time() - step5:.2f}s")

    print(f"✅ Prediction ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Total time: {time.time() - overall_start:.2f}s")

    return {
        "prediction": pred_label,
        "waveform_image_base64": waveform_uri
    }
