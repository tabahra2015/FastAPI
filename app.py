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
    start = time.time()
    print(f"\n🕒 Prediction started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Download audio
    response = requests.get(req.audio_url)
    if response.status_code != 200:
        print("❌ Failed to download file.")
        return {"error": "Failed to download file."}

    # Step 2: Save to temp file
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    # Step 3: Load audio
    y, sr = librosa.load(tmp_path, sr=16000, duration=2.0)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Step 4: Extract features
    features = extract_features(y, sr)
    features = (features - X_train_mean) / X_train_std
    features = features.reshape(1, -1)

    # Step 5: Predict
    pred_index = model.predict(features)[0]
    pred_label = label_encoder.inverse_transform([pred_index])[0]

    end = time.time()
    duration = end - start

    print(f"✅ Prediction ended at:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Total processing time: {duration:.2f} seconds")

    return {"prediction": pred_label}
