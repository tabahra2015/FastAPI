# app.py

import time
from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import os
import numpy as np
import librosa
import joblib 
from tempfile import NamedTemporaryFile

app = FastAPI()

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

    # === Ensure consistent size ===
    expected_feature_size = 180
    if features.shape[0] != expected_feature_size:
        features = np.resize(features, expected_feature_size)

    return features

# 
class PredictRequest(BaseModel):
    audio_url: str

@app.post("/predict")
def predict(req: PredictRequest):
    start = time.time()

    response = requests.get(req.audio_url)
    if response.status_code != 200:
        return {"error": "Failed to download file."}

    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    y, sr = librosa.load(tmp_path, sr=16000, duration=2.0)
    y = y / np.max(np.abs(y))

    features = extract_features(y, sr)
    features = (features - X_train_mean) / X_train_std
    features = features.reshape(1, -1)

    pred_index = model.predict(features)[0]
    pred_label = label_encoder.inverse_transform([pred_index])[0]

    return {"prediction": pred_label}
