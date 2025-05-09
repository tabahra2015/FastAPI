# import os
# import librosa
# import librosa.display
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import sounddevice as sd
# from scipy.io.wavfile import write
# from IPython.display import display
# import ipywidgets as widgets

# # === Settings ===
# wav_folder = "converted_wav"
# output_file = "custom_isolet_style.csv"

# # === Feature Extraction ===
# def extract_features(y, sr):
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=256, hop_length=128)
#     if mfcc.shape[1] < 9:
#         mfcc = np.pad(mfcc, ((0, 0), (0, 9 - mfcc.shape[1])), mode='edge')

#     delta = librosa.feature.delta(mfcc)
#     delta2 = librosa.feature.delta(mfcc, order=2)
#     spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
#     spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#     spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#     spectral_flatness = librosa.feature.spectral_flatness(y=y)
#     zcr = librosa.feature.zero_crossing_rate(y)
#     rms = librosa.feature.rms(y=y)

#     def normalize_block(block):
#         return (block - np.min(block)) / (np.max(block) - np.min(block) + 1e-8)

#     def agg(matrix):
#         return np.concatenate([
#             np.mean(matrix, axis=1),
#             np.std(matrix, axis=1),
#             np.min(matrix, axis=1),
#             np.max(matrix, axis=1)
#         ])

#     features = np.concatenate([
#         normalize_block(agg(mfcc)),
#         normalize_block(agg(delta)),
#         normalize_block(agg(delta2)),
#         normalize_block(agg(spectral_centroid)),
#         normalize_block(agg(spectral_bandwidth)),
#         normalize_block(agg(spectral_rolloff)),
#         normalize_block(agg(spectral_flatness)),
#         normalize_block(agg(zcr)),
#         normalize_block(agg(rms))
#     ])
#     return features

# # === Create Dataset ===
# rows = []
# for file in os.listdir(wav_folder):
#     if file.endswith(".wav"):
#         try:
#             path = os.path.join(wav_folder, file)
#             y, sr = librosa.load(path, sr=16000)
#             y = y / np.max(np.abs(y))
#             features = extract_features(y, sr)
#             label = file[0].upper()
#             row = np.append(features, label)
#             rows.append(row)
#         except Exception as e:
#             print(f"⚠️ Error processing {file}: {e}")

# # === Save dataset ===
# feature_dim = len(rows[0]) - 1
# columns = [f'f{i+1}' for i in range(feature_dim)] + ['label']
# df = pd.DataFrame(rows, columns=columns)
# df.to_csv(output_file, index=False)
# print(f"✅ Data saved in ISOLET style with shape: {df.shape}")

# # === Prepare Data for Training ===
# df = pd.read_csv(output_file)
# label_encoder = LabelEncoder()
# df['label_encoded'] = label_encoder.fit_transform(df['label'])

# X = df.drop(columns=['label', 'label_encoded']).values
# y = df['label_encoded'].values

# # Normalize
# X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

# # === Split Data 70% Train / 30% Test ===
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # === Build Random Forest Model ===
# model = RandomForestClassifier(n_estimators=200, random_state=42)
# model.fit(X_train, y_train)

# # === Evaluate ===
# y_pred = model.predict(X_test)
# acc = accuracy_score(y_test, y_pred)
# print(f"\n✅ Random Forest Test Accuracy: {acc:.2f}")
# print("\n📄 Classification Report:")
# print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# # === Predict New File ===
# def predict_file(filename):
#     y, sr = librosa.load(filename, sr=16000)
#     y = y / np.max(np.abs(y))
#     features = extract_features(y, sr)
#     features = (features - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + 1e-8)  # normalize like training set
#     features = features.reshape(1, -1)
#     pred_index = model.predict(features)[0]
#     pred_label = label_encoder.inverse_transform([pred_index])[0]
#     return pred_label

# # === Plot a file
# filepath = "B23.wav"
# predicted_letter = predict_file(filepath)

# print(f"\n📂 File: {filepath}")
# print(f"📢 Predicted letter: {predicted_letter}")

# y, sr = librosa.load(filepath, sr=16000)
# plt.figure(figsize=(10, 4))
# librosa.display.waveshow(y, sr=sr)
# plt.title(f"Waveform of {filepath}", fontsize=14)
# plt.text(0.01, 0.88, f"📢 Predicted: {predicted_letter}", transform=plt.gca().transAxes, fontsize=12, color='blue')
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.tight_layout()
# plt.show()

import traceback
import numpy as np
import librosa
import joblib
import os
from pydantic import BaseModel
from tempfile import NamedTemporaryFile
import soundfile as sf
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import time
from datetime import datetime


app = FastAPI()

class PredictRequest(BaseModel):
    audio_url: str

# Load model and preprocessing
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
X_train_mean = np.load("x_train_mean.npy")
X_train_std = np.load("x_train_std.npy")

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



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        overall_start = time.time()
        print(f"\n🕒 Prediction started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # === Save the uploaded file
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name
        print(f"📁 Temp file path: {tmp_path}")

        # === Load audio (try soundfile, fallback to librosa)
        try:
            y, sr = sf.read(tmp_path)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000
        except Exception as e:
            print("⚠️ soundfile.read() failed, falling back to librosa.load()")
            y, sr = librosa.load(tmp_path, sr=16000, duration=2.0)

        # === Normalize and truncate
        y = y[:sr * 2]
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        print(f"🎧 Audio loaded. sr={sr}, samples={len(y)}")

        # === Feature extraction
        features = extract_features(y, sr)
        features = (features - X_train_mean) / X_train_std
        features = features.reshape(1, -1)

        # === Prediction
        pred_index = model.predict(features)[0]
        pred_label = label_encoder.inverse_transform([pred_index])[0]

        print(f"🎯 Prediction: {pred_label}")
        print(f"✅ Done in {time.time() - overall_start:.2f} seconds")

        return JSONResponse(content={
            "prediction": pred_label,
        })

    except Exception as e:
        print("❌ Prediction failed:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Prediction crashed on server.")
