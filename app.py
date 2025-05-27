import traceback
import numpy as np
import librosa
import os
from tempfile import NamedTemporaryFile
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from scipy.io import wavfile
from tensorflow.keras.models import load_model
import joblib  # لا زلنا نستخدمه لتحميل label_encoder

app = FastAPI()

# Optional health check route
@app.get("/")
def health():
    return {"status": "✅ Server is live"}

# Load model and preprocessing data
try:
# في السيرفر:
    model = load_model("keras_model.h5")
    label_encoder = joblib.load("label_encoder.pkl")  # تحميل الترميز
    X_train_mean = np.load("x_train_mean.npy")
    X_train_std = np.load("x_train_std.npy")
except Exception as e:
    print("❌ Model or preprocessing files could not be loaded.")
    traceback.print_exc()
    raise RuntimeError("Server initialization failed")


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

    # ضبط حجم الميزات بناءً على عدد ميزات نموذج Keras
    expected_feature_size = model.input_shape[-1]  # نموذج Keras عادة يأخذ input shape مثل (None, features)
    if features.shape[0] > expected_feature_size:
        features = features[:expected_feature_size]
    elif features.shape[0] < expected_feature_size:
        features = np.pad(features, (0, expected_feature_size - features.shape[0]), mode='constant')

    return features


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        start_time = time.time()
        print(f"\n🕒 Prediction started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Read audio
        audio_data = await file.read()

        # Load with librosa
        y, sr = librosa.load(BytesIO(audio_data), sr=16000, mono=True, duration=2.0)

        # Save as PCM 16-bit WAV
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            wavfile.write(tmp_wav.name, sr, (y * 32767).astype(np.int16))
            tmp_path = tmp_wav.name

        # Reload with soundfile (stable for feature extraction)
        y, sr = sf.read(tmp_path)
        y = y[:sr * 2]
        y = y / (np.max(np.abs(y)) + 1e-6)

        # Generate waveform image
        plt.figure(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr)
        plt.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        waveform_base64 = base64.b64encode(buf.getvalue()).decode()
        waveform_uri = f"data:image/png;base64,{waveform_base64}"

        # Extract features and scale
        features = extract_features(y, sr)
        features = (features - X_train_mean) / X_train_std

        # النموذج Keras يحتاج عادة مدخلات بشكل (batch_size, features)
        features = features.reshape(1, -1)

        # Predict
        pred_probs = model.predict(features)
        pred_index = np.argmax(pred_probs, axis=1)[0]

        pred_label = label_encoder.inverse_transform([pred_index])[0]

        print(f"🎯 Prediction: {pred_label}")
        print(f"✅ Done in {time.time() - start_time:.2f} seconds")

        return JSONResponse(content={
            "prediction": pred_label,
            "waveform_image_base64": waveform_uri
        })

    except Exception as e:
        print("❌ Prediction failed:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Prediction crashed on server.")
