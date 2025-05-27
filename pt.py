import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import joblib

# === Settings ===
wav_folder = "converted_wav"
sr = 16000
feature_size = 180

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

    # Resize to ensure fixed feature length
    if features.shape[0] != feature_size:
        features = np.resize(features, feature_size)

    return features

# === Load and Split Files ===
all_files = [f for f in os.listdir(wav_folder) if f.endswith(".wav")]
random.shuffle(all_files)
split_point = int(0.7 * len(all_files))
train_files = all_files[:split_point]
test_files = all_files[split_point:]

print(f"✅ Total files: {len(all_files)}, Train: {len(train_files)}, Test: {len(test_files)}")

# === Extract Features ===
train_rows, test_rows = [], []

for file_list, target in [(train_files, train_rows), (test_files, test_rows)]:
    for file in file_list:
        try:
            path = os.path.join(wav_folder, file)
            y, _ = librosa.load(path, sr=sr)
            y = y / np.max(np.abs(y))
            features = extract_features(y, sr)
            label = file[0].upper().strip()
            row = list(features) + [label, file]
            target.append(row)
        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")

# === Create DataFrames ===
feature_columns = [f'f{i+1}' for i in range(feature_size)]
columns = feature_columns + ['label', 'filename']
df_train = pd.DataFrame(train_rows, columns=columns).dropna()
df_test = pd.DataFrame(test_rows, columns=columns).dropna()

# === Encode Labels ===
le = LabelEncoder()
df_train['label'] = df_train['label'].astype(str).str.strip().str.upper()
df_test['label'] = df_test['label'].astype(str).str.strip().str.upper()
df_train['label_encoded'] = le.fit_transform(df_train['label'])
df_test['label_encoded'] = le.transform(df_test['label'])

# === Prepare Features and Labels ===
X_train = df_train[feature_columns].astype(float).values
X_test = df_test[feature_columns].astype(float).values
y_train = to_categorical(df_train['label_encoded'].values)
y_test = to_categorical(df_test['label_encoded'].values)
filenames_test = df_test['filename'].values

# === Normalize ===
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# === Build Neural Network Model ===
model = Sequential([
    Dense(256, input_shape=(feature_size,), activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Train Model ===
print("\n🚀 Training the model...")
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# === Evaluate ===
print("\n✅ Evaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n🎯 Test Accuracy: {test_acc:.2f}")

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("\n📄 Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=le.classes_))

# === Prediction on Files ===
print("\n🔍 Predictions on test files:")
for i in range(len(filenames_test)):
    pred_label = le.inverse_transform([y_pred_labels[i]])[0]
    print(f"📂 File: {filenames_test[i]}  -->  📢 Predicted Letter: {pred_label}")

# === Save Model and Scaler ===
model.save("keras_model.h5")
joblib.dump(le, "label_encoder.pkl")
np.save("x_train_mean.npy", mean)
np.save("x_train_std.npy", std)

print("\n💾 Saved model and encoders.")
