import numpy as np
np.complex = complex  # ✅ إصلاح مؤقت لمشكلة np.complex في librosa

import os
import librosa
import pandas as pd
import random
import joblib  # ✅ For saving Random Forest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# === Settings ===
wav_folder = "converted_wav"
sr = 16000
feature_size = 180

# === Feature extraction ===
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

    if features.shape[0] != feature_size:
        features = np.resize(features, feature_size)
    return features

# === Load & prepare dataset ===
def prepare_dataset(file_list):
    rows = []
    for file in file_list:
        try:
            path = os.path.join(wav_folder, file)
            y, _ = librosa.load(path, sr=sr)
            y = y / np.max(np.abs(y))
            features = extract_features(y, sr)
            label = file[0].upper().strip()
            rows.append([*features, label, file])
        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")
    return rows

# === Read WAV files ===
all_files = [f for f in os.listdir(wav_folder) if f.endswith(".wav")]
random.shuffle(all_files)
split = int(0.7 * len(all_files))
train_files = all_files[:split]
test_files = all_files[split:]

train_rows = prepare_dataset(train_files)
test_rows = prepare_dataset(test_files)

feature_cols = [f'f{i+1}' for i in range(feature_size)]
columns = feature_cols + ['label', 'filename']
df_train = pd.DataFrame(train_rows, columns=columns).dropna()
df_test = pd.DataFrame(test_rows, columns=columns).dropna()

# === Label encoding ===
le = LabelEncoder()
df_train['label'] = df_train['label'].str.upper().str.strip()
df_test['label'] = df_test['label'].str.upper().str.strip()
df_train['label_encoded'] = le.fit_transform(df_train['label'])
df_test['label_encoded'] = le.transform(df_test['label'])

# === Features and labels ===
X_train = df_train[feature_cols].astype(float).values
X_test = df_test[feature_cols].astype(float).values
y_train_int = df_train['label_encoded'].values
y_test_int = df_test['label_encoded'].values
y_train_cat = to_categorical(y_train_int)
y_test_cat = to_categorical(y_test_int)

# === Normalization ===
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8
X_train_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std

# === Random Forest ===
print("\n🔵 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train_int)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test_int, rf_preds)

# === MLP Model ===
print("\n🟢 Training MLP...")
mlp_model = Sequential([
    Dense(256, input_shape=(feature_size,), activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])
mlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mlp_model.fit(X_train_norm, y_train_cat, epochs=30, batch_size=32, validation_split=0.2, verbose=1)
mlp_acc = mlp_model.evaluate(X_test_norm, y_test_cat, verbose=0)[1]
mlp_preds = np.argmax(mlp_model.predict(X_test_norm), axis=1)

# === CNN Model ===
print("\n🟠 Training CNN...")
X_train_cnn = X_train_norm.reshape(-1, 15, 12, 1)
X_test_cnn = X_test_norm.reshape(-1, 15, 12, 1)
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(15, 12, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_cnn, y_train_cat, epochs=30, batch_size=32, validation_split=0.2, verbose=1)
cnn_acc = cnn_model.evaluate(X_test_cnn, y_test_cat, verbose=0)[1]
cnn_preds = np.argmax(cnn_model.predict(X_test_cnn), axis=1)

# === Save Models ===
print("\n💾 Saving all models...")
np.save("x_train_mean.npy", mean)
np.save("x_train_std.npy", std)

# Save Random Forest
joblib.dump(rf, "random_forest_model.pkl")
print("✅ Random Forest model saved as random_forest_model.pkl")

# Save MLP
mlp_model.save("mlp_model.h5")
print("✅ MLP model saved as mlp_model.h5")

# Save CNN
cnn_model.save("cnn_model.h5")
print("✅ CNN model saved as cnn_model.h5")

# === Results ===
print("\n✅ Accuracy Comparison:")
print(f"Random Forest: {rf_acc:.4f}")
print(f"MLP:           {mlp_acc:.4f}")
print(f"CNN:           {cnn_acc:.4f}")

print("\n📊 Classification Reports:")
print("\nRandom Forest:\n", classification_report(y_test_int, rf_preds, target_names=le.classes_))
print("\nMLP:\n", classification_report(y_test_int, mlp_preds, target_names=le.classes_))
print("\nCNN:\n", classification_report(y_test_int, cnn_preds, target_names=le.classes_))
