# import os
# import librosa
# import librosa.display
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import random

# # === Settings ===
# wav_folder = "converted_wav"

# # === Feature Extraction Function ===
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

# # === Split Files 70% Train / 30% Test ===
# all_files = [f for f in os.listdir(wav_folder) if f.endswith(".wav")]
# random.shuffle(all_files)
# split_point = int(0.7 * len(all_files))
# train_files = all_files[:split_point]
# test_files = all_files[split_point:]

# print(f"✅ Total files: {len(all_files)}, Training: {len(train_files)}, Testing: {len(test_files)}")

# # === Extract Features for Train and Test ===
# train_rows = []
# for file in train_files:
#     try:
#         path = os.path.join(wav_folder, file)
#         y, sr = librosa.load(path, sr=16000)
#         y = y / np.max(np.abs(y))
#         features = extract_features(y, sr)
#         label = file[0].upper()
#         row = np.append(features, [label, file])  # keep filename for both train and test
#         train_rows.append(row)
#     except Exception as e:
#         print(f"⚠️ Error processing {file}: {e}")

# test_rows = []
# for file in test_files:
#     try:
#         path = os.path.join(wav_folder, file)
#         y, sr = librosa.load(path, sr=16000)
#         y = y / np.max(np.abs(y))
#         features = extract_features(y, sr)
#         label = file[0].upper()
#         row = np.append(features, [label, file])
#         test_rows.append(row)
#     except Exception as e:
#         print(f"⚠️ Error processing {file}: {e}")

# # === Prepare Data ===
# feature_dim = len(train_rows[0]) - 2  # features only (label + filename at the end)
# columns = [f'f{i+1}' for i in range(feature_dim)] + ['label', 'filename']

# df_train = pd.DataFrame(train_rows, columns=columns)
# df_test = pd.DataFrame(test_rows, columns=columns)

# # Encode labels
# label_encoder = LabelEncoder()
# df_train['label_encoded'] = label_encoder.fit_transform(df_train['label'])
# df_test['label_encoded'] = label_encoder.transform(df_test['label'])  # Important: same encoder

# # Features and Labels
# X_train = df_train.drop(columns=['label', 'filename']).values.astype(float)
# y_train = df_train['label_encoded'].values

# X_test = df_test.drop(columns=['label', 'filename']).values.astype(float)
# y_test = df_test['label_encoded'].values
# filenames_test = df_test['filename'].values

# # Normalize (Important: using train statistics)
# X_train_mean = np.mean(X_train, axis=0)
# X_train_std = np.std(X_train, axis=0) + 1e-8

# X_train = (X_train - X_train_mean) / X_train_std
# X_test = (X_test - X_train_mean) / X_train_std

# # === Train Model ===
# model = RandomForestClassifier(n_estimators=200, random_state=42)
# model.fit(X_train, y_train)

# # === Test Model ===
# y_pred = model.predict(X_test)
# acc = accuracy_score(y_test, y_pred)
# print(f"\n✅ Random Forest Test Accuracy: {acc:.2f}")
# print("\n📄 Classification Report:")
# print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# # === Print each test file prediction ===
# print("\n🎯 Predictions on Test Set:")
# for i in range(len(X_test)):
#     pred_label = label_encoder.inverse_transform([y_pred[i]])[0]
#     print(f"📂 File: {filenames_test[i]}  -->  📢 Predicted Letter: {pred_label}")


# # === Predict and Plot a Specific File ===
# filepath = "converted_wav/A8.wav"
# y, sr = librosa.load(filepath, sr=16000)
# y = y / np.max(np.abs(y))
# features = extract_features(y, sr)
# features = features.reshape(1, -1) 
# features = (features - X_train_mean) / X_train_std
# predicted_label = label_encoder.inverse_transform(model.predict(features))[0]
# print(f"\n📂 File: {filepath}")
# print(f"📢 Predicted letter: {predicted_label}")

# plt.figure(figsize=(10, 4))
# librosa.display.waveplot(y, sr=sr)
# plt.title(f"Waveform of {filepath}", fontsize=14)
# plt.text(0.01, 0.88, f"📢 Predicted: {predicted_label}", transform=plt.gca().transAxes, fontsize=12, color='blue')
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.tight_layout()
# plt.show()


import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import random

# === Settings ===
wav_folder = "converted_wav"

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

# === Split Files 70% Train / 30% Test ===
all_files = [f for f in os.listdir(wav_folder) if f.endswith(".wav")]
random.shuffle(all_files)
split_point = int(0.7 * len(all_files))
train_files = all_files[:split_point]
test_files = all_files[split_point:]

print(f"✅ Total files: {len(all_files)}, Training: {len(train_files)}, Testing: {len(test_files)}")

# === Extract Features for Train and Test ===
train_rows = []
for file in train_files:
    try:
        path = os.path.join(wav_folder, file)
        y, sr = librosa.load(path, sr=16000)
        y = y / np.max(np.abs(y))
        features = extract_features(y, sr)
        label = file[0].upper()
        row = np.append(features, [label, file])
        train_rows.append(row)
    except Exception as e:
        print(f"⚠️ Error processing {file}: {e}")

test_rows = []
for file in test_files:
    try:
        path = os.path.join(wav_folder, file)
        y, sr = librosa.load(path, sr=16000)
        y = y / np.max(np.abs(y))
        features = extract_features(y, sr)
        label = file[0].upper()
        row = np.append(features, [label, file])
        test_rows.append(row)
    except Exception as e:
        print(f"⚠️ Error processing {file}: {e}")

# === Prepare Data ===
feature_dim = len(train_rows[0]) - 2  # features only (label + filename at the end)
columns = [f'f{i+1}' for i in range(feature_dim)] + ['label', 'filename']

df_train = pd.DataFrame(train_rows, columns=columns)
df_test = pd.DataFrame(test_rows, columns=columns)

# Encode labels
label_encoder = LabelEncoder()
df_train['label_encoded'] = label_encoder.fit_transform(df_train['label'])
df_test['label_encoded'] = label_encoder.transform(df_test['label'])

# Features and Labels
X_train = df_train.drop(columns=['label', 'filename']).values.astype(float)
y_train = df_train['label_encoded'].values

X_test = df_test.drop(columns=['label', 'filename']).values.astype(float)
y_test = df_test['label_encoded'].values
filenames_test = df_test['filename'].values

# === Normalize ===
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0) + 1e-8

X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

# === Train Model ===
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# === Test Model ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Random Forest Test Accuracy: {acc:.2f}")
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# === Print each test file prediction ===
print("\n🎯 Predictions on Test Set:")
for i in range(len(X_test)):
    pred_label = label_encoder.inverse_transform([y_pred[i]])[0]
    print(f"📂 File: {filenames_test[i]}  -->  📢 Predicted Letter: {pred_label}")

# === Predict and Plot a Specific File ===
def predict_and_plot(filepath):
    try:
        # Load and preprocess the specific file
        y, sr = librosa.load(filepath, sr=16000)
        y = y / np.max(np.abs(y))  # Normalize amplitude
        features = extract_features(y, sr)

        # Force feature size to match training
        expected_feature_size = X_train.shape[1]
        if features.shape[0] != expected_feature_size:
            features = np.resize(features, expected_feature_size)

        # Normalize
        features = (features - X_train_mean) / X_train_std
        features = features.reshape(1, -1)

        # Predict
        pred_index = model.predict(features)[0]
        pred_label = label_encoder.inverse_transform([pred_index])[0]

        # Output result
        print(f"\n📂 File: {os.path.basename(filepath)}")
        print(f"📢 Predicted Letter: {pred_label}")

        # Plot waveform
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr)
        plt.title(f"Waveform of {os.path.basename(filepath)}", fontsize=14)
        plt.text(0.01, 0.88, f"📢 Predicted: {pred_label}", transform=plt.gca().transAxes,
                 fontsize=12, color='blue')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"⚠️ Error predicting {filepath}: {e}")

import joblib

joblib.dump(model, "model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
np.save("x_train_mean.npy", X_train_mean)
np.save("x_train_std.npy", X_train_std)
