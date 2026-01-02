# kws_train.py
# Kullanım: python kws_train.py --data_dir path/to/FSDD --epochs 30
import os
import argparse
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="FSDD", help="FSDD wav files directory")
parser.add_argument("--sr", type=int, default=8000)
parser.add_argument("--n_mfcc", type=int, default=13)
parser.add_argument("--time_frames", type=int, default=32)  # örnek
parser.add_argument("--epochs", type=int, default=30)
args = parser.parse_args()

def load_wavs_mfcc(data_dir, sr, n_mfcc, time_frames):
    X = []
    y = []
    files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]
    for f in files:
        path = os.path.join(data_dir, f)
        # FSDD file name formati: <digit>_<speaker>_<index>.wav
        label = int(f.split("_")[0])
        wav, _ = librosa.load(path, sr=sr)
        # MFCC
        mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc)
        # mfcc shape = (n_mfcc, frames). Sabitle: pad veya trim frames
        if mfcc.shape[1] < time_frames:
            pad_width = time_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :time_frames]
        X.append(mfcc)
        y.append(label)
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.int32)
    # normalize per-feature
    X = (X - np.mean(X)) / (np.std(X)+1e-8)
    # add channel dim
    X = X[..., np.newaxis]  # shape: (N, n_mfcc, time_frames, 1)
    return X, y

print("Loading data...")
X, y = load_wavs_mfcc(args.data_dir, args.sr, args.n_mfcc, args.time_frames)
num_classes = len(np.unique(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Basit CNN
input_shape = X_train.shape[1:]
model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_split=0.1, epochs=args.epochs, batch_size=32)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc*100:.2f}%")

# Save Keras .h5
out_dir = "models_kws"
os.makedirs(out_dir, exist_ok=True)
keras_path = os.path.join(out_dir, "kws_cnn.h5")
model.save(keras_path)
print("Saved Keras model:", keras_path)

# Convert to TFLite (float)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(os.path.join(out_dir, "kws_cnn.tflite"), "wb") as f:
    f.write(tflite_model)
print("Saved TFLite model (float).")

# Post-training quantization (dynamic range)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant = converter.convert()
with open(os.path.join(out_dir, "kws_cnn_quant.tflite"), "wb") as f:
    f.write(tflite_quant)
print("Saved quantized TFLite model.")