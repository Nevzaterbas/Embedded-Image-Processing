import os
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mnist import load_images, load_labels

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt

# --- Paths (kitaptaki gibi) ---
train_img_path   = os.path.join("MNIST-dataset", "train-images.idx3-ubyte")
train_label_path = os.path.join("MNIST-dataset", "train-labels.idx1-ubyte")
test_img_path    = os.path.join("MNIST-dataset", "t10k-images.idx3-ubyte")
test_label_path  = os.path.join("MNIST-dataset", "t10k-labels.idx1-ubyte")

# --- Load MNIST ---
train_images = load_images(train_img_path)
train_labels = load_labels(train_label_path)
test_images  = load_images(test_img_path)
test_labels  = load_labels(test_label_path)

# --- Feature extraction: Hu moments (7 features) ---
train_huMoments = np.empty((len(train_images), 7), dtype=np.float64)
test_huMoments  = np.empty((len(test_images), 7), dtype=np.float64)

for i, img in enumerate(train_images):
    m = cv2.moments(img, binaryImage=True)
    train_huMoments[i] = cv2.HuMoments(m).reshape(7)

for i, img in enumerate(test_images):
    m = cv2.moments(img, binaryImage=True)
    test_huMoments[i] = cv2.HuMoments(m).reshape(7)

# --- (ÖNEMLİ) Standardization: 10.9'da yaptığımız gibi ---
mean = np.mean(train_huMoments, axis=0)
std  = np.std(train_huMoments, axis=0)
std[std == 0] = 1.0

train_huMoments = (train_huMoments - mean) / std
test_huMoments  = (test_huMoments  - mean) / std

# --- MLP Model (3 katman) ---
model = keras.models.Sequential([
    keras.layers.Dense(100, input_shape=(7,), activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

# Kategoriler 0..9
categories = np.unique(test_labels)

# --- Compile ---
# Kitap: SparseCategoricalCrossentropy
# Optimizer: Adam (kitapta 1e4 yazıyor ama o genellikle patlatır; 1e-3 stabil)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["sparse_categorical_accuracy"]
)

# --- Callbacks ---
# ModelCheckpoint: en iyi modeli kaydet
mc_callback = ModelCheckpoint(
    "mlp_mnist_model.h5",
    monitor="loss",
    save_best_only=True,
    verbose=1
)

# EarlyStopping: iyileşme durunca durdur
es_callback = EarlyStopping(
    monitor="loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# --- Train ---
history = model.fit(
    train_huMoments,
    train_labels,
    epochs=1000,
    batch_size=128,
    verbose=1,
    callbacks=[mc_callback, es_callback]
)

# --- Predict & Confusion Matrix ---
nn_probs = model.predict(test_huMoments, verbose=0)
predicted_classes = np.argmax(nn_probs, axis=1)

conf_matrix = confusion_matrix(test_labels, predicted_classes, labels=categories)
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=categories)
cm_display.plot(values_format="d")
cm_display.ax_.set_title("Neural Network Confusion Matrix")
plt.show()

print("Saved best model as: mlp_mnist_model.h5")
