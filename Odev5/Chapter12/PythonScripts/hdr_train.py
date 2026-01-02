# hdr_train.py
# Kullanım: python hdr_train.py --epochs 10
import os
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()

# Load MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

input_shape = train_images.shape[1:]  # (28,28,1)
num_classes = 10

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(train_images, train_labels, validation_split=0.1, epochs=args.epochs, batch_size=128)

loss, acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"MNIST test accuracy: {acc*100:.2f}%")

out_dir = "models_mnist"
os.makedirs(out_dir, exist_ok=True)
keras_path = os.path.join(out_dir, "mnist_cnn.h5")
model.save(keras_path)
print("Saved Keras model:", keras_path)

# Convert and quantize
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Full integer quantization (requires representative dataset)
def representative_data_gen():
    for i in range(100):
        img = train_images[i:i+1]
        yield [img]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
with open(os.path.join(out_dir, "mnist_cnn_int8.tflite"), "wb") as f:
    f.write(tflite_model)
print("Saved int8 TFLite model.")