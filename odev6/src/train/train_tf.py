import os
import json
import time
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix


# -------------------------
# Helpers
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# -------------------------
# Dataset (MNIST baseline, RAM şişirmeden tf.data ile)
# Not: Kitaptaki "offline dataset" farklıysa sadece bu fonksiyonu değiştiririz.
# -------------------------
def make_mnist_datasets(img_size: int = 96, batch_size: int = 32):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    def preprocess(x, y):
        # x: (28,28) uint8
        x = tf.cast(x, tf.float32) / 255.0           # (28,28)
        x = tf.expand_dims(x, axis=-1)               # (28,28,1)
        x = tf.image.resize(x, (img_size, img_size)) # (img,img,1)
        x = tf.repeat(x, 3, axis=-1)                 # (img,img,3) -> CNN'ler için
        return x, y

    train_ds = (
        train_ds
        .shuffle(20000)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        test_ds
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, test_ds, y_test


# -------------------------
# Models
# -------------------------
def fire_module(x, squeeze=16, expand=64, name="fire"):
    s = tf.keras.layers.Conv2D(squeeze, 1, activation="relu", padding="same", name=f"{name}_squeeze")(x)
    e1 = tf.keras.layers.Conv2D(expand, 1, activation="relu", padding="same", name=f"{name}_expand1x1")(s)
    e3 = tf.keras.layers.Conv2D(expand, 3, activation="relu", padding="same", name=f"{name}_expand3x3")(s)
    return tf.keras.layers.Concatenate(name=f"{name}_concat")([e1, e3])


def build_squeezenet_like(input_shape=(96, 96, 3), num_classes=10):
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(inp)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = fire_module(x, squeeze=16, expand=64, name="fire2")
    x = fire_module(x, squeeze=16, expand=64, name="fire3")
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = fire_module(x, squeeze=32, expand=128, name="fire4")
    x = fire_module(x, squeeze=32, expand=128, name="fire5")
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = fire_module(x, squeeze=48, expand=192, name="fire6")
    x = fire_module(x, squeeze=48, expand=192, name="fire7")
    x = fire_module(x, squeeze=64, expand=256, name="fire8")
    x = fire_module(x, squeeze=64, expand=256, name="fire9")

    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Conv2D(num_classes, 1, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Activation("softmax")(x)

    return tf.keras.Model(inp, out, name="SqueezeNetLike")


def build_efficientnet_b0(input_shape=(96, 96, 3), num_classes=10):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,         # offline eğitim için None
        input_shape=input_shape,
        pooling="avg",
    )
    inp = tf.keras.Input(shape=input_shape)
    x = base(inp)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="EfficientNetB0")


def build_model(model_name: str, input_shape, num_classes: int):
    if model_name == "squeezenet":
        return build_squeezenet_like(input_shape=input_shape, num_classes=num_classes)
    if model_name == "efficientnet_b0":
        return build_efficientnet_b0(input_shape=input_shape, num_classes=num_classes)
    raise ValueError(f"Unknown model_name: {model_name}")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="squeezenet",
                        choices=["squeezenet", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img", type=int, default=96)  # 96 ile başla (RAM/Speed iyi)
    args = parser.parse_args()

    # Reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    results_dir = os.path.join("results", args.model)
    ensure_dir(results_dir)

    # Dataset
    train_ds, test_ds, y_test = make_mnist_datasets(img_size=args.img, batch_size=args.batch)

    # Model
    model = build_model(args.model, input_shape=(args.img, args.img, 3), num_classes=10)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train
    t0 = time.time()
    history = model.fit(train_ds, epochs=args.epochs, validation_data=test_ds, verbose=1)
    train_time = time.time() - t0

    # Evaluate
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    # Predict for confusion matrix
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    # Save artifacts
    model.save(os.path.join(results_dir, "saved_model"))

    metrics = {
        "model": args.model,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "img": args.img,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "train_time_sec": float(train_time),
        "params": int(model.count_params()),
    }
    save_json(os.path.join(results_dir, "metrics.json"), metrics)
    np.savetxt(os.path.join(results_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    # Print summary
    print("\n=== DONE ===")
    print(f"Model: {args.model}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Params: {model.count_params()}")
    print(f"Train time (s): {train_time:.1f}")
    print(f"Saved to: {results_dir}")


if __name__ == "__main__":
    main()
