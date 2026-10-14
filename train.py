"""
Food image classification: EfficientNetB0 vs. MobileNet transfer learning.

Dataset: Food-101 (Bossard, Guillaumin & Van Gool, 2014) - 101 food
categories, ~1000 images each. Download it with download_data.sh before
running this (not bundled here - the full dataset is several GB).

Approach: both models use ImageNet-pretrained weights as a frozen feature
extractor with a new classification head on top, trained and evaluated
identically so the comparison between the two architectures is fair.
EfficientNetB0 and MobileNet are both designed for efficiency (far fewer
params than VGG16), which is the reason they're commonly compared for
mobile/edge deployment - this script reports which one actually wins on
accuracy vs. training time on this dataset.

Usage:
    python train.py --data-dir food-101/images --epochs 10
"""
import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import efficientnet, mobilenet

IMG_SIZE = (224, 224)


def build_model(base_constructor, preprocess_fn, num_classes):
    base = base_constructor(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = preprocess_fn(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


MODEL_REGISTRY = {
    "efficientnetb0": (efficientnet.EfficientNetB0, efficientnet.preprocess_input),
    "mobilenet": (mobilenet.MobileNet, mobilenet.preprocess_input),
}


def train_one(name, train_ds, val_ds, test_ds, num_classes, epochs):
    constructor, preprocess_fn = MODEL_REGISTRY[name]
    model = build_model(constructor, preprocess_fn, num_classes)

    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)
    train_time = time.time() - start

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    n_params = model.count_params()

    print(f"\n{name}: test_acc={test_acc:.4f}  train_time={train_time:.1f}s  params={n_params:,}")
    return {"name": name, "history": history, "test_acc": test_acc, "train_time": train_time, "params": n_params}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="food-101/images")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir, validation_split=0.3, subset="training", seed=42,
        image_size=IMG_SIZE, batch_size=args.batch_size,
    )
    val_ds_full = tf.keras.utils.image_dataset_from_directory(
        args.data_dir, validation_split=0.3, subset="validation", seed=42,
        image_size=IMG_SIZE, batch_size=args.batch_size,
    )
    class_names = train_ds.class_names
    print("Classes:", class_names)

    val_batches = tf.data.experimental.cardinality(val_ds_full)
    test_ds = val_ds_full.take(val_batches // 2).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds_full.skip(val_batches // 2).prefetch(tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    results = [
        train_one(name, train_ds, val_ds, test_ds, len(class_names), args.epochs)
        for name in MODEL_REGISTRY
    ]

    print("\n=== Comparison ===")
    print(f"{'Model':<16}{'Test acc':<12}{'Train time (s)':<18}{'Params':<12}")
    for r in results:
        print(f"{r['name']:<16}{r['test_acc']:<12.4f}{r['train_time']:<18.1f}{r['params']:<12,}")

    plt.figure(figsize=(8, 5))
    for r in results:
        plt.plot(r["history"].history["val_accuracy"], label=r["name"])
    plt.title("Validation accuracy: EfficientNetB0 vs MobileNet")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    print("Saved comparison plot to comparison.png")


if __name__ == "__main__":
    main()
