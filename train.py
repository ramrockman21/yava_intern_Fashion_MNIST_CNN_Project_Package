"""Train and evaluate a regularized CNN on Fashion-MNIST.

The script is intentionally self-contained so it can be run directly from VS Code:
    python train.py --epochs 20 --batch-size 128 --output-dir outputs
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def set_reproducibility(seed: int) -> None:
    """Set seeds to make experiments repeatable where the backend permits it."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_data(seed: int):
    """Load, scale, reshape, and stratify-split Fashion-MNIST."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train, test_size=0.10, random_state=seed, stratify=y_train
    )
    return x_fit, y_fit, x_val, y_val, x_test, y_test


def build_model(input_shape=(28, 28, 1), num_classes=10) -> tf.keras.Model:
    """Create a compact CNN with augmentation, batch normalization, dropout, and L2 weight decay."""
    l2 = tf.keras.regularizers.l2(1e-4)
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="reflect", name="augment_translation")(inputs)
    x = tf.keras.layers.RandomZoom(0.10, name="augment_zoom")(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=l2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.20)(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=l2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=l2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=l2)(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    model = tf.keras.Model(inputs, outputs, name="fashion_mnist_regularized_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )
    return model


def save_training_curves(history: tf.keras.callbacks.History, out_dir: Path) -> None:
    hist = pd.DataFrame(history.history)
    hist.to_csv(out_dir / "history.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = np.arange(1, len(hist) + 1)
    axes[0].plot(epochs, hist["loss"], label="Train", linewidth=2)
    axes[0].plot(epochs, hist["val_loss"], label="Validation", linewidth=2)
    axes[0].set_title("Loss across epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Sparse cross-entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, hist["accuracy"], label="Train", linewidth=2)
    axes[1].plot(epochs, hist["val_accuracy"], label="Validation", linewidth=2)
    axes[1].set_title("Accuracy across epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle("Fashion-MNIST CNN learning curves", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, out_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_title("Test-set confusion matrix", fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_sample_predictions(x_test, y_test, probabilities, out_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x_test), size=12, replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(11, 8))
    for ax, idx in zip(axes.ravel(), indices):
        true_idx = int(y_test[idx])
        pred_idx = int(np.argmax(probabilities[idx]))
        confidence = float(np.max(probabilities[idx]))
        color = "#2E8B57" if true_idx == pred_idx else "#C0392B"
        ax.imshow(x_test[idx].squeeze(), cmap="gray")
        ax.set_title(f"True: {CLASS_NAMES[true_idx]}\nPred: {CLASS_NAMES[pred_idx]} ({confidence:.0%})",
                     fontsize=8, color=color)
        ax.axis("off")
    fig.suptitle("Qualitative test-set predictions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "sample_predictions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate(model, x_test, y_test, out_dir: Path, seed: int) -> dict:
    probabilities = model.predict(x_test, batch_size=256, verbose=0)
    y_pred = probabilities.argmax(axis=1)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(out_dir / "classification_report.csv")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "test_examples": int(len(y_test)),
        "test_accuracy": float(np.mean(y_pred == y_test)),
        "test_top3_accuracy": float(np.mean([y_test[i] in np.argsort(probabilities[i])[-3:] for i in range(len(y_test))])),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "best_validation_accuracy": float(max(model.history.history.get("val_accuracy", [float("nan")])) if hasattr(model, "history") else float("nan")),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(y_test, y_pred, out_dir)
    save_sample_predictions(x_test, y_test, probabilities, out_dir, seed)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_reproducibility(args.seed)
    x_fit, y_fit, x_val, y_val, x_test, y_test = load_data(args.seed)
    model = build_model()
    model.summary()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.output_dir / "best_model.keras", monitor="val_accuracy", mode="max", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(args.output_dir / "training_log.csv"),
    ]
    history = model.fit(
        x_fit, y_fit,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    save_training_curves(history, args.output_dir)
    best_model = tf.keras.models.load_model(args.output_dir / "best_model.keras")
    # Attach history so the evaluation JSON includes the best validation score.
    best_model.history = history
    metrics = evaluate(best_model, x_test, y_test, args.output_dir, args.seed)
    metadata = {
        "dataset": "Fashion-MNIST",
        "classes": CLASS_NAMES,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "train_examples": int(len(y_fit)),
        "validation_examples": int(len(y_val)),
        "test_examples": int(len(y_test)),
        "metrics_file": str(args.output_dir / "metrics.json"),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nRun complete. Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
