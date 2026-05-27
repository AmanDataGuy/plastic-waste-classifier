# =============================================================================
# train.py — Plastic Waste Classifier
# Implements both tasks from:
# "A Vision-Based Automated System for Plastic Waste Segregation
#  Using Deep Learning" (IEEE AIEI 2026)
#
# Task 1: Binary Classification  (Plastic vs Non-Plastic)  — Xception
# Task 2: Multiclass Classification (Bottle / Box / Polythene) — DenseNet121
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.applications import Xception, DenseNet121
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# =============================================================================
# 1. PATHS
# =============================================================================

BINARY_DATA_DIR     = r"D:\projects\plastic-waste-classifier\data\binary"
MULTICLASS_DATA_DIR = r"D:\projects\plastic-waste-classifier\data\multiclass"
MODELS_DIR          = r"D:\projects\plastic-waste-classifier\models"

os.makedirs(MODELS_DIR, exist_ok=True)


# =============================================================================
# 2. HYPERPARAMETERS
# Paper: 10 epochs, Adam optimizer, augmentation as per Table III
# =============================================================================

EPOCHS        = 10
BATCH_SIZE    = 32
LEARNING_RATE = 0.001

# Native input sizes for each architecture
XCEPTION_IMG_SIZE  = (299, 299)
DENSENET_IMG_SIZE  = (224, 224)


# =============================================================================
# 3. DATA AUGMENTATION
# Paper Table III:
#   - Rotation       : ±15°
#   - Horizontal Flip: probability 0.5
#   - Width Shift    : 0.1 fraction
#   - Height Shift   : 0.1 fraction
#   - Zoom           : ±10%
#   - Brightness     : 0.8 – 1.2
# Augmentation applied to train split only. Test split uses rescale only.
# 80/20 train/test split as per paper.
# =============================================================================

def get_generators(data_dir, img_size, batch_size, class_mode):
    """
    Returns train and validation ImageDataGenerators from the same directory.
    Validation split = 0.2 (80/20 as per paper).
    """

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2]
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset="training",
        shuffle=True,
        seed=42
    )

    test_gen = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset="validation",
        shuffle=False,
        seed=42
    )

    return train_gen, test_gen


# =============================================================================
# 4. CLASS WEIGHTS  (binary only)
# Binary dataset: non_plastic ~10,662 | plastic ~1,597 → ~6:1 imbalance.
# Without class weights the model learns to always predict non_plastic and
# still achieves ~87% accuracy without ever learning plastic features.
# Balanced class weights force equal penalty for both classes.
# =============================================================================

def get_class_weights(train_gen):
    """
    Computes balanced class weights from training generator labels.
    Returns dict: {class_index: weight}
    """
    labels  = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )
    return dict(enumerate(weights))


# =============================================================================
# 5. MODEL DEFINITIONS
# =============================================================================

def build_xception(num_classes=1):
    """
    Binary classifier — Xception backbone.

    Architecture (paper):
      Pretrained Xception (ImageNet) → FROZEN
      GlobalAveragePooling2D
      Dropout(0.2)
      BatchNormalization
      FC(1024, relu) → FC(1024, relu)
      Dense(1, sigmoid)              ← binary output

    Note: GAP already outputs a 1D tensor; no Flatten needed after it.
    """
    base = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(*XCEPTION_IMG_SIZE, 3)
    )
    base.trainable = False          # Freeze backbone — transfer learning only

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(1024, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_densenet(num_classes=3):
    """
    Multiclass classifier — DenseNet121 backbone.

    Architecture:
      Pretrained DenseNet121 (ImageNet) → FROZEN
        Dense blocks: 6-12-24-16 layers
        Transition layers: 1×1 conv + 2×2 avg pool
      GlobalAveragePooling2D
      Dropout(0.2)
      FC(1024, relu)
      Dense(3, softmax)              ← 3-class output

    Lighter head than Xception intentionally: DenseNet121 already has
    deep dense connections internally. A heavy FC stack on top overfits
    on a ~1,200-image multiclass dataset.
    """
    base = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(*DENSENET_IMG_SIZE, 3)
    )
    base.trainable = False          # Freeze backbone — transfer learning only

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# =============================================================================
# 6. TRAINING
# =============================================================================

def train_model(model, train_gen, test_gen, model_name, class_weights=None):
    """
    Trains model for up to EPOCHS epochs.
    Saves best checkpoint based on val_accuracy.
    EarlyStopping (patience=5) prevents wasted epochs on plateaus.
    """
    save_path = os.path.join(MODELS_DIR, f"{model_name}.h5")

    callbacks = [
        ModelCheckpoint(
            filepath=save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=test_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    print(f"\n{model_name} — Best val_accuracy: "
          f"{max(history.history['val_accuracy']):.4f}")

    return history


# =============================================================================
# 7. TRAINING CURVES
# Plots training & validation accuracy and loss on a dual-axis chart.
# Matches Fig. 7 and Fig. 8(a) from the paper.
# Saved as PNG to models/ directory.
# =============================================================================

def plot_history(history, model_name):
    acc      = history.history["accuracy"]
    val_acc  = history.history["val_accuracy"]
    loss     = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Left y-axis — Accuracy
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy", color="blue")
    ax1.plot(epochs_range, acc,     color="blue",  label="Training Accuracy")
    ax1.plot(epochs_range, val_acc, color="cyan",  label="Validation Accuracy",
             linestyle="--")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Right y-axis — Loss
    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss", color="red")
    ax2.plot(epochs_range, loss,     color="red",    label="Training Loss")
    ax2.plot(epochs_range, val_loss, color="orange", label="Validation Loss",
             linestyle="--")
    ax2.tick_params(axis="y", labelcolor="red")

    fig.suptitle(f"Training Accuracy vs Training Loss — {model_name}")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.tight_layout()

    save_path = os.path.join(MODELS_DIR, f"{model_name}_training_curve.png")
    plt.savefig(save_path)
    print(f"Training curve saved → {save_path}")
    plt.show()


# =============================================================================
# 8. MAIN
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # TASK 1 — Binary Classification (Xception)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TASK 1 — Binary Classification (Xception)")
    print("=" * 60)

    binary_train_gen, binary_test_gen = get_generators(
        data_dir   = BINARY_DATA_DIR,
        img_size   = XCEPTION_IMG_SIZE,
        batch_size = BATCH_SIZE,
        class_mode = "binary"
    )

    print(f"\nClass indices : {binary_train_gen.class_indices}")
    print(f"Train samples : {binary_train_gen.samples}")
    print(f"Test samples  : {binary_test_gen.samples}")

    class_weights = get_class_weights(binary_train_gen)
    print(f"Class weights : {class_weights}")

    xception_model = build_xception()
    xception_model.summary()

    binary_history = train_model(
        model         = xception_model,
        train_gen     = binary_train_gen,
        test_gen      = binary_test_gen,
        model_name    = "xception_best",
        class_weights = class_weights
    )

    plot_history(binary_history, "Xception_Binary")

    # -------------------------------------------------------------------------
    # TASK 2 — Multiclass Classification (DenseNet121)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TASK 2 — Multiclass Classification (DenseNet121)")
    print("=" * 60)

    multi_train_gen, multi_test_gen = get_generators(
        data_dir   = MULTICLASS_DATA_DIR,
        img_size   = DENSENET_IMG_SIZE,
        batch_size = BATCH_SIZE,
        class_mode = "categorical"
    )

    print(f"\nClass indices : {multi_train_gen.class_indices}")
    print(f"Train samples : {multi_train_gen.samples}")
    print(f"Test samples  : {multi_test_gen.samples}")

    densenet_model = build_densenet(num_classes=3)
    densenet_model.summary()

    multi_history = train_model(
        model      = densenet_model,
        train_gen  = multi_train_gen,
        test_gen   = multi_test_gen,
        model_name = "densenet_best"
    )

    plot_history(multi_history, "DenseNet121_Multiclass")

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Models saved to : {MODELS_DIR}")
    print("=" * 60)