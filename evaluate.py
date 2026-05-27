# =============================================================================
# evaluate.py — Plastic Waste Classifier
# Implements evaluation for both tasks from:
# "A Vision-Based Automated System for Plastic Waste Segregation
#  Using Deep Learning" (IEEE AIEI 2026)
#
# Generates:
#   Task 1 (Xception)     — Confusion matrix, Accuracy, FP/FN/FD/FO rates
#   Task 2 (DenseNet121)  — Confusion matrix, Class-wise Precision/Recall/F1
#
# Run AFTER train.py — requires xception_best.h5 and densenet_best.h5
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# =============================================================================
# 1. PATHS
# =============================================================================

BINARY_DATA_DIR     = r"D:\projects\plastic-waste-classifier\data\binary"
MULTICLASS_DATA_DIR = r"D:\projects\plastic-waste-classifier\data\multiclass"
MODELS_DIR          = r"D:\projects\plastic-waste-classifier\models"

XCEPTION_MODEL_PATH  = os.path.join(MODELS_DIR, "xception_best.h5")
DENSENET_MODEL_PATH  = os.path.join(MODELS_DIR, "densenet_best.h5")

# Image sizes — must match train.py
XCEPTION_IMG_SIZE  = (299, 299)
DENSENET_IMG_SIZE  = (224, 224)
BATCH_SIZE         = 32


# =============================================================================
# 2. TEST GENERATOR
# Identical to train.py validation split — same seed ensures same images.
# No augmentation on test set — rescale only.
# =============================================================================

def get_test_generator(data_dir, img_size, batch_size, class_mode):
    """
    Recreates the exact validation split used during training.
    seed=42 and validation_split=0.2 must match train.py exactly.
    """
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    test_gen = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset="validation",
        shuffle=False,          # Must be False for correct label alignment
        seed=42
    )

    return test_gen


# =============================================================================
# 3. CONFUSION MATRIX PLOT
# Matches visual style of Fig. 6(a) and Fig. 8(b) from the paper.
# =============================================================================

def plot_confusion_matrix(cm, class_names, model_name):
    """
    Plots and saves a seaborn heatmap confusion matrix.
    Rows = Actual, Columns = Predicted (paper convention).
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    save_path = os.path.join(MODELS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(save_path)
    print(f"Confusion matrix saved → {save_path}")
    plt.show()


# =============================================================================
# 4. BINARY METRICS
# Computes and prints:
#   - Overall accuracy
#   - FP rate  (False Positive Rate)  = FP / (FP + TN)
#   - FN rate  (False Negative Rate)  = FN / (FN + TP)
#   - FD rate  (False Discovery Rate) = FP / (FP + TP)
#   - FO rate  (False Omission Rate)  = FN / (FN + TN)
# Matches Fig. 6(b) from the paper.
# Paper values: FP ~0.03, FN ~0.07
# =============================================================================

def evaluate_binary(model, test_gen):
    """
    Full binary evaluation — Xception.
    Predicts on test set, computes confusion matrix and error rates.
    """
    print("\n" + "=" * 60)
    print("TASK 1 — Binary Evaluation (Xception)")
    print("=" * 60)

    # Get predictions
    y_pred_prob = model.predict(test_gen, verbose=1)
    y_pred      = (y_pred_prob > 0.5).astype(int).flatten()
    y_true      = test_gen.classes

    # Class names from generator (alphabetical order)
    class_names = list(test_gen.class_indices.keys())

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\nTesting Accuracy : {acc * 100:.2f}%")
    print(f"Paper Target     : 95.00%")

    # Confusion matrix
    # For binary: assume plastic=1, non_plastic=0
    # Actual positive = plastic
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, "Xception_Binary")

    # Extract TP, FP, TN, FN from confusion matrix
    # cm[actual][predicted]
    # non_plastic=0, plastic=1
    plastic_idx     = test_gen.class_indices.get("plastic", 1)
    non_plastic_idx = test_gen.class_indices.get("non_plastic", 0)

    TN = cm[non_plastic_idx][non_plastic_idx]
    FP = cm[non_plastic_idx][plastic_idx]
    FN = cm[plastic_idx][non_plastic_idx]
    TP = cm[plastic_idx][plastic_idx]

    print(f"\nConfusion Matrix Values:")
    print(f"  TP (Plastic   → Plastic)     : {TP}")
    print(f"  TN (NonPlastic→ NonPlastic)  : {TN}")
    print(f"  FP (NonPlastic→ Plastic)     : {FP}")
    print(f"  FN (Plastic   → NonPlastic)  : {FN}")

    # Error rates — Fig. 6(b) from paper
    FP_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
    FN_rate = FN / (FN + TP) if (FN + TP) > 0 else 0
    FD_rate = FP / (FP + TP) if (FP + TP) > 0 else 0
    FO_rate = FN / (FN + TN) if (FN + TN) > 0 else 0

    print(f"\nError Rates (Fig. 6b):")
    print(f"  FP Rate (False Positive Rate) : {FP_rate:.4f}  | Paper: ~0.03")
    print(f"  FN Rate (False Negative Rate) : {FN_rate:.4f}  | Paper: ~0.07")
    print(f"  FD Rate (False Discovery Rate): {FD_rate:.4f}")
    print(f"  FO Rate (False Omission Rate) : {FO_rate:.4f}")

    # Plot error rates bar chart — matches Fig. 6(b)
    plot_error_rates(FP_rate, FN_rate, FD_rate, FO_rate)

    # Full classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def plot_error_rates(fp, fn, fd, fo):
    """
    Bar chart of FP, FN, FD, FO rates.
    Matches Fig. 6(b) from the paper.
    """
    labels = ["FP Rate", "FN Rate", "FD Rate", "FO Rate"]
    values = [fp, fn, fd, fo]
    colors = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors, width=0.5)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=11
        )

    plt.title("FP, FN, FD and FO Rates — Xception Model")
    plt.ylabel("Rate")
    plt.ylim(0, max(values) * 1.3)
    plt.tight_layout()

    save_path = os.path.join(MODELS_DIR, "Xception_error_rates.png")
    plt.savefig(save_path)
    print(f"Error rates chart saved → {save_path}")
    plt.show()


# =============================================================================
# 5. MULTICLASS METRICS
# Computes and prints:
#   - Overall accuracy  (paper: 89.84%)
#   - Class-wise Precision, Recall, F1 — matches Table VII from paper
#   - Confusion matrix  — matches Fig. 8(b) from paper
# Paper values:
#   Plastic Bottle: P=0.84, R=0.98, F1=0.90
#   Plastic Box   : P=0.86, R=0.89, F1=0.87
#   Polythene     : P=0.91, R=0.80, F1=0.85
# =============================================================================

def evaluate_multiclass(model, test_gen):
    """
    Full multiclass evaluation — DenseNet121.
    Predicts on test set, prints class-wise metrics and confusion matrix.
    """
    print("\n" + "=" * 60)
    print("TASK 2 — Multiclass Evaluation (DenseNet121)")
    print("=" * 60)

    # Get predictions
    y_pred_prob = model.predict(test_gen, verbose=1)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    y_true      = test_gen.classes

    # Class names from generator (alphabetical: plastic_bottle, plastic_box, polythene)
    class_names = list(test_gen.class_indices.keys())

    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\nTesting Accuracy : {acc * 100:.2f}%")
    print(f"Paper Target     : 89.84%")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, "DenseNet121_Multiclass")

    # Class-wise metrics — Table VII from paper
    print(f"\nClass-wise Metrics (Table VII):")
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 55)

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )

    paper_targets = {
        "plastic_bottle": {"precision": 0.84, "recall": 0.98, "f1-score": 0.90},
        "plastic_box":    {"precision": 0.86, "recall": 0.89, "f1-score": 0.87},
        "polythene":      {"precision": 0.91, "recall": 0.80, "f1-score": 0.85},
    }

    for cls in class_names:
        p  = report[cls]["precision"]
        r  = report[cls]["recall"]
        f1 = report[cls]["f1-score"]
        print(f"{cls:<20} {p:>10.2f} {r:>10.2f} {f1:>10.2f}")

    print("\nPaper Targets:")
    for cls, vals in paper_targets.items():
        print(f"  {cls:<18} P={vals['precision']}  R={vals['recall']}  F1={vals['f1-score']}")

    # Full report
    print(f"\nFull Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


# =============================================================================
# 6. MAIN
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # TASK 1 — Binary Evaluation (Xception)
    # -------------------------------------------------------------------------
    print("\nLoading Xception model...")
    xception_model = load_model(XCEPTION_MODEL_PATH)

    binary_test_gen = get_test_generator(
        data_dir   = BINARY_DATA_DIR,
        img_size   = XCEPTION_IMG_SIZE,
        batch_size = BATCH_SIZE,
        class_mode = "binary"
    )

    evaluate_binary(xception_model, binary_test_gen)

    # -------------------------------------------------------------------------
    # TASK 2 — Multiclass Evaluation (DenseNet121)
    # -------------------------------------------------------------------------
    print("\nLoading DenseNet121 model...")
    densenet_model = load_model(DENSENET_MODEL_PATH)

    multi_test_gen = get_test_generator(
        data_dir   = MULTICLASS_DATA_DIR,
        img_size   = DENSENET_IMG_SIZE,
        batch_size = BATCH_SIZE,
        class_mode = "categorical"
    )

    evaluate_multiclass(densenet_model, multi_test_gen)

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print(f"All outputs saved to : {MODELS_DIR}")
    print("=" * 60)