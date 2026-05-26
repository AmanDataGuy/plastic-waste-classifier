# Plastic Waste Classifier
Vision-based plastic waste classification using deep learning — binary and multiclass.
Based on: *A Vision-Based Automated System for Plastic Waste Segregation Using Deep Learning* (IEEE AIEI 2026)

---

## Architecture

### Binary Classification (Xception)
```
Input Image
     │
     ▼
┌─────────────────────────────────────┐
│           Preprocessing             │
│  Resize 299x299 │ Normalize [0,1]   │
│     Augmentation (train only)       │
│  rotation │ flip │ zoom │ brightness│
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         Xception Backbone           │
│      (pretrained, ImageNet)         │
│      Depthwise Separable Conv       │
│           [FROZEN]                  │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│        Classification Head          │
│   GlobalAvgPool → Dropout(0.2)      │
│   BatchNorm → FC(1024) → FC(1024)   │
│         → Sigmoid Output            │
└─────────────────┬───────────────────┘
                  │
                  ▼
         Plastic / Non-Plastic
```

### Multiclass Classification (DenseNet121)
```
Input Image (if Plastic)
     │
     ▼
┌─────────────────────────────────────┐
│           Preprocessing             │
│  Resize 224x224 │ Normalize [0,1]   │
│     Augmentation (train only)       │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│        DenseNet121 Backbone         │
│      (pretrained, ImageNet)         │
│   Dense Blocks: 6-12-24-16 layers   │
│           [FROZEN]                  │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│        Classification Head          │
│     GlobalAvgPool → Dropout(0.2)    │
│      FC(1024) → Softmax Output      │
└─────────────────┬───────────────────┘
                  │
                  ▼
     Plastic Bottle / Plastic Box / Polythene
```

---

## Full Pipeline
```
Input Image
     │
     ▼
  Xception
     │
     ├── Non-Plastic → Rejected
     │
     └── Plastic
              │
              ▼
         DenseNet121
              │
              ├── Plastic Bottle
              ├── Plastic Box
              └── Polythene
```

---

## Dataset

### Binary
| Split | Plastic | Non-Plastic | Total  |
|-------|---------|-------------|--------|
| Train | ~1,278  | ~8,530      | ~9,808 |
| Test  | ~319    | ~2,132      | ~2,451 |

Source: Garbage Classification Dataset (Kaggle) — 10 classes merged into 2.
Class imbalance (~6.7:1) handled via balanced class weights during training.

### Multiclass
| Split | Plastic Bottle | Plastic Box | Polythene | Total |
|-------|----------------|-------------|-----------|-------|
| Train | 400            | 400         | 400       | 1,200 |
| Test  | 100            | 100         | 100       | 300   |

Source: Recyclable and Household Waste Classification Dataset (Kaggle) — 7 subfolders merged into 3 classes.

---

## Results

### Binary — Xception

| Metric            | Paper  | Ours       |
|-------------------|--------|------------|
| Training Accuracy | 96%    | **97.17%** |
| Testing Accuracy  | 95%    | **94.86%** |
| Best Epoch        | 10     | 6          |

**Confusion Matrix**

|                     | Predicted: Non-Plastic | Predicted: Plastic |
|---------------------|------------------------|--------------------|
| Actual: Non-Plastic | 2029 (TN)              | 103 (FP)           |
| Actual: Plastic     | 23 (FN)                | 296 (TP)           |

**Error Rates**

| Metric                   | Formula         | Ours       | Paper |
|--------------------------|-----------------|------------|-------|
| FP Rate (False Positive) | FP / (FP + TN)  | **0.0483** | ~0.03 |
| FN Rate (False Negative) | FN / (FN + TP)  | **0.0721** | ~0.07 |
| FD Rate (False Discovery)| FP / (FP + TP)  | **0.2581** | —     |
| FO Rate (False Omission) | FN / (FN + TN)  | **0.0112** | —     |

---

### Multiclass — DenseNet121

| Metric            | Paper      | Ours                   |
|-------------------|------------|------------------------|
| Training Accuracy | 96%        | **97.67%**             |
| Testing Accuracy  | 89.84%     | **97.67%**             |
| Best Epoch        | 10         | 1 (EarlyStopping at 6) |

**Confusion Matrix**

|                   | Predicted: Bottle | Predicted: Box | Predicted: Polythene |
|-------------------|-------------------|----------------|----------------------|
| Actual: Bottle    | **96**            | 0              | 4                    |
| Actual: Box       | 0                 | **100**        | 0                    |
| Actual: Polythene | 1                 | 2              | **97**               |

**Class-wise Metrics**

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Plastic Bottle | 0.99      | 0.96   | 0.97     | 100     |
| Plastic Box    | 0.97      | 1.00   | 0.98     | 100     |
| Polythene      | 0.96      | 0.97   | 0.97     | 100     |
| **Overall**    |           |        | **97.67%** | 300   |

---

## Project Structure
```
plastic-waste-classifier/
│
├── data/
│   ├── binary/
│   │   ├── plastic/          ← 1,597 images
│   │   └── non_plastic/      ← 10,662 images
│   │
│   └── multiclass/
│       ├── plastic_bottle/   ← 500 images
│       ├── plastic_box/      ← 500 images
│       └── polythene/        ← 500 images
│
├── models/
│   ├── xception_best.h5      ← 94.86% val accuracy
│   └── densenet_best.h5      ← 97.67% val accuracy
│
├── train.py
├── evaluate.py
└── app.py
```

---

## Pipeline
```
train.py    →    evaluate.py    →    app.py
  │                  │                 │
Data Prep        Confusion         Streamlit UI
+ Training        Matrix +         Upload →
10 epochs        Metrics          Predict
```

---

## Stack
- Python 3.11
- TensorFlow 2.13.0 / Keras
- Streamlit
- NumPy, Matplotlib, scikit-learn, seaborn, Pillow

---

## Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train both models
python train.py

# 3. Evaluate
python evaluate.py

# 4. Launch UI
streamlit run app.py
```
