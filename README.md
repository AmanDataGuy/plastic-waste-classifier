# Plastic Waste Classifier
Vision-based binary classification of plastic vs non-plastic waste using deep learning.

Based on: *A Vision-Based Automated System for Plastic Waste Segregation Using Deep Learning* (IEEE AIEI 2026)

---

## Architecture

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

---

## Dataset

| Split    | Plastic | Non-Plastic | Total  |
|----------|---------|-------------|--------|
| Train    | ~1,278  | ~9,401      | ~10,679|
| Test     | ~319    | ~2,350      | ~2,669 |

Source: Garbage Classification Dataset (Kaggle) — 10 classes merged into 2.

---

## Project Structure

```
plastic-waste-classifier/
│
├── data/
│   ├── plastic/
│   └── non_plastic/
│
├── models/
│   └── xception_best.h5
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
Data Prep        Confusion         Gradio UI
+ Training        Matrix +         Upload →
10 epochs        Metrics          Predict
```

---

## Results

| Metric           | Paper | Ours |
|------------------|-------|------|
| Training Accuracy| 96%   | TBD  |
| Testing Accuracy | 95%   | TBD  |
| FP Rate          | ~0.03 | TBD  |
| FN Rate          | ~0.07 | TBD  |

---

## Stack

- Python 3.10+
- TensorFlow / Keras
- Gradio
- NumPy, Matplotlib, scikit-learn

---

## Run

```bash
# 1. Train
python train.py

# 2. Evaluate
python evaluate.py

# 3. Launch UI
python app.py
```
