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
| Split | Plastic | Non-Plastic | Total   |
|-------|---------|-------------|---------|
| Train | ~1,278  | ~8,530      | ~9,808  |
| Test  | ~319    | ~2,132      | ~2,451  |

Source: Garbage Classification Dataset (Kaggle) — 10 classes merged into 2.

### Multiclass
| Split | Plastic Bottle | Plastic Box | Polythene | Total |
|-------|---------------|-------------|-----------|-------|
| Train | 400           | 400         | 400       | 1,200 |
| Test  | 100           | 100         | 100       | 300   |

Source: Recyclable and Household Waste Classification Dataset (Kaggle) — 7 subfolders merged into 3 classes.

---

## Project Structure
```
plastic-waste-classifier/
│
├── data/
│   ├── binary/
│   │   ├── plastic/
│   │   └── non_plastic/
│   │
│   └── multiclass/
│       ├── plastic_bottle/
│       ├── plastic_box/
│       └── polythene/
│
├── models/
│   ├── xception_best.h5
│   └── densenet_best.h5
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

### Binary (Xception)
| Metric            | Paper | Ours |
|-------------------|-------|------|
| Training Accuracy | 96%   | TBD  |
| Testing Accuracy  | 95%   | TBD  |
| FP Rate           | ~0.03 | TBD  |
| FN Rate           | ~0.07 | TBD  |

### Multiclass (DenseNet121)
| Class          | Precision | Recall | F1   |
|----------------|-----------|--------|------|
| Plastic Bottle | 0.84      | 0.98   | 0.90 |
| Plastic Box    | 0.86      | 0.89   | 0.87 |
| Polythene      | 0.91      | 0.80   | 0.85 |
| **Overall**    |           |        |**89.84%**|

---

## Stack
- Python 3.11
- TensorFlow / Keras
- Gradio
- NumPy, Matplotlib, scikit-learn

---

## Run
```bash
# 1. Train both models
python train.py

# 2. Evaluate
python evaluate.py

# 3. Launch UI
python app.py
```
