# =============================================================================
# app.py — Plastic Waste Classifier
# Streamlit interface for real-time inference.
# Chains Xception (binary) → DenseNet121 (multiclass if plastic detected).
# Run: streamlit run app.py
# =============================================================================

import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# =============================================================================
# 1. CONFIG
# =============================================================================

MODELS_DIR          = r"D:\projects\plastic-waste-classifier\models"
XCEPTION_MODEL_PATH = os.path.join(MODELS_DIR, "xception_best.h5")
DENSENET_MODEL_PATH = os.path.join(MODELS_DIR, "densenet_best.h5")

XCEPTION_IMG_SIZE  = (299, 299)
DENSENET_IMG_SIZE  = (224, 224)
BINARY_THRESHOLD   = 0.5

MULTICLASS_LABELS = ["Plastic Bottle", "Plastic Box", "Polythene"]

st.set_page_config(
    page_title="Plastic Waste Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# 2. STYLING
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0e0e0e;
        color: #e8e8e8;
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2rem;
        max-width: 740px;
    }

    h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 0.85rem;
        color: #666;
        font-family: 'IBM Plex Mono', monospace;
        margin-bottom: 2rem;
    }

    .divider {
        border: none;
        border-top: 1px solid #222;
        margin: 1.5rem 0;
    }

    /* Pipeline steps */
    .pipeline {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1.2rem 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #555;
    }

    .step {
        padding: 0.35rem 0.75rem;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        background: #161616;
        color: #888;
        white-space: nowrap;
    }

    .step.active {
        border-color: #4ade80;
        color: #4ade80;
        background: #0d1f14;
    }

    .step.active-warn {
        border-color: #facc15;
        color: #facc15;
        background: #1a1700;
    }

    .arrow {
        color: #333;
        font-size: 0.9rem;
    }

    /* Result cards */
    .result-card {
        border: 1px solid #2a2a2a;
        border-radius: 6px;
        padding: 1.2rem 1.4rem;
        margin: 0.6rem 0;
        background: #141414;
    }

    .result-card.plastic {
        border-left: 3px solid #4ade80;
    }

    .result-card.non-plastic {
        border-left: 3px solid #f87171;
    }

    .result-card.type {
        border-left: 3px solid #60a5fa;
    }

    .result-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }

    .result-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff;
    }

    .result-conf {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #555;
        margin-top: 0.2rem;
    }

    /* Confidence bar */
    .conf-bar-bg {
        background: #1e1e1e;
        border-radius: 2px;
        height: 4px;
        margin-top: 0.6rem;
        overflow: hidden;
    }

    .conf-bar-fill {
        height: 100%;
        border-radius: 2px;
        background: #4ade80;
    }

    .conf-bar-fill.warn {
        background: #f87171;
    }

    .conf-bar-fill.blue {
        background: #60a5fa;
    }

    /* Upload zone */
    [data-testid="stFileUploader"] {
        border: 1px dashed #2a2a2a !important;
        border-radius: 6px !important;
        background: #111 !important;
        padding: 1rem !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #444 !important;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 3. MODEL LOADING (cached — loads once per session)
# =============================================================================

@st.cache_resource
def load_models():
    xception  = load_model(XCEPTION_MODEL_PATH)
    densenet  = load_model(DENSENET_MODEL_PATH)
    return xception, densenet


# =============================================================================
# 4. INFERENCE
# =============================================================================

def preprocess(image: Image.Image, target_size: tuple) -> np.ndarray:
    """Resize, normalize, add batch dimension."""
    img = image.convert("RGB").resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_binary(model, image: Image.Image):
    """Returns (label, confidence). label: 'Plastic' or 'Non-Plastic'."""
    x    = preprocess(image, XCEPTION_IMG_SIZE)
    prob = model.predict(x, verbose=0)[0][0]

    # class_indices: non_plastic=0, plastic=1 (alphabetical)
    if prob >= BINARY_THRESHOLD:
        return "Plastic", float(prob)
    else:
        return "Non-Plastic", float(1 - prob)


def predict_multiclass(model, image: Image.Image):
    """Returns (label, confidence). label: one of MULTICLASS_LABELS."""
    x     = preprocess(image, DENSENET_IMG_SIZE)
    probs = model.predict(x, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return MULTICLASS_LABELS[idx], float(probs[idx])


# =============================================================================
# 5. UI
# =============================================================================

st.markdown("<h1>Plastic Waste Classifier</h1>", unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">IEEE AIEI 2026 &nbsp;·&nbsp; Xception + DenseNet121 &nbsp;·&nbsp; Transfer Learning</div>',
    unsafe_allow_html=True
)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Load models
with st.spinner("Loading models..."):
    try:
        xception_model, densenet_model = load_models()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

# Upload
uploaded = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded:
    image = Image.open(uploaded)

    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.image(image, caption=uploaded.name, use_column_width=True)

    with col2:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Step 1: Binary classification ────────────────────────────────────
        binary_label, binary_conf = predict_binary(xception_model, image)
        is_plastic = binary_label == "Plastic"

        # Pipeline indicator
        if is_plastic:
            st.markdown("""
            <div class="pipeline">
                <span class="step active">Xception</span>
                <span class="arrow">→</span>
                <span class="step active">DenseNet121</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="pipeline">
                <span class="step active-warn">Xception</span>
                <span class="arrow">→</span>
                <span class="step">DenseNet121</span>
            </div>
            """, unsafe_allow_html=True)

        # Binary result card
        card_class  = "plastic" if is_plastic else "non-plastic"
        bar_class   = "" if is_plastic else "warn"
        bar_width   = int(binary_conf * 100)

        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-label">Binary Classification</div>
            <div class="result-value">{binary_label}</div>
            <div class="result-conf">Confidence: {binary_conf * 100:.1f}%</div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill {bar_class}" style="width:{bar_width}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Step 2: Multiclass (only if plastic detected) ────────────────────
        if is_plastic:
            type_label, type_conf = predict_multiclass(densenet_model, image)
            type_bar_width = int(type_conf * 100)

            st.markdown(f"""
            <div class="result-card type">
                <div class="result-label">Plastic Type</div>
                <div class="result-value">{type_label}</div>
                <div class="result-conf">Confidence: {type_conf * 100:.1f}%</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill blue" style="width:{type_bar_width}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="result-card" style="border-left:3px solid #333;">
                <div class="result-label">Plastic Type</div>
                <div class="result-value" style="color:#333;">—</div>
                <div class="result-conf">Not applicable</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; color:#333; font-family:'IBM Plex Mono',monospace;
                font-size:0.8rem; padding: 3rem 0;">
        no image uploaded
    </div>
    """, unsafe_allow_html=True)