import os

import numpy as np
import streamlit as st
import torch
from PIL import Image

from predict import load_model, predict_image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 NeuroScan AI — MRI Brain Tumor Classifier")
st.caption("NeuroBridge AI | SuHack 2026 — Track 2: MRI Analysis")

# ── Sidebar settings ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Ayarlar")
    checkpoint_path = st.text_input(
        "Model Checkpoint",
        value="checkpoints/best_model.pth",
    )
    ollama_model = st.selectbox(
        "Ollama Modeli",
        options=["llama3", "mistral", "llama3.2", "gemma2"],
        index=0,
    )
    use_gradcam = st.toggle("Grad-CAM Göster", value=True)
    use_report = st.toggle("Ollama Raporu Üret", value=True)

    st.divider()
    st.caption("Ollama çalışmıyorsa: `ollama serve`")

# ── Model cache ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Model yükleniyor...")
def get_model_cached(ckpt: str):
    if not os.path.exists(ckpt):
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_model(ckpt, device), device


# ── Main UI ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "MRI görüntüsü yükleyin (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    pil_image = Image.open(uploaded).convert("RGB")

    loaded = get_model_cached(checkpoint_path)
    if loaded is None:
        st.error(f"Checkpoint bulunamadı: `{checkpoint_path}`")
        st.stop()

    model, device = loaded

    with st.spinner("Analiz yapılıyor..."):
        result = predict_image(
            pil_image, model, device,
            with_gradcam=use_gradcam,
            with_report=use_report,
            ollama_model=ollama_model,
        )

    # ── Results layout ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Yüklenen Görüntü")
        st.image(pil_image.resize((224, 224)), use_container_width=True)

    with col2:
        st.subheader("Grad-CAM Isı Haritası")
        if result["gradcam_overlay"] is not None:
            st.image(result["gradcam_overlay"], use_container_width=True)
        else:
            st.info("Grad-CAM devre dışı.")

    with col3:
        st.subheader("Tahmin")
        cls = result["predicted_class"].upper()
        conf = result["confidence"]

        color = {
            "GLIOMA": "🔴",
            "MENINGIOMA": "🟠",
            "NOTUMOR": "🟢",
            "PITUITARY": "🟡",
        }.get(cls, "⚪")

        st.markdown(f"## {color} {cls}")
        st.metric("Güven", f"{conf:.1%}")

        st.divider()
        st.caption("Sınıf olasılıkları")
        for name, prob in sorted(
            result["class_probabilities"].items(), key=lambda x: -x[1]
        ):
            st.progress(prob, text=f"{name}: {prob:.1%}")

    # ── Ollama Report ─────────────────────────────────────────────────────────
    if result["report"]:
        st.divider()
        st.subheader("📋 Yapay Zeka Ön Raporu")
        st.info(result["report"])
        st.caption("Bu rapor bir ön değerlendirmedir. Kesin tanı için uzmana başvurun.")
