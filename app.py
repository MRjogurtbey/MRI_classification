"""
NeuroBridge AI Hackathon - SuHack 2026
MRI Sınıflandırma Sistemi - Streamlit Arayüzü
4 Sınıf: Glioma, Meningioma, Pituitary, NoTumor
"""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

from config import (
    CHECKPOINTS_DIR, CLASS_COLORS, CLASS_DESCRIPTIONS, CLASS_NAMES,
    DEFAULT_CONFIDENCE_THRESHOLD, DEVICE_AUTO, GRADCAM_ALPHA,
    H5_DATASET_NAME, IMAGE_SIZE, LAYOUT, LOG_LEVEL,
    MODEL_PATH, MODELS_DIR, OUTPUTS_DIR, PAGE_ICON, PAGE_TITLE,
    PLOT_HEIGHT, SHOW_GRADCAM,
)
from utils import (
    generate_gradcam,
    load_h5_image,
    load_model,
    overlay_gradcam,
    preprocess_image,
    predict_image,
)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

ACTIVE_LEARNING_LOG = OUTPUTS_DIR / "active_learning_corrections.json"
ACTIVE_LEARNING_THRESHOLD = 90  # % — below this triggers doctor review

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
for key in ("model", "model_loaded"):
    if key not in st.session_state:
        st.session_state[key] = None if key == "model" else False

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header { font-size:3rem; font-weight:bold; color:#1f77b4; text-align:center; margin-bottom:2rem; }
.sub-header  { font-size:1.5rem; color:#666; text-align:center; margin-bottom:3rem; }
.stButton>button { width:100%; background-color:#1f77b4; color:white;
                   font-size:1.2rem; padding:0.8rem; border-radius:10px; }
</style>
""", unsafe_allow_html=True)


# ── Model cache ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Model yükleniyor...")
def load_mri_model_v4(model_path: str):
    try:
        if not Path(model_path).exists():
            st.warning(f"Model bulunamadı: {model_path}")
            return None
        return load_model(
            model_path=model_path,
            class_names=CLASS_NAMES,
            device=DEVICE_AUTO,
            image_size=IMAGE_SIZE,
        )
    except Exception as e:
        st.error(f"Model yüklenirken hata: {e}")
        return None


# ── Active Learning helper ────────────────────────────────────────────────────
def save_correction(image_name: str, predicted: str, correct: str, confidence: float):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {"image": image_name, "predicted": predicted,
              "correct": correct, "confidence": confidence,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    existing = []
    if ACTIVE_LEARNING_LOG.exists():
        try:
            existing = json.loads(ACTIVE_LEARNING_LOG.read_text())
        except Exception:
            pass
    existing.append(record)
    ACTIVE_LEARNING_LOG.write_text(json.dumps(existing, indent=2))
    return len(existing)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem;
         background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
         border-radius:10px; margin-bottom:1rem;'>
      <h1 style='color:white; margin:0; font-size:1.8rem;'>🧠 NeuroBridge AI</h1>
      <p style='color:#f0f0f0; margin:0; font-size:0.9rem;'>MRI Classification System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## ⚙️ Model Ayarları")
    model_files = (
        list(CHECKPOINTS_DIR.glob("*.pt")) + list(CHECKPOINTS_DIR.glob("*.pth")) +
        list(MODELS_DIR.glob("*.pt")) + list(MODELS_DIR.glob("*.pth"))
    )
    if model_files:
        model_file_map = {f.name: f for f in model_files}
        selected_model = st.selectbox("Model Seç", list(model_file_map.keys()))
        model_path = model_file_map[selected_model]
    else:
        st.warning("⚠️ Model dosyası bulunamadı!")
        model_path = MODEL_PATH

    confidence_threshold = st.slider("Güven Eşiği (%)", 0, 100, DEFAULT_CONFIDENCE_THRESHOLD)
    show_gradcam = st.checkbox("Grad-CAM Görselleştirme", value=SHOW_GRADCAM)
    use_tta = st.checkbox("Test-Time Augmentation (TTA)", value=False,
                          help="5 augmentation ile tahmin yap, daha doğru ama yavaş")

    st.markdown("---")
    st.markdown("## 🔬 Aktif Öğrenme")
    st.info(f"Güven < %{ACTIVE_LEARNING_THRESHOLD} ise doktor düzeltmesi istenir ve vaka loglanır.")
    if ACTIVE_LEARNING_LOG.exists():
        try:
            n = len(json.loads(ACTIVE_LEARNING_LOG.read_text()))
            st.metric("Biriken düzeltme", n)
        except Exception:
            pass

    st.markdown("---")
    if st.button("🔄 Modeli Yükle/Yenile"):
        with st.spinner("Model yükleniyor..."):
            st.session_state.model = load_mri_model_v4(str(model_path))
            if st.session_state.model is not None:
                st.session_state.model_loaded = True
                st.success("✅ Model başarıyla yüklendi!")
            else:
                st.session_state.model_loaded = False

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🧠 NeuroBridge AI - MRI Sınıflandırma</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Beyin MRI Görüntülerinde Tümör Tespiti</p>', unsafe_allow_html=True)

# ── Upload + Results ──────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 MRI Görüntüsü Yükle")
    uploaded_file = st.file_uploader(
        "JPG, PNG, JPEG veya H5 seçin",
        type=["jpg", "jpeg", "png", "h5"],
    )

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".h5"):
            st.warning("H5 dosyası algılandı.")
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="Yüklenen MRI", use_container_width=True)
            st.info(f"Boyut: {image.size[0]}×{image.size[1]} | Mod: {image.mode}")

with col2:
    st.markdown("### 🔍 Analiz Sonuçları")

    if uploaded_file is not None:
        # Reset state on new file
        if st.session_state.get("last_file") != uploaded_file.name:
            st.session_state.last_file = uploaded_file.name
            for k in ("predictions", "predicted_class", "confidence", "inference_time", "cam_map"):
                st.session_state.pop(k, None)

        if st.button("🚀 Analizi Başlat", type="primary"):
            if not st.session_state.model_loaded:
                st.warning("⚠️ Önce sol menüden modeli yükleyin!")
            else:
                with st.spinner("🔬 Analiz ediliyor..."):
                    try:
                        t0 = time.time()
                        img_input = (
                            load_h5_image(uploaded_file, H5_DATASET_NAME)
                            if uploaded_file.name.endswith(".h5")
                            else image
                        )
                        result = (
                            st.session_state.model.predict_with_tta(img_input, num_augmentations=5, return_probabilities=True)
                            if use_tta
                            else st.session_state.model.predict(img_input, return_probabilities=True)
                        )
                        if show_gradcam:
                            try:
                                cam_map, _ = generate_gradcam(
                                    model=st.session_state.model.model,
                                    image=img_input,
                                    target_class=result["predicted_index"],
                                    image_size=IMAGE_SIZE,
                                )
                                st.session_state.cam_map = cam_map
                            except Exception as e:
                                logger.warning(f"Grad-CAM hatası: {e}")

                        st.session_state.predictions = result["probabilities"]
                        st.session_state.predicted_class = result["predicted_class"]
                        st.session_state.confidence = result["confidence"] * 100
                        st.session_state.inference_time = time.time() - t0
                    except Exception as e:
                        st.error(f"Hata: {e}")
                        st.stop()

        # ── Show results ──────────────────────────────────────────────────────
        if "predictions" in st.session_state:
            preds = st.session_state.predictions
            pred_cls = st.session_state.predicted_class
            conf = st.session_state.confidence

            if conf >= confidence_threshold:
                st.success(f"### ✅ {pred_cls}  —  {conf:.1f}%\n\n{CLASS_DESCRIPTIONS[pred_cls]}")
            else:
                st.warning(f"### ⚠️ {pred_cls}  —  {conf:.1f}%\n\nGüven düşük, uzman değerlendirmesi önerilir.")

            if "inference_time" in st.session_state:
                st.info(f"⏱️ {st.session_state.inference_time:.3f} sn")

            colors = [CLASS_COLORS[n] for n in preds]
            fig = go.Figure(go.Bar(
                x=list(preds.keys()),
                y=[v * 100 for v in preds.values()],
                marker_color=colors,
                text=[f"{v*100:.1f}%" for v in preds.values()],
                textposition="auto",
            ))
            fig.update_layout(title="Sınıf Olasılıkları", yaxis_title="%",
                              yaxis_range=[0, 100], height=PLOT_HEIGHT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Sol taraftan MRI görüntüsü yükleyin.")

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
if uploaded_file is not None and show_gradcam and "cam_map" in st.session_state:
    st.markdown("---")
    st.markdown("### 🎯 Grad-CAM — Modelin Odak Noktası")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Orijinal")
        if not uploaded_file.name.endswith(".h5"):
            st.image(image, use_container_width=True)
    with c2:
        st.markdown("#### Isı Haritası")
        st.image(st.session_state.cam_map, use_container_width=True, clamp=True)
    with c3:
        st.markdown("#### Üst Üste")
        try:
            img_arr = np.array(image.resize(IMAGE_SIZE))
            st.image(overlay_gradcam(st.session_state.cam_map, img_arr, alpha=GRADCAM_ALPHA),
                     use_container_width=True)
        except Exception as e:
            st.error(f"Overlay hatası: {e}")

# ── Active Learning ───────────────────────────────────────────────────────────
if "predicted_class" in st.session_state and st.session_state.confidence < ACTIVE_LEARNING_THRESHOLD:
    st.markdown("---")
    st.markdown("### 🔬 Aktif Öğrenme — Düşük Güven Vakası")
    st.warning(
        f"Model bu vakada yalnızca **{st.session_state.confidence:.1f}%** güven bildirdi. "
        f"Doktor düzeltmesi bu vakayı yeniden eğitim setine ekler."
    )
    with st.form("correction_form"):
        correct_class = st.selectbox(
            "Doğru teşhis nedir?",
            CLASS_NAMES,
            index=CLASS_NAMES.index(st.session_state.predicted_class)
            if st.session_state.predicted_class in CLASS_NAMES else 0,
        )
        notes = st.text_area("Notlar (isteğe bağlı)")
        submitted = st.form_submit_button("✅ Düzeltmeyi Kaydet")
        if submitted:
            n = save_correction(
                image_name=uploaded_file.name if uploaded_file else "unknown",
                predicted=st.session_state.predicted_class,
                correct=correct_class,
                confidence=st.session_state.confidence,
            )
            st.success(f"Kaydedildi! Toplam {n} düzeltme birikti — yeniden eğitimde kullanılacak.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#666;'>
  <p><strong>NeuroBridge AI — SuHack 2026</strong></p>
  <p>⚠️ Klinik karar vermek için uzman doktor değerlendirmesi şarttır.</p>
</div>
""", unsafe_allow_html=True)
