"""
NeuroBridge AI Hackathon - SuHack 2026
MRI Sınıflandırma Sistemi - Streamlit Arayüzü
4 Sınıf: Glioma, Meningioma, Pituitary, NoTumor
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import time
import logging

# Proje modüllerini import et
from config import *
from utils import (
    preprocess_image, 
    load_h5_image, 
    load_model, 
    predict_image,
    generate_gradcam,
    overlay_gradcam
)

# Logging ayarları
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Sayfa yapılandırması
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Session state başlat
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# CSS ile özel stil
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.8rem;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Model yükleme fonksiyonu
@st.cache_resource
def load_mri_model(model_path: str):
    """Model'i yükle ve cache'le"""
    try:
        if not Path(model_path).exists():
            st.warning(f"⚠️ Model dosyası bulunamadı: {model_path}")
            st.info("""
            **Model dosyası yok!**
            
            Lütfen eğitilmiş model dosyanızı `models/` klasörüne yerleştirin.
            
            Model dosya formatı: `.pt` veya `.pth`
            
            Örnek: `models/best_model.pt`
            """)
            return None
        
        model_inference = load_model(
            model_path=model_path,
            class_names=CLASS_NAMES,
            device=DEVICE_AUTO,
            image_size=IMAGE_SIZE
        )
        logger.info(f"Model başarıyla yüklendi: {model_path}")
        return model_inference
    except Exception as e:
        st.error(f"Model yüklenirken hata: {str(e)}")
        logger.error(f"Model yükleme hatası: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=NeuroBridge+AI", 
             use_container_width=True)
    st.markdown("## 🧠 Hakkında")
    st.info("""
    **SuHack 2026 - NeuroBridge AI**
    
    Bu sistem, beyin MRI görüntülerini analiz ederek 4 farklı kategoride sınıflandırma yapar:
    - 🔴 Glioma
    - 🟡 Meningioma
    - 🟢 Pituitary
    - 🔵 No Tumor
    
    **Teknolojiler:**
    - PyTorch
    - MONAI
    - Streamlit
    - Grad-CAM
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Ayarları")
    
    # Model seçimi
    model_files = list(MODELS_DIR.glob("*.pt")) + list(MODELS_DIR.glob("*.pth"))
    if model_files:
        selected_model = st.selectbox(
            "Model Seç",
            options=[f.name for f in model_files],
            help="Kullanılacak model dosyası"
        )
        model_path = MODELS_DIR / selected_model
    else:
        st.warning("⚠️ Model dosyası bulunamadı!")
        model_path = MODEL_PATH
    
    confidence_threshold = st.slider(
        "Güven Eşiği (%)", 
        min_value=0, 
        max_value=100, 
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Tahmin için minimum güven seviyesi"
    )
    
    show_gradcam = st.checkbox(
        "Grad-CAM Görselleştirme", 
        value=SHOW_GRADCAM,
        help="Modelin odaklandığı bölgeleri göster"
    )
    
    apply_clahe = st.checkbox(
        "CLAHE Kontrast İyileştirme",
        value=APPLY_CLAHE,
        help="Görüntü kontrastını artır"
    )
    
    # Model yükle butonu
    if st.button("🔄 Modeli Yükle/Yenile"):
        with st.spinner("Model yükleniyor..."):
            st.session_state.model = load_mri_model(str(model_path))
            if st.session_state.model is not None:
                st.session_state.model_loaded = True
                st.success("✅ Model başarıyla yüklendi!")
            else:
                st.session_state.model_loaded = False

# Ana başlık
st.markdown('<p class="main-header">🧠 NeuroBridge AI - MRI Sınıflandırma Sistemi</p>', 
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Beyin MRI Görüntülerinde Tümör Tespiti ve Sınıflandırma</p>', 
            unsafe_allow_html=True)

# Ana içerik alanı
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 MRI Görüntüsü Yükle")
    
    # Dosya yükleme
    uploaded_file = st.file_uploader(
        "MRI görüntüsünü seçin (JPG, PNG, JPEG veya H5)",
        type=["jpg", "jpeg", "png", "h5"],
        help="Desteklenen formatlar: JPG, PNG, JPEG, H5"
    )
    
    if uploaded_file is not None:
        # Görüntüyü göster
        try:
            if uploaded_file.name.endswith('.h5'):
                st.warning("H5 dosyası yüklendi. İşleniyor...")
                # H5 dosyası işleme kodu buraya eklenecek
            else:
                image = Image.open(uploaded_file)
                st.image(image, caption="Yüklenen MRI Görüntüsü", use_container_width=True)
                
                # Görüntü bilgileri
                st.info(f"""
                **Görüntü Bilgileri:**
                - Boyut: {image.size[0]} x {image.size[1]}
                - Format: {image.format}
                - Mod: {image.mode}
                """)
        except Exception as e:
            st.error(f"Görüntü yüklenirken hata oluştu: {str(e)}")

with col2:
    st.markdown("### 🔍 Analiz Sonuçları")
    
    if uploaded_file is not None:
        # Analiz butonu
        if st.button("🚀 Analizi Başlat", type="primary"):
            # Model kontrolü
            if st.session_state.model is None:
                st.warning("⚠️ Önce modeli yükleyin!")
                st.info("👈 Sol menüden 'Modeli Yükle/Yenile' butonuna tıklayın.")
            else:
                with st.spinner("🔬 Model çalışıyor... Lütfen bekleyin."):
                    try:
                        start_time = time.time()
                        
                        # Görüntüyü yükle ve hazırla
                        if uploaded_file.name.endswith('.h5'):
                            # H5 dosyası
                            h5_path = f"temp_{uploaded_file.name}"
                            with open(h5_path, 'wb') as f:
                                f.write(uploaded_file.getbuffer())
                            
                            image_array = load_h5_image(h5_path, H5_DATASET_NAME)
                            image_for_prediction = image_array
                            
                            # Temizlik
                            import os
                            os.remove(h5_path)
                        else:
                            # Normal görüntü dosyası
                            image_for_prediction = image
                        
                        # Tahmin yap
                        result = st.session_state.model.predict(
                            image_for_prediction,
                            return_probabilities=True
                        )
                        
                        predictions = result['probabilities']
                        predicted_class = result['predicted_class']
                        confidence = result['confidence'] * 100
                        
                        # Grad-CAM oluştur (eğer seçiliyse)
                        cam_map = None
                        if show_gradcam:
                            try:
                                cam_map, _ = generate_gradcam(
                                    model=st.session_state.model.model,
                                    image=image_for_prediction,
                                    target_class=result['predicted_index'],
                                    image_size=IMAGE_SIZE
                                )
                                st.session_state.cam_map = cam_map
                            except Exception as e:
                                logger.warning(f"Grad-CAM oluşturulamadı: {str(e)}")
                        
                        inference_time = time.time() - start_time
                        st.session_state.predictions = predictions
                        st.session_state.predicted_class = predicted_class
                        st.session_state.confidence = confidence
                        st.session_state.inference_time = inference_time
                        
                    except Exception as e:
                        st.error(f"❌ Tahmin yapılırken hata oluştu: {str(e)}")
                        logger.error(f"Prediction error: {str(e)}")
                        st.stop()
        
        # Sonuçları göster (eğer varsa)
        if 'predictions' in st.session_state:
            predictions = st.session_state.predictions
            predicted_class = st.session_state.predicted_class
            confidence = st.session_state.confidence
            
            # Sonuç kutusu
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                
            # Tahmin edilen sınıf
            if confidence >= confidence_threshold:
                st.success(f"""
                ### ✅ Tahmin: **{predicted_class}**
                **Güven Oranı:** {confidence:.2f}%
                
                **Açıklama:** {CLASS_DESCRIPTIONS[predicted_class]}
                """)
            else:
                st.warning(f"""
                ### ⚠️ Düşük Güven Seviyesi
                **Tahmin:** {predicted_class} ({confidence:.2f}%)
                
                Güven seviyesi belirlenen eşiğin altında. 
                Lütfen uzman değerlendirmesi yapınız.
                """)
            
            # İşlem süresi
            if 'inference_time' in st.session_state:
                st.info(f"⏱️ **İşlem Süresi:** {st.session_state.inference_time:.3f} saniye")
            
            st.markdown('</div>', unsafe_allow_html=True)
                
            # Olasılık dağılımı grafiği
            st.markdown("### 📊 Olasılık Dağılımı")
            
            # Plotly bar chart
            colors = [CLASS_COLORS[name] for name in predictions.keys()]
            fig = go.Figure(data=[
                go.Bar(
                    x=list(predictions.keys()),
                    y=[v * 100 for v in predictions.values()],
                    marker_color=colors,
                    text=[f"{v*100:.1f}%" for v in predictions.values()],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Sınıf Olasılıkları",
                xaxis_title="Sınıf",
                yaxis_title="Olasılık (%)",
                yaxis_range=[0, 100],
                height=PLOT_HEIGHT,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Pasta grafiği
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(predictions.keys()),
                values=list(predictions.values()),
                marker_colors=colors,
                hole=.3
            )])
            
            fig_pie.update_layout(
                title="Sınıf Dağılımı",
                height=PLOT_HEIGHT
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("👈 Lütfen sol taraftan bir MRI görüntüsü yükleyin.")

# Grad-CAM bölümü
if uploaded_file is not None and show_gradcam and 'cam_map' in st.session_state:
    st.markdown("---")
    st.markdown("### 🎯 Grad-CAM Görselleştirme")
    st.info("🔍 Model bu bölgelere odaklanarak karar verdi:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Orijinal Görüntü")
        if not uploaded_file.name.endswith('.h5'):
            st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("#### Grad-CAM Haritası")
        cam_map = st.session_state.cam_map
        st.image(cam_map, use_container_width=True, clamp=True)
    
    with col3:
        st.markdown("#### Üst Üste Bindirme")
        try:
            if not uploaded_file.name.endswith('.h5'):
                image_array = np.array(image.resize(IMAGE_SIZE))
                overlayed = overlay_gradcam(cam_map, image_array, alpha=GRADCAM_ALPHA)
                st.image(overlayed, use_container_width=True)
        except Exception as e:
            st.error(f"Overlay oluşturulamadı: {str(e)}")

# Alt bilgi
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>NeuroBridge AI - SuHack 2026</strong></p>
    <p>⚠️ Bu sistem eğitim amaçlıdır. Klinik karar verme için uzman doktor değerlendirmesi şarttır.</p>
</div>
""", unsafe_allow_html=True)
