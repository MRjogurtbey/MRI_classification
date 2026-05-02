"""
Konfigürasyon Dosyası
Proje ayarları ve sabitler
"""

from pathlib import Path

# Proje yolları
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data"
TRAINING_DIR = PROJECT_ROOT / "Training"
TESTING_DIR = PROJECT_ROOT / "Testing"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Model ayarları
MODEL_PATH = MODELS_DIR / "best_model.pth"  # Varsayılan model
IMAGE_SIZE = (224, 224)  # Görüntü boyutu (H, W)
BATCH_SIZE = 32

# Sınıf isimleri ve renkleri
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "NoTumor"]
CLASS_COLORS = {
    "Glioma": "#ff6b6b",
    "Meningioma": "#ffd93d",
    "Pituitary": "#6bcf7f",
    "NoTumor": "#4d96ff"
}

CLASS_DESCRIPTIONS = {
    "Glioma": "Beyin ve omurilik dokusundan kaynaklanan tümör. "
              "En yaygın malign beyin tümörü tipidir.",
    "Meningioma": "Beyin zarlarından (meninges) kaynaklanan genellikle iyi huylu tümör. "
                  "Yavaş büyüme gösterir.",
    "Pituitary": "Hipofiz bezinde gelişen tümör. "
                 "Hormon üretimini etkileyebilir.",
    "NoTumor": "MRI taramasında tümör tespit edilmedi. "
               "Normal beyin görüntüsü."
}

# Cihaz ayarları
DEVICE_AUTO = "auto"  # 'auto', 'cuda', 'cpu'

# Grad-CAM ayarları
GRADCAM_ALPHA = 0.5  # Overlay şeffaflığı
GRADCAM_COLORMAP = "jet"  # matplotlib colormap

# Streamlit ayarları
PAGE_TITLE = "NeuroBridge AI - MRI Sınıflandırma"
PAGE_ICON = "🧠"
LAYOUT = "wide"

# Güven eşiği
DEFAULT_CONFIDENCE_THRESHOLD = 50  # %

# H5 dataset ayarları
H5_DATASET_NAME = "image"  # H5 dosyasındaki dataset ismi

# Preprocessing ayarları
APPLY_CLAHE = False  # CLAHE kontrast iyileştirme
NORMALIZE = True  # ImageNet normalizasyonu

# Görselleştirme
SHOW_GRADCAM = True
PLOT_HEIGHT = 400
DPI = 300

# Logging
LOG_LEVEL = "INFO"
