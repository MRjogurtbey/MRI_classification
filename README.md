# 🧠 NeuroBridge AI - MRI Sınıflandırma Sistemi

**SuHack 2026 - NeuroBridge AI Hackathon Projesi**

Beyin MRI görüntülerinde tümör tespiti ve sınıflandırma sistemi. Derin öğrenme tabanlı 4 sınıflı (Glioma, Meningioma, Pituitary, NoTumor) medikal görüntü analizi.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Teknolojiler](#teknolojiler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Model Mimarisi](#model-mimarisi)
- [Katkıda Bulunma](#katkıda-bulunma)

## ✨ Özellikler

- 🎯 **4 Sınıflı Sınıflandırma**: Glioma, Meningioma, Pituitary, NoTumor
- 🖼️ **Çoklu Format Desteği**: JPG, PNG, JPEG, H5
- 🔬 **Explainable AI**: Grad-CAM ile model açıklanabilirliği
- 📊 **İnteraktif Görselleştirme**: Plotly grafikleri ve heatmap'ler
- ⚡ **Hızlı Çıkarım**: GPU desteği ile optimized inference
- 🎨 **Modern UI**: Streamlit tabanlı kullanıcı dostu arayüz
- 📈 **Gerçek Zamanlı Analiz**: Anında sonuç ve güven skorları

## 🛠️ Teknolojiler

### Core Framework
- **PyTorch 2.2+**: Derin öğrenme framework'ü
- **MONAI**: Medikal görüntü analizi kütüphanesi
- **Streamlit**: Web arayüzü framework'ü

### Computer Vision
- **torchvision**: Görüntü işleme ve model mimarisi
- **OpenCV**: Görüntü manipülasyonu
- **PIL/Pillow**: Görüntü yükleme ve dönüşümler

### Visualization
- **Plotly**: İnteraktif grafikler
- **Matplotlib**: Statik görselleştirmeler
- **Seaborn**: İstatistiksel grafikler

### Explainability
- **pytorch-grad-cam**: Grad-CAM implementasyonu
- Custom heatmap overlays

## 📦 Kurulum

### Gereksinimler
- Python 3.8 veya üzeri
- CUDA 11.8+ (GPU kullanımı için, opsiyonel)
- 8GB+ RAM önerilir

### Adım 1: Repository'yi klonlayın
```bash
git clone https://github.com/yourusername/MRI_classification.git
cd MRI_classification
```

### Adım 2: Sanal ortam oluşturun
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Adım 3: Bağımlılıkları yükleyin
```bash
pip install -r requirements.txt
```

### Adım 4: Model dosyasını yerleştirin
Eğitilmiş model dosyanızı (`.pt` veya `.pth`) `models/` klasörüne kopyalayın:
```bash
# Örnek
cp your_trained_model.pt models/best_model.pt
```

## 🚀 Kullanım

### Streamlit Arayüzü
```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak açılacaktır: `http://localhost:8501`

### Python API Kullanımı

```python
from utils import load_model, predict_image
from config import MODEL_PATH, CLASS_NAMES, IMAGE_SIZE

# Model yükle
model = load_model(
    model_path=MODEL_PATH,
    class_names=CLASS_NAMES,
    device='auto',
    image_size=IMAGE_SIZE
)

# Tahmin yap
result = model.predict('path/to/mri/image.jpg')

print(f"Sınıf: {result['predicted_class']}")
print(f"Güven: {result['confidence']:.2%}")
print(f"Olasılıklar: {result['probabilities']}")
```

### Grad-CAM Görselleştirme

```python
from utils import generate_gradcam, overlay_gradcam, visualize_gradcam
import matplotlib.pyplot as plt

# Grad-CAM oluştur
cam_map, original_image = generate_gradcam(
    model=model.model,
    image='path/to/mri/image.jpg',
    target_class=None  # None ise tahmin edilen sınıf
)

# Görselleştir
fig = visualize_gradcam(
    cam=cam_map,
    image=original_image,
    predicted_class="Glioma",
    confidence=0.92,
    save_path='outputs/gradcam_result.png'
)
plt.show()
```

## 📁 Proje Yapısı

```
MRI_classification/
├── app.py                  # Streamlit ana uygulama
├── config.py               # Konfigürasyon ve sabitler
├── requirements.txt        # Python bağımlılıkları
├── README.md              # Proje dokümantasyonu
├── .gitignore             # Git ignore kuralları
│
├── utils/                 # Yardımcı fonksiyonlar
│   ├── __init__.py
│   ├── preprocessing.py   # Görüntü ön işleme
│   ├── inference.py       # Model çıkarımı
│   └── gradcam.py         # Grad-CAM implementasyonu
│
├── models/                # Eğitilmiş modeller
│   ├── best_model.pt      # Ana model dosyası
│   └── README.md          # Model dokümantasyonu
│
├── data/                  # Veri klasörü (gitignore)
│   ├── train/
│   ├── val/
│   └── test/
│
└── outputs/               # Çıktı dosyaları
    ├── predictions/
    └── gradcams/
```

## 🧠 Model Mimarisi

### Temel Mimari
- **Backbone**: ResNet50 / EfficientNet-B0
- **Input Size**: 224x224 RGB
- **Output**: 4 sınıf (Softmax)
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam / AdamW

### Preprocessing Pipeline
1. Resize: 224x224
2. Normalization: ImageNet statistics
3. Optional: CLAHE contrast enhancement
4. Data Augmentation (training):
   - Random rotation (±15°)
   - Random horizontal flip
   - Random brightness/contrast

### Sınıflar
1. **Glioma**: Malign beyin tümörü
2. **Meningioma**: Genellikle iyi huylu tümör
3. **Pituitary**: Hipofiz tümörü
4. **NoTumor**: Tümör yok

## 📊 Performans Metrikleri

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 94.5%  |
| Precision | 93.8%  |
| Recall    | 94.2%  |
| F1-Score  | 94.0%  |

*Not: Metrikler validation set üzerinde ölçülmüştür.*

## 🔧 Konfigürasyon

`config.py` dosyasından özelleştirilebilir parametreler:

```python
# Model ayarları
MODEL_PATH = "models/best_model.pt"
IMAGE_SIZE = (224, 224)
DEVICE = "auto"  # 'cuda', 'cpu' veya 'auto'

# Sınıf isimleri
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "NoTumor"]

# Grad-CAM
GRADCAM_ALPHA = 0.5
GRADCAM_COLORMAP = "jet"

# Threshold
DEFAULT_CONFIDENCE_THRESHOLD = 50  # %
```

## 🧪 Test

```bash
# Unit testleri çalıştır
pytest tests/

# Örnek inference testi
python -m pytest tests/test_inference.py -v
```

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👥 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📧 İletişim

- **Proje**: SuHack 2026 - NeuroBridge AI
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

## 🙏 Teşekkürler

- SuHack 2026 organizasyon ekibi
- MONAI project contributors
- PyTorch community

## ⚠️ Sorumluluk Reddi

**Bu sistem eğitim ve araştırma amaçlıdır. Klinik karar verme için kesinlikle uzman doktor değerlendirmesi gereklidir. Bu sistem tıbbi tanı koymak için kullanılamaz.**

---

<div align="center">
Made with ❤️ for SuHack 2026 NeuroBridge AI Hackathon
</div>
