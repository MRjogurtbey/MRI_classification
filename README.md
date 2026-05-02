# NeuroBridge AI - MRI Siniflandirma Sistemi

**SuHack 2026 - NeuroBridge AI Hackathon Projesi**

Beyin MRI goruntülerinde tümör tespiti ve siniflandirma sistemi. Derin öğrenme tabanli 4 sinifli (Glioma, Meningioma, Pituitary, NoTumor) medikal görüntü analizi.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Özellikler

- 4 Sinifli Siniflandirma: Glioma, Meningioma, Pituitary, NoTumor
- Çoklu Format Destegi: JPG, PNG, JPEG, H5
- Explainable AI: Grad-CAM ile model açiklanabilirligi
- Interaktif Görselleştirme: Plotly grafikleri ve heatmap'ler
- Hizli Çikarim: GPU destegi ile optimize edilmiş inference
- Modern UI: Streamlit tabanli kullanici dostu arayüz

## Teknolojiler

**Core Framework**
- PyTorch 2.2+
- MONAI (medikal görüntü analizi)
- Streamlit (web arayüzü)

**Computer Vision**
- torchvision, OpenCV, PIL/Pillow

**Visualization**
- Plotly, Matplotlib, Seaborn

## Kurulum

### Gereksinimler
- Python 3.8+
- CUDA 11.8+ (opsiyonel, GPU için)
- 8GB+ RAM önerilir

```bash
git clone https://github.com/MRjogurtbey/MRI_classification.git
cd MRI_classification
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

Model dosyasini (`best_model.pth`) `checkpoints/` klasörüne koyun.

## Kullanim

### Streamlit Arayüzü
```bash
streamlit run app.py
```
Tarayicida açilir: `http://localhost:8501`

### REST API (FastAPI)
```bash
python api.py
```
`http://localhost:8000` adresinde çalişir. Swagger: `http://localhost:8000/docs`

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/mri_image.jpg"
```

### Python Kütüphanesi
```python
from utils import load_model, predict_image
from config import MODEL_PATH, CLASS_NAMES, IMAGE_SIZE

model = load_model(model_path=MODEL_PATH, class_names=CLASS_NAMES,
                   device='auto', image_size=IMAGE_SIZE)
result = model.predict('path/to/mri/image.jpg')
print(result['predicted_class'], result['confidence'])
```

## Proje Yapisi

```
MRI_classification/
├── app.py              # Streamlit web UI
├── api.py              # FastAPI REST API
├── config.py           # Konfigurasyon
├── requirements.txt
├── utils/
│   ├── preprocessing.py
│   ├── inference.py
│   └── gradcam.py
├── checkpoints/
│   └── best_model.pth
└── outputs/
```

## Model Mimarisi

| Parametre | Deger |
|-----------|-------|
| Backbone | ResNet-18 (ImageNet pretrained) |
| Parametreler | 11.4M |
| Girdi | 224x224 RGB |
| Çikti | 4 sinif (Softmax) |
| Loss | Focal Loss (gamma=2.0) |
| Optimizer | Adam (lr=1e-4, weight_decay=1e-4) |
| Egitim süresi | ~33 dakika (RTX 3060) |
| En iyi epoch | 28 (Val Acc: %99.68) |

**Dataset**
- Egitim: 9,457 örnek (%80)
- Validasyon: 1,890 örnek (%20)
- Test: 3,670 örnek (ayri tutuldu)
- Toplam: 13,127 MRI görüntüsü (+%82 genisletme)

## Performans

### Genel Test Sonuclari

| Metrik | Skor |
|--------|------|
| Test Accuracy | **%96.81** |
| Weighted F1 | %96.80 |
| Macro F1 | %96.94 |

### Sinif Bazinda (Test Set — 3,670 örnek)

| Sinif | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Glioma | %99 | **%92** | %95 |
| Meningioma | %92 | %98 | %95 |
| NoTumor | %97 | %100 | %98 |
| Pituitary | %100 | %99 | %99 |

### Model Evrimi

| Metrik | v1.0 | v2.0 | v3.0 |
|--------|------|------|------|
| Test Accuracy | %95.13 | %95.37 | **%96.81** |
| Glioma Recall | %83 ⚠️ | %83 ⚠️ | **%92** ✅ |
| Overfitting Gap | %0.63 | %0.40 | **%0.26** |
| Dataset | 7,200 | 7,200 | **13,127** |

Glioma recall iyileştirmesinin anahtari dataset genislemesiydi (+%106 Glioma egitim verisi).

## Gelecek Çalismalar

- [ ] Ensemble modeller (ResNet50 + EfficientNet)
- [ ] DICOM format destegi
- [ ] Aktif öğrenme pipeline otomasyonu
- [ ] Model quantization (düsük donanim uyumu)
- [ ] Federe öğrenme (hastaneler arasi)

## Konfigurasyon

`config.py` üzerinden düzenlenebilir parametreler:

```python
IMAGE_SIZE = (224, 224)
DEVICE_AUTO = "auto"           # 'cuda', 'cpu', 'auto'
DEFAULT_CONFIDENCE_THRESHOLD = 50
GRADCAM_ALPHA = 0.5
```

## Lisans

MIT License

## Sorumluluk Reddi

> Bu sistem egitim ve arastirma amaçlidir. Klinik karar vermek için uzman doktor degerlendirmesi sarttir.

---

SuHack 2026 — NeuroBridge AI | [@MRjogurtbey](https://github.com/MRjogurtbey)
