# 📖 Kullanım Kılavuzu

## 🚀 Hızlı Başlangıç

### 1. Kurulum Testi
```bash
python test_setup.py
```

Bu komut:
- Python versiyonunu kontrol eder
- Gerekli kütüphanelerin yüklü olup olmadığını kontrol eder
- Proje yapısını doğrular
- Model dosyalarını kontrol eder
- GPU kullanılabilirliğini test eder

### 2. Uygulamayı Başlatma

#### Seçenek A: Quick Start (Önerilen)
```bash
python run.py
```

#### Seçenek B: Streamlit Komutu
```bash
streamlit run app.py
```

Uygulama `http://localhost:8501` adresinde açılacaktır.

## 🎯 Streamlit Arayüzü Kullanımı

### Adım 1: Model Yükleme
1. Sol sidebar'dan model dosyasını seçin
2. "🔄 Modeli Yükle/Yenile" butonuna tıklayın
3. Model yüklenene kadar bekleyin

### Adım 2: Görüntü Yükleme
1. Ana ekranda "📤 MRI Görüntüsü Yükle" bölümünden dosya seçin
2. Desteklenen formatlar: JPG, PNG, JPEG, H5
3. Görüntü önizlemesi otomatik gösterilir

### Adım 3: Analiz
1. "🚀 Analizi Başlat" butonuna tıklayın
2. Model çalışırken bekleme animasyonu görünür
3. Sonuçlar otomatik olarak gösterilir:
   - Tahmin edilen sınıf
   - Güven skoru (%)
   - Olasılık dağılımı (Bar ve Pie chart)
   - Grad-CAM görselleştirmesi

### Ayarlar (Sidebar)

#### Güven Eşiği
- Minimum güven seviyesini ayarlayın (0-100%)
- Eşiğin altındaki tahminler uyarı ile gösterilir

#### Grad-CAM Görselleştirme
- Açık: Modelin odaklandığı bölgeleri gösterir
- Kapalı: Sadece tahmin sonuçları gösterilir

#### CLAHE Kontrast İyileştirme
- MRI görüntülerinde kontrast artırma
- Düşük kontrastlı görüntüler için önerilir

## 💻 Python API Kullanımı

### Basit Tahmin
```python
from utils import load_model, predict_image
from config import MODEL_PATH, CLASS_NAMES, IMAGE_SIZE

# Model yükle
model = load_model(
    model_path=str(MODEL_PATH),
    class_names=CLASS_NAMES,
    device='auto',
    image_size=IMAGE_SIZE
)

# Tek görüntü tahmini
result = model.predict('path/to/image.jpg')

print(f"Sınıf: {result['predicted_class']}")
print(f"Güven: {result['confidence']:.2%}")

# Tüm olasılıklar
for class_name, prob in result['probabilities'].items():
    print(f"{class_name}: {prob:.2%}")
```

### Batch Tahmin
```python
# Birden fazla görüntü
images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = model.predict_batch(images, batch_size=8)

for i, result in enumerate(results):
    print(f"Görüntü {i+1}: {result['predicted_class']} ({result['confidence']:.2%})")
```

### Grad-CAM Oluşturma
```python
from utils import generate_gradcam, overlay_gradcam, visualize_gradcam
import matplotlib.pyplot as plt

# Grad-CAM oluştur
cam_map, original_image = generate_gradcam(
    model=model.model,
    image='path/to/image.jpg',
    target_class=None,  # None = tahmin edilen sınıf
    image_size=(224, 224)
)

# Görselleştir ve kaydet
fig = visualize_gradcam(
    cam=cam_map,
    image=original_image,
    predicted_class="Glioma",
    confidence=0.92,
    save_path='outputs/gradcam_result.png'
)
plt.show()

# Veya sadece overlay
overlayed = overlay_gradcam(cam_map, original_image, alpha=0.5)
plt.imshow(overlayed)
plt.axis('off')
plt.savefig('outputs/overlay.png', dpi=300, bbox_inches='tight')
```

### H5 Dosyaları
```python
from utils import load_h5_image, preprocess_image

# H5 dosyasını yükle
image_array = load_h5_image('path/to/file.h5', dataset_name='image')

# Ön işleme uygula
tensor = preprocess_image(image_array, image_size=(224, 224), apply_clahe=True)

# Tahmin
result = model.predict(tensor)
```

### Özelleştirilmiş Preprocessing
```python
from utils.preprocessing import MRIPreprocessor
from PIL import Image

# Preprocessor oluştur
preprocessor = MRIPreprocessor(image_size=(224, 224))

# Görüntü yükle
image = Image.open('mri.jpg')

# CLAHE uygula
import numpy as np
image_array = np.array(image)
enhanced = preprocessor.apply_clahe(image_array)

# Model için hazırla
tensor = preprocessor.preprocess_numpy_array(enhanced)

# Tahmin
result = model.predict(tensor)
```

## 🔧 Konfigürasyon

`config.py` dosyasında ayarları değiştirin:

```python
# Model
MODEL_PATH = MODELS_DIR / "your_model.pt"
IMAGE_SIZE = (224, 224)  # veya (240, 240)

# Sınıflar (modelinize göre)
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "NoTumor"]

# Device
DEVICE_AUTO = "auto"  # 'cuda', 'cpu' veya 'auto'

# Grad-CAM
GRADCAM_ALPHA = 0.5  # Overlay şeffaflığı (0-1)
GRADCAM_COLORMAP = "jet"  # 'jet', 'hot', 'viridis', vb.
```

## 📊 Model Gereksinimleri

Modeliniz şu formatta olmalıdır:

### Minimum Gereksinimler
- **Framework**: PyTorch
- **Giriş**: (batch_size, 3, 224, 224) veya (batch_size, 3, 240, 240)
- **Çıkış**: (batch_size, 4) - 4 sınıf için logits
- **Format**: `.pt` veya `.pth`

### Önerilen Format
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'class_names': ['Glioma', 'Meningioma', 'Pituitary', 'NoTumor'],
    'image_size': (224, 224),
    'accuracy': 0.945,
    'epoch': 50
}
torch.save(checkpoint, 'models/best_model.pt')
```

## 🐛 Sorun Giderme

### Model Yüklenmiyor
```bash
# Model dosyasını kontrol edin
ls models/

# Model yolunu doğrulayın
python -c "from config import MODEL_PATH; print(MODEL_PATH)"
```

### CUDA Hatası
```python
# config.py'de device'ı 'cpu' yapın
DEVICE_AUTO = "cpu"

# Veya sadece CPU için PyTorch yükleyin
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Import Hatası
```bash
# Kütüphaneleri tekrar yükleyin
pip install -r requirements.txt --upgrade

# Veya eksik paketi manuel yükleyin
pip install streamlit torch torchvision monai
```

### Görüntü Formatı Hatası
- Görüntünüzün RGB veya Grayscale olduğundan emin olun
- H5 dosyaları için dataset ismini kontrol edin
- Desteklenen formatlar: JPG, PNG, JPEG, H5

## 💡 İpuçları

### Performans
- GPU kullanımı için CUDA'lı PyTorch yükleyin
- Batch prediction daha hızlıdır
- Model cache'lenir (ilk yükleme yavaş, sonrakiler hızlı)

### Doğruluk
- Görüntü kalitesi önemlidir
- CLAHE düşük kontrastlı görüntüler için faydalıdır
- Güven eşiğini ihtiyacınıza göre ayarlayın

### Görselleştirme
- Grad-CAM her görüntü için ~2-3 saniye ekler
- Büyük batch'lerde Grad-CAM'i kapatabilirsiniz
- Farklı colormap'ler deneyin: 'jet', 'hot', 'coolwarm'

## 📚 Daha Fazla Bilgi

- [README.md](README.md) - Proje genel bakış
- [models/README.md](models/README.md) - Model formatı
- [SuHack 2026 Website](https://suhack.io) - Hackathon bilgileri

## 🆘 Destek

Sorun yaşıyorsanız:
1. `test_setup.py` çalıştırın
2. Log'ları kontrol edin
3. Issue açın veya email gönderin

---

**Not**: Tıbbi kararlar için mutlaka uzman doktor görüşü alınız. Bu sistem sadece eğitim amaçlıdır.
