# Modeller Klasörü

Bu klasör eğitilmiş PyTorch modellerini içerir.

## Model Dosyaları

Eğitilmiş model dosyalarınızı (`.pt` veya `.pth`) bu klasöre koyun:

```
models/
├── best_model.pt          # En iyi performanslı model
├── resnet50_mri.pt        # ResNet50 tabanlı model
├── efficientnet_mri.pt    # EfficientNet tabanlı model
└── README.md              # Bu dosya
```

## Model Formatı

Model dosyaları şu formatta olmalıdır:

### Seçenek 1: Sadece state_dict
```python
torch.save(model.state_dict(), 'model.pt')
```

### Seçenek 2: Tam checkpoint (Önerilen)
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
    'accuracy': accuracy,
    'class_names': ['Glioma', 'Meningioma', 'Pituitary', 'NoTumor']
}
torch.save(checkpoint, 'model.pt')
```

## Model Gereksinimleri

- **Giriş boyutu**: (1, 3, 224, 224) veya (1, 3, 240, 240)
- **Çıkış boyutu**: 4 sınıf (Glioma, Meningioma, Pituitary, NoTumor)
- **Framework**: PyTorch 2.2.0+

## Kullanım

```python
from utils import load_model, predict_image

# Model yükle
model_inference = load_model(
    model_path='models/best_model.pt',
    class_names=['Glioma', 'Meningioma', 'Pituitary', 'NoTumor'],
    device='auto',
    image_size=(224, 224)
)

# Tahmin yap
result = predict_image(model_inference, 'path/to/mri/image.jpg')
print(f"Tahmin: {result['predicted_class']}")
print(f"Güven: {result['confidence']:.2%}")
```

## Not

Model dosyaları `.gitignore` ile git'e eklenmez. 
Büyük model dosyalarını paylaşmak için:
- Google Drive
- Hugging Face Hub
- Git LFS
kullanabilirsiniz.
