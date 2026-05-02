"""
Model Çıkarım Modülü
Eğitilmiş modeli yükleme ve tahmin yapma
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image
import logging

from .preprocessing import preprocess_image

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInference:
    """Model çıkarım sınıfı"""
    
    def __init__(
        self,
        model_path: str,
        class_names: list = None,
        device: str = 'auto',
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            model_path: Eğitilmiş model dosyası (.pt veya .pth)
            class_names: Sınıf isimleri listesi
            device: 'cuda', 'cpu' veya 'auto'
            image_size: Görüntü boyutu
        """
        self.model_path = Path(model_path)
        self.image_size = image_size
        
        # Sınıf isimleri
        if class_names is None:
            self.class_names = ["Glioma", "Meningioma", "Pituitary", "NoTumor"]
        else:
            self.class_names = class_names
        
        # Device seçimi
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Kullanılan device: {self.device}")
        
        # Model yükle
        self.model = self._load_model()
        self.model.eval()
    
    def _load_model(self) -> nn.Module:
        """Model dosyasını yükler"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {self.model_path}")
        
        try:
            # Checkpoint yükle
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Model state_dict'i al
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Model mimarisini yükle veya oluştur
            # Not: Gerçek projede model mimarisi ayrı dosyadan import edilmeli
            model = self._create_model_architecture()
            model.load_state_dict(state_dict)
            model.to(self.device)
            
            logger.info(f"Model başarıyla yüklendi: {self.model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Model yüklenirken hata: {str(e)}")
            raise
    
    def _create_model_architecture(self) -> nn.Module:
        """
        Model mimarisini oluşturur
        Not: Bu fonksiyon projeye özel güncellenmeli
        """
        # ResNet18 tabanlı model
        from torchvision import models
        
        model = models.resnet18(weights=None)
        num_classes = len(self.class_names)
        
        # FC layer: Sequential(Linear(512, 512), ReLU, Dropout, Linear(512, 4))
        model.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        return model
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Image.Image, np.ndarray, torch.Tensor],
        return_probabilities: bool = True
    ) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Görüntü üzerinde tahmin yapar
        
        Args:
            image: Görüntü (dosya yolu, PIL Image, NumPy array veya Tensor)
            return_probabilities: Tüm sınıflar için olasılık dön
            
        Returns:
            Dict: Tahmin sonuçları
                - predicted_class: str
                - confidence: float (0-1 arası)
                - probabilities: Dict[str, float] (opsiyonel)
        """
        self.model.eval()
        
        # Görüntüyü hazırla
        if isinstance(image, torch.Tensor):
            input_tensor = image.to(self.device)
        else:
            input_tensor = preprocess_image(image, self.image_size)
            input_tensor = input_tensor.to(self.device)
        
        # Tahmin yap
        outputs = self.model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # En yüksek olasılığı bul
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = self.class_names[predicted_idx.item()]
        confidence_value = confidence.item()
        
        result = {
            'predicted_class': predicted_class,
            'predicted_index': predicted_idx.item(),
            'confidence': confidence_value
        }
        
        # Tüm sınıf olasılıkları
        if return_probabilities:
            probs_dict = {
                name: prob.item() 
                for name, prob in zip(self.class_names, probabilities[0])
            }
            result['probabilities'] = probs_dict
        
        return result
    
    def predict_batch(
        self,
        images: list,
        batch_size: int = 32
    ) -> list:
        """
        Birden fazla görüntü üzerinde tahmin yapar
        
        Args:
            images: Görüntü listesi
            batch_size: Batch boyutu
            
        Returns:
            list: Her görüntü için tahmin sonuçları
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Batch'i hazırla
            tensors = []
            for img in batch:
                if isinstance(img, torch.Tensor):
                    tensors.append(img)
                else:
                    tensor = preprocess_image(img, self.image_size)
                    tensors.append(tensor)
            
            # Stack et ve device'a gönder
            batch_tensor = torch.cat(tensors, dim=0).to(self.device)
            
            # Tahmin yap
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                for probs in probabilities:
                    confidence, predicted_idx = torch.max(probs, 0)
                    predicted_class = self.class_names[predicted_idx.item()]
                    
                    result = {
                        'predicted_class': predicted_class,
                        'predicted_index': predicted_idx.item(),
                        'confidence': confidence.item(),
                        'probabilities': {
                            name: prob.item() 
                            for name, prob in zip(self.class_names, probs)
                        }
                    }
                    results.append(result)
        
        return results
    
    def get_feature_maps(
        self,
        image: Union[str, Image.Image, np.ndarray, torch.Tensor],
        layer_name: str = None
    ) -> torch.Tensor:
        """
        Belirli bir katmanın feature map'lerini döndürür
        Grad-CAM için kullanılır
        
        Args:
            image: Görüntü
            layer_name: Katman ismi (None ise son conv katmanı)
            
        Returns:
            torch.Tensor: Feature maps
        """
        # Görüntüyü hazırla
        if isinstance(image, torch.Tensor):
            input_tensor = image.to(self.device)
        else:
            input_tensor = preprocess_image(image, self.image_size)
            input_tensor = input_tensor.to(self.device)
        
        # Hook ile feature map'leri yakala
        features = []
        
        def hook_fn(module, input, output):
            features.append(output)
        
        # Son conv layer'ı bul ve hook ekle
        if layer_name is None:
            # ResNet için örnek: layer4
            target_layer = self.model.layer4[-1]
        else:
            target_layer = dict(self.model.named_modules())[layer_name]
        
        handle = target_layer.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        handle.remove()
        
        return features[0]


def load_model(
    model_path: str,
    class_names: list = None,
    device: str = 'auto',
    image_size: Tuple[int, int] = (224, 224)
) -> ModelInference:
    """
    Model çıkarım objesi oluşturur
    
    Args:
        model_path: Model dosya yolu
        class_names: Sınıf isimleri
        device: Cihaz
        image_size: Görüntü boyutu
        
    Returns:
        ModelInference: Çıkarım objesi
    """
    return ModelInference(
        model_path=model_path,
        class_names=class_names,
        device=device,
        image_size=image_size
    )


def predict_image(
    model: ModelInference,
    image: Union[str, Image.Image, np.ndarray, torch.Tensor]
) -> Dict:
    """
    Basitleştirilmiş tahmin fonksiyonu
    
    Args:
        model: ModelInference objesi
        image: Görüntü
        
    Returns:
        Dict: Tahmin sonuçları
    """
    return model.predict(image, return_probabilities=True)
