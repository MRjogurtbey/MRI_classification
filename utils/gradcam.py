"""
Grad-CAM Görselleştirme Modülü
Modelin hangi bölgelere odaklandığını gösterir
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Union, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .preprocessing import preprocess_image, denormalize_tensor


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM)"""
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: PyTorch modeli
            target_layer: Grad-CAM uygulanacak katman (genellikle son conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks kaydet
        self._register_hooks()
    
    def _register_hooks(self):
        """Forward ve backward hook'ları kaydeder"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Grad-CAM haritası oluşturur
        
        Args:
            input_tensor: Giriş tensor (1, C, H, W)
            target_class: Hedef sınıf (None ise tahmin edilen sınıf)
            
        Returns:
            np.ndarray: Grad-CAM haritası (H, W)
        """
        self.model.eval()
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Hedef sınıf belirle
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Grad-CAM hesapla
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """generate() için kısayol"""
        return self.generate(input_tensor, target_class)


def generate_gradcam(
    model: torch.nn.Module,
    image: Union[str, Image.Image, np.ndarray, torch.Tensor],
    target_layer: torch.nn.Module = None,
    target_class: Optional[int] = None,
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Grad-CAM haritası oluşturur
    
    Args:
        model: PyTorch modeli
        image: Giriş görüntüsü
        target_layer: Hedef katman (None ise otomatik seçilir)
        target_class: Hedef sınıf (None ise tahmin edilen)
        image_size: Görüntü boyutu
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (cam_map, original_image)
            - cam_map: Grad-CAM haritası (H, W)
            - original_image: Orijinal görüntü (H, W, 3)
    """
    device = next(model.parameters()).device
    
    # Görüntüyü hazırla
    if isinstance(image, torch.Tensor):
        input_tensor = image.to(device)
        original_image = denormalize_tensor(input_tensor[0].cpu())
    else:
        input_tensor = preprocess_image(image, image_size).to(device)
        # Orijinal görüntüyü de kaydet
        if isinstance(image, str):
            original_image = np.array(Image.open(image).resize(image_size))
        elif isinstance(image, Image.Image):
            original_image = np.array(image.resize(image_size))
        elif isinstance(image, np.ndarray):
            original_image = cv2.resize(image, image_size)
        else:
            original_image = denormalize_tensor(input_tensor[0].cpu())
    
    # Target layer otomatik seç
    if target_layer is None:
        # ResNet için son conv layer
        if hasattr(model, 'layer4'):
            target_layer = model.layer4[-1]
        # EfficientNet için
        elif hasattr(model, 'features'):
            target_layer = model.features[-1]
        else:
            raise ValueError("Target layer belirtilmeli veya model ResNet/EfficientNet olmalı")
    
    # Grad-CAM oluştur
    gradcam = GradCAM(model, target_layer)
    cam = gradcam(input_tensor, target_class)
    
    # Cam'i orijinal boyuta getir
    cam_resized = cv2.resize(cam, (image_size[1], image_size[0]))
    
    return cam_resized, original_image


def overlay_gradcam(
    cam: np.ndarray,
    image: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Grad-CAM haritasını orijinal görüntü üzerine bindire
    
    Args:
        cam: Grad-CAM haritası (H, W) veya (H, W, 1)
        image: Orijinal görüntü (H, W, 3)
        alpha: Karıştırma oranı (0-1)
        colormap: OpenCV color map
        
    Returns:
        np.ndarray: Birleştirilmiş görüntü (H, W, 3)
    """
    # Cam'i 0-255 arası uint8'e çevir
    if cam.max() <= 1.0:
        cam = (cam * 255).astype(np.uint8)
    else:
        cam = cam.astype(np.uint8)
    
    # 2D ise 3D yap
    if len(cam.shape) == 2:
        cam = np.expand_dims(cam, axis=2)
    
    # Color map uygula
    heatmap = cv2.applyColorMap(cam[:, :, 0], colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Görüntüyü normalize et
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Grayscale ise RGB'ye çevir
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    
    # Boyutları eşitle
    if image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Overlay
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    return overlayed


def visualize_gradcam(
    cam: np.ndarray,
    image: np.ndarray,
    predicted_class: str = None,
    confidence: float = None,
    save_path: str = None
) -> plt.Figure:
    """
    Grad-CAM görselleştirmesi oluşturur
    
    Args:
        cam: Grad-CAM haritası
        image: Orijinal görüntü
        predicted_class: Tahmin edilen sınıf
        confidence: Güven skoru
        save_path: Kayıt yolu (opsiyonel)
        
    Returns:
        plt.Figure: Matplotlib figürü
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Orijinal görüntü
    axes[0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    axes[0].set_title('Orijinal MRI Görüntüsü', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Grad-CAM haritası
    im = axes[1].imshow(cam, cmap='jet', alpha=0.8)
    axes[1].set_title('Grad-CAM Isı Haritası', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    overlayed = overlay_gradcam(cam, image, alpha=0.5)
    axes[2].imshow(overlayed)
    axes[2].set_title('Üst Üste Bindirme', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Başlık
    if predicted_class and confidence:
        fig.suptitle(
            f'Tahmin: {predicted_class} (Güven: {confidence:.2%})',
            fontsize=14,
            fontweight='bold',
            y=0.98
        )
    
    plt.tight_layout()
    
    # Kaydet
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_gradcam_comparison(
    model: torch.nn.Module,
    images: list,
    class_names: list,
    target_layer: torch.nn.Module = None,
    save_path: str = None
) -> plt.Figure:
    """
    Birden fazla görüntü için Grad-CAM karşılaştırması
    
    Args:
        model: PyTorch modeli
        images: Görüntü listesi
        class_names: Sınıf isimleri
        target_layer: Hedef katman
        save_path: Kayıt yolu
        
    Returns:
        plt.Figure: Karşılaştırma figürü
    """
    n_images = len(images)
    fig, axes = plt.subplots(n_images, 3, figsize=(15, 5 * n_images))
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, image in enumerate(images):
        # Grad-CAM oluştur
        cam, orig_image = generate_gradcam(model, image, target_layer)
        overlayed = overlay_gradcam(cam, orig_image)
        
        # Orijinal
        axes[i, 0].imshow(orig_image, cmap='gray' if len(orig_image.shape) == 2 else None)
        axes[i, 0].set_title(f'Görüntü {i+1}', fontweight='bold')
        axes[i, 0].axis('off')
        
        # Grad-CAM
        axes[i, 1].imshow(cam, cmap='jet')
        axes[i, 1].set_title('Grad-CAM', fontweight='bold')
        axes[i, 1].axis('off')
        
        # Overlay
        axes[i, 2].imshow(overlayed)
        axes[i, 2].set_title('Overlay', fontweight='bold')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
