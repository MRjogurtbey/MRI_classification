"""
Görüntü Ön İşleme Modülü
MRI görüntülerini model için hazırlama
"""

import numpy as np
import torch
from PIL import Image
import cv2
import h5py
from typing import Union, Tuple
from torchvision import transforms


class MRIPreprocessor:
    """MRI görüntüleri için ön işleme sınıfı"""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            image_size: Hedef görüntü boyutu (height, width)
        """
        self.image_size = image_size
        
        # PyTorch transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalizasyonu
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Grayscale için alternatif transform
        self.transform_gray = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),  # 1 kanallı -> 3 kanallı
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_pil_image(self, image: Image.Image) -> torch.Tensor:
        """
        PIL Image objesini model için hazırlar
        
        Args:
            image: PIL Image objesi
            
        Returns:
            torch.Tensor: (1, 3, H, W) boyutunda tensor
        """
        # Görüntü moduna göre transform seç
        if image.mode == 'L' or image.mode == 'I':
            # Grayscale görüntü
            transformed = self.transform_gray(image)
        elif image.mode == 'RGB':
            transformed = self.transform(image)
        elif image.mode == 'RGBA':
            # RGBA -> RGB dönüşümü
            rgb_image = image.convert('RGB')
            transformed = self.transform(rgb_image)
        else:
            # Diğer modları RGB'ye çevir
            rgb_image = image.convert('RGB')
            transformed = self.transform(rgb_image)
        
        # Batch dimension ekle
        return transformed.unsqueeze(0)
    
    def preprocess_numpy_array(self, array: np.ndarray) -> torch.Tensor:
        """
        NumPy array'ini model için hazırlar
        
        Args:
            array: NumPy array (H, W) veya (H, W, C)
            
        Returns:
            torch.Tensor: (1, 3, H, W) boyutunda tensor
        """
        # Normalize et
        if array.dtype != np.uint8:
            # Float array ise 0-255 aralığına getir
            array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
        
        # PIL Image'a çevir
        if len(array.shape) == 2:
            # Grayscale
            image = Image.fromarray(array, mode='L')
        elif len(array.shape) == 3:
            if array.shape[2] == 1:
                # Single channel
                image = Image.fromarray(array[:, :, 0], mode='L')
            elif array.shape[2] == 3:
                # RGB
                image = Image.fromarray(array, mode='RGB')
            elif array.shape[2] == 4:
                # RGBA
                image = Image.fromarray(array, mode='RGBA')
            else:
                raise ValueError(f"Desteklenmeyen kanal sayısı: {array.shape[2]}")
        else:
            raise ValueError(f"Desteklenmeyen array boyutu: {array.shape}")
        
        return self.preprocess_pil_image(image)
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization) uygular
        MRI görüntülerinde kontrast artırma için
        
        Args:
            image: Grayscale NumPy array
            
        Returns:
            np.ndarray: CLAHE uygulanmış görüntü
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)


def preprocess_image(
    image: Union[Image.Image, np.ndarray, str],
    image_size: Tuple[int, int] = (224, 224),
    apply_clahe: bool = False
) -> torch.Tensor:
    """
    Görüntüyü model için hazırlar
    
    Args:
        image: PIL Image, NumPy array veya dosya yolu
        image_size: Hedef boyut
        apply_clahe: CLAHE uygulansın mı
        
    Returns:
        torch.Tensor: Hazırlanmış tensor
    """
    preprocessor = MRIPreprocessor(image_size=image_size)
    
    # Dosya yolu ise yükle
    if isinstance(image, str):
        image = Image.open(image)
    
    # NumPy array ise
    if isinstance(image, np.ndarray):
        if apply_clahe:
            image = preprocessor.apply_clahe(image)
        return preprocessor.preprocess_numpy_array(image)
    
    # PIL Image ise
    if isinstance(image, Image.Image):
        if apply_clahe:
            # PIL -> NumPy -> CLAHE -> PIL
            np_image = np.array(image)
            np_image = preprocessor.apply_clahe(np_image)
            image = Image.fromarray(np_image)
        return preprocessor.preprocess_pil_image(image)
    
    raise TypeError(f"Desteklenmeyen görüntü tipi: {type(image)}")


def load_h5_image(h5_path: str, dataset_name: str = 'image') -> np.ndarray:
    """
    H5 dosyasından görüntü yükler
    
    Args:
        h5_path: H5 dosya yolu
        dataset_name: H5 içindeki dataset ismi
        
    Returns:
        np.ndarray: Görüntü array'i
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            # Dataset isimlerini kontrol et
            available_keys = list(f.keys())
            
            if dataset_name in f:
                image = np.array(f[dataset_name])
            elif len(available_keys) == 1:
                # Tek dataset varsa onu al
                image = np.array(f[available_keys[0]])
            else:
                raise KeyError(
                    f"Dataset '{dataset_name}' bulunamadı. "
                    f"Mevcut dataset'ler: {available_keys}"
                )
            
            return image
    except Exception as e:
        raise RuntimeError(f"H5 dosyası yüklenirken hata: {str(e)}")


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Normalize edilmiş tensor'ı görselleştirme için geri çevirir
    
    Args:
        tensor: Normalize edilmiş tensor (C, H, W)
        
    Returns:
        np.ndarray: 0-255 arası uint8 array (H, W, C)
    """
    # ImageNet normalizasyonunu geri al
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # (C, H, W) -> (H, W, C) ve 0-255 arası
    array = tensor.permute(1, 2, 0).numpy()
    array = (array * 255).astype(np.uint8)
    
    return array
