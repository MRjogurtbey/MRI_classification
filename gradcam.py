import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """Grad-CAM for ResNet-18. Hooks onto the last BasicBlock of layer4."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        target = model.layer4[-1]
        self._fwd_hook = target.register_forward_hook(self._save_activation)
        self._bwd_hook = target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        """
        Returns:
            cam       : np.ndarray (H, W), values in [0, 1]
            class_idx : int, predicted or specified class index
        """
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        output[0, class_idx].backward()

        # (C,) weights via global average pooling of gradients
        weights = self._gradients[0].mean(dim=(1, 2))
        cam = (weights[:, None, None] * self._activations[0]).sum(dim=0)
        cam = torch.relu(cam)

        cam_np = cam.detach().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        cam_np = cv2.resize(cam_np, (input_tensor.shape[3], input_tensor.shape[2]))

        return cam_np, class_idx

    @staticmethod
    def overlay(original_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Blend a JET heatmap onto an RGB image (values 0-255)."""
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        blended = heatmap * alpha + original_rgb * (1 - alpha)
        return np.clip(blended, 0, 255).astype(np.uint8)
