import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from gradcam import GradCAM
from model import get_model
from ollama_report import generate_report

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = get_model(num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    return model


def predict_image(
    pil_image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    with_gradcam: bool = True,
    with_report: bool = True,
    ollama_model: str = "llama3",
) -> dict:
    """
    Args:
        pil_image   : PIL.Image (RGB)
        model       : loaded model (from load_model)
        device      : torch device
    Returns dict with keys:
        predicted_class, confidence, class_probabilities,
        gradcam_overlay (np.ndarray or None),
        report (str or None)
    """
    input_tensor = _TRANSFORM(pil_image).unsqueeze(0).to(device)

    gradcam = GradCAM(model) if with_gradcam else None

    model.eval()
    with torch.no_grad() if not with_gradcam else torch.enable_grad():
        if with_gradcam:
            cam, predicted_idx = gradcam.generate(input_tensor)
        else:
            output = model(input_tensor)
            predicted_idx = int(output.argmax(dim=1).item())
            cam = None

    # Probabilities (run a clean forward pass without grad tape)
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probs[predicted_idx])

    overlay = None
    if cam is not None:
        img_np = np.array(pil_image.resize((224, 224)))
        overlay = GradCAM.overlay(img_np, cam)

    if gradcam:
        gradcam.remove_hooks()

    report = generate_report(predicted_class, ollama_model) if with_report else None

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "class_probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "gradcam_overlay": overlay,
        "report": report,
    }


def _cli():
    parser = argparse.ArgumentParser(description="MRI Inference — single image prediction")
    parser.add_argument("image", help="Path to MRI image")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--no_gradcam", action="store_true")
    parser.add_argument("--no_report", action="store_true")
    parser.add_argument("--ollama_model", default="llama3")
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    pil_image = Image.open(args.image).convert("RGB")

    result = predict_image(
        pil_image, model, device,
        with_gradcam=not args.no_gradcam,
        with_report=not args.no_report,
        ollama_model=args.ollama_model,
    )

    print(f"\nPrediction  : {result['predicted_class'].upper()}")
    print(f"Confidence  : {result['confidence']:.2%}")
    print("\nClass probabilities:")
    for cls, prob in sorted(result["class_probabilities"].items(), key=lambda x: -x[1]):
        print(f"  {cls:<15} {prob:.2%}")

    if result["gradcam_overlay"] is not None:
        import matplotlib.pyplot as plt
        out_path = os.path.join(args.output_dir, "gradcam_result.png")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(pil_image.resize((224, 224)))
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(result["gradcam_overlay"])
        axes[1].set_title(f"Grad-CAM — {result['predicted_class']}")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"\nGrad-CAM saved → {out_path}")

    if result["report"]:
        print(f"\n--- Ön Rapor ---\n{result['report']}")


if __name__ == "__main__":
    _cli()
