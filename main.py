import argparse
import os

import torch

from dataset import get_dataloaders
from evaluate import evaluate, plot_training_curves
from model import get_model
from train import train


def parse_args():
    parser = argparse.ArgumentParser(description="MRI Brain Tumor Classification")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--no_pretrain", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_DIR = os.path.join(BASE_DIR, "Training")
    TEST_DIR = os.path.join(BASE_DIR, "Testing")
    SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[Data]")
    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders(
        TRAIN_DIR, TEST_DIR,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("\n[Model]")
    model = get_model(num_classes=len(class_names), pretrained=not args.no_pretrain)
    model = model.to(device)

    if args.mode in ("train", "both"):
        print("\n[Training]")
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,  # Use validation set for model selection
            num_epochs=args.epochs,
            device=device,
            class_weights=class_weights,
            save_dir=SAVE_DIR,
        )
        plot_training_curves(history, save_dir=BASE_DIR)

    if args.mode in ("eval", "both"):
        ckpt = args.checkpoint
        if os.path.exists(ckpt):
            print(f"\n[Eval] Loading checkpoint: {ckpt}")
            model.load_state_dict(torch.load(ckpt, map_location=device))
        else:
            print(f"\n[Eval] Checkpoint not found ({ckpt}), using current weights.")

        print("\n[Final Test Evaluation]")
        print("=" * 55)
        print("NOTE: This is the FINAL test set evaluation.")
        print("This data was not used during training or model selection.")
        print("=" * 55)
        evaluate(model, test_loader, device, class_names, save_dir=BASE_DIR)


if __name__ == "__main__":
    main()
