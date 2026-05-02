import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm


def evaluate(model, loader, device, class_names, save_dir: str = "."):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    accuracy = (all_preds == all_labels).mean()

    print("\n" + "=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Weighted F1    : {weighted_f1:.4f}")
    print(f"Macro F1       : {macro_f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    _plot_confusion_matrix(cm, class_names, save_dir)

    return {"accuracy": accuracy, "weighted_f1": weighted_f1, "macro_f1": macro_f1, "cm": cm}


def _plot_confusion_matrix(cm, class_names, save_dir: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()

    out = f"{save_dir}/confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {out}")


def plot_training_curves(history: dict, save_dir: str = "."):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    out = f"{save_dir}/training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Training curves saved → {out}")
