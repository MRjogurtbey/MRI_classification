import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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
    plot_per_class_metrics(all_labels, all_preds, class_names, save_dir)

    return {"accuracy": accuracy, "weighted_f1": weighted_f1, "macro_f1": macro_f1, "cm": cm}


def _plot_confusion_matrix(cm, class_names, save_dir: str):
    # Row-normalize for percentages (each row = true class)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = cm / row_sums * 100

    # Build annotation: "count\n(pct%)"
    annot = np.array([
        [f"{cm[i, j]}\n({cm_pct[i, j]:.0f}%)" for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ])

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_pct,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=100,
        ax=ax,
    )
    ax.set_title("Confusion Matrix (count & row %)", fontsize=14)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()

    out = f"{save_dir}/confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved -> {out}")


def plot_per_class_metrics(all_labels, all_preds, class_names, save_dir: str):
    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    f1 = f1_score(all_labels, all_preds, average=None)

    x = np.arange(len(class_names))
    bar_w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_p = ax.bar(x - bar_w, precision, bar_w, label="Precision", color="#4C72B0")
    bars_r = ax.bar(x,          recall,    bar_w, label="Recall",    color="#DD8452")
    bars_f = ax.bar(x + bar_w,  f1,        bar_w, label="F1-Score",  color="#55A868")

    # Value labels on top of each bar
    for bars in (bars_p, bars_r, bars_f):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics: Precision / Recall / F1-Score", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = f"{save_dir}/performance_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Performance metrics saved → {out}")


def plot_training_curves(history: dict, save_dir: str = "."):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = f"{save_dir}/training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Training curves saved -> {out}")
