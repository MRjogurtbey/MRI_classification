import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def _run_epoch(model, loader, criterion, optimizer, device, training: bool, scaler):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = scaler is not None

    ctx = torch.enable_grad() if training else torch.no_grad()
    desc = "Train" if training else "Val  "

    with ctx:
        for inputs, labels in tqdm(loader, desc=desc, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pass  # autocast already applied above

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / total, correct / total


def train(
    model,
    train_loader,
    val_loader,
    num_epochs: int,
    device,
    class_weights,
    save_dir: str = "checkpoints",
):
    os.makedirs(save_dir, exist_ok=True)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Mixed Precision (AMP): {'enabled' if use_amp else 'disabled'}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*55}")
        print(f"Epoch {epoch}/{num_epochs}  |  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*55}")

        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device, training=True, scaler=scaler)
        vl_loss, vl_acc = _run_epoch(model, val_loader, criterion, optimizer, device, training=False, scaler=scaler)

        scheduler.step(vl_acc)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(f"Train  loss={tr_loss:.4f}  acc={tr_acc:.4f}")
        print(f"Val    loss={vl_loss:.4f}  acc={vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            ckpt_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Checkpoint saved  (best val_acc={best_val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    return history
