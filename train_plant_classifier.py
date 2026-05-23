"""
Task 2: Plant Disease Image Classification
==========================================
Dataset : plantvillage dataset/  (38 classes)
Models  : ResNet-50 & MobileNet-V2  (full fine-tune, no freeze)
Split   : 70 % train / 15 % val / 15 % test  (stratified, random)
Training: Mixed Precision (AMP) + OneCycleLR scheduler
Saves   : plant_resnet50.pth   (state_dict + class_names)
          plant_mobilenet.pth  (state_dict + class_names)
          plant_class_names.pkl
"""

import os, json, pickle, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────── Config ───────────────────────────
DATA_DIR   = "plantvillage dataset"
IMG_SIZE   = 224
BATCH_SIZE = 64
EPOCHS     = 25
LR_MAX     = 1e-3
WEIGHT_DECAY = 1e-4
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ─────────────────────────── Transforms ───────────────────────
train_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────── Dataset & Split ──────────────────
full_dataset = datasets.ImageFolder(DATA_DIR, transform=val_tfm)  # base tfm; overridden below
class_names  = full_dataset.classes
NUM_CLASSES  = len(class_names)
print(f"[INFO] Classes: {NUM_CLASSES} | Images: {len(full_dataset)}")

# Stratified 70/15/15 split on indices
labels = np.array([s[1] for s in full_dataset.samples])
all_idx = np.arange(len(full_dataset))

sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
train_idx, temp_idx = next(sss1.split(all_idx, labels))
temp_labels = labels[temp_idx]
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
val_rel_idx, test_rel_idx = next(sss2.split(temp_idx, temp_labels))
val_idx  = temp_idx[val_rel_idx]
test_idx = temp_idx[test_rel_idx]

print(f"[INFO] Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

# Separate datasets with correct transforms
train_ds_base = datasets.ImageFolder(DATA_DIR, transform=train_tfm)
val_ds_base   = datasets.ImageFolder(DATA_DIR, transform=val_tfm)

train_ds = Subset(train_ds_base, train_idx)
val_ds   = Subset(val_ds_base,   val_idx)
test_ds  = Subset(val_ds_base,   test_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=(DEVICE.type == "cuda"))
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=(DEVICE.type == "cuda"))
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=(DEVICE.type == "cuda"))

# ─────────────────────────── Model factory ────────────────────
def build_model(arch: str) -> nn.Module:
    """Build full fine-tune model (no frozen layers)."""
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)        # 512 → 38
    elif arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)  # 1024 → 38
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return m.to(DEVICE)

# ─────────────────────────── Train one model ──────────────────
def train_model(arch: str, save_path: str):
    print(f"\n{'='*60}")
    print(f"  Training: {arch.upper()}")
    print(f"{'='*60}")

    model = build_model(arch)
    print(f"[INFO] Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX,
        steps_per_epoch=len(train_loader), epochs=EPOCHS,
        pct_start=0.1, div_factor=10, final_div_factor=100
    )

    use_amp  = DEVICE.type == "cuda"
    gscaler  = GradScaler(device='cuda' if use_amp else 'cpu', enabled=use_amp)

    best_val_acc = 0.0
    best_state   = None

    print(f"\n{'Ep':>4} | {'Train Loss':>11} | {'Train Acc':>10} | {'Val Acc':>9} | {'Time':>7}")
    print("-" * 52)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        # ── train
        model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0
        for imgs, labels_b in train_loader:
            imgs, labels_b = imgs.to(DEVICE), labels_b.to(DEVICE)
            optimizer.zero_grad()
            with autocast('cuda' if use_amp else 'cpu'):
                logits = model(imgs)
                loss   = criterion(logits, labels_b)
            gscaler.scale(loss).backward()
            gscaler.step(optimizer)
            gscaler.update()
            scheduler.step()
            run_loss    += loss.item() * imgs.size(0)
            run_correct += (logits.argmax(1) == labels_b).sum().item()
            run_total   += imgs.size(0)

        train_loss = run_loss / run_total
        train_acc  = run_correct / run_total

        # ── validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels_b in val_loader:
                imgs, labels_b = imgs.to(DEVICE), labels_b.to(DEVICE)
                with autocast('cuda' if use_amp else 'cpu'):
                    logits = model(imgs)
                val_correct += (logits.argmax(1) == labels_b).sum().item()
                val_total   += imgs.size(0)
        val_acc = val_correct / val_total

        elapsed = time.time() - t0
        print(f"{epoch:>4} | {train_loss:>11.4f} | {train_acc:>10.4f} | {val_acc:>9.4f} | {elapsed:>6.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    # ── test evaluation
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for imgs, labels_b in test_loader:
            imgs, labels_b = imgs.to(DEVICE), labels_b.to(DEVICE)
            with autocast('cuda' if use_amp else 'cpu'):
                logits = model(imgs)
            preds = logits.argmax(1)
            test_correct += (preds == labels_b).sum().item()
            test_total   += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_b.cpu().numpy())

    test_acc = test_correct / test_total
    print(f"\n[RESULT] {arch.upper()} — Best Val Acc: {best_val_acc:.4f}  |  Test Acc: {test_acc:.4f}")

    # Full classification report
    print("\n[Classification Report]")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

    # Save
    torch.save({
        "arch": arch,
        "state_dict": best_state,
        "class_names": class_names,
        "num_classes": NUM_CLASSES,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
    }, save_path)
    print(f"[SAVED] {save_path}")

    return best_val_acc, test_acc, all_preds, all_labels

# ─────────────────────────── Run ──────────────────────────────
res_val, res_test, res_preds, res_labels = train_model("resnet18",         "plant_resnet18.pth")
mob_val, mob_test, mob_preds, mob_labels = train_model("mobilenet_v3_small","plant_mobilenet_v3s.pth")

# Save class names separately for convenience
with open("plant_class_names.pkl", "wb") as f:
    pickle.dump(class_names, f)

print("\n[DONE] Saved:")
print("  • plant_resnet18.pth")
print("  • plant_mobilenet_v3s.pth")
print("  • plant_class_names.pkl")

print("\n[SUMMARY]")
print(f"  ResNet-18         : Val Acc = {res_val:.4f}  |  Test Acc = {res_test:.4f}")
print(f"  MobileNet-V3-Small: Val Acc = {mob_val:.4f}  |  Test Acc = {mob_test:.4f}")
