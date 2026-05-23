"""
Quantum Transfer Learning — Plant Disease Classification
=========================================================
Dataset  : plantvillage dataset/  (38 classes, 54 305 images)
Strategy : Quantum Transfer Learning
  1. Frozen ImageNet ResNet-18 as feature extractor  (512-dim, ONE forward pass)
  2. Cache ALL features to RAM (fast; no CNN re-run during quantum training)
  3. Quantum head:
       Linear(512 → N_QUBITS=8) + Tanh * π
       → AngleEmbedding + StronglyEntanglingLayers (2 layers)
       → 8 PauliZ expectation values
       → Linear(8 → 38)
  4. Classical baseline on same cached features for fair comparison:
       Linear(512 → 38)  — shows how much the VQC adds

Why this design?
  • Running VQC through 54 k CNN forward passes every epoch = impossibly slow.
  • Pre-extraction decouples CNN from quantum head — same approach used in
    Mari et al. "Transfer Learning in Hybrid Classical-Quantum Neural Networks"
    (2020, arXiv:1912.08278).

Saves: plant_qnn.pth  (quantum head weights)
       plant_class_names.pkl  (already exists — reused)
"""

import os, pickle, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score

import pennylane as qml

# ─────────────────────────── Config ───────────────────────────
DATA_DIR    = "plantvillage dataset"
IMG_SIZE    = 224
BATCH_SIZE  = 128
EPOCHS      = 5
LR          = 3e-3
WEIGHT_DECAY = 1e-4
SEED        = 42
N_QUBITS    = 8
N_LAYERS    = 1

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device  : {DEVICE}")
print(f"[INFO] Qubits  : {N_QUBITS}  |  VQC layers : {N_LAYERS}")

# ─────────────────────────── 1. Dataset splits (same as classical) ─
val_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

base_ds     = datasets.ImageFolder(DATA_DIR, transform=val_tfm)
class_names = base_ds.classes
NUM_CLASSES = len(class_names)
labels_all  = np.array([s[1] for s in base_ds.samples])
all_idx     = np.arange(len(base_ds))

sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
train_idx, temp_idx = next(sss1.split(all_idx, labels_all))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
val_rel, test_rel = next(sss2.split(temp_idx, labels_all[temp_idx]))
val_idx  = temp_idx[val_rel]
test_idx = temp_idx[test_rel]
print(f"[INFO] Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

# ─────────────────────────── 2. Feature extractor (frozen) ────
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
feat_extractor = nn.Sequential(*list(resnet18.children())[:-1], nn.Flatten())
feat_extractor = feat_extractor.to(DEVICE).eval()
for p in feat_extractor.parameters():
    p.requires_grad = False
print(f"[INFO] ResNet-18 feature extractor: 512-dim (frozen ImageNet weights)")

# ─────────────────────────── 3. Pre-extract all features ──────
def extract_features(dataset, indices, max_samples=None):
    if max_samples and len(indices) > max_samples:
        indices = np.random.choice(indices, max_samples, replace=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=4)
    feats, labs = [], []
    with torch.no_grad():
        for imgs, lbs in loader:
            feats.append(feat_extractor(imgs.to(DEVICE)).cpu())
            labs.append(lbs)
    return torch.cat(feats), torch.cat(labs)

print("[INFO] Extracting features …")
t0 = time.time()
# Use max 2000 samples for train, 1000 for val/test to keep quantum training fast
X_train_f, y_train = extract_features(base_ds, train_idx, max_samples=2000)
X_val_f,   y_val   = extract_features(base_ds, val_idx,   max_samples=1000)
X_test_f,  y_test  = extract_features(base_ds, test_idx,  max_samples=1000)
print(f"[INFO] Done in {time.time()-t0:.1f}s  |  Train Features: {X_train_f.shape}")

def make_loader(X, y, shuffle=False):
    return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE,
                      shuffle=shuffle, num_workers=0)

train_loader = make_loader(X_train_f, y_train, shuffle=True)
val_loader   = make_loader(X_val_f,   y_val)
test_loader  = make_loader(X_test_f,  y_test)

# ─────────────────────────── 4. Quantum circuit ────────────────
try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    diff_m = "adjoint"
    print("[INFO] Using lightning.qubit (adjoint diff)")
except Exception:
    dev = qml.device("default.qubit", wires=N_QUBITS)
    diff_m = "best"
    print("[INFO] Using default.qubit")

@qml.qnode(dev, diff_method=diff_m, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# ─────────────────────────── 5. Quantum classifier ────────────
class QuantumHead(nn.Module):
    """
    512-dim ResNet-18 features
      → Linear(512 → N_QUBITS) + Tanh * π   (angle compression)
      → VQC (AngleEmbedding + StronglyEntanglingLayers)
      → Linear(N_QUBITS → NUM_CLASSES)
    """
    def __init__(self):
        super().__init__()
        self.compress = nn.Sequential(nn.Linear(512, N_QUBITS), nn.Tanh())
        self.qlayer   = qlayer
        self.head     = nn.Linear(N_QUBITS, NUM_CLASSES)

    def forward(self, x):
        x = self.compress(x) * torch.pi   # scale to (-π, π)
        x = self.qlayer(x)                 # [B, N_QUBITS] ∈ (-1,1)
        return self.head(x)

# ─────────────────────────── 6. Classical baseline (fair cmp) ─
class ClassicalHead(nn.Module):
    """Linear probe on the same 512-dim features — zero-quantum reference."""
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(512, NUM_CLASSES)
    def forward(self, x):
        return self.net(x)

# ─────────────────────────── 7. Generic train loop ────────────
def train_head(model, tag, save_path=None):
    model = model.to(DEVICE)
    q_p  = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*58}")
    print(f"  {tag}   ({q_p:,} params)")
    print(f"{'='*58}")

    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS,
        pct_start=0.1, div_factor=10, final_div_factor=100)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val, best_state = 0.0, None

    print(f"\n{'Ep':>4} | {'Train Acc':>10} | {'Val Acc':>9} | {'Time':>7}")
    print("-" * 40)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        correct, total = 0, 0
        for i, (Xb, yb) in enumerate(train_loader):
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            correct += (model(Xb).argmax(1) == yb).sum().item()
            total   += Xb.size(0)
            if i % 2 == 0:
                print(f"  [Epoch {epoch} Batch {i}] loss: {loss.item():.4f}", end='\r')
        print(" " * 50, end='\r')

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                vc += (model(Xb.to(DEVICE)).argmax(1) == yb.to(DEVICE)).sum().item()
                vt += Xb.size(0)
        val_acc = vc / vt
        print(f"{epoch:>4} | {correct/total:>10.4f} | {val_acc:>9.4f} | {time.time()-t0:>6.1f}s")

        if val_acc > best_val:
            best_val   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            all_p.extend(model(Xb.to(DEVICE)).argmax(1).cpu().numpy())
            all_l.extend(yb.numpy())

    test_acc = accuracy_score(all_l, all_p)
    macro_f1 = f1_score(all_l, all_p, average="macro")
    print(f"\n[RESULT] {tag}  |  Best Val: {best_val:.4f}  |  Test Acc: {test_acc:.4f}  |  Macro F1: {macro_f1:.4f}")
    print(classification_report(all_l, all_p, target_names=class_names, digits=3))

    if save_path:
        torch.save({"state_dict": best_state, "class_names": class_names,
                    "n_qubits": N_QUBITS, "n_layers": N_LAYERS,
                    "best_val_acc": best_val, "test_acc": test_acc},
                   save_path)
        print(f"[SAVED] {save_path}")

    return best_val, test_acc, macro_f1

# ─────────────────────────── 8. Run both ──────────────────────
cl_val, cl_test, cl_f1 = train_head(ClassicalHead(), "Classical Linear Head (baseline)")
qn_val, qn_test, qn_f1 = train_head(QuantumHead(),   "Quantum Head (VQC)",
                                     save_path="plant_qnn.pth")

print("\n" + "="*58)
print("  FINAL COMPARISON (same ResNet-18 features, 512-dim)")
print("="*58)
print(f"  Classical Linear : Val={cl_val:.4f}  Test={cl_test:.4f}  F1={cl_f1:.4f}")
print(f"  Quantum VQC      : Val={qn_val:.4f}  Test={qn_test:.4f}  F1={qn_f1:.4f}")
