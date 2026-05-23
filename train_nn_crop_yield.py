"""
Task 1: Neural Network for Crop Yield Prediction
=================================================
Dataset  : crop_yield.csv
Target   : Yield (tonnes/hectare) — log1p transformed
Encoding : Target Encoding for categorical columns (Crop, Season, State)
Model    : Deep residual MLP with BatchNorm + Dropout
Training : Mixed Precision (AMP) via torch.cuda.amp
Saves    : crop_yield_nn_assets.pkl  (encoder maps, scaler, label info)
           crop_yield_nn_model.pth   (model weights + config)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast

# ─────────────────────────── Config ───────────────────────────
SEED = 42
BATCH_SIZE = 512
EPOCHS = 150
LR = 3e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 20          # early-stopping patience

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ─────────────────────────── 1. Load & Clean ──────────────────
df = pd.read_csv("crop_yield.csv")
print(f"[INFO] Dataset shape: {df.shape}")

df.drop(["Crop_Year", "Production"], axis=1, inplace=True)

for col in ["Crop", "Season", "State"]:
    df[col] = df[col].str.strip()

X = df.drop(columns=["Yield"])
y_raw = df["Yield"].values
y = np.log1p(y_raw)

# ─────────────────────────── 2. Split ─────────────────────────
X_train_df, X_temp_df, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED)
X_val_df, X_test_df, y_val, y_test = train_test_split(
    X_temp_df, y_temp, test_size=0.50, random_state=SEED)

X_train_df = X_train_df.copy()
X_val_df   = X_val_df.copy()
X_test_df  = X_test_df.copy()

# ─────────────────────────── 3. Target Encoding ───────────────
global_mean = float(y_train.mean())
target_maps = {}

for col in ["Crop", "Season", "State"]:
    mean_map = pd.Series(y_train, index=X_train_df.index).groupby(X_train_df[col]).mean()
    target_maps[col] = mean_map.to_dict()
    X_train_df[col] = X_train_df[col].map(mean_map).fillna(global_mean)
    X_val_df[col]   = X_val_df[col].map(mean_map).fillna(global_mean)
    X_test_df[col]  = X_test_df[col].map(mean_map).fillna(global_mean)

# ─────────────────────────── 4. Standard Scale ────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df).astype(np.float32)
X_val   = scaler.transform(X_val_df).astype(np.float32)
X_test  = scaler.transform(X_test_df).astype(np.float32)

y_train = y_train.astype(np.float32)
y_val   = y_val.astype(np.float32)
y_test  = y_test.astype(np.float32)

print(f"[INFO] Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# ─────────────────────────── 5. DataLoaders ───────────────────
def make_loader(X, y, shuffle=False):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=(DEVICE.type == "cuda"))

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val,   y_val)
test_loader  = make_loader(X_test,  y_test)

# ─────────────────────────── 6. Model ─────────────────────────
class ResBlock(nn.Module):
    """Residual block: Linear → BN → SiLU → Dropout → Linear → BN, + skip."""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class CropYieldNN(nn.Module):
    def __init__(self, in_features, hidden=512, n_blocks=6, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


IN_FEATURES = X_train.shape[1]
model = CropYieldNN(in_features=IN_FEATURES, hidden=512, n_blocks=6, dropout=0.2).to(DEVICE)
print(f"[INFO] Model params: {sum(p.numel() for p in model.parameters()):,}")

# ─────────────────────────── 7. Training ──────────────────────
criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

use_amp = DEVICE.type == "cuda"
amp_scaler = GradScaler(device='cuda' if use_amp else 'cpu', enabled=use_amp)

best_val_loss = float("inf")
patience_ctr  = 0
best_state    = None

print("\n[INFO] Starting training …")
print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>10} | {'Val R²':>8}")
print("-" * 48)

for epoch in range(1, EPOCHS + 1):
    # ── train
    model.train()
    running = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        with autocast('cuda' if use_amp else 'cpu'):
            pred = model(Xb)
            loss = criterion(pred, yb)
        amp_scaler.scale(loss).backward()
        amp_scaler.step(optimizer)
        amp_scaler.update()
        running += loss.item() * Xb.size(0)
    train_loss = running / len(train_loader.dataset)

    # ── validate
    model.eval()
    val_preds, val_trues = [], []
    val_running = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            with autocast('cuda' if use_amp else 'cpu'):
                pred = model(Xb)
                loss = criterion(pred, yb)
            val_running += loss.item() * Xb.size(0)
            val_preds.append(pred.cpu().numpy())
            val_trues.append(yb.cpu().numpy())
    val_loss = val_running / len(val_loader.dataset)
    val_preds = np.concatenate(val_preds).ravel()
    val_trues = np.concatenate(val_trues).ravel()
    val_r2    = r2_score(val_trues, val_preds)

    scheduler.step()

    if epoch % 10 == 0 or epoch == 1:
        print(f"{epoch:>6} | {train_loss:>11.6f} | {val_loss:>10.6f} | {val_r2:>8.4f}")

    # ── early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_ctr  = 0
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\n[INFO] Early stopping at epoch {epoch}.")
            break

# ─────────────────────────── 8. Evaluate ──────────────────────
model.load_state_dict(best_state)
model.eval()

def evaluate(loader, split_name):
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE)
            with autocast('cuda' if use_amp else 'cpu'):
                p = model(Xb)
            preds.append(p.cpu().numpy())
            trues.append(yb.numpy())
    preds = np.concatenate(preds).ravel()
    trues = np.concatenate(trues).ravel()
    preds_raw = np.expm1(preds)
    trues_raw = np.expm1(trues)
    r2_log = r2_score(trues, preds)
    r2_raw = r2_score(trues_raw, preds_raw)
    mae    = mean_absolute_error(trues_raw, preds_raw)
    rmse   = np.sqrt(mean_squared_error(trues_raw, preds_raw))
    print(f"{split_name:<12} {r2_log:>+10.4f} {r2_raw:>+10.4f} {mae:>12.3f} {rmse:>13.3f}")

print(f"\n{'Split':<12} {'R² (log)':>10} {'R² (raw)':>10} {'MAE (t/ha)':>12} {'RMSE (t/ha)':>13}")
print("-" * 60)
evaluate(train_loader, "Train")
evaluate(val_loader,   "Validation")
evaluate(test_loader,  "Test")

# ─────────────────────────── 9. Save assets ───────────────────
# 9a — model config + weights
model_cfg = {"in_features": IN_FEATURES, "hidden": 512, "n_blocks": 6, "dropout": 0.2}
torch.save({"config": model_cfg, "state_dict": best_state}, "crop_yield_nn_model.pth")

# 9b — preprocessing assets
assets = {
    "scaler": scaler,
    "target_maps": target_maps,
    "global_mean": global_mean,
}
with open("crop_yield_nn_assets.pkl", "wb") as f:
    pickle.dump(assets, f)

print("\n[DONE] Saved:")
print("  • crop_yield_nn_model.pth")
print("  • crop_yield_nn_assets.pkl")
