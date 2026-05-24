"""
Ablation Study: Quantum Features + Large Classical Neural Network
=================================================================
This script takes the exact 3.3M parameter classical architecture from 
train_nn_crop_yield.py but trains it EXCLUSIVELY on the Quantum Features 
extracted and cached by the QNN!

Since the quantum features were already pre-computed and saved to disk 
(quantum_features_cache.pt), this script bypasses PennyLane entirely. 
It trains a massive deep neural network on the quantum-embedded data 
to see if we can achieve an even higher R² score.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ─────────────────────────── Config ───────────────────────────
SEED         = 42
BATCH_SIZE   = 256
EPOCHS       = 150
LR           = 3e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 20
OUT_DIR      = "results/crop_yield"
CACHE_PATH   = os.path.join(OUT_DIR, "quantum_features_cache.pt")
MODEL_PATH   = os.path.join(OUT_DIR, "crop_yield_qnn_large_model.pth")

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device : {DEVICE}")

# ─────────────────────────── 1. Load Cached Quantum Data ──────
if not os.path.exists(CACHE_PATH):
    raise FileNotFoundError(f"Cannot find {CACHE_PATH}! You must run the QNN extraction first.")

print(f"[INFO] Loading pre-extracted quantum features from {CACHE_PATH}...")
cache = torch.load(CACHE_PATH, map_location="cpu")

# Convert labels to 2D for MSE calculation if they aren't already
def ensure_2d(tensor):
    return tensor.view(-1, 1) if tensor.dim() == 1 else tensor

Q_train = cache["Q_train"]
yQ_train = ensure_2d(cache["yQ_train"])
Q_val = cache["Q_val"]
yQ_val = ensure_2d(cache["yQ_val"])
Q_test = cache["Q_test"]
yQ_test = ensure_2d(cache["yQ_test"])

print(f"[INFO] Quantum Train shape : {Q_train.shape}")
print(f"[INFO] Quantum Val shape   : {Q_val.shape}")

train_loader = DataLoader(TensorDataset(Q_train, yQ_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(Q_val, yQ_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(Q_test, yQ_test), batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────── 2. Large Classical Architecture ──
class ResBlock(nn.Module):
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

class LargeCropYieldNN(nn.Module):
    """The exact architecture from train_nn_crop_yield.py"""
    def __init__(self, in_features=7, hidden=512, n_blocks=6, dropout=0.2):
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

model = LargeCropYieldNN().to(DEVICE)
print(f"[INFO] Large Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

# ─────────────────────────── 3. Training Loop ─────────────────
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

best_val_loss = float("inf")
patience_ctr  = 0
best_state    = None

print("\n" + "="*50)
print(f" FAST CLASSICAL TRAINING ON QUANTUM FEATURES")
print("="*50)

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for Qb, yb in train_loader:
        Qb, yb = Qb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Qb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * Qb.size(0)
    train_loss = running / len(train_loader.dataset)

    model.eval()
    val_run = 0.0
    val_preds, val_trues = [], []
    with torch.no_grad():
        for Qb, yb in val_loader:
            Qb, yb = Qb.to(DEVICE), yb.to(DEVICE)
            p = model(Qb)
            val_run += criterion(p, yb).item() * Qb.size(0)
            val_preds.append(p.cpu())
            val_trues.append(yb.cpu())
            
    val_loss = val_run / len(val_loader.dataset)
    val_r2 = r2_score(torch.cat(val_trues).numpy(), torch.cat(val_preds).numpy())
    scheduler.step()

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:>3}/{EPOCHS} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val R²: {val_r2:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_ctr = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        torch.save(best_state, MODEL_PATH)
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\n[INFO] Early stopping triggered at epoch {epoch}.")
            break

# ─────────────────────────── 4. Final Evaluation ──────────────
model.load_state_dict(best_state)
model.eval()

def evaluate(loader, name):
    preds, trues = [], []
    with torch.no_grad():
        for Qb, yb in loader:
            preds.append(model(Qb.to(DEVICE)).cpu())
            trues.append(yb)
    p = torch.cat(preds).numpy().ravel()
    t = torch.cat(trues).numpy().ravel()
    r2_log = r2_score(t, p)
    r2_raw = r2_score(np.expm1(t), np.expm1(p))
    mae    = mean_absolute_error(np.expm1(t), np.expm1(p))
    rmse   = np.sqrt(mean_squared_error(np.expm1(t), np.expm1(p)))
    print(f"{name:<12} {r2_log:>+10.4f} {r2_raw:>+10.4f} {mae:>12.3f} {rmse:>13.3f}")

print(f"\n{'Split':<12} {'R² (log)':>10} {'R² (raw)':>10} {'MAE (t/ha)':>12} {'RMSE (t/ha)':>13}")
print("-" * 60)
evaluate(train_loader, "Train")
evaluate(val_loader,   "Validation")
evaluate(test_loader,  "Test")
print(f"\n[DONE] Best large Q-Hybrid model saved to: {MODEL_PATH}")
