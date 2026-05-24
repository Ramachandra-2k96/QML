"""
Quantum Hybrid NN — Two-Stage Crop Yield Prediction
====================================================
Dataset  : crop_yield.csv
Strategy : TWO-STAGE TRAINING
           1. Train the entire Hybrid QNN (Classical Pre + VQC + Classical Post)
              end-to-end for 30 epochs. This lets the quantum circuit adapt its
              rotations to the actual dataset distribution.
           2. Freeze the VQC and the Pre-layer. Pass the entire dataset through
              them to extract and cache the 'Quantum Features'.
           3. Train ONLY the Classical Post-layer on these static quantum
              features for many more epochs, getting maximum performance instantly!
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml

# ─────────────────────────── Config ───────────────────────────
SEED         = 42
BATCH_SIZE   = 256
STAGE1_EPOCHS = 30       # Slow quantum end-to-end training
STAGE2_EPOCHS = 40      # Fast classical fine-tuning
LR           = 5e-3
WEIGHT_DECAY = 1e-4
N_QUBITS     = 7
N_LAYERS     = 3
OUT_DIR      = "results/crop_yield"

os.makedirs(OUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUT_DIR, "crop_yield_qnn_model.pth")
CACHE_PATH = os.path.join(OUT_DIR, "quantum_features_cache.pt")

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device : {DEVICE}")
print(f"[INFO] Qubits : {N_QUBITS}  |  VQC layers : {N_LAYERS}")

# ─────────────────────────── 1. Data ──────────────────────────
df = pd.read_csv("crop_yield.csv")
df.drop(["Crop_Year", "Production"], axis=1, inplace=True)
for col in ["Crop", "Season", "State"]:
    df[col] = df[col].str.strip()

X = df.drop(columns=["Yield"])
y = np.log1p(df["Yield"].values)

X_train_df, X_temp_df, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=SEED)
X_val_df, X_test_df, y_val, y_test = train_test_split(X_temp_df, y_temp, test_size=0.50, random_state=SEED)

for df_ in [X_train_df, X_val_df, X_test_df]: df_ = df_.copy()
X_train_df, X_val_df, X_test_df = X_train_df.copy(), X_val_df.copy(), X_test_df.copy()

global_mean = float(y_train.mean())
for col in ["Crop", "Season", "State"]:
    mean_map = pd.Series(y_train, index=X_train_df.index).groupby(X_train_df[col]).mean()
    X_train_df[col] = X_train_df[col].map(mean_map).fillna(global_mean)
    X_val_df[col]   = X_val_df[col].map(mean_map).fillna(global_mean)
    X_test_df[col]  = X_test_df[col].map(mean_map).fillna(global_mean)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df).astype(np.float32)
X_val   = scaler.transform(X_val_df).astype(np.float32)
X_test  = scaler.transform(X_test_df).astype(np.float32)

y_train = y_train.astype(np.float32)
y_val   = y_val.astype(np.float32)
y_test  = y_test.astype(np.float32)

print(f"[INFO] Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

def make_loader(X, y, shuffle=False):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    
    # Ensure y is always [N, 1] without stacking extra dimensions
    y = y.view(-1, 1)
    
    return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val,   y_val)
test_loader  = make_loader(X_test,  y_test)

# ─────────────────────────── 2. Hybrid Model ──────────────────
try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    is_lightning = True
except Exception:
    dev = qml.device("default.qubit", wires=N_QUBITS)
    is_lightning = False

@qml.qnode(dev, diff_method="adjoint" if is_lightning else "best", interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

class HybridQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(N_QUBITS, N_QUBITS), nn.Tanh())
        self.qlayer = qlayer
        self.post = nn.Sequential(nn.Linear(N_QUBITS, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        x = self.pre(x) * torch.pi
        x = self.qlayer(x)
        return self.post(x)

    def extract_quantum_features(self, x):
        """Passes data through pre-layer and quantum layer only."""
        with torch.no_grad():
            x = self.pre(x) * torch.pi
            x = self.qlayer(x)
        return x

model = HybridQNN().to(DEVICE)
criterion = nn.MSELoss()

# ─────────────────────────── 3. STAGE 1: End-to-End Quantum ───
print("\n" + "="*50)
print(f" STAGE 1: End-to-End Quantum Training ({STAGE1_EPOCHS} Epochs)")
print("="*50)

opt1 = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=STAGE1_EPOCHS, eta_min=1e-5)

for epoch in range(1, STAGE1_EPOCHS + 1):
    model.train()
    running = 0.0
    pbar = tqdm(train_loader, desc=f"Stage 1 Epoch {epoch}/{STAGE1_EPOCHS}", leave=False)
    for Xb, yb in pbar:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        opt1.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        opt1.step()
        running += loss.item() * Xb.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    train_loss = running / len(train_loader.dataset)

    model.eval()
    val_run = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            val_run += criterion(model(Xb.to(DEVICE)), yb.to(DEVICE)).item() * Xb.size(0)
    val_loss = val_run / len(val_loader.dataset)
    sched1.step()

    print(f"Stage 1 Epoch {epoch:>2} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

# ─────────────────────────── 4. Extract & Cache Data ──────────
print("\n[INFO] STAGE 1 Complete. Freezing Quantum Layer & Extracting Features...")

def get_quantum_features(loader):
    features, labels = [], []
    model.eval()
    for Xb, yb in tqdm(loader, desc="Extracting", leave=False):
        features.append(model.extract_quantum_features(Xb.to(DEVICE)).cpu())
        labels.append(yb)
    return torch.cat(features), torch.cat(labels)

Q_train, yQ_train = get_quantum_features(train_loader)
Q_val,   yQ_val   = get_quantum_features(val_loader)
Q_test,  yQ_test  = get_quantum_features(test_loader)

# Save the extracted features to disk
torch.save({
    "Q_train": Q_train, "yQ_train": yQ_train,
    "Q_val": Q_val, "yQ_val": yQ_val,
    "Q_test": Q_test, "yQ_test": yQ_test
}, CACHE_PATH)
print(f"[INFO] Quantum features safely written to disk: {CACHE_PATH}")

# Load back from disk (to simulate disk-based reading, though they easily fit in RAM)
cache = torch.load(CACHE_PATH)
fast_train_loader = make_loader(cache["Q_train"], cache["yQ_train"], shuffle=True)
fast_val_loader   = make_loader(cache["Q_val"],   cache["yQ_val"])
fast_test_loader  = make_loader(cache["Q_test"],  cache["yQ_test"])

# ─────────────────────────── 5. STAGE 2: Classical Fine-Tune ──
print("\n" + "="*50)
print(f" STAGE 2: Fast Classical Fine-Tuning ({STAGE2_EPOCHS} Epochs)")
print("="*50)

# Optimizer strictly for the POST layer (Classical Regression Head)
opt2 = torch.optim.AdamW(model.post.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=STAGE2_EPOCHS, eta_min=1e-5)

best_val_loss = float("inf")
best_state = None

for epoch in range(1, STAGE2_EPOCHS + 1):
    model.post.train()
    running = 0.0
    for Qb, yb in fast_train_loader:
        Qb, yb = Qb.to(DEVICE), yb.to(DEVICE)
        opt2.zero_grad()
        loss = criterion(model.post(Qb), yb)  # Only pass through POST layer
        loss.backward()
        opt2.step()
        running += loss.item() * Qb.size(0)
    train_loss = running / len(fast_train_loader.dataset)

    model.post.eval()
    val_preds, val_trues, val_run = [], [], 0.0
    with torch.no_grad():
        for Qb, yb in fast_val_loader:
            Qb, yb = Qb.to(DEVICE), yb.to(DEVICE)
            p = model.post(Qb)
            val_run += criterion(p, yb).item() * Qb.size(0)
            val_preds.append(p.cpu()); val_trues.append(yb.cpu())
    
    val_loss = val_run / len(fast_val_loader.dataset)
    val_r2 = r2_score(torch.cat(val_trues).numpy(), torch.cat(val_preds).numpy())
    sched2.step()

    if epoch % 10 == 0 or epoch == 1:
        print(f"Stage 2 Epoch {epoch:>3} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val R²: {val_r2:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        torch.save({"state_dict": best_state, "n_qubits": N_QUBITS, "n_layers": N_LAYERS}, MODEL_PATH)

# ─────────────────────────── 6. Final Evaluation ──────────────
model.load_state_dict(best_state)
model.eval()

def evaluate(loader, name):
    preds, trues = [], []
    with torch.no_grad():
        for Qb, yb in loader:
            preds.append(model.post(Qb.to(DEVICE)).cpu())
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
evaluate(fast_train_loader, "Train")
evaluate(fast_val_loader,   "Validation")
evaluate(fast_test_loader,  "Test")
print(f"\n[DONE] Best model saved to: {MODEL_PATH}")