"""
Quantum Hybrid NN — Crop Yield Prediction
==========================================
Dataset  : crop_yield.csv  (same preprocessing as classical baseline)
QML      : PennyLane VQC embedded as a TorchLayer
           • 7 qubits  (one per input feature)
           • AngleEmbedding  →  StronglyEntanglingLayers (3 layers)
           • PauliZ expectation values  →  classical regression head
Saves    : crop_yield_qnn_model.pth   (weights + config)
           crop_yield_nn_assets.pkl   (shared with classical — same preprocessing)
"""

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
SEED        = 42
BATCH_SIZE  = 256
EPOCHS      = 120
LR          = 5e-3
WEIGHT_DECAY = 1e-4
PATIENCE    = 20
N_QUBITS    = 7       # one per input feature
N_LAYERS    = 3       # StronglyEntanglingLayers depth

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device : {DEVICE}")
print(f"[INFO] Qubits : {N_QUBITS}  |  VQC layers : {N_LAYERS}")

# ─────────────────────────── 1. Data (identical to classical) ──
df = pd.read_csv("crop_yield.csv")
df.drop(["Crop_Year", "Production"], axis=1, inplace=True)
for col in ["Crop", "Season", "State"]:
    df[col] = df[col].str.strip()

X     = df.drop(columns=["Yield"])
y_raw = df["Yield"].values
y     = np.log1p(y_raw)

X_train_df, X_temp_df, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED)
X_val_df, X_test_df, y_val, y_test = train_test_split(
    X_temp_df, y_temp, test_size=0.50, random_state=SEED)

for df_ in [X_train_df, X_val_df, X_test_df]:
    df_ = df_.copy()   # already returned as copies from split

X_train_df = X_train_df.copy()
X_val_df   = X_val_df.copy()
X_test_df  = X_test_df.copy()

global_mean = float(y_train.mean())
target_maps = {}
for col in ["Crop", "Season", "State"]:
    mean_map = pd.Series(y_train, index=X_train_df.index).groupby(X_train_df[col]).mean()
    target_maps[col] = mean_map.to_dict()
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
    ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val,   y_val)
test_loader  = make_loader(X_test,  y_test)

# ─────────────────────────── 2. Quantum Circuit ────────────────
# Try fast Lightning simulator, fall back to default
try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    print("[INFO] Using lightning.qubit")
except Exception:
    dev = qml.device("default.qubit", wires=N_QUBITS)
    print("[INFO] Using default.qubit")

@qml.qnode(dev, diff_method="adjoint" if "lightning" in dev.name else "best",
           interface="torch")
def quantum_circuit(inputs, weights):
    # Encode scaled features as rotation angles
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    # Variational ansatz
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    # Measure each qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# ─────────────────────────── 3. Hybrid Model ──────────────────
class HybridQNN(nn.Module):
    """
    Classical pre-layer → VQC (7 qubits) → Classical head.

    Pre-layer compresses + scales to [-π, π] suitable for angle encoding.
    Post-layer regresses to log-yield from quantum measurement outputs.
    """
    def __init__(self):
        super().__init__()
        # Map 7 features → 7 angles in [-π, π]
        self.pre = nn.Sequential(
            nn.Linear(N_QUBITS, N_QUBITS),
            nn.Tanh(),          # output in (-1, 1)
        )
        self.qlayer = qlayer
        # Quantum outputs → regression
        self.post = nn.Sequential(
            nn.Linear(N_QUBITS, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # Scale to (-π, π) for AngleEmbedding
        x = self.pre(x) * torch.pi
        x = self.qlayer(x)          # [B, N_QUBITS]  expectation values ∈ (-1,1)
        return self.post(x)

model = HybridQNN().to(DEVICE)
q_params  = sum(p.numel() for p in model.qlayer.parameters())
cl_params = sum(p.numel() for p in model.pre.parameters()) + \
            sum(p.numel() for p in model.post.parameters())
print(f"[INFO] Quantum params : {q_params}   Classical params : {cl_params}")

# ─────────────────────────── 4. Training ──────────────────────
criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5)

best_val_loss = float("inf")
patience_ctr  = 0
best_state    = None

print(f"\n{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>10} | {'Val R²':>8}")
print("-" * 48)

for epoch in range(1, EPOCHS + 1):
    # train
    model.train()
    running = 0.0
    
    # Wrap dataloader with tqdm for real-time progress
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
    for Xb, yb in train_pbar:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * Xb.size(0)
        train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    train_loss = running / len(train_loader.dataset)

    # validate
    model.eval()
    val_preds, val_trues, val_running = [], [], 0.0
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", leave=False)
    with torch.no_grad():
        for Xb, yb in val_pbar:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            p = model(Xb)
            loss = criterion(p, yb)
            val_running += loss.item() * Xb.size(0)
            val_preds.append(p.cpu()); val_trues.append(yb.cpu())
            val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    val_loss = val_running / len(val_loader.dataset)
    val_r2   = r2_score(
        torch.cat(val_trues).numpy(), torch.cat(val_preds).numpy())

    scheduler.step()

    if epoch % 10 == 0 or epoch == 1:
        print(f"{epoch:>6} | {train_loss:>11.6f} | {val_loss:>10.6f} | {val_r2:>8.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss; patience_ctr = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        # Save to disk immediately
        torch.save({"state_dict": best_state,
                    "n_qubits": N_QUBITS, "n_layers": N_LAYERS},
                   "crop_yield_qnn_model.pth")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\n[INFO] Early stopping at epoch {epoch}.")
            break

# ─────────────────────────── 5. Evaluate ──────────────────────
model.load_state_dict(best_state)
model.eval()

def evaluate(loader, name):
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            preds.append(model(Xb.to(DEVICE)).cpu())
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

# ─────────────────────────── 6. Save ──────────────────────────
print(f"\n[DONE] Best model was saved intermediately to: crop_yield_qnn_model.pth")