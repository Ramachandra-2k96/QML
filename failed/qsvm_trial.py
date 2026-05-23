import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVR

# ─── 1. Load & preprocess ───────────────────────────────────────────────────
dataset = pd.read_csv("crop_yield.csv")
dataset.drop(["Crop_Year", "Production"], inplace=True, axis=1)
for col in ["Crop", "Season", "State"]:
    dataset[col] = dataset[col].str.strip()

# Label encode categoricals (neutral integer codes — no target leakage)
le_crop   = LabelEncoder()
le_season = LabelEncoder()
le_state  = LabelEncoder()
dataset["Crop"]   = le_crop.fit_transform(dataset["Crop"])
dataset["Season"] = le_season.fit_transform(dataset["Season"])
dataset["State"]  = le_state.fit_transform(dataset["State"])

X = dataset.drop(columns=["Yield"])
y = np.log1p(dataset["Yield"].values)

# ─── 2. Split ────────────────────────────────────────────────────────────────
X_train_df, X_temp_df, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val_df, X_test_df, y_val, y_test = train_test_split(
    X_temp_df, y_temp, test_size=0.50, random_state=42
)

# ─── 3. MinMaxScaler to [0, π] (correct for quantum rotation angle encoding) ─
# Quantum gates use feature values as rotation angles — must be in [0, π]
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train_scaled = scaler.fit_transform(X_train_df)
X_val_scaled   = scaler.transform(X_val_df)
X_test_scaled  = scaler.transform(X_test_df)

# ─── 4. PCA to N_QUBITS dims ─────────────────────────────────────────────────
N_QUBITS = 4
pca = PCA(n_components=N_QUBITS, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca   = pca.transform(X_val_scaled)
X_test_pca  = pca.transform(X_test_scaled)

# Re-scale PCA output back to [0, π] since PCA can shift the range
# clip val/test in case their PCA projections fall outside the training min/max
pca_scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train_pca = pca_scaler.fit_transform(X_train_pca)
X_val_pca   = np.clip(pca_scaler.transform(X_val_pca),  0, np.pi)
X_test_pca  = np.clip(pca_scaler.transform(X_test_pca), 0, np.pi)

print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.4f}")
print(f"Train feature range after PCA rescale: [{X_train_pca.min():.3f}, {X_train_pca.max():.3f}]")
print(f"Val   feature range after clip:         [{X_val_pca.min():.3f}, {X_val_pca.max():.3f}]")
print(f"Test  feature range after clip:         [{X_test_pca.min():.3f}, {X_test_pca.max():.3f}]")

# ─── 5. Subsample ALL splits ─────────────────────────────────────────────────
N_SUBSAMPLE = 150
rng = np.random.default_rng(42)

idx_train = rng.choice(len(X_train_pca), N_SUBSAMPLE, replace=False)
X_q_train = X_train_pca[idx_train];  y_q_train = y_train[idx_train]

idx_val   = rng.choice(len(X_val_pca), N_SUBSAMPLE, replace=False)
X_q_val   = X_val_pca[idx_val];      y_q_val   = y_val[idx_val]

idx_test  = rng.choice(len(X_test_pca), N_SUBSAMPLE, replace=False)
X_q_test  = X_test_pca[idx_test];    y_q_test  = y_test[idx_test]

# ─── 6. Sweep C values (select best by val R² only — no test snooping) ──────
configs = [
    dict(C=0.1,  epsilon=0.5, reps=1),
    dict(C=1.0,  epsilon=0.3, reps=1),
    dict(C=5.0,  epsilon=0.2, reps=1),
    dict(C=10.0, epsilon=0.1, reps=1),
    dict(C=1.0,  epsilon=0.3, reps=2),
]

print(f"\n{'Config':<30} {'Train R²':>9} {'Val R²':>9}")
print("-" * 52)

best_val_r2 = -np.inf
best_qsvr   = None
best_cfg    = None

for i, cfg in enumerate(configs, 1):
    label = f"C={cfg['C']}, eps={cfg['epsilon']}, reps={cfg['reps']}"
    print(f"[{i}/{len(configs)}] Fitting {label}...", flush=True)
    fm = zz_feature_map(feature_dimension=N_QUBITS, reps=cfg["reps"])
    qk = FidelityQuantumKernel(feature_map=fm)
    q  = QSVR(quantum_kernel=qk, C=cfg["C"], epsilon=cfg["epsilon"])
    q.fit(X_q_train, y_q_train)

    r2_tr  = r2_score(y_q_train, q.predict(X_q_train))
    r2_val = r2_score(y_q_val,   q.predict(X_q_val))
    print(f"{label:<30} {r2_tr:>+9.4f} {r2_val:>+9.4f}")

    if r2_val > best_val_r2:
        best_val_r2 = r2_val
        best_cfg    = cfg
        best_qsvr   = q

# ─── 7. Best model full eval ─────────────────────────────────────────────────
print(f"\nBest config: {best_cfg}  (val R²={best_val_r2:+.4f})")
print(f"\n{'Split':<14} {'R²(log)':>9} {'R²(raw)':>9} {'MAE':>10} {'RMSE':>10}")
print("-" * 56)
for name, y_true, X_s in [
    ("Train(sub)",  y_q_train, X_q_train),
    ("Val(sub)",    y_q_val,   X_q_val),
    ("Test(sub)",   y_q_test,  X_q_test),
]:
    y_pred_log = best_qsvr.predict(X_s)
    y_true_raw = np.expm1(y_true)
    y_pred_raw = np.expm1(y_pred_log)
    print(
        f"{name:<14}"
        f" {r2_score(y_true, y_pred_log):>+9.4f}"
        f" {r2_score(y_true_raw, y_pred_raw):>+9.4f}"
        f" {mean_absolute_error(y_true_raw, y_pred_raw):>10.2f}"
        f" {np.sqrt(mean_squared_error(y_true_raw, y_pred_raw)):>10.2f}"
    )

