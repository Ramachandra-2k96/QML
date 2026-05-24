import torch
import torch.nn as nn
import pennylane as qml

# ─────────────────────────── CLASSICAL NN ───────────────────────────
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

class CropYieldNN(nn.Module):
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

# ─────────────────────────── HYBRID QNN ───────────────────────────
N_QUBITS = 7
N_LAYERS = 3

dev = qml.device("default.qubit", wires=N_QUBITS)
@qml.qnode(dev, interface="torch")
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

# ─────────────────────────── CALCULATION ───────────────────────────
if __name__ == "__main__":
    nn_model = CropYieldNN()
    qnn_model = HybridQNN()

    nn_total = sum(p.numel() for p in nn_model.parameters())
    
    q_params  = sum(p.numel() for p in qnn_model.qlayer.parameters())
    cl_params = sum(p.numel() for p in qnn_model.pre.parameters()) + sum(p.numel() for p in qnn_model.post.parameters())
    qnn_total = q_params + cl_params

    print("=" * 60)
    print(" 🧠 CLASSICAL MODEL (CropYieldNN)")
    print("=" * 60)
    print(f"  • Network architecture : 6-Block Residual MLP (Hidden=512)")
    print(f"  • Total Parameters   : {nn_total:,}")
    print("\n" + "=" * 60)
    print(" 🌌 HYBRID QUANTUM MODEL (HybridQNN)")
    print("=" * 60)
    print(f"  • Network architecture : Pre-layer -> 7-Qubit VQC -> Post-layer")
    print(f"  • Quantum Parameters   : {q_params:,}")
    print(f"  • Classical Parameters : {cl_params:,}")
    print(f"  • Total Parameters     : {qnn_total:,}")
    print("=" * 60)
    
    ratio = nn_total / qnn_total
    print(f"\n[CONCLUSION] The Classical Model has {ratio:,.1f}x MORE parameters than the Quantum Hybrid Model!\n")
