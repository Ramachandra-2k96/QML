"""
FastAPI — Agricultural ML Inference Server
==========================================
Two endpoints:
  POST /predict/yield   → Crop Yield prediction (Classical NN or Quantum Hybrid)
  POST /predict/disease → Plant Disease classification (ResNet-18 or MobileNet-V3-Small)

Run with:
  uv run uvicorn main:app --reload
"""

import io
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from PIL import Image
from contextlib import asynccontextmanager
from torchvision import transforms, models

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal

# ─────────────────────────── Paths ────────────────────────────
CROP_DIR    = "results/crop_yield"
PLANT_DIR   = "results/plant_disease"

NN_MODEL_PATH    = os.path.join(CROP_DIR, "crop_yield_nn_model.pth")
NN_ASSETS_PATH   = os.path.join(CROP_DIR, "crop_yield_nn_assets.pkl")
QNN_MODEL_PATH   = os.path.join(CROP_DIR, "crop_yield_qnn_model.pth")
QNN_LARGE_PATH   = os.path.join(CROP_DIR, "crop_yield_qnn_large_model.pth")
QNN_CACHE_PATH   = os.path.join(CROP_DIR, "quantum_features_cache.pt")

RESNET_PATH      = os.path.join(PLANT_DIR, "plant_resnet18.pth")
MOBILENET_PATH   = os.path.join(PLANT_DIR, "plant_mobilenet_v3s.pth")
CLASS_NAMES_PATH = os.path.join(PLANT_DIR, "plant_class_names.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_PLANT_CLASSES = 38
IMG_SIZE = 224

# ─────────────────────────── Global Model Store ───────────────
store = {}

# ─────────────────────────── Architecture Definitions ─────────

# — Classical Crop Yield NN —
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class CropYieldNN(nn.Module):
    def __init__(self, in_features=7, hidden=512, n_blocks=6, dropout=0.2):
        super().__init__()
        self.stem   = nn.Sequential(nn.Linear(in_features, hidden), nn.BatchNorm1d(hidden), nn.SiLU())
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head   = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 256), nn.SiLU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


# — Quantum + Large Classical Yield NN —
def build_quantum_layer(n_qubits=7, n_layers=3):
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        diff = "adjoint"
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)
        diff = "best"

    @qml.qnode(dev, diff_method=diff, interface="torch")
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return qml.qnn.TorchLayer(quantum_circuit, weight_shapes)


class HybridQNNSmall(nn.Module):
    """Original small QNN (Pre → VQC → tiny post head). Loaded from crop_yield_qnn_model.pth."""
    def __init__(self, n_qubits=7, qlayer=None):
        super().__init__()
        self.pre    = nn.Sequential(nn.Linear(n_qubits, n_qubits), nn.Tanh())
        self.qlayer = qlayer
        self.post   = nn.Sequential(nn.Linear(n_qubits, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        x = self.pre(x) * torch.pi
        x = self.qlayer(x)
        return self.post(x)


class LargeCropYieldNN(nn.Module):
    """3.3M param classical head that runs on quantum features."""
    def __init__(self, in_features=7, hidden=512, n_blocks=6, dropout=0.2):
        super().__init__()
        self.stem   = nn.Sequential(nn.Linear(in_features, hidden), nn.BatchNorm1d(hidden), nn.SiLU())
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head   = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 256), nn.SiLU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class EndToEndHybridQNN(nn.Module):
    """Full end-to-end: Pre → VQC → LargeClassical. Uses weights from both .pth files."""
    def __init__(self, qlayer):
        super().__init__()
        self.pre    = nn.Sequential(nn.Linear(7, 7), nn.Tanh())
        self.qlayer = qlayer
        self.large_classical = LargeCropYieldNN(in_features=7)

    def forward(self, x):
        x = self.pre(x) * torch.pi
        x = self.qlayer(x)
        return self.large_classical(x)


# ─────────────────────────── Image Transform ──────────────────
infer_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────── Lifespan (load models once) ──────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Loading models...")

    # 1. Classical crop yield NN
    with open(NN_ASSETS_PATH, "rb") as f:
        assets = pickle.load(f)
    store["scaler"]      = assets["scaler"]
    store["target_maps"] = assets["target_maps"]
    store["global_mean"] = assets.get("global_mean", None)

    nn_model = CropYieldNN().to(DEVICE)
    ckpt = torch.load(NN_MODEL_PATH, map_location=DEVICE)
    state = ckpt.get("state_dict", ckpt)
    nn_model.load_state_dict(state)
    nn_model.eval()
    store["nn_model"] = nn_model

    # 2. Small Quantum Hybrid QNN (for standalone QNN endpoint)
    qnn_ckpt = torch.load(QNN_MODEL_PATH, map_location=DEVICE)
    n_qubits = qnn_ckpt.get("n_qubits", 7)
    n_layers = qnn_ckpt.get("n_layers", 3)
    qlayer_small = build_quantum_layer(n_qubits, n_layers)
    small_qnn = HybridQNNSmall(n_qubits=n_qubits, qlayer=qlayer_small).to(DEVICE)
    small_qnn.load_state_dict(qnn_ckpt["state_dict"])
    small_qnn.eval()
    store["small_qnn"] = small_qnn

    # 3. End-to-End Hybrid (Large head on quantum features)
    qlayer_large = build_quantum_layer(n_qubits, n_layers)
    e2e_model = EndToEndHybridQNN(qlayer=qlayer_large).to(DEVICE)

    # Load pre + qlayer weights from small QNN checkpoint
    pre_dict    = {k.replace("pre.", "", 1): v for k, v in qnn_ckpt["state_dict"].items() if k.startswith("pre.")}
    qlayer_dict = {k.replace("qlayer.", "", 1): v for k, v in qnn_ckpt["state_dict"].items() if k.startswith("qlayer.")}
    e2e_model.pre.load_state_dict(pre_dict)
    e2e_model.qlayer.load_state_dict(qlayer_dict)

    # Load large classical head weights
    large_state = torch.load(QNN_LARGE_PATH, map_location=DEVICE)
    e2e_model.large_classical.load_state_dict(large_state)
    e2e_model.eval()
    store["e2e_qnn"] = e2e_model

    # 4. Plant disease — ResNet-18
    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, NUM_PLANT_CLASSES)
    resnet_ckpt = torch.load(RESNET_PATH, map_location=DEVICE)
    resnet.load_state_dict(resnet_ckpt["state_dict"])
    resnet = resnet.to(DEVICE).eval()
    store["resnet18"] = resnet

    # 5. Plant disease — MobileNet-V3-Small
    mobilenet = models.mobilenet_v3_small(weights=None)
    mobilenet.classifier[3] = nn.Linear(mobilenet.classifier[3].in_features, NUM_PLANT_CLASSES)
    mobilenet_ckpt = torch.load(MOBILENET_PATH, map_location=DEVICE)
    mobilenet.load_state_dict(mobilenet_ckpt["state_dict"])
    mobilenet = mobilenet.to(DEVICE).eval()
    store["mobilenet_v3s"] = mobilenet

    # 6. Class names
    with open(CLASS_NAMES_PATH, "rb") as f:
        store["class_names"] = pickle.load(f)

    print(f"[INFO] All models loaded on {DEVICE}. Server ready!")
    yield
    store.clear()


app = FastAPI(
    title="Agricultural ML Inference API",
    description="Crop Yield Prediction (Classical NN / Quantum Hybrid) + Plant Disease Classification",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────── Pydantic Schemas ─────────────────
class YieldInput(BaseModel):
    # Raw categorical strings
    crop: str
    season: str
    state: str
    # Numerical features
    area: float            # hectares
    annual_rainfall: float # mm
    fertilizer: float      # kg/ha
    pesticide: float       # kg/ha
    model: Literal["classical", "qnn_small", "qnn_large"] = "classical"


class YieldResponse(BaseModel):
    model_used: str
    predicted_yield_log: float
    predicted_yield_tonne_per_ha: float


class DiseaseResponse(BaseModel):
    model_used: str
    predicted_class: str
    confidence: float


# ─────────────────────────── Preprocessing Helper ─────────────
def preprocess_yield(data: YieldInput) -> torch.Tensor:
    """Apply target encoding + StandardScaler to match training pipeline."""
    target_maps = store["target_maps"]
    scaler      = store["scaler"]
    global_mean = store.get("global_mean")

    # Get global mean from training (stored when available, else default)
    gm = global_mean if global_mean is not None else 5.5   # rough log1p mean

    crop_enc   = target_maps["Crop"].get(data.crop.strip(),     gm)
    season_enc = target_maps["Season"].get(data.season.strip(), gm)
    state_enc  = target_maps["State"].get(data.state.strip(),   gm)

    raw = np.array([[
        crop_enc, season_enc, state_enc,
        data.area, data.annual_rainfall,
        data.fertilizer, data.pesticide
    ]], dtype=np.float32)

    scaled = scaler.transform(raw).astype(np.float32)
    return torch.tensor(scaled).to(DEVICE)


# ─────────────────────────── Endpoints ────────────────────────

@app.post("/predict/yield", response_model=YieldResponse, tags=["Crop Yield"])
async def predict_yield(data: YieldInput):
    """
    Predict crop yield from tabular features.

    - **model=classical** → 3.3M param Deep Residual MLP  
    - **model=qnn_small** → 7-qubit VQC + tiny classical head (408 params)  
    - **model=qnn_large** → 7-qubit VQC + 3.3M classical residual head (R²≈0.91)
    """
    try:
        x = preprocess_yield(data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {e}")

    with torch.no_grad():
        if data.model == "classical":
            pred_log = store["nn_model"](x).item()
        elif data.model == "qnn_small":
            pred_log = store["small_qnn"](x).item()
        elif data.model == "qnn_large":
            pred_log = store["e2e_qnn"](x).item()
        else:
            raise HTTPException(status_code=400, detail="Invalid model choice.")

    pred_raw = float(np.expm1(pred_log))

    return YieldResponse(
        model_used=data.model,
        predicted_yield_log=round(pred_log, 6),
        predicted_yield_tonne_per_ha=round(pred_raw, 4),
    )


@app.post("/predict/disease", response_model=DiseaseResponse, tags=["Plant Disease"])
async def predict_disease(
    file: UploadFile = File(..., description="Plant leaf image (jpg/png)"),
    model: Literal["resnet18", "mobilenet_v3_small"] = "resnet18",
):
    """
    Classify plant leaf disease from an uploaded image.

    - **model=resnet18** → ResNet-18 fine-tuned on PlantVillage (38 classes)  
    - **model=mobilenet_v3_small** → MobileNet-V3-Small fine-tuned on PlantVillage
    """
    # Validate image content type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Uploaded file must be an image (jpg/png).")

    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = infer_tfm(img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image processing error: {e}")

    key = "resnet18" if model == "resnet18" else "mobilenet_v3s"
    with torch.no_grad():
        logits = store[key](tensor)
        probs  = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)

    class_name = store["class_names"][idx.item()]
    confidence = round(conf.item() * 100, 2)

    return DiseaseResponse(
        model_used=model,
        predicted_class=class_name,
        confidence=confidence,
    )


@app.get("/", tags=["Info"])
async def root():
    return {
        "status": "running",
        "endpoints": {
            "yield_prediction": "POST /predict/yield",
            "disease_detection": "POST /predict/disease",
            "docs": "/docs",
        }
    }
