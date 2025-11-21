from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from ...ml.fallnet_model import FallNet, FallNetConfig  # type: ignore


class RadarSequence(BaseModel):
    """One radar episode: frames x features."""

    data: List[List[float]]


class PredictRequest(BaseModel):
    sequences: List[RadarSequence]


class PredictResponseItem(BaseModel):
    label: str
    prob_fall: float
    prob_normal: float


class PredictResponse(BaseModel):
    results: List[PredictResponseItem]


app = FastAPI(title="mmWave Fall Detection Demo API", version="0.1.0")


_model: FallNet | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model() -> FallNet:
    global _model
    if _model is not None:
        return _model

    model_path = Path(os.getenv("MODEL_PATH", "ml/fallnet.pt"))
    if not model_path.exists():
        raise RuntimeError(
            f"Model file not found at {model_path}. Did you run ml/train_fallnet.py?"
        )

    checkpoint = torch.load(model_path, map_location=_device)
    cfg = FallNetConfig(**checkpoint["config"])
    model = FallNet(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_device)
    model.eval()
    _model = model
    return model


@app.get("/health" )
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = _load_model()

    arrs = [np.array(seq.data, dtype="float32") for seq in req.sequences]
    max_frames = max(a.shape[0] for a in arrs)
    features = arrs[0].shape[1]

    # Pad sequences to the same length (simple zero padding).
    padded = np.zeros((len(arrs), max_frames, features), dtype="float32")
    for i, a in enumerate(arrs):
        padded[i, : a.shape[0]] = a

    x = torch.from_numpy(padded).to(_device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    results: List[PredictResponseItem] = []
    for p in probs:
        prob_normal = float(p[0])
        prob_fall = float(p[1])
        label = "fall" if prob_fall >= prob_normal else "normal"
        results.append(
            PredictResponseItem(
                label=label,
                prob_fall=prob_fall,
                prob_normal=prob_normal,
            )
        )

    return PredictResponse(results=results)
