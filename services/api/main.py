from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from ...ml.fallnet_model import FallNet, FallNetConfig  # type: ignore


class Position(BaseModel):
    x: float
    y: float


class RadarSequence(BaseModel):
    """One radar episode: frames x features."""

    data: List[List[float]]


class PredictRequest(BaseModel):
    sequences: List[RadarSequence]


class PredictResponseItem(BaseModel):
    label: str
    probabilities: Dict[str, float]


class PredictResponse(BaseModel):
    results: List[PredictResponseItem]


class FallEvent(BaseModel):
    id: str
    timestamp: float  # unix seconds
    label: str
    probabilities: Dict[str, float]
    position: Position


class PredictionWithPosition(BaseModel):
    sequence: RadarSequence
    position: Position


app = FastAPI(title="mmWave Fall Detection Demo API", version="0.3.0")


_model: FallNet | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_LABEL_MAP: Dict[int, str] | None = None
_EVENTS: List[FallEvent] = []
_MAX_EVENTS = 200


def _load_model() -> FallNet:
    global _model, _LABEL_MAP
    if _model is not None:
        return _model

    model_path = Path(os.getenv("MODEL_PATH", "ml/fallnet.pt"))
    if not model_path.exists():
        raise RuntimeError(
            f"Model file not found at {model_path}. Did you run ml/train_fallnet.py?"
        )

    checkpoint = torch.load(model_path, map_location=_device)
    cfg = FallNetConfig(**checkpoint["config"])
    _LABEL_MAP = cfg.label_map
    model = FallNet(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_device)
    model.eval()
    _model = model
    return model


def _get_label_map() -> Dict[int, str]:
    global _LABEL_MAP
    if _LABEL_MAP is None:
        _load_model()
    assert _LABEL_MAP is not None
    return _LABEL_MAP


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = _load_model()
    label_map = _get_label_map()

    arrs = [np.array(seq.data, dtype="float32") for seq in req.sequences]
    max_frames = max(a.shape[0] for a in arrs)
    features = arrs[0].shape[1]

    padded = np.zeros((len(arrs), max_frames, features), dtype="float32")
    for i, a in enumerate(arrs):
        padded[i, : a.shape[0]] = a

    x = torch.from_numpy(padded).to(_device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    results: List[PredictResponseItem] = []
    for p in probs:
        idx = int(p.argmax())
        label = label_map.get(idx, str(idx))
        prob_dict = {label_map.get(i, str(i)): float(p_i) for i, p_i in enumerate(p)}
        results.append(
            PredictResponseItem(
                label=label,
                probabilities=prob_dict,
            )
        )

    return PredictResponse(results=results)


@app.post("/events/from_prediction", response_model=FallEvent)
def create_event_from_prediction(body: PredictionWithPosition) -> FallEvent:
    """Run prediction on a single sequence and log it as an event.

    This endpoint is convenient for wiring Omniverse / Isaac Sim directly
    into the API: send a radar episode + (x, y) position and get back
    the classified event while it is appended to the in-memory event log.
    """
    model = _load_model()
    label_map = _get_label_map()

    arr = np.array(body.sequence.data, dtype="float32")[None, ...]
    x_tensor = torch.from_numpy(arr).to(_device)
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    idx = int(probs.argmax())
    label = label_map.get(idx, str(idx))
    prob_dict = {label_map.get(i, str(i)): float(p_i) for i, p_i in enumerate(probs)}

    ev = FallEvent(
        id=str(len(_EVENTS) + 1),
        timestamp=float(time.time()),
        label=label,
        probabilities=prob_dict,
        position=body.position,
    )
    _EVENTS.append(ev)
    if len(_EVENTS) > _MAX_EVENTS:
        del _EVENTS[0 : len(_EVENTS) - _MAX_EVENTS]

    return ev


class EventsResponse(BaseModel):
    events: List[FallEvent]


@app.get("/events/recent", response_model=EventsResponse)
def get_recent_events() -> EventsResponse:
    """Return recent events for the dashboard.

    The front-end dashboard polls this endpoint periodically to update
    the top-down map and timeline.
    """
    return EventsResponse(events=_EVENTS)
