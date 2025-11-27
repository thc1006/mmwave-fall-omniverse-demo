"""mmWave Fall Detection Demo API with WebSocket support."""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml.fallnet_model import FallNet, FallNetConfig, ModelFactory, ModelType  # type: ignore
from services.api.websocket_manager import ws_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class Position(BaseModel):
    x: float
    y: float


class RadarSequence(BaseModel):
    """One radar episode: frames x features."""

    data: list[list[float]]


class PredictRequest(BaseModel):
    sequences: list[RadarSequence]


class PredictResponseItem(BaseModel):
    label: str
    probabilities: dict[str, float]
    explanation: str


class PredictResponse(BaseModel):
    results: list[PredictResponseItem]


class FallEvent(BaseModel):
    id: str
    timestamp: float  # unix seconds
    label: str
    probabilities: dict[str, float]
    position: Position


class PredictionWithPosition(BaseModel):
    sequence: RadarSequence
    position: Position


class EventsResponse(BaseModel):
    events: list[FallEvent]


class StatsResponse(BaseModel):
    total_events: int
    label_counts: dict[str, int]
    recent_falls: list[FallEvent]
    websocket_connections: int


class AlertRequest(BaseModel):
    event_id: str
    recipient: str  # e.g., phone number or email (simulated)
    message: str | None = None


class AlertResponse(BaseModel):
    success: bool
    alert_id: str
    message: str


# ---------------------------------------------------------------------------
# Application Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="mmWave Fall Detection Demo API",
    version="0.4.0",
    description="Real-time fall detection API with WebSocket support for live event streaming.",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_model: torch.nn.Module | None = None  # Can be FallNet, FallNetCNN, or FallNetLSTM
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_LABEL_MAP: dict[int, str] | None = None
_NORM_MEAN: torch.Tensor | None = None
_NORM_STD: torch.Tensor | None = None
_MAX_EVENTS = 200
_EVENTS: deque[FallEvent] = deque(maxlen=_MAX_EVENTS)  # Auto-evicting FIFO queue
_ALERTS: list[dict[str, Any]] = []


@app.on_event("startup")
async def startup_event():
    """Preload model on startup to avoid cold start latency."""
    logger.info("Preloading model on startup...")
    try:
        _load_model()
        logger.info("Model preloaded successfully")
    except Exception as e:
        logger.warning(f"Model preload failed (will retry on first request): {e}")


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------


def _load_model() -> FallNet:
    """Load the FallNet model from disk."""
    global _model, _LABEL_MAP, _NORM_MEAN, _NORM_STD
    if _model is not None:
        return _model

    model_path = Path(os.getenv("MODEL_PATH", "ml/fallnet_lstm.pt"))
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}")
        raise HTTPException(
            status_code=503,
            detail=f"Model file not found at {model_path}. Did you run ml/train_fallnet.py?",
        )

    try:
        checkpoint = torch.load(model_path, map_location=_device, weights_only=False)
        config_dict = checkpoint["config"]
        # Convert model_type string to enum if needed
        if isinstance(config_dict.get("model_type"), str):
            config_dict["model_type"] = ModelType(config_dict["model_type"])
        cfg = FallNetConfig(**config_dict)
        _LABEL_MAP = checkpoint.get("idx_to_label", cfg.label_map)
        # Load normalization stats if available
        if "norm_mean" in checkpoint:
            _NORM_MEAN = checkpoint["norm_mean"].to(_device)
            _NORM_STD = checkpoint["norm_std"].to(_device)
            logger.info("Loaded normalization stats from checkpoint")
        # Use ModelFactory to create correct model type (MLP, CNN, or LSTM)
        model = ModelFactory.create(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(_device)
        model.eval()
        _model = model
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to load model: {e}")


def _get_label_map() -> dict[int, str]:
    """Get the label map, loading model if necessary."""
    global _LABEL_MAP
    if _LABEL_MAP is None:
        _load_model()
    assert _LABEL_MAP is not None
    return _LABEL_MAP


def _generate_explanation(label: str, probabilities: dict[str, float]) -> str:
    """Generate a natural language explanation based on prediction label and confidence.

    Args:
        label: The predicted label (e.g., "fall", "normal", "rehab_bad_posture", "chest_abnormal")
        probabilities: Dictionary mapping labels to their probabilities

    Returns:
        A natural language explanation string
    """
    # Get the probability of the predicted label
    label_prob = probabilities.get(label, 0.0)
    label_lower = label.lower()

    if label_lower == "fall":
        # Fall detection with confidence levels
        if label_prob >= 0.8:
            return f"Fall detected with high confidence ({label_prob:.1%}). Immediate attention may be required."
        elif label_prob >= 0.6:
            return f"Possible fall detected ({label_prob:.1%}). Please verify the situation."
        else:
            return f"Uncertain fall detection ({label_prob:.1%}). Manual verification recommended."
    elif label_lower == "rehab_bad_posture":
        # Bad posture during rehabilitation exercises
        if label_prob >= 0.8:
            return f"Poor rehabilitation posture detected ({label_prob:.1%}). Corrective guidance recommended."
        elif label_prob >= 0.6:
            return f"Possible posture issue during exercise ({label_prob:.1%}). Please verify form."
        else:
            return f"Uncertain posture analysis ({label_prob:.1%}). Manual verification recommended."
    elif label_lower == "chest_abnormal":
        # Abnormal chest/breathing patterns
        if label_prob >= 0.8:
            return f"Abnormal breathing/chest pattern detected ({label_prob:.1%}). Medical attention may be required."
        elif label_prob >= 0.6:
            return f"Possible respiratory anomaly ({label_prob:.1%}). Please monitor closely."
        else:
            return f"Uncertain chest pattern ({label_prob:.1%}). Further monitoring recommended."
    else:
        # Normal activity or other non-fall labels
        return f"Normal activity detected ({label_prob:.1%}). No concerns."


# ---------------------------------------------------------------------------
# Health & Info Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """Return statistics summary including label counts and recent anomalies."""
    label_counts = Counter(ev.label for ev in _EVENTS)

    # Get recent fall events (last 10 falls)
    recent_falls = [ev for ev in reversed(_EVENTS) if ev.label == "fall"][:10]

    return StatsResponse(
        total_events=len(_EVENTS),
        label_counts=dict(label_counts),
        recent_falls=recent_falls,
        websocket_connections=ws_manager.connection_count,
    )


# ---------------------------------------------------------------------------
# Prediction Endpoints
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    """Run fall detection on radar sequences."""
    if not req.sequences:
        raise HTTPException(status_code=400, detail="No sequences provided")

    model = _load_model()
    label_map = _get_label_map()

    try:
        arrs = [np.array(seq.data, dtype="float32") for seq in req.sequences]
        max_frames = max(a.shape[0] for a in arrs)
        features = arrs[0].shape[1]

        padded = np.zeros((len(arrs), max_frames, features), dtype="float32")
        for i, a in enumerate(arrs):
            padded[i, : a.shape[0]] = a

        x = torch.from_numpy(padded).to(_device)
        # Apply z-score normalization if available
        if _NORM_MEAN is not None and _NORM_STD is not None:
            x = (x - _NORM_MEAN) / _NORM_STD
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results: list[PredictResponseItem] = []
        for p in probs:
            idx = int(p.argmax())
            label = label_map.get(idx, str(idx))
            prob_dict = {label_map.get(i, str(i)): float(p_i) for i, p_i in enumerate(p)}
            explanation = _generate_explanation(label, prob_dict)
            results.append(PredictResponseItem(label=label, probabilities=prob_dict, explanation=explanation))

        logger.debug(f"Predicted {len(results)} sequences")
        return PredictResponse(results=results)

    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/events/from_prediction", response_model=FallEvent)
async def create_event_from_prediction(body: PredictionWithPosition) -> FallEvent:
    """Run prediction on a single sequence and log it as an event.

    This endpoint is convenient for wiring Omniverse / Isaac Sim directly
    into the API: send a radar episode + (x, y) position and get back
    the classified event while it is appended to the in-memory event log.

    If the event is a fall, it will be broadcast to all WebSocket clients.
    """
    model = _load_model()
    label_map = _get_label_map()

    try:
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
        _EVENTS.append(ev)  # deque auto-evicts oldest when maxlen reached

        # Broadcast fall events via WebSocket
        if label == "fall":
            logger.warning(f"Fall detected! Event ID: {ev.id}, Position: ({ev.position.x}, {ev.position.y})")
            await ws_manager.broadcast({"type": "fall_event", "event": ev.model_dump()})
        else:
            await ws_manager.broadcast({"type": "event", "event": ev.model_dump()})

        return ev

    except Exception as e:
        logger.exception(f"Event creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Event creation failed: {e}")


@app.get("/events/recent", response_model=EventsResponse)
async def get_recent_events() -> EventsResponse:
    """Return recent events for the dashboard.

    The front-end dashboard polls this endpoint periodically to update
    the top-down map and timeline.
    """
    return EventsResponse(events=list(_EVENTS))


# ---------------------------------------------------------------------------
# Alert Endpoints
# ---------------------------------------------------------------------------


@app.post("/alerts", response_model=AlertResponse)
async def send_alert(req: AlertRequest) -> AlertResponse:
    """Send a fall alert notification (simulated).

    In production, this would integrate with SMS/push notification services.
    """
    # Find the event
    event = next((ev for ev in _EVENTS if ev.id == req.event_id), None)
    if not event:
        raise HTTPException(status_code=404, detail=f"Event {req.event_id} not found")

    # Simulate sending alert
    alert_id = f"alert_{len(_ALERTS) + 1}_{int(time.time())}"
    default_message = f"Fall detected at position ({event.position.x:.1f}, {event.position.y:.1f})"
    message = req.message or default_message

    alert_record = {
        "alert_id": alert_id,
        "event_id": req.event_id,
        "recipient": req.recipient,
        "message": message,
        "sent_at": time.time(),
    }
    _ALERTS.append(alert_record)

    logger.info(f"Alert sent: {alert_id} to {req.recipient}")

    # Broadcast alert via WebSocket
    await ws_manager.broadcast({"type": "alert_sent", "alert": alert_record})

    return AlertResponse(
        success=True,
        alert_id=alert_id,
        message=f"Alert sent to {req.recipient}: {message}",
    )


@app.get("/alerts/history")
async def get_alert_history() -> dict[str, Any]:
    """Get history of sent alerts."""
    return {"alerts": _ALERTS, "total": len(_ALERTS)}


# ---------------------------------------------------------------------------
# WebSocket Endpoints
# ---------------------------------------------------------------------------


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time event streaming.

    Clients connect to this endpoint to receive live updates when:
    - New events are detected
    - Falls are detected (high priority)
    - Alerts are sent
    """
    await ws_manager.connect(websocket)
    try:
        # Send current stats on connect
        await websocket.send_json({
            "type": "connected",
            "stats": {
                "total_events": len(_EVENTS),
                "connections": ws_manager.connection_count,
            },
        })

        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            # Echo back or handle client messages if needed
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
    finally:
        await ws_manager.disconnect(websocket)
