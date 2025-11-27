"""赤土崎多功能館 mmWave Fall Detection API.

FastAPI server for fall detection inference using FallNet model.
Includes WebSocket support for real-time alerts and status updates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml.fallnet_model import FallNet


# =============================================================================
# Pydantic Models
# =============================================================================

class PredictRequest(BaseModel):
    """Request body for /predict endpoint."""
    sequences: list[list[float]]  # [frames, features] or [features]


class PredictResponse(BaseModel):
    """Response body for /predict endpoint."""
    label: str
    label_index: int
    probabilities: dict[str, float]
    confidence: float
    explanation: str


class BatchPredictRequest(BaseModel):
    """Request body for /predict/batch endpoint."""
    sequences: list[list[list[float]]]  # [batch, frames, features]


class BatchPredictResponse(BaseModel):
    """Response body for /predict/batch endpoint."""
    results: list[PredictResponse]


class ZoneStatusRequest(BaseModel):
    """Request body for zone status update."""
    zone_id: str
    radar_id: str
    sequences: list[list[float]]


class ZoneStatusResponse(BaseModel):
    """Response body for zone status."""
    zone_id: str
    radar_id: str
    prediction: PredictResponse
    alert_required: bool
    alert_message: str | None


class WebSocketMessage(BaseModel):
    """WebSocket message format for alerts and status updates."""
    type: str  # "alert" | "status_update"
    zone_id: str
    radar_id: str
    prediction: dict[str, Any]
    timestamp: str


class SimulationConfig(BaseModel):
    """Configuration for background simulation mode."""
    enabled: bool = False
    interval_seconds: float = 5.0
    zones: list[str] = ["zone_01", "zone_02", "zone_03"]
    radars: list[str] = ["R1", "R2", "R3"]


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time broadcasting."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and store a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return

        message_json = json.dumps(message)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to send message to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    async def send_personal_message(self, message: dict[str, Any], websocket: WebSocket) -> None:
        """Send a message to a specific client."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.warning(f"Failed to send personal message: {e}")


# =============================================================================
# Application Setup
# =============================================================================

app = FastAPI(
    title="赤土崎多功能館 mmWave Fall Detection API",
    description="Real-time fall detection API for elderly care facility using mmWave radar. Supports WebSocket for real-time alerts.",
    version="1.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_model: FallNet | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# WebSocket connection manager
manager = ConnectionManager()

# Background simulation state
_simulation_task: asyncio.Task | None = None
_simulation_config = SimulationConfig()

# Label mapping (must match training)
LABEL_MAP: dict[int, str] = {
    0: "normal",
    1: "fall",
    2: "rehab_bad_posture",
    3: "chest_abnormal",
}

# Alert priority by label
ALERT_PRIORITY: dict[str, str] = {
    "normal": "none",
    "fall": "critical",
    "rehab_bad_posture": "medium",
    "chest_abnormal": "high",
}


# =============================================================================
# Model Loading
# =============================================================================

def _load_model() -> FallNet:
    """Load the FallNet model from disk."""
    global _model
    if _model is not None:
        return _model

    model_path = Path(os.getenv("MODEL_PATH", "ml/fallnet.pt"))
    if not model_path.exists():
        # Try relative to project root
        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / "ml" / "fallnet.pt"

    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}")
        raise HTTPException(
            status_code=503,
            detail=f"Model file not found. Please train the model first.",
        )

    try:
        checkpoint = torch.load(model_path, map_location=_device, weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            config = checkpoint.get("config", {})
            input_dim = config.get("input_dim", 256)
            num_classes = config.get("num_classes", 4)
            model_type = config.get("model_type", "mlp")
            model = FallNet(input_dim=input_dim, num_classes=num_classes, model_type=model_type)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded {model_type.upper()} model from checkpoint")
        else:
            # Direct state dict - try to infer model type from keys
            state_dict = checkpoint
            if any("lstm" in k for k in state_dict.keys()):
                model_type = "lstm"
            elif any("conv" in k for k in state_dict.keys()):
                model_type = "cnn"
            else:
                model_type = "mlp"
            model = FallNet(model_type=model_type)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded {model_type.upper()} model (inferred from state dict)")

        model.to(_device)
        model.eval()
        _model = model
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to load model: {e}")


def _generate_explanation(label: str, confidence: float) -> str:
    """Generate natural language explanation based on prediction."""
    if label == "fall":
        if confidence >= 0.8:
            return f"Fall detected with high confidence ({confidence:.1%}). Immediate attention required!"
        elif confidence >= 0.6:
            return f"Possible fall detected ({confidence:.1%}). Please verify the situation."
        else:
            return f"Uncertain fall detection ({confidence:.1%}). Manual verification recommended."
    elif label == "rehab_bad_posture":
        if confidence >= 0.8:
            return f"Poor rehabilitation posture detected ({confidence:.1%}). Corrective guidance recommended."
        elif confidence >= 0.6:
            return f"Possible posture issue during exercise ({confidence:.1%}). Please verify form."
        else:
            return f"Uncertain posture analysis ({confidence:.1%}). Manual verification recommended."
    elif label == "chest_abnormal":
        if confidence >= 0.8:
            return f"Abnormal breathing/chest pattern detected ({confidence:.1%}). Medical attention may be required."
        elif confidence >= 0.6:
            return f"Possible respiratory anomaly ({confidence:.1%}). Please monitor closely."
        else:
            return f"Uncertain chest pattern ({confidence:.1%}). Further monitoring recommended."
    else:  # normal
        return f"Normal activity detected ({confidence:.1%}). No concerns."


# =============================================================================
# Background Simulation
# =============================================================================

def _generate_fake_radar_data(input_dim: int = 256) -> list[list[float]]:
    """Generate fake radar data for simulation mode."""
    # Generate random radar-like features
    # Simulate 10 frames of radar data
    num_frames = 10
    data = np.random.randn(num_frames, input_dim).astype(np.float32)
    # Add some structure to make it look more realistic
    data = data * 0.1  # Scale down
    return data.tolist()


def _generate_fake_prediction() -> dict[str, Any]:
    """Generate a fake prediction for simulation mode without using the model."""
    # Weighted random selection - mostly normal, occasionally abnormal
    weights = [0.85, 0.08, 0.04, 0.03]  # normal, fall, rehab_bad_posture, chest_abnormal
    label_idx = random.choices(range(4), weights=weights)[0]
    label = LABEL_MAP[label_idx]

    # Generate random probabilities that sum to 1
    probs = np.random.dirichlet([10 if i == label_idx else 1 for i in range(4)])
    confidence = float(probs[label_idx])

    prob_dict = {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)}
    explanation = _generate_explanation(label, confidence)

    return {
        "label": label,
        "label_index": label_idx,
        "probabilities": prob_dict,
        "confidence": confidence,
        "explanation": explanation,
    }


async def _run_model_prediction(sequences: list[list[float]]) -> dict[str, Any]:
    """Run model prediction and return result as dict."""
    try:
        model = _load_model()
        arr = np.array(sequences, dtype="float32")
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim == 2:
            arr = arr[None, ...]

        x = torch.from_numpy(arr).to(_device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        idx = int(probs.argmax())
        label = LABEL_MAP.get(idx, str(idx))
        confidence = float(probs[idx])
        prob_dict = {LABEL_MAP.get(i, str(i)): float(p) for i, p in enumerate(probs)}
        explanation = _generate_explanation(label, confidence)

        return {
            "label": label,
            "label_index": idx,
            "probabilities": prob_dict,
            "confidence": confidence,
            "explanation": explanation,
        }
    except Exception as e:
        logger.warning(f"Model prediction failed in simulation, using fake data: {e}")
        return _generate_fake_prediction()


async def _simulation_loop():
    """Background task that generates fake radar data and broadcasts predictions."""
    logger.info("Starting background simulation loop")

    while _simulation_config.enabled:
        try:
            # Select random zone and radar
            zone_id = random.choice(_simulation_config.zones)
            radar_id = random.choice(_simulation_config.radars)

            # Generate fake radar data
            fake_data = _generate_fake_radar_data()

            # Try to run model prediction, fall back to fake prediction
            try:
                prediction = await _run_model_prediction(fake_data)
            except Exception:
                prediction = _generate_fake_prediction()

            # Determine message type
            is_alert = prediction["label"] != "normal" and prediction["confidence"] >= 0.6
            msg_type = "alert" if is_alert else "status_update"

            # Create WebSocket message
            message = {
                "type": msg_type,
                "zone_id": zone_id,
                "radar_id": radar_id,
                "prediction": prediction,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Broadcast to all connected clients
            await manager.broadcast(message)

            if is_alert:
                logger.info(f"Simulation alert: {prediction['label']} in {zone_id}")

        except Exception as e:
            logger.error(f"Simulation loop error: {e}")

        # Wait for next interval
        await asyncio.sleep(_simulation_config.interval_seconds)

    logger.info("Background simulation loop stopped")


# =============================================================================
# REST Endpoints
# =============================================================================

@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/info")
async def info() -> dict[str, Any]:
    """Get API and model information."""
    model_path = Path(os.getenv("MODEL_PATH", "ml/fallnet.pt"))
    return {
        "api_version": "1.1.0",
        "facility": "赤土崎多功能館",
        "model_path": str(model_path),
        "model_loaded": _model is not None,
        "device": str(_device),
        "labels": LABEL_MAP,
        "websocket_endpoint": "/ws",
        "active_connections": len(manager.active_connections),
        "simulation_enabled": _simulation_config.enabled,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    """Run fall detection on a single radar sequence.

    Args:
        req: Request containing radar sequence data [frames, features] or [features]

    Returns:
        Prediction result with label, probabilities, and explanation
    """
    model = _load_model()

    try:
        # Convert to numpy then tensor
        arr = np.array(req.sequences, dtype="float32")
        if arr.ndim == 1:
            arr = arr[None, :]  # [features] -> [1, features]
        if arr.ndim == 2:
            arr = arr[None, ...]  # [frames, features] -> [1, frames, features]

        x = torch.from_numpy(arr).to(_device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # Get prediction
        idx = int(probs.argmax())
        label = LABEL_MAP.get(idx, str(idx))
        confidence = float(probs[idx])

        # Build probability dict
        prob_dict = {LABEL_MAP.get(i, str(i)): float(p) for i, p in enumerate(probs)}

        # Generate explanation
        explanation = _generate_explanation(label, confidence)

        return PredictResponse(
            label=label,
            label_index=idx,
            probabilities=prob_dict,
            confidence=confidence,
            explanation=explanation,
        )

    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    """Run fall detection on multiple radar sequences.

    Args:
        req: Request containing batch of radar sequences [batch, frames, features]

    Returns:
        Batch prediction results
    """
    model = _load_model()

    try:
        results = []
        for seq in req.sequences:
            arr = np.array(seq, dtype="float32")
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.ndim == 2:
                arr = arr[None, ...]

            x = torch.from_numpy(arr).to(_device)

            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            idx = int(probs.argmax())
            label = LABEL_MAP.get(idx, str(idx))
            confidence = float(probs[idx])
            prob_dict = {LABEL_MAP.get(i, str(i)): float(p) for i, p in enumerate(probs)}
            explanation = _generate_explanation(label, confidence)

            results.append(PredictResponse(
                label=label,
                label_index=idx,
                probabilities=prob_dict,
                confidence=confidence,
                explanation=explanation,
            ))

        return BatchPredictResponse(results=results)

    except Exception as e:
        logger.exception(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


@app.post("/zone/status", response_model=ZoneStatusResponse)
async def zone_status(req: ZoneStatusRequest) -> ZoneStatusResponse:
    """Update zone status with radar prediction.

    This endpoint is designed for integration with the facility monitoring system.
    It returns whether an alert should be triggered based on the prediction.
    Also broadcasts the status update to all connected WebSocket clients.
    """
    # Get prediction
    predict_req = PredictRequest(sequences=req.sequences)
    prediction = await predict(predict_req)

    # Determine if alert is required
    alert_required = prediction.label != "normal" and prediction.confidence >= 0.6
    alert_message = None

    if alert_required:
        priority = ALERT_PRIORITY.get(prediction.label, "medium")
        alert_message = f"[{priority.upper()}] Zone {req.zone_id}: {prediction.explanation}"

    # Broadcast to WebSocket clients
    ws_message = {
        "type": "alert" if alert_required else "status_update",
        "zone_id": req.zone_id,
        "radar_id": req.radar_id,
        "prediction": prediction.model_dump(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    await manager.broadcast(ws_message)

    return ZoneStatusResponse(
        zone_id=req.zone_id,
        radar_id=req.radar_id,
        prediction=prediction,
        alert_required=alert_required,
        alert_message=alert_message,
    )


@app.get("/labels")
async def get_labels() -> dict[str, Any]:
    """Get label definitions and alert priorities."""
    return {
        "labels": LABEL_MAP,
        "alert_priorities": ALERT_PRIORITY,
    }


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time fall detection alerts.

    Clients connecting to this endpoint will receive JSON messages with the format:
    {
        "type": "alert" | "status_update",
        "zone_id": "zone_01",
        "radar_id": "R1",
        "prediction": {
            "label": "fall",
            "label_index": 1,
            "probabilities": {...},
            "confidence": 0.95,
            "explanation": "..."
        },
        "timestamp": "2025-01-01T00:00:00Z"
    }
    """
    await manager.connect(websocket)

    # Send welcome message
    welcome_msg = {
        "type": "status_update",
        "zone_id": "system",
        "radar_id": "system",
        "prediction": {
            "label": "normal",
            "label_index": 0,
            "probabilities": {"normal": 1.0, "fall": 0.0, "rehab_bad_posture": 0.0, "chest_abnormal": 0.0},
            "confidence": 1.0,
            "explanation": "WebSocket connection established successfully.",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    await manager.send_personal_message(welcome_msg, websocket)

    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                # Handle ping/pong for keepalive
                if message.get("type") == "ping":
                    await manager.send_personal_message({"type": "pong"}, websocket)

                # Handle radar data submission via WebSocket
                elif message.get("type") == "radar_data":
                    zone_id = message.get("zone_id", "unknown")
                    radar_id = message.get("radar_id", "unknown")
                    sequences = message.get("sequences", [])

                    if sequences:
                        prediction = await _run_model_prediction(sequences)
                        is_alert = prediction["label"] != "normal" and prediction["confidence"] >= 0.6

                        response = {
                            "type": "alert" if is_alert else "status_update",
                            "zone_id": zone_id,
                            "radar_id": radar_id,
                            "prediction": prediction,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }

                        # Broadcast to all clients
                        await manager.broadcast(response)

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {data}")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# =============================================================================
# Simulation Control Endpoints
# =============================================================================

@app.post("/simulation/start")
async def start_simulation(
    interval_seconds: float = 5.0,
    zones: list[str] | None = None,
    radars: list[str] | None = None,
) -> dict[str, Any]:
    """Start background simulation mode that generates fake radar data.

    Args:
        interval_seconds: Time between simulated readings (default 5.0)
        zones: List of zone IDs to simulate (default: zone_01, zone_02, zone_03)
        radars: List of radar IDs to simulate (default: R1, R2, R3)

    Returns:
        Simulation status
    """
    global _simulation_task, _simulation_config

    if _simulation_config.enabled:
        return {"status": "already_running", "config": _simulation_config.model_dump()}

    _simulation_config = SimulationConfig(
        enabled=True,
        interval_seconds=interval_seconds,
        zones=zones or ["zone_01", "zone_02", "zone_03"],
        radars=radars or ["R1", "R2", "R3"],
    )

    _simulation_task = asyncio.create_task(_simulation_loop())

    logger.info(f"Simulation started with interval {interval_seconds}s")

    return {
        "status": "started",
        "config": _simulation_config.model_dump(),
    }


@app.post("/simulation/stop")
async def stop_simulation() -> dict[str, str]:
    """Stop background simulation mode."""
    global _simulation_task, _simulation_config

    if not _simulation_config.enabled:
        return {"status": "not_running"}

    _simulation_config.enabled = False

    if _simulation_task:
        _simulation_task.cancel()
        try:
            await _simulation_task
        except asyncio.CancelledError:
            pass
        _simulation_task = None

    logger.info("Simulation stopped")

    return {"status": "stopped"}


@app.get("/simulation/status")
async def simulation_status() -> dict[str, Any]:
    """Get current simulation status."""
    return {
        "enabled": _simulation_config.enabled,
        "config": _simulation_config.model_dump(),
        "active_connections": len(manager.active_connections),
    }


# =============================================================================
# Startup Event
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Preload model on startup."""
    logger.info("Preloading model on startup...")
    try:
        _load_model()
        logger.info("Model preloaded successfully")
    except Exception as e:
        logger.warning(f"Model preload failed (will retry on first request): {e}")
