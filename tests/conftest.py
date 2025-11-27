"""Pytest configuration and fixtures for the fall detection API tests."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.fallnet_model import FallNet, FallNetConfig


# ---------------------------------------------------------------------------
# Pytest Configuration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Model Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def model_config() -> FallNetConfig:
    """Create a test model configuration."""
    return FallNetConfig(
        input_dim=6,  # Smaller for faster tests
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
        num_classes=4,
        seq_len=10,
        label_map={
            0: "normal",
            1: "fall",
            2: "rehab_bad_posture",
            3: "chest_abnormal",
        },
    )


@pytest.fixture(scope="session")
def test_model(model_config: FallNetConfig) -> FallNet:
    """Create a test FallNet model."""
    model = FallNet(model_config)
    model.eval()
    return model


@pytest.fixture(scope="session")
def model_checkpoint_path(test_model: FallNet, model_config: FallNetConfig) -> Generator[str, None, None]:
    """Create a temporary model checkpoint file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint = {
            "model_state_dict": test_model.state_dict(),
            "config": {
                "input_dim": model_config.input_dim,
                "hidden_dim": model_config.hidden_dim,
                "num_layers": model_config.num_layers,
                "dropout": model_config.dropout,
                "num_classes": model_config.num_classes,
                "seq_len": model_config.seq_len,
                "model_type": model_config.model_type.value,
                "label_map": model_config.label_map,
            },
            "idx_to_label": model_config.label_map,
        }
        torch.save(checkpoint, f.name)
        yield f.name

    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


# ---------------------------------------------------------------------------
# Test Client Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app_with_mock_model(model_checkpoint_path: str):
    """Create a FastAPI test app with the mock model loaded.

    This fixture patches the MODEL_PATH environment variable and resets
    the global model state for each test.
    """
    # Reset global state before importing/creating the app
    import services.api.main as main_module

    # Reset global state
    main_module._model = None
    main_module._LABEL_MAP = None
    main_module._NORM_MEAN = None
    main_module._NORM_STD = None
    main_module._EVENTS.clear()
    main_module._ALERTS.clear()

    with patch.dict(os.environ, {"MODEL_PATH": model_checkpoint_path}):
        yield main_module.app

    # Cleanup after test
    main_module._model = None
    main_module._LABEL_MAP = None
    main_module._EVENTS.clear()
    main_module._ALERTS.clear()


@pytest.fixture
def test_client(app_with_mock_model):
    """Create a TestClient for the FastAPI app."""
    from fastapi.testclient import TestClient

    with TestClient(app_with_mock_model) as client:
        yield client


@pytest.fixture
def async_client(app_with_mock_model):
    """Create an async HTTP client for testing async endpoints."""
    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=app_with_mock_model)
    return AsyncClient(transport=transport, base_url="http://test")


# ---------------------------------------------------------------------------
# Test Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_radar_sequence(model_config: FallNetConfig) -> list[list[float]]:
    """Generate a sample radar sequence for testing.

    Returns a 2D list representing [frames, features].
    """
    np.random.seed(42)
    frames = model_config.seq_len
    features = model_config.input_dim
    data = np.random.randn(frames, features).tolist()
    return data


@pytest.fixture
def sample_position() -> dict[str, float]:
    """Generate a sample position for testing."""
    return {"x": 5.0, "y": 10.0}


@pytest.fixture
def sample_predict_request(sample_radar_sequence: list[list[float]]) -> dict[str, Any]:
    """Create a sample predict request payload."""
    return {
        "sequences": [
            {"data": sample_radar_sequence}
        ]
    }


@pytest.fixture
def sample_prediction_with_position(
    sample_radar_sequence: list[list[float]],
    sample_position: dict[str, float]
) -> dict[str, Any]:
    """Create a sample prediction with position payload."""
    return {
        "sequence": {"data": sample_radar_sequence},
        "position": sample_position
    }


@pytest.fixture
def multiple_radar_sequences(model_config: FallNetConfig) -> list[dict[str, list[list[float]]]]:
    """Generate multiple radar sequences for batch testing."""
    np.random.seed(123)
    frames = model_config.seq_len
    features = model_config.input_dim

    sequences = []
    for _ in range(3):
        data = np.random.randn(frames, features).tolist()
        sequences.append({"data": data})

    return sequences


# ---------------------------------------------------------------------------
# Event and Alert Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_alert_request() -> dict[str, Any]:
    """Create a sample alert request payload."""
    return {
        "event_id": "1",
        "recipient": "+1234567890",
        "message": "Fall detected in the hall!"
    }


# ---------------------------------------------------------------------------
# WebSocket Test Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def websocket_test_client(app_with_mock_model):
    """Create a TestClient configured for WebSocket testing."""
    from fastapi.testclient import TestClient

    # Return a fresh TestClient for WebSocket tests
    return TestClient(app_with_mock_model)
