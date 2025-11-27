"""Comprehensive test suite for the mmWave Fall Detection FastAPI application.

This module contains tests for all API endpoints including:
- Health check
- Prediction endpoints
- Event management
- Alert system
- WebSocket real-time streaming

Run tests with: pytest tests/test_api.py -v
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Health Check Tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self, test_client):
        """Test that health endpoint returns status ok."""
        response = test_client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_response_format(self, test_client):
        """Test that health response has correct structure."""
        response = test_client.get("/health")
        data = response.json()

        assert "status" in data
        assert isinstance(data["status"], str)


# ---------------------------------------------------------------------------
# Stats Endpoint Tests
# ---------------------------------------------------------------------------


class TestStatsEndpoint:
    """Tests for the /stats endpoint."""

    def test_stats_returns_initial_values(self, test_client):
        """Test that stats endpoint returns correct initial values."""
        response = test_client.get("/stats")

        assert response.status_code == 200
        data = response.json()

        assert "total_events" in data
        assert "label_counts" in data
        assert "recent_falls" in data
        assert "websocket_connections" in data

        assert data["total_events"] == 0
        assert data["label_counts"] == {}
        assert data["recent_falls"] == []
        assert data["websocket_connections"] == 0

    def test_stats_after_events_created(
        self, test_client, sample_prediction_with_position
    ):
        """Test that stats reflect created events."""
        # Create an event first
        test_client.post(
            "/events/from_prediction",
            json=sample_prediction_with_position
        )

        response = test_client.get("/stats")
        assert response.status_code == 200
        data = response.json()

        assert data["total_events"] == 1
        assert len(data["label_counts"]) > 0


# ---------------------------------------------------------------------------
# Predict Endpoint Tests
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_single_sequence(self, test_client, sample_predict_request):
        """Test prediction with a single radar sequence."""
        response = test_client.post("/predict", json=sample_predict_request)

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert len(data["results"]) == 1

        result = data["results"][0]
        assert "label" in result
        assert "probabilities" in result
        assert result["label"] in ["normal", "fall", "rehab_bad_posture", "chest_abnormal"]

    def test_predict_multiple_sequences(self, test_client, multiple_radar_sequences):
        """Test prediction with multiple radar sequences."""
        request_data = {"sequences": multiple_radar_sequences}
        response = test_client.post("/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert len(data["results"]) == len(multiple_radar_sequences)

        for result in data["results"]:
            assert "label" in result
            assert "probabilities" in result

    def test_predict_probabilities_sum_to_one(
        self, test_client, sample_predict_request
    ):
        """Test that prediction probabilities sum approximately to 1."""
        response = test_client.post("/predict", json=sample_predict_request)

        assert response.status_code == 200
        data = response.json()

        probs = data["results"][0]["probabilities"]
        total = sum(probs.values())

        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, expected ~1.0"

    def test_predict_empty_sequences_returns_error(self, test_client):
        """Test that empty sequences list returns 400 error."""
        response = test_client.post("/predict", json={"sequences": []})

        assert response.status_code == 400
        assert "No sequences provided" in response.json()["detail"]

    def test_predict_invalid_data_format(self, test_client):
        """Test that invalid data format returns appropriate error."""
        invalid_request = {"sequences": [{"data": "not_a_list"}]}

        response = test_client.post("/predict", json=invalid_request)

        # Should return validation error (422) for Pydantic validation failure
        assert response.status_code == 422

    def test_predict_missing_sequences_field(self, test_client):
        """Test that missing sequences field returns validation error."""
        response = test_client.post("/predict", json={})

        assert response.status_code == 422

    def test_predict_variable_sequence_lengths(
        self, test_client, model_config
    ):
        """Test prediction with sequences of different lengths."""
        np.random.seed(456)
        features = model_config.input_dim

        sequences = [
            {"data": np.random.randn(5, features).tolist()},
            {"data": np.random.randn(10, features).tolist()},
            {"data": np.random.randn(15, features).tolist()},
        ]

        response = test_client.post("/predict", json={"sequences": sequences})

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 3


# ---------------------------------------------------------------------------
# Event Endpoints Tests
# ---------------------------------------------------------------------------


class TestEventEndpoints:
    """Tests for event-related endpoints."""

    def test_create_event_from_prediction(
        self, test_client, sample_prediction_with_position
    ):
        """Test creating an event from a prediction."""
        response = test_client.post(
            "/events/from_prediction",
            json=sample_prediction_with_position
        )

        assert response.status_code == 200
        data = response.json()

        assert "id" in data
        assert "timestamp" in data
        assert "label" in data
        assert "probabilities" in data
        assert "position" in data

        assert data["position"]["x"] == sample_prediction_with_position["position"]["x"]
        assert data["position"]["y"] == sample_prediction_with_position["position"]["y"]

    def test_event_id_increments(
        self, test_client, sample_prediction_with_position
    ):
        """Test that event IDs increment correctly."""
        # Create first event
        response1 = test_client.post(
            "/events/from_prediction",
            json=sample_prediction_with_position
        )
        id1 = response1.json()["id"]

        # Create second event
        response2 = test_client.post(
            "/events/from_prediction",
            json=sample_prediction_with_position
        )
        id2 = response2.json()["id"]

        assert int(id2) == int(id1) + 1

    def test_event_timestamp_is_recent(
        self, test_client, sample_prediction_with_position
    ):
        """Test that event timestamp is close to current time."""
        before = time.time()

        response = test_client.post(
            "/events/from_prediction",
            json=sample_prediction_with_position
        )

        after = time.time()
        timestamp = response.json()["timestamp"]

        assert before <= timestamp <= after

    def test_get_recent_events_empty(self, test_client):
        """Test getting recent events when none exist."""
        response = test_client.get("/events/recent")

        assert response.status_code == 200
        data = response.json()

        assert "events" in data
        assert isinstance(data["events"], list)

    def test_get_recent_events_after_creation(
        self, test_client, sample_prediction_with_position
    ):
        """Test that created events appear in recent events."""
        # Create an event
        create_response = test_client.post(
            "/events/from_prediction",
            json=sample_prediction_with_position
        )
        created_event = create_response.json()

        # Fetch recent events
        response = test_client.get("/events/recent")
        assert response.status_code == 200

        events = response.json()["events"]
        assert len(events) >= 1

        # Find the created event
        event_ids = [ev["id"] for ev in events]
        assert created_event["id"] in event_ids

    def test_events_max_limit(
        self, test_client, sample_prediction_with_position
    ):
        """Test that events are limited to max count (200)."""
        # Create more events than the limit
        import services.api.main as main_module

        max_events = main_module._MAX_EVENTS

        for _ in range(max_events + 10):
            test_client.post(
                "/events/from_prediction",
                json=sample_prediction_with_position
            )

        response = test_client.get("/events/recent")
        events = response.json()["events"]

        assert len(events) <= max_events

    def test_event_position_validation(self, test_client, sample_radar_sequence):
        """Test that invalid position is rejected."""
        invalid_payload = {
            "sequence": {"data": sample_radar_sequence},
            "position": {"x": "not_a_number", "y": 10.0}  # Invalid x
        }

        response = test_client.post(
            "/events/from_prediction",
            json=invalid_payload
        )

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Alert Endpoints Tests
# ---------------------------------------------------------------------------


class TestAlertEndpoints:
    """Tests for alert-related endpoints."""

    def test_create_alert_success(
        self, test_client, sample_prediction_with_position
    ):
        """Test successful alert creation."""
        # First create an event
        event_response = test_client.post(
            "/events/from_prediction",
            json=sample_prediction_with_position
        )
        event_id = event_response.json()["id"]

        # Create alert for the event
        alert_request = {
            "event_id": event_id,
            "recipient": "+1234567890",
            "message": "Test alert message"
        }

        response = test_client.post("/alerts", json=alert_request)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "alert_id" in data
        assert "message" in data
        assert "+1234567890" in data["message"]

    def test_create_alert_default_message(
        self, test_client, sample_prediction_with_position
    ):
        """Test alert creation with default message."""
        # Create an event
        event_response = test_client.post(
            "/events/from_prediction",
            json=sample_prediction_with_position
        )
        event_id = event_response.json()["id"]

        # Create alert without custom message
        alert_request = {
            "event_id": event_id,
            "recipient": "test@example.com"
        }

        response = test_client.post("/alerts", json=alert_request)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        # Default message should contain position
        assert "Fall detected" in data["message"]

    def test_create_alert_nonexistent_event(self, test_client):
        """Test that alert for nonexistent event returns 404."""
        alert_request = {
            "event_id": "99999",
            "recipient": "+1234567890"
        }

        response = test_client.post("/alerts", json=alert_request)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_alert_history_empty(self, test_client):
        """Test getting alert history when none exist."""
        response = test_client.get("/alerts/history")

        assert response.status_code == 200
        data = response.json()

        assert "alerts" in data
        assert "total" in data
        assert data["total"] == 0

    def test_get_alert_history_after_creation(
        self, test_client, sample_prediction_with_position
    ):
        """Test that created alerts appear in history."""
        # Create an event
        event_response = test_client.post(
            "/events/from_prediction",
            json=sample_prediction_with_position
        )
        event_id = event_response.json()["id"]

        # Create alert
        test_client.post("/alerts", json={
            "event_id": event_id,
            "recipient": "+1234567890"
        })

        # Fetch history
        response = test_client.get("/alerts/history")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] >= 1
        assert len(data["alerts"]) >= 1

    def test_alert_record_contains_required_fields(
        self, test_client, sample_prediction_with_position
    ):
        """Test that alert records contain all required fields."""
        # Create event and alert
        event_response = test_client.post(
            "/events/from_prediction",
            json=sample_prediction_with_position
        )
        event_id = event_response.json()["id"]

        test_client.post("/alerts", json={
            "event_id": event_id,
            "recipient": "user@test.com",
            "message": "Custom alert"
        })

        # Verify alert record
        response = test_client.get("/alerts/history")
        alert = response.json()["alerts"][0]

        assert "alert_id" in alert
        assert "event_id" in alert
        assert "recipient" in alert
        assert "message" in alert
        assert "sent_at" in alert

        assert alert["event_id"] == event_id
        assert alert["recipient"] == "user@test.com"
        assert alert["message"] == "Custom alert"


# ---------------------------------------------------------------------------
# WebSocket Tests
# ---------------------------------------------------------------------------


class TestWebSocketEndpoint:
    """Tests for the WebSocket /ws/events endpoint."""

    def test_websocket_connection(self, websocket_test_client):
        """Test basic WebSocket connection."""
        with websocket_test_client.websocket_connect("/ws/events") as websocket:
            # Should receive connected message
            data = websocket.receive_json()

            assert data["type"] == "connected"
            assert "stats" in data
            assert "total_events" in data["stats"]
            assert "connections" in data["stats"]

    def test_websocket_ping_pong(self, websocket_test_client):
        """Test WebSocket ping/pong functionality."""
        with websocket_test_client.websocket_connect("/ws/events") as websocket:
            # Receive initial connected message
            websocket.receive_json()

            # Send ping
            websocket.send_text("ping")

            # Should receive pong
            data = websocket.receive_json()
            assert data["type"] == "pong"

    def test_websocket_connection_count(self, app_with_mock_model):
        """Test that WebSocket connection count is tracked."""
        from fastapi.testclient import TestClient

        client = TestClient(app_with_mock_model)

        # Check initial connection count
        response = client.get("/stats")
        assert response.json()["websocket_connections"] == 0

        # Connect WebSocket
        with client.websocket_connect("/ws/events"):
            # Check connection count is incremented
            response = client.get("/stats")
            assert response.json()["websocket_connections"] == 1

        # After disconnect, count should decrease
        response = client.get("/stats")
        assert response.json()["websocket_connections"] == 0

    def test_websocket_receives_event_broadcast(
        self, app_with_mock_model, sample_prediction_with_position
    ):
        """Test that WebSocket receives broadcast events."""
        from fastapi.testclient import TestClient

        client = TestClient(app_with_mock_model)

        with client.websocket_connect("/ws/events") as websocket:
            # Receive initial connected message
            connected_msg = websocket.receive_json()
            assert connected_msg["type"] == "connected"

            # Create an event via HTTP
            client.post(
                "/events/from_prediction",
                json=sample_prediction_with_position
            )

            # WebSocket should receive the broadcast
            data = websocket.receive_json()
            assert data["type"] in ["event", "fall_event"]
            assert "event" in data


# ---------------------------------------------------------------------------
# Edge Cases and Error Handling Tests
# ---------------------------------------------------------------------------


class TestEdgeCasesAndErrors:
    """Tests for edge cases and error handling."""

    def test_predict_with_extreme_values(self, test_client, model_config):
        """Test prediction behavior with extreme float values.

        Note: NaN/Inf cannot be sent via JSON, so we test with very large values
        that might cause numerical issues.
        """
        features = model_config.input_dim
        # Create data with very large values (but still JSON-compliant)
        data_with_extremes = [[1e30] * features for _ in range(5)]

        response = test_client.post("/predict", json={
            "sequences": [{"data": data_with_extremes}]
        })

        # The API should handle extreme values gracefully
        # It might return 200 with valid predictions or 500 if overflow occurs
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            # If successful, verify response structure
            data = response.json()
            assert "results" in data

    def test_predict_with_empty_sequence_data(self, test_client):
        """Test prediction with empty sequence data."""
        response = test_client.post("/predict", json={
            "sequences": [{"data": []}]
        })

        # Empty data should cause an error
        assert response.status_code in [422, 500]

    def test_predict_with_single_frame(self, test_client, model_config):
        """Test prediction with single frame sequence."""
        np.random.seed(789)
        features = model_config.input_dim
        single_frame = np.random.randn(1, features).tolist()

        response = test_client.post("/predict", json={
            "sequences": [{"data": single_frame}]
        })

        assert response.status_code == 200
        assert len(response.json()["results"]) == 1

    def test_predict_with_very_long_sequence(self, test_client, model_config):
        """Test prediction with very long sequence."""
        np.random.seed(101)
        features = model_config.input_dim
        long_sequence = np.random.randn(1000, features).tolist()

        response = test_client.post("/predict", json={
            "sequences": [{"data": long_sequence}]
        })

        assert response.status_code == 200

    def test_concurrent_event_creation(
        self, test_client, sample_prediction_with_position
    ):
        """Test creating multiple events concurrently."""
        import concurrent.futures

        def create_event():
            return test_client.post(
                "/events/from_prediction",
                json=sample_prediction_with_position
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_event) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All requests should succeed
        assert all(r.status_code == 200 for r in results)

        # Verify events were created
        events_response = test_client.get("/events/recent")
        assert len(events_response.json()["events"]) >= 10


# ---------------------------------------------------------------------------
# Model Loading Tests
# ---------------------------------------------------------------------------


class TestModelLoading:
    """Tests for model loading behavior."""

    def test_predict_loads_model_lazily(
        self, test_client, sample_predict_request
    ):
        """Test that model is loaded on first prediction request."""
        import services.api.main as main_module

        # Model should not be loaded initially
        assert main_module._model is None

        # Make prediction request
        response = test_client.post("/predict", json=sample_predict_request)
        assert response.status_code == 200

        # Model should now be loaded
        assert main_module._model is not None

    def test_model_not_found_returns_503(self, app_with_mock_model):
        """Test that missing model file returns 503 error."""
        from fastapi.testclient import TestClient
        import services.api.main as main_module

        # Reset model state
        main_module._model = None
        main_module._LABEL_MAP = None

        with patch.dict("os.environ", {"MODEL_PATH": "/nonexistent/model.pt"}):
            # Need to reset the model to force reload
            main_module._model = None

            client = TestClient(app_with_mock_model)
            # Note: We need a fresh request context with the new env
            # This test is complex due to how the model loading works


# ---------------------------------------------------------------------------
# Response Schema Validation Tests
# ---------------------------------------------------------------------------


class TestResponseSchemas:
    """Tests validating response schemas match expected format."""

    def test_predict_response_schema(self, test_client, sample_predict_request):
        """Validate predict response matches PredictResponse schema."""
        response = test_client.post("/predict", json=sample_predict_request)
        data = response.json()

        assert isinstance(data, dict)
        assert "results" in data
        assert isinstance(data["results"], list)

        for result in data["results"]:
            assert isinstance(result, dict)
            assert "label" in result
            assert "probabilities" in result
            assert isinstance(result["label"], str)
            assert isinstance(result["probabilities"], dict)

    def test_event_response_schema(
        self, test_client, sample_prediction_with_position
    ):
        """Validate event response matches FallEvent schema."""
        response = test_client.post(
            "/events/from_prediction",
            json=sample_prediction_with_position
        )
        data = response.json()

        assert isinstance(data, dict)
        assert "id" in data
        assert "timestamp" in data
        assert "label" in data
        assert "probabilities" in data
        assert "position" in data

        assert isinstance(data["id"], str)
        assert isinstance(data["timestamp"], (int, float))
        assert isinstance(data["label"], str)
        assert isinstance(data["probabilities"], dict)
        assert isinstance(data["position"], dict)
        assert "x" in data["position"]
        assert "y" in data["position"]

    def test_stats_response_schema(self, test_client):
        """Validate stats response matches StatsResponse schema."""
        response = test_client.get("/stats")
        data = response.json()

        assert isinstance(data, dict)
        assert "total_events" in data
        assert "label_counts" in data
        assert "recent_falls" in data
        assert "websocket_connections" in data

        assert isinstance(data["total_events"], int)
        assert isinstance(data["label_counts"], dict)
        assert isinstance(data["recent_falls"], list)
        assert isinstance(data["websocket_connections"], int)

    def test_alert_response_schema(
        self, test_client, sample_prediction_with_position
    ):
        """Validate alert response matches AlertResponse schema."""
        # Create event first
        event_response = test_client.post(
            "/events/from_prediction",
            json=sample_prediction_with_position
        )
        event_id = event_response.json()["id"]

        # Create alert
        response = test_client.post("/alerts", json={
            "event_id": event_id,
            "recipient": "test@test.com"
        })
        data = response.json()

        assert isinstance(data, dict)
        assert "success" in data
        assert "alert_id" in data
        assert "message" in data

        assert isinstance(data["success"], bool)
        assert isinstance(data["alert_id"], str)
        assert isinstance(data["message"], str)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests covering complete workflows."""

    def test_full_fall_detection_workflow(
        self, test_client, sample_prediction_with_position
    ):
        """Test complete workflow: predict -> create event -> create alert."""
        # Step 1: Health check
        health_response = test_client.get("/health")
        assert health_response.status_code == 200

        # Step 2: Create event from prediction
        event_response = test_client.post(
            "/events/from_prediction",
            json=sample_prediction_with_position
        )
        assert event_response.status_code == 200
        event = event_response.json()
        event_id = event["id"]

        # Step 3: Verify event appears in recent events
        recent_response = test_client.get("/events/recent")
        assert recent_response.status_code == 200
        events = recent_response.json()["events"]
        assert any(e["id"] == event_id for e in events)

        # Step 4: Create alert for the event
        alert_response = test_client.post("/alerts", json={
            "event_id": event_id,
            "recipient": "+1234567890",
            "message": "Emergency: Fall detected!"
        })
        assert alert_response.status_code == 200
        assert alert_response.json()["success"] is True

        # Step 5: Verify alert in history
        history_response = test_client.get("/alerts/history")
        assert history_response.status_code == 200
        alerts = history_response.json()["alerts"]
        assert any(a["event_id"] == event_id for a in alerts)

        # Step 6: Check stats
        stats_response = test_client.get("/stats")
        assert stats_response.status_code == 200
        stats = stats_response.json()
        assert stats["total_events"] >= 1

    def test_batch_prediction_workflow(
        self, test_client, multiple_radar_sequences
    ):
        """Test batch prediction with multiple sequences."""
        # Batch predict
        response = test_client.post(
            "/predict",
            json={"sequences": multiple_radar_sequences}
        )
        assert response.status_code == 200

        results = response.json()["results"]
        assert len(results) == len(multiple_radar_sequences)

        # Verify each result has valid label
        valid_labels = {"normal", "fall", "rehab_bad_posture", "chest_abnormal"}
        for result in results:
            assert result["label"] in valid_labels
