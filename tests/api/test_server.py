from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from nuclear_mass_predictor.api.main import app, inference_service


@pytest.fixture
def client():
    """Provides a FastAPI test client."""
    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["winning_model"] == "ANN7 (JAX)"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "ANN7"
    assert data["framework"] == "jax"


def test_predict_single_path_mocked(client):
    with patch.object(inference_service, "predict_single", return_value=(418.7, 8.723)):
        # Simulate loaded state
        inference_service.is_loaded = True
        response = client.get("/predict/20/28")
        assert response.status_code == 200
        data = response.json()
        assert data["z"] == 20
        assert data["n"] == 28
        assert data["a"] == 48
        assert data["predicted_binding_energy_total_mev"] == 418.7
        assert data["predicted_binding_energy_per_nucleon_mev"] == 8.723


def test_predict_post_mocked(client):
    with patch.object(inference_service, "predict_single", return_value=(418.7, 8.723)):
        inference_service.is_loaded = True
        payload = {"nuclei": [{"z": 20, "n": 28}, {"z": 82, "n": 126}]}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
        assert data["predictions"][0]["z"] == 20
        assert data["predictions"][1]["z"] == 82


def test_predict_invalid_input(client):
    payload = {"nuclei": [{"z": -5, "n": 28}]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
