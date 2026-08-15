from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from nuclear_mass_predictor.api.main import app, inference_service
from nuclear_mass_predictor.api.service import ModelInferenceService


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


def test_service_extract_features():
    svc = ModelInferenceService()
    feats = svc.extract_features(20, 28)
    assert len(feats) == 7
    assert feats[0] == 20.0
    assert feats[1] == 28.0


def test_service_missing_model_file(tmp_path):
    svc = ModelInferenceService(model_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        svc.load_model()


def test_service_real_prediction():
    svc = ModelInferenceService()
    if svc.model_path.exists() and svc.scaler_path.exists():
        total_be, be_per_a = svc.predict_single(20, 28)
        assert total_be > 0
        assert be_per_a > 0


def test_predict_single_path_real(client):
    if inference_service.model_path.exists() and inference_service.scaler_path.exists():
        response = client.get("/predict/20/28")
        assert response.status_code == 200
        data = response.json()
        assert data["z"] == 20
        assert data["n"] == 28
        assert data["a"] == 48
        assert data["predicted_binding_energy_total_mev"] > 0


def test_predict_post_real(client):
    if inference_service.model_path.exists() and inference_service.scaler_path.exists():
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
