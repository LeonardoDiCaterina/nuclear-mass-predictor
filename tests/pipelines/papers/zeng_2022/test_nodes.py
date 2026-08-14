import pandas as pd
import numpy as np
import pytest
from nuclear_mass_predictor.pipelines.papers.zeng_2022.nodes import (
    scale_features,
    train_jax_model,
    train_pytorch_model,
    split_historical_data,
    evaluate_all_models
)

@pytest.fixture
def dummy_features():
    return pd.DataFrame({
        "z": np.random.rand(7),
        "n": np.random.rand(7),
        "a": np.random.rand(7),
        "mass_excess": np.random.rand(7)
    })

def test_scale_features():
    df = pd.DataFrame({
        "z": [2, 4],
        "n": [2, 4]
    })
    y = pd.Series([10.0, 20.0])
    
    X_tr, X_te, y_tr, y_te, params = scale_features(df, df, y, y, {"scaler_type": "standard", "scale_target": True})
    
    assert params["feature_names"] == ["z", "n"]
    assert "x_mean" in params
    assert X_tr.shape == (2, 2)
    
def test_train_jax_model():
    # Smoke test the JAX model loop
    X_train_scaled = np.random.rand(4, 3)
    y_train_scaled = pd.Series(np.random.rand(4))
    scaler_params = {"feature_names": ["z", "n", "asy"]}
    
    params = {
        "smoke_run": True, # This forces epochs to 5
        "learning_rate": 0.01,
        "batch_size": 2,
        "model_suite": {
            "test_model": {
                "features": ["z", "n", "asy"],
                "hidden_dims": [4, 4]
            }
        }
    }
    
    state, loss_df = train_jax_model(
        X_train_scaled, y_train_scaled, scaler_params, params, "test_model"
    )
    
    assert "params" in state
    assert len(loss_df) == 5 # 5 epochs

def test_train_pytorch_model():
    X_train_scaled = np.random.rand(4, 3)
    y_train_scaled = pd.Series(np.random.rand(4))
    scaler_params = {"feature_names": ["z", "n", "asy"]}
    
    params = {
        "smoke_run": True, # This forces epochs to 5
        "learning_rate": 0.01,
        "batch_size": 2,
        "model_suite": {
            "test_model": {
                "features": ["z", "n", "asy"],
                "hidden_dims": [4, 4]
            }
        }
    }
    
    model, loss_df = train_pytorch_model(
        X_train_scaled, y_train_scaled, scaler_params, params, "test_model"
    )
    
    assert model is not None
    assert len(loss_df) == 5

def test_split_historical_data(dummy_features):
    # dummy_features has z, n, a, mass_excess. Add is_test20 for split logic
    df = dummy_features.copy()
    df["is_test20"] = [False, False, False, False, False, True, True]
    df["is_ws4_subset"] = [True] * 7
    df["z_eo"] = 1
    df["n_eo"] = 1
    df["delta_z"] = 0
    df["delta_n"] = 0
    df["asy"] = 0.5
    df["binding_energy"] = df["mass_excess"]
    
    params = {
        "test_set_definition": "default",
        "ws4_subset_only": True,
        "target_col": "binding_energy"
    }
    
    X_train, X_test, y_train, y_test = split_historical_data(df, params)
    
    assert len(X_train) == 5
    assert len(X_test) == 2
    assert "z" in X_train.columns

def test_evaluate_all_models():
    # evaluate_all_models takes X_test_scaled, y_test, X_test_raw, scaler_params, params, **models
    X_test_scaled = np.random.rand(2, 2)
    y_test = pd.Series([10.0, 20.0])
    X_test_raw = pd.DataFrame({"z": [2, 4], "n": [2, 4]})
    scaler_params = {"y_mean": 0.0, "y_scale": 1.0, "feature_names": ["z", "n"]}
    params = {
        "model_suite": {
            "model1": {"features": ["z", "n"], "hidden_dims": [4]}
        }
    }
    
    class MockResult:
        def numpy(self): return np.array([[10.0], [20.0]])
        
    class MockTorchModel:
        def eval(self): pass
        def __call__(self, x): return MockResult()
    
    models = {"model1_pytorch": MockTorchModel()}
    
    results = evaluate_all_models(X_test_scaled, y_test, X_test_raw, scaler_params, params, **models)
    
    assert len(results) == 2
    assert "prediction" in results.columns
    assert "residual" in results.columns
