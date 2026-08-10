import numpy as np
import pandas as pd
import pytest

from nuclear_mass_predictor.pipelines.papers.zeng_2022.nodes import (
    scale_features,
    split_historical_data,
)


def test_split_historical_data():
    # Arrange: Create a mock dataframe with more than 122 rows
    data = {
        "z": [1] * 150,
        "n": [1] * 150,
        "z_eo": [1] * 150,
        "n_eo": [1] * 150,
        "delta_z": [0.0] * 150,
        "delta_n": [0.0] * 150,
        "asy": [0.1] * 150,
        "binding_energy": [2.0] * 150,
        "discovery": list(range(150)) # Mock discovery order
    }
    df = pd.DataFrame(data)
    params = {"test_size_target": 20} # Test with a smaller holdout size

    # Act
    X_train, X_test, y_train, y_test = split_historical_data(df, params)

    # Assert
    assert len(X_test) == 20
    assert len(X_train) == 130
    assert len(y_test) == 20
    assert len(y_train) == 130
    assert list(X_train.columns) == ["z", "n", "z_eo", "n_eo", "delta_z", "delta_n", "asy"]


def test_scale_features():
    # Arrange: Create simple numeric training and test frames
    X_train = pd.DataFrame({
        "feat1": [0.0, 10.0, 20.0],
        "feat2": [5.0, 5.0, 5.0]
    })
    X_test = pd.DataFrame({
        "feat1": [10.0],
        "feat2": [5.0]
    })

    # Act
    X_train_scaled, _X_test_scaled, scaler_params = scale_features(X_train, X_test)

    # Assert
    # Check that feature means are centered around 0 for training data
    assert np.allclose(X_train_scaled.mean(axis=0), 0.0)
    
    # Check structure of exportable YAML parameters
    assert "mean" in scaler_params
    assert "scale" in scaler_params
    assert scaler_params["feature_names"] == ["feat1", "feat2"]
    assert len(scaler_params["mean"]) == 2
    assert len(scaler_params["scale"]) == 2

from nuclear_mass_predictor.pipelines.papers.zeng_2022.nodes import (
    compute_summary_metrics,
)


def test_compute_summary_metrics():
    """
    Test that the summary metrics node correctly calculates RMSD and MAE
    grouped by framework.
    """
    # Arrange: Create dummy residuals (True - Pred)
    # PyTorch residuals: [3.0, -3.0] -> MAE: 3.0, RMSD: 3.0
    # JAX residuals: [4.0, -4.0] -> MAE: 4.0, RMSD: 4.0
    dummy_unified_df = pd.DataFrame({
        "framework": ["pytorch", "pytorch", "jax", "jax"],
        "residual": [3.0, -3.0, 4.0, -4.0]
    })

    # Act
    metrics = compute_summary_metrics(dummy_unified_df)

    # Assert
    assert metrics["pytorch_test_mae_mev"] == pytest.approx(3.0)
    assert metrics["pytorch_test_rmsd_mev"] == pytest.approx(3.0)
    assert metrics["jax_test_mae_mev"] == pytest.approx(4.0)
    assert metrics["jax_test_rmsd_mev"] == pytest.approx(4.0)


from nuclear_mass_predictor.pipelines.papers.zeng_2022.nodes import (
    train_jax_model,
    train_pytorch_model,
)


def test_train_pytorch_model_returns_tuple():
    """Test that PyTorch training returns a model and a loss history dataframe."""
    X_train = np.random.randn(20, 7)
    y_train = pd.Series([100.0] * 20)
    params = {"epochs": 2, "batch_size": 10}

    model, loss_df = train_pytorch_model(X_train, y_train, params)

    assert model is not None
    assert not loss_df.empty
    assert "loss" in loss_df.columns
    assert len(loss_df) == 2  # 2 epochs tested

def test_train_jax_model_returns_tuple():
    """Test that JAX training returns params and a loss history dataframe."""
    X_train = np.random.randn(20, 7)
    y_train = pd.Series([100.0] * 20)
    params = {"epochs": 2, "batch_size": 10}

    jax_output, loss_df = train_jax_model(X_train, y_train, params)

    assert "params" in jax_output
    assert not loss_df.empty
    assert "loss" in loss_df.columns
    assert len(loss_df) == 2  # 2 epochs tested
