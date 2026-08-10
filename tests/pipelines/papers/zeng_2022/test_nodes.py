import numpy as np
import pandas as pd

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
