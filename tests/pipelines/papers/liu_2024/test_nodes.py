import pandas as pd
import numpy as np
import pytest
from nuclear_mass_predictor.pipelines.papers.liu_2024.nodes import split_data_for_kan, train_and_evaluate_kan

def test_split_data_for_kan(dummy_features):
    params = {
        "target_col": "mass_excess",
        "model_suite": {
            "KAN_2": ["z", "n"],
            "KAN_4": ["z", "n", "a", "i"]
        }
    }
    
    # Passing dummy data as primary features
    dataset, train_df, test_df = split_data_for_kan(dummy_features, params)
    
    assert len(train_df) + len(test_df) == len(dataset)
    assert "mass_excess" in train_df.columns
    # Check that it split reasonably (e.g. at least 1 row in test if random state is fixed)
    assert isinstance(dataset, pd.DataFrame)

def test_train_and_evaluate_kan():
    # Smoke test the KAN model training loop
    train_df = pd.DataFrame({"z": [12, 12, 12, 12], "n": [12, 14, 16, 18], "mass_excess": [1.0, 2.0, 3.0, 4.0]})
    test_df = pd.DataFrame({"z": [12, 12], "n": [20, 22], "mass_excess": [5.0, 6.0]})
    
    params = {
        "smoke_run": True,
        "grid_size": 3,
        "epochs": 2,
        "learning_rate": 0.05,
        "batch_size": 2,
        "model_suite": {
            "KAN_2": ["z", "n"]
        }
    }
    
    preds = train_and_evaluate_kan(train_df, test_df, params)
    
    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == 2
    assert "prediction" in preds.columns
    assert "residual" in preds.columns
