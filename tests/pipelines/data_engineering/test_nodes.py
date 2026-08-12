from unittest.mock import patch

import pandas as pd
import pytest

from nuclear_mass_predictor.pipelines.data_engineering.nodes import fetch_iaea_data


@patch("nuclear_mass_predictor.pipelines.data_engineering.nodes.requests.get")
def test_fetch_iaea_data(mock_get):
    # 1. Arrange: Create a mock response object with fake CSV data
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = "z,n,binding_energy\n1,1,2.22\n2,2,28.3"

    # Fake parameters to pass to the node
    dummy_params = {
        "url": "https://fake-url.com",
        "headers": {"User-Agent": "test-agent"}
    }

    # 2. Act: Call the node function
    result_df = fetch_iaea_data(dummy_params)

    # 3. Assert: Check that the API was called with the right params
    mock_get.assert_called_once_with("https://fake-url.com", headers={"User-Agent": "test-agent"})
    
    # Assert that the node correctly transformed the text into a DataFrame
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 2
    assert list(result_df.columns) == ["z", "n", "binding_energy"]


from nuclear_mass_predictor.pipelines.data_engineering.nodes import (
    create_engineered_features,
)


def test_create_engineered_features():
    # 1. Arrange: Create a dummy dataframe that satisfies the RawNuclearSchema
    raw_data = pd.DataFrame({
        "z": [8, 20, 9],            # 8 and 20 are magic, 9 is odd
        "n": [8, 21, 10],           # 8 is magic, 21 is odd, 10 is even
        "binding_energy": [60.0, 160.0, 75.0]
    })
    
    ws4_params = {
        "kappa": 0.0, 
        "xi": 0.0, 
        "fs": 1.0
    }
    baseline_params = {
        "model_type": "none"
    }

    # 2. Act: Pass the data through the node
    result_df = create_engineered_features(raw_data, ws4_params, baseline_params)

    # 3. Assert: Verify the new columns exist and the logic mapped correctly
    expected_columns = [
        "z", "n", "binding_energy", "binding_energy_total_mev",
        "z_eo", "n_eo", "delta_z", "delta_n", "asy",
        "macroscopic_energy", "residual_energy"
    ]
    assert all(col in result_df.columns for col in expected_columns)
    
    # Check total binding energy conversion: BE_total = BE * (z + n) / 1000.0
    # For z=8, n=8, BE=60.0 -> 60.0 * 16 / 1000 = 0.96 MeV
    assert result_df.loc[0, "binding_energy_total_mev"] == pytest.approx(0.96)
    assert result_df.loc[0, "residual_energy"] == pytest.approx(0.96)

    # Check pairing logic
    assert result_df.loc[0, "z_eo"] == 0  # z=8 (even)
    assert result_df.loc[2, "z_eo"] == 1  # z=9 (odd)
    assert result_df.loc[1, "n_eo"] == 1  # n=21 (odd)

    # Check shell distance logic
    assert result_df.loc[0, "delta_z"] == 0  # z=8 is exactly magic
    assert result_df.loc[1, "delta_z"] == 0  # z=20 is exactly magic
    assert result_df.loc[2, "delta_z"] == 1  # z=9 is 1 away from 8
    assert result_df.loc[1, "delta_n"] == 1  # n=21 is 1 away from 20

    # Ensure no rows were dropped during the transformation
    assert len(result_df) == 3
