import pandas as pd
from unittest.mock import patch
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
