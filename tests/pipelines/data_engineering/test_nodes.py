import pandas as pd

from nuclear_mass_predictor.pipelines.data_engineering.nodes import (
    create_ame_historical_dataset,
    create_engineered_features,
    fetch_iaea_data,
    parse_amdc_fixed_width,
)


def test_create_ame_historical_dataset(dummy_ame_raw):
    """Test that primary features are calculated correctly from raw AME data"""
    ws4_params = {"kappa": 1.0, "xi": 1.0, "fs": 1.0}
    baseline_params = {"c1": 1.0, "c2": 1.0}
    
    # We pass the same raw df for both 2016 and 2020 just to test the logic
    # In reality, they'd be different, but for testing transformation logic, this is fine
    primary_features, train_df, test_df = create_ame_historical_dataset(
        dummy_ame_raw, dummy_ame_raw, ws4_params, baseline_params
    )
    
    assert isinstance(primary_features, pd.DataFrame)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    
    # Check if Magic number distance logic worked (Pb-208 is doubly magic, distance should be 0)
    pb208 = primary_features[(primary_features['z'] == 82) & (primary_features['n'] == 126)]
    assert pb208['delta_z'].iloc[0] == 0
    assert pb208['delta_n'].iloc[0] == 0

    # Check parity logic
    he5 = primary_features[(primary_features['z'] == 2) & (primary_features['n'] == 3)]
    assert he5['z_eo'].iloc[0] == 0 # Z=2 is even -> 0
    assert he5['n_eo'].iloc[0] == 1 # N=3 is odd -> 1

def test_parse_amdc_fixed_width():
    # Mock AME text format (very long fixed width)
    # n_str: 4:9, z_str: 9:14, a_str: 14:19, el: 20:23, me: 28:41, be: 54:65
    mock_text = "\n" * 36 
    
    def make_line(n, z, a, el, me, be):
        c = [" "] * 100
        c[4:9] = f"{n:5d}"
        c[9:14] = f"{z:5d}"
        c[14:19] = f"{a:5d}"
        c[20:23] = f"{el:<3}"
        c[28:28+len(me)] = list(me)
        c[54:54+len(be)] = list(be)
        return "".join(c) + "\n"
        
    mock_text += make_line(2, 2, 4, "He", "2424.915", "7073.915")
    mock_text += make_line(1, 0, 1, "n", "8071.318", "0.000")
    mock_text += make_line(126, 82, 208, "Pb", "-21759.", "7867#7")
    
    df = parse_amdc_fixed_width(mock_text, start_line=36)
    
    assert len(df) == 2 
    assert df.iloc[0]["z"] == 2
    assert df.iloc[1]["is_extrapolated"] == True

def test_create_engineered_features(dummy_ame_raw):
    ws4_params = {"kappa": 1.0, "xi": 1.0, "fs": 1.0}
    baseline_params = {"model_type": "none"}
    
    # Rename for legacy structure
    dummy_ame_raw["binding_energy"] = dummy_ame_raw["binding_energy_total_mev"] * 1000 / dummy_ame_raw["a"]
    
    df = create_engineered_features(dummy_ame_raw, ws4_params, baseline_params)
    assert "binding_energy_total_mev" in df.columns
    assert "macroscopic_energy" in df.columns
    assert "residual_energy" in df.columns

def test_fetch_iaea_data():
    import requests
    class MockResponse:
        def __init__(self, text): self.text = text
        def raise_for_status(self): pass
    
    mock_csv = "z,n,binding,other\n1,1,1000,x\n2,2,2000,y\n"
    
    # Monkeypatch requests.get
    old_get = requests.get
    def mock_get(url, **kwargs): return MockResponse(mock_csv)
    requests.get = mock_get
    
    try:
        df = fetch_iaea_data({"url": "http://fake.url", "headers": {}})
        assert len(df) == 2
        assert "binding_energy" in df.columns
        assert "z" in df.columns
    finally:
        requests.get = old_get
