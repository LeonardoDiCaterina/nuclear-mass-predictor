from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nuclear_mass_predictor.pipelines.data_engineering.nodes import (
    create_ame_historical_dataset,
    create_engineered_features,
    fetch_and_parse_ame_data,
    fetch_iaea_data,
    parse_amdc_fixed_width,
)


@patch("nuclear_mass_predictor.pipelines.data_engineering.nodes.requests.get")
def test_fetch_iaea_data(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = "z,n,binding_energy\n1,1,2.22\n2,2,28.3"

    dummy_params = {
        "url": "https://fake-url.com",
        "headers": {"User-Agent": "test-agent"}
    }

    result_df = fetch_iaea_data(dummy_params)
    mock_get.assert_called_once_with("https://fake-url.com", headers={"User-Agent": "test-agent"}, timeout=30)
    
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 2
    assert list(result_df.columns) == ["z", "n", "binding_energy"]


def test_parse_amdc_fixed_width():
    # Construct mock AMDC fixed-width format lines (matching mass16.txt / mass_1.mas20.txt format)
    # Fortran format:
    # 0:4 cc/NZ, 4:9 N, 9:14 Z, 14:19 A, 20:23 EL, 54:65 BINDING/A (keV)
    header = "\n" * 36
    # Line 1: H-1 (Z=1, N=0, A=1) -> single nucleon, should be skipped
    line_h1 = "0 -1    0    1    1 H          7288.97061    0.00009      0.0      0.0   B-      *                1 007825.03224    0.00009"
    # Line 2: H-2 (Z=1, N=1, A=2) -> Deuterium, BE/A = 1112.283 keV -> BE_total = 2.224566 MeV
    line_h2 = "0  0    1    1    2 H         13135.72176    0.00011   1112.283    0.000 B-      *                2 014101.77811    0.00012"
    # Line 3: He-4 (Z=2, N=2, A=4) -> Alpha, BE/A = 7073.915 keV -> BE_total = 28.29566 MeV
    line_he4 = "   0    2    2    4 He         2424.91561    0.00006   7073.915    0.000 B- -22898.273  212.132   4 002603.25413    0.00006"

    mock_text = header + "\n" + line_h1 + "\n" + line_h2 + "\n" + line_he4

    df = parse_amdc_fixed_width(mock_text, start_line=36)

    assert len(df) == 2  # H-1 is filtered out
    assert df.loc[0, "z"] == 1
    assert df.loc[0, "n"] == 1
    assert df.loc[0, "a"] == 2
    assert df.loc[0, "el"] == "H"
    assert df.loc[0, "binding_energy_per_a_kev"] == pytest.approx(1112.283)
    assert df.loc[0, "binding_energy_total_mev"] == pytest.approx(2.224566)

    assert df.loc[1, "z"] == 2
    assert df.loc[1, "n"] == 2
    assert df.loc[1, "a"] == 4
    assert df.loc[1, "el"] == "He"
    assert df.loc[1, "binding_energy_total_mev"] == pytest.approx(28.29566)


def test_create_ame_historical_dataset():
    # Mock AME2016 (Nuclide A and Nuclide B)
    df_2016 = pd.DataFrame({
        "z": [8, 20],
        "n": [8, 20],
        "a": [16, 40],
        "el": ["O", "Ca"],
        "binding_energy_per_a_kev": [7976.2, 8551.3],
        "binding_energy": [7976.2, 8551.3],
        "binding_energy_total_mev": [127.619, 342.052],
    })

    # Mock AME2020 (Nuclide A, B, and newly discovered Nuclide C)
    df_2020 = pd.DataFrame({
        "z": [8, 20, 26],
        "n": [8, 20, 30],
        "a": [16, 40, 56],
        "el": ["O", "Ca", "Fe"],
        "binding_energy_per_a_kev": [7976.2, 8551.3, 8790.3],
        "binding_energy": [7976.2, 8551.3, 8790.3],
        "binding_energy_total_mev": [127.619, 342.052, 492.256],
    })

    ws4_params = {"kappa": 1.139, "xi": 1.250, "fs": 1.0}
    baseline_params = {"model_type": "none"}

    full_df, train_df, test_df = create_ame_historical_dataset(df_2016, df_2020, ws4_params, baseline_params)

    assert len(full_df) == 3
    assert len(train_df) == 2
    assert len(test_df) == 1

    # Check that Fe-56 (Z=26, N=30) was flagged as test20
    assert test_df.loc[0, "z"] == 26
    assert test_df.loc[0, "n"] == 30
    assert test_df.loc[0, "is_test20"] == True

    # Check physical features presence
    expected_cols = ["z_eo", "n_eo", "delta_z", "delta_n", "asy", "is_ws4_subset", "residual_energy"]
    for col in expected_cols:
        assert col in full_df.columns
        assert col in train_df.columns
        assert col in test_df.columns


@patch("nuclear_mass_predictor.pipelines.data_engineering.nodes.requests.get")
def test_fetch_and_parse_ame_data(mock_get):
    header = "\n" * 36
    line_h2 = "0  0    1    1    2 H         13135.72176    0.00011   1112.283    0.000 B-      *                2 014101.77811    0.00012"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = header + "\n" + line_h2
    mock_get.return_value = mock_resp

    ame_params = {
        "ame2016_url": "https://test.org/2016",
        "ame2020_url": "https://test.org/2020",
        "headers": {"User-Agent": "test"}
    }

    df16, df20 = fetch_and_parse_ame_data(ame_params)
    assert len(df16) == 1
    assert len(df20) == 1
    assert df16.loc[0, "z"] == 1


def test_create_engineered_features():
    raw_data = pd.DataFrame({
        "z": [8, 20, 9],
        "n": [8, 21, 10],
        "binding_energy": [60.0, 160.0, 75.0]
    })
    ws4_params = {"kappa": 0.0, "xi": 0.0, "fs": 1.0}
    baseline_params = {"model_type": "none"}

    result_df = create_engineered_features(raw_data, ws4_params, baseline_params)

    expected_columns = [
        "z", "n", "binding_energy", "binding_energy_total_mev",
        "z_eo", "n_eo", "delta_z", "delta_n", "asy",
        "macroscopic_energy", "residual_energy"
    ]
    assert all(col in result_df.columns for col in expected_columns)
    assert len(result_df) == 3

