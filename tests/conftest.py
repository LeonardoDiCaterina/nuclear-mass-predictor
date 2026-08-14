import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def dummy_ame_raw():
    """Returns a dummy raw AME DataFrame"""
    return pd.DataFrame({
        "n": [2, 3, 4, 126, 127],
        "z": [2, 2, 2, 82, 82],
        "a": [4, 5, 6, 208, 209],
        "mass_excess": [2.42, 11.39, 17.59, -21.75, -21.45],
        "binding_energy_total_mev": [28.3, 27.2, 29.2, 1636.4, 1640.2],
        "is_extrapolated": [False, False, True, False, True]
    })

@pytest.fixture
def dummy_features():
    """Returns a dummy computed features DataFrame"""
    return pd.DataFrame({
        "n": [2, 3, 4, 126, 127],
        "z": [2, 2, 2, 82, 82],
        "a": [4, 5, 6, 208, 209],
        "i": [0.0, 0.2, 0.33, 0.21, 0.215],
        "mass_excess": [2.42, 11.39, 17.59, -21.75, -21.45],
        "S_n": [20.5, 0.0, 0.0, 7.3, 3.9],
        "S_p": [19.8, 0.0, 0.0, 8.0, 7.8],
        "delta_z": [0.0, 0.0, 0.0, 0.0, 0.0],
        "delta_n": [0.0, 1.0, 2.0, 0.0, 1.0],
        "z_eo": [1, 1, 1, 1, 1],
        "n_eo": [1, -1, 1, 1, -1]
    })

@pytest.fixture
def dummy_liu_features():
    """Returns a small dataset for JAX KAN smoke testing"""
    return pd.DataFrame({
        "z": [12, 82, 50, 20],
        "n": [12, 126, 82, 28],
        "a": [24, 208, 132, 48],
        "i": [0.0, 0.21, 0.24, 0.16],
        "mass_excess": [2.4, -21.7, -76.5, -44.2]
    })
