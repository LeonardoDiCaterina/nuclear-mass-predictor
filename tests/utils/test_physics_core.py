import pytest

from nuclear_mass_predictor.utils.physics_core import (
    calculate_asy,
    calculate_pairing,
    distance_to_magic,
)


def test_calculate_pairing():
    # Odd numbers should return 1
    assert calculate_pairing(7) == 1
    assert calculate_pairing(121) == 1
    
    # Even numbers should return 0
    assert calculate_pairing(8) == 0
    assert calculate_pairing(0) == 0

def test_distance_to_magic():
    # Exact magic numbers should return a distance of 0
    assert distance_to_magic(8) == 0
    assert distance_to_magic(126) == 0
    
    # Numbers close to 20
    assert distance_to_magic(22) == 2  # 22 - 20
    assert distance_to_magic(17) == 3  # 20 - 17
    
    # Numbers close to 28
    assert distance_to_magic(26) == 2  # 28 - 26
    
    # Numbers beyond the highest magic number (184)
    assert distance_to_magic(190) == 6 # 190 - 184

def test_calculate_asy():
    # Edge case: No nucleons (A=0) prevents division by zero
    assert calculate_asy(0, 0, 1.0, 1.0, 1.0) == 0.0
    
    # Symmetric nuclei (Z == N) have an isospin asymmetry of 0, so ASY must be 0
    assert calculate_asy(20, 20, 1.5, 2.0, 1.0) == 0.0
    
    # Deterministic math check for asymmetric nuclei
    # If Z=10, N=20 -> A=30, I=(20-10)/30 = 1/3
    # With kappa=0.0, xi=0.0, fs=1.0: term1 = 1, term2 = 0
    # ASY = 1 * (1/3)^2 * 30 * 1 = 30 / 9 = 3.3333...
    result = calculate_asy(10, 20, 0.0, 0.0, 1.0)
    assert pytest.approx(result, 0.0001) == 3.3333
