import pytest

from nuclear_mass_predictor.utils.physics_core import (
    calculate_asy,
    calculate_liquid_drop_energy,
    calculate_pairing,
    calculate_ws4_macroscopic_energy,
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

def test_calculate_liquid_drop_energy():
    # Edge case: A = 0
    assert calculate_liquid_drop_energy(0, 0, {}) == 0.0
    
    # Typical LDM parameters
    ldm_params = {
        "a_v": 15.75,
        "a_s": 17.8,
        "a_c": 0.711,
        "a_a": 23.7,
        "delta_0": 11.18
    }
    
    # Fe-56 (Z=26, N=30 -> even-even, A=56)
    # Energy should be strongly positive (around ~490 MeV)
    be_fe56 = calculate_liquid_drop_energy(26, 30, ldm_params)
    assert be_fe56 > 450.0
    assert be_fe56 < 520.0
    
    # Check even-even has higher binding energy than odd-odd with similar A
    # Even-even (Z=26, N=30) vs odd-odd (Z=27, N=29)
    be_odd_odd = calculate_liquid_drop_energy(27, 29, ldm_params)
    # Due to pairing delta (+delta vs -delta), even-even is strictly more bound
    assert be_fe56 > be_odd_odd

def test_calculate_ws4_macroscopic_energy():
    # Edge case: A = 0
    assert calculate_ws4_macroscopic_energy(0, 0, {}) == 0.0
    
    ws4_params = {
        "a_v": 15.5906,
        "a_s": 17.0251,
        "a_c": 0.7053,
        "kappa": 1.139,
        "delta_0": 11.2
    }
    
    # Fe-56 (Z=26, N=30)
    be_fe56 = calculate_ws4_macroscopic_energy(26, 30, ws4_params)
    assert be_fe56 > 450.0
    assert be_fe56 < 520.0

