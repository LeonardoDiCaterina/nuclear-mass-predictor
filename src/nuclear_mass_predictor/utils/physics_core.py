import numpy as np

MAGIC_NUMBERS = np.array([8, 20, 28, 50, 82, 126, 184])

def calculate_pairing(number: int) -> int:
    """
    Calculates the pairing effect for protons (Z) or neutrons (N).
    Returns 1 if the number is odd, 0 otherwise.
    """
    return 1 if number % 2 != 0 else 0

def distance_to_magic(number: int) -> int:
    """
    Calculates the absolute difference between the nucleon number and the closest magic number.
    """
    distances = np.abs(MAGIC_NUMBERS - number)
    return int(np.min(distances))

def calculate_asy(z: int, n: int, kappa: float, xi: float, fs: float) -> float:
    """
    Calculates the isospin-asymmetry feature (ASY) based on the WS4 model parameters.
    """
    a = z + n
    if a == 0:
        return 0.0

    i_val = (n - z) / a
    abs_i = abs(i_val)

    term1 = 1 - (kappa / (a ** (1/3)))
    term2 = xi * ((2 - abs_i) / (2 + (abs_i * a)))

    asy = (term1 + term2) * (i_val ** 2) * a * fs
    return float(asy)


