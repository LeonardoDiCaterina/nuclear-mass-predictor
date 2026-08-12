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


def calculate_liquid_drop_energy(z: int, n: int, ldm_params: dict[str, float]) -> float:
    """
    Computes the macroscopic binding energy using the Bethe-Weizsäcker Liquid Drop Model.
    """
    a = z + n
    if a == 0:
        return 0.0

    a_v = ldm_params.get("a_v", 15.75)
    a_s = ldm_params.get("a_s", 17.8)
    a_c = ldm_params.get("a_c", 0.711)
    a_a = ldm_params.get("a_a", 23.7)
    delta_0 = ldm_params.get("delta_0", 11.18)

    # Pairing term (delta)
    if z % 2 == 0 and n % 2 == 0:
        pairing = delta_0 / (a ** 0.5)
    elif z % 2 != 0 and n % 2 != 0:
        pairing = -delta_0 / (a ** 0.5)
    else:
        pairing = 0.0

    vol = a_v * a
    surf = a_s * (a ** (2 / 3))
    coul = a_c * (z * (z - 1)) / (a ** (1 / 3))
    asym = a_a * ((a - 2 * z) ** 2) / a

    return vol - surf - coul - asym + pairing

def calculate_ws4_macroscopic_energy(z: int, n: int, ws4_coeffs: dict[str, float]) -> float:
    """
    Computes the macroscopic binding energy using the Weizsäcker-Skyrme (WS4)
    parametrization, incorporating surface asymmetry.
    """
    a = z + n
    if a == 0:
        return 0.0

    # Isospin asymmetry
    i = (n - z) / a

    # WS4 empirical parameters
    a_v = ws4_coeffs.get("a_v", 15.5906)
    a_s = ws4_coeffs.get("a_s", 17.0251)
    a_c = ws4_coeffs.get("a_c", 0.7053)
    kappa = ws4_coeffs.get("kappa", 1.139)
    delta_0 = ws4_coeffs.get("delta_0", 11.2)

    # The WS4 innovation: asymmetry affects both volume and surface
    sym_factor = 1.0 - kappa * (i ** 2)

    vol = a_v * a * sym_factor
    surf = a_s * (a ** (2/3)) * sym_factor

    # Coulomb term (using Z^2 instead of Z(Z-1) in some WS variants, but we keep standard)
    coul = a_c * (z ** 2) / (a ** (1/3))

    # Pairing
    if z % 2 == 0 and n % 2 == 0:
        pairing = delta_0 / (a ** 0.5)
    elif z % 2 != 0 and n % 2 != 0:
        pairing = -delta_0 / (a ** 0.5)
    else:
        pairing = 0.0

    return vol - surf - coul + pairing
