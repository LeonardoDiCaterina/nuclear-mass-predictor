"""
Data engineering pipeline nodes for Nuclear Mass Predictor.
Ingests and processes Atomic Mass Data Center (AMDC) AME2016 and AME2020 mass evaluations.
"""

import io
from typing import Any

import numpy as np
import pandas as pd
import requests

from nuclear_mass_predictor.utils.physics_core import (
    calculate_asy,
    calculate_liquid_drop_energy,
    calculate_pairing,
    calculate_shell_index,
    calculate_ws4_macroscopic_energy,
    distance_to_magic,
)


def parse_amdc_fixed_width(raw_text: str, start_line: int = 36) -> pd.DataFrame:
    """
    Parses official Atomic Mass Data Center (AMDC) fixed-width ASCII tables (mass16.txt, mass_1.mas20.txt).
    Extracts Z, N, A, Element, and binding energy per nucleon (keV/A).
    Calculates direct total binding energy in MeV: BE_total = (BE_per_A * A) / 1000.0.
    """
    records = []
    for line in raw_text.splitlines()[start_line:]:
        if len(line) < 65:
            continue
        n_str = line[4:9].strip()
        z_str = line[9:14].strip()
        a_str = line[14:19].strip()
        el = line[20:23].strip()
        be_str_raw = line[54:65].strip()
        mass_excess_str_raw = line[28:41].strip()
        is_extrapolated = "#" in be_str_raw or "#" in mass_excess_str_raw
        be_str = be_str_raw.replace("#", ".")
        me_str = mass_excess_str_raw.replace("#", ".")

        if n_str.isdigit() and z_str.isdigit() and a_str.isdigit():
            n = int(n_str)
            z = int(z_str)
            a = int(a_str)
            # Exclude unbound single nucleons (A=1, BE=0)
            if a == 1 and (z == 0 or (z == 1 and n == 0)):
                continue
            be_val = float(be_str.replace("*", "")) if be_str else np.nan
            be_total = (be_val * a) / 1000.0 if not np.isnan(be_val) else np.nan
            me_val = float(me_str.replace("*", "")) / 1000.0 if me_str else np.nan # AME gives Mass Excess in keV, convert to MeV
            records.append({
                "z": z,
                "n": n,
                "a": a,
                "el": el,
                "mass_excess": me_val,
                "binding_energy_per_a_kev": be_val,
                "binding_energy": be_val,
                "binding_energy_total_mev": be_total,
                "is_extrapolated": is_extrapolated,
            })
    return pd.DataFrame(records)


def fetch_and_parse_ame_data(ame_params: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches raw AMDC AME2016 and AME2020 mass tables from IAEA / AMDC endpoints,
    and returns parsed dataframes.
    """
    headers = ame_params.get("headers", {"User-Agent": "Mozilla/5.0"})
    url_2016 = ame_params.get("ame2016_url", "https://www-nds.iaea.org/amdc/ame2016/mass16.txt")
    url_2020 = ame_params.get("ame2020_url", "https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt")

    r16 = requests.get(url_2016, headers=headers, timeout=30)
    r16.raise_for_status()
    df_16 = parse_amdc_fixed_width(r16.text)

    r20 = requests.get(url_2020, headers=headers, timeout=30)
    r20.raise_for_status()
    df_20 = parse_amdc_fixed_width(r20.text)

    return df_16, df_20


def create_ame_historical_dataset(
    df_2016: pd.DataFrame,
    df_2020: pd.DataFrame,
    ws4_params: dict[str, Any],
    baseline_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Constructs the feature-engineered dataset and applies the exact historical split:
    - Training set: Nuclides in AME2016 (3434 nuclides)
    - Test set (test20): Newly compiled nuclides in AME2020 not in AME2016 (122 nuclides)
    - WS4 subset flag: Z >= 8 and N >= 8 (3336 train, 120 test)
    """
    df = df_2020.copy()

    # 1. Historical split tagging
    keys_16 = set(zip(df_2016["z"], df_2016["n"]))
    keys_16_extrap = set(zip(df_2016[df_2016["is_extrapolated"]]["z"], df_2016[df_2016["is_extrapolated"]]["n"]))
    
    df["is_test20"] = [(row.z, row.n) not in keys_16 for row in df.itertuples()]
    df["is_promoted_test"] = [
        ((row.z, row.n) in keys_16_extrap) and (not row.is_extrapolated)
        for row in df.itertuples()
    ]
    df["is_ws4_subset"] = (df["z"] >= 8) & (df["n"] >= 8)

    # 2. Physics priors (5 features)
    df["z_eo"] = df["z"].apply(calculate_pairing)
    df["n_eo"] = df["n"].apply(calculate_pairing)
    df["delta_z"] = df["z"].apply(distance_to_magic)
    df["delta_n"] = df["n"].apply(distance_to_magic)
    df["a_2_3"] = df["a"] ** (2/3)
    df["isospin_asym_absolute"] = np.abs(df["n"] - df["z"])
    df["z_shell"] = df["z"].apply(calculate_shell_index)
    df["n_shell"] = df["n"].apply(calculate_shell_index)
    df["asy"] = df.apply(lambda row: calculate_asy(row["z"], row["n"], **ws4_params), axis=1)

    # 3. Macroscopic baseline & residual
    model_type = baseline_params.get("model_type", "none")
    if model_type == "ldm":
        df["macroscopic_energy"] = df.apply(
            lambda row: calculate_liquid_drop_energy(row["z"], row["n"], baseline_params.get("ldm_coeffs", {})),
            axis=1,
        )
    elif model_type == "ws4":
        df["macroscopic_energy"] = df.apply(
            lambda row: calculate_ws4_macroscopic_energy(row["z"], row["n"], baseline_params.get("ws4_coeffs", {})),
            axis=1,
        )
    else:
        df["macroscopic_energy"] = 0.0

    df["residual_energy"] = df["binding_energy_total_mev"] - df["macroscopic_energy"]

    train_df = df[~df["is_test20"]].copy().reset_index(drop=True)
    test_df = df[df["is_test20"]].copy().reset_index(drop=True)

    return df, train_df, test_df


def fetch_iaea_data(api_params: dict[str, Any]) -> pd.DataFrame:
    """
    Legacy IAEA Live Chart fetcher node.
    """
    url = api_params["url"]
    headers = api_params["headers"]

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    csv_data = io.StringIO(response.text)
    df = pd.read_csv(csv_data)
    df = df.rename(columns={"binding": "binding_energy"})
    df = df[["z", "n", "binding_energy"]].dropna(subset=["binding_energy"])

    return df


def create_engineered_features(
    raw_data: pd.DataFrame,
    ws4_params: dict[str, Any],
    baseline_params: dict[str, Any],
) -> pd.DataFrame:
    """
    Legacy engineered features node.
    """
    df = raw_data.copy()
    a = df["z"] + df["n"]
    df["binding_energy_total_mev"] = (df["binding_energy"] * a) / 1000.0

    df["z_eo"] = df["z"].apply(calculate_pairing)
    df["n_eo"] = df["n"].apply(calculate_pairing)
    df["delta_z"] = df["z"].apply(distance_to_magic)
    df["delta_n"] = df["n"].apply(distance_to_magic)
    df["asy"] = df.apply(lambda row: calculate_asy(row["z"], row["n"], **ws4_params), axis=1)

    model_type = baseline_params.get("model_type", "none")
    if model_type == "ldm":
        df["macroscopic_energy"] = df.apply(
            lambda row: calculate_liquid_drop_energy(row["z"], row["n"], baseline_params["ldm_coeffs"]),
            axis=1,
        )
    elif model_type == "ws4":
        df["macroscopic_energy"] = df.apply(
            lambda row: calculate_ws4_macroscopic_energy(row["z"], row["n"], baseline_params["ws4_coeffs"]),
            axis=1,
        )
    else:
        df["macroscopic_energy"] = 0.0

    df["residual_energy"] = df["binding_energy_total_mev"] - df["macroscopic_energy"]
    return df


