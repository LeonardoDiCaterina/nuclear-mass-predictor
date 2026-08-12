"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.19.15
"""


import io
from typing import Any

import pandas as pd
import pandera.pandas as pa
import requests

from nuclear_mass_predictor.schemas.ame2020 import RawNuclearSchema
from nuclear_mass_predictor.utils.physics_core import (
    calculate_asy,
    calculate_liquid_drop_energy,
    calculate_pairing,
    calculate_ws4_macroscopic_energy,
    distance_to_magic,
)


def fetch_iaea_data(api_params: dict[str, Any]) -> pd.DataFrame:
    """
    Node 1: Fetches ground states data from the IAEA API.
    Uses headers to avoid 403 Forbidden errors.
    Cleans and subsets the data for the pipeline.
    """
    url = api_params["url"]
    headers = api_params["headers"]

    response = requests.get(url, headers=headers)
    response.raise_for_status() 

    csv_data = io.StringIO(response.text)
    df = pd.read_csv(csv_data)

    # Clean the dataframe to match our Pandera schema
    df = df.rename(columns={"binding": "binding_energy"})
    
    # Keep only the features we need
    df = df[["z", "n", "binding_energy"]]
    
    # Drop rows where binding energy might be missing
    df = df.dropna(subset=["binding_energy"])

    return df



@pa.check_types
def create_engineered_features(
    raw_data: pa.typing.DataFrame[RawNuclearSchema],
    ws4_params: dict[str, Any],
    baseline_params: dict[str, Any]
) -> pd.DataFrame:
    df = raw_data.copy()

    # 0. Convert API Binding Energy (keV/nucleon) to Total Binding Energy (MeV)
    a = df['z'] + df['n']
    df['binding_energy_total_mev'] = (df['binding_energy'] * a) / 1000.0

    # 1. Physical features
    df['z_eo'] = df['z'].apply(calculate_pairing)
    df['n_eo'] = df['n'].apply(calculate_pairing)
    df['delta_z'] = df['z'].apply(distance_to_magic)
    df['delta_n'] = df['n'].apply(distance_to_magic)
    df['asy'] = df.apply(lambda row: calculate_asy(row['z'], row['n'], **ws4_params), axis=1)

    # 2. Configurable Macroscopic Baseline
    model_type = baseline_params.get("model_type", "none")
    
    if model_type == "ldm":
        df['macroscopic_energy'] = df.apply(
            lambda row: calculate_liquid_drop_energy(row['z'], row['n'], baseline_params['ldm_coeffs']), 
            axis=1
        )
    elif model_type == "ws4":
        df['macroscopic_energy'] = df.apply(
            lambda row: calculate_ws4_macroscopic_energy(row['z'], row['n'], baseline_params['ws4_coeffs']), 
            axis=1
        )
    else:
        df['macroscopic_energy'] = 0.0

    # 3. Target Definition (Apples to Apples!)
    df['residual_energy'] = df['binding_energy_total_mev'] - df['macroscopic_energy']

    return df

