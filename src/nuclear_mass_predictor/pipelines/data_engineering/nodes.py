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
    calculate_pairing,
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
    ws4_params: dict[str, Any]
) -> pd.DataFrame:
    """
    Node 2: Takes the validated raw data and calculates the physical priors.
    """
    df = raw_data.copy()

    # 1. Pairing effects
    df['z_eo'] = df['z'].apply(calculate_pairing)
    df['n_eo'] = df['n'].apply(calculate_pairing)

    # 2. Shell effects
    df['delta_z'] = df['z'].apply(distance_to_magic)
    df['delta_n'] = df['n'].apply(distance_to_magic)

    # 3. Isospin-asymmetry effect
    df['asy'] = df.apply(
        lambda row: calculate_asy(
            z=row['z'],
            n=row['n'],
            kappa=ws4_params['kappa'],
            xi=ws4_params['xi'],
            fs=ws4_params['fs']
        ),
        axis=1
    )

    return df
