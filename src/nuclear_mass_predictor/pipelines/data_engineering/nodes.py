"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.19.15
"""


import pandas as pd
import requests
import io
from typing import Dict, Any

def fetch_iaea_data(api_params: Dict[str, Any]) -> pd.DataFrame:
    """
    Node 1: Fetches ground states data from the IAEA API.
    Uses headers to avoid 403 Forbidden errors.
    """
    url = api_params["url"]
    headers = api_params["headers"]

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    csv_data = io.StringIO(response.text)

    return pd.read_csv(csv_data)


