from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def split_historical_data(
    df: pd.DataFrame, 
    params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into a historical training set (AME2016 proxy, ~3434 samples)
    and a holdout test set (AME2020 additions, 122 samples).
    """
    test_size = params.get("test_size_target", 122)
    
    if "discovery" in df.columns:
        df = df.sort_values(by="discovery")
        
    test_df = df.tail(test_size)
    train_df = df.head(len(df) - test_size)
    
    feature_cols = ["z", "n", "z_eo", "n_eo", "delta_z", "delta_n", "asy"]
    target_col = "binding_energy"
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Fits a StandardScaler on training data, transforms both splits, and exports
    pure statistical parameters (mean, scale) as a dictionary for YAML serialization.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": X_train.columns.tolist()
    }
    
    return X_train_scaled, X_test_scaled, scaler_params
