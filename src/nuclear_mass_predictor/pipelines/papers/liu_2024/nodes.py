import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from kan import KAN
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def split_data_for_kan(
    df: pd.DataFrame, 
    params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into training and testing sets based on Liu et al. (2024).
    Nuclei with Z >= 8 and N >= 8 are selected.
    Randomly divided into 2856 for training and 600 for testing.
    """
    filtered_df = df[(df["z"] >= 8) & (df["n"] >= 8)].copy()
    
    # Shuffle and split
    filtered_df = filtered_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    test_size = 600
    train_df = filtered_df.iloc[:-test_size].copy()
    test_df = filtered_df.iloc[-test_size:].copy()
    
    return filtered_df, train_df, test_df


def train_and_evaluate_kan(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    params: dict[str, Any]
) -> pd.DataFrame:
    """
    Trains and evaluates the 4 KAN models specified in Liu et al. 2024.
    """
    target_col = params.get("target_col", "mass_excess")
    grid_size = params.get("grid_size", 30)
    epochs = params.get("epochs", 500)
    learning_rate = params.get("learning_rate", 0.01)
    batch_size = params.get("batch_size", 64)
    model_suite = params.get("model_suite", {})
    
    results = []
    
    for model_name, features in model_suite.items():
        logging.info(f"Training {model_name} with features: {features}")
        
        # Data preparation
        X_train = train_df[features].values
        y_train = train_df[target_col].values
        X_test = test_df[features].values
        y_test = test_df[target_col].values
        
        # Scaling
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        X_test_scaled = scaler_X.transform(X_test)
        
        dataset = {
            'train_input': torch.tensor(X_train_scaled, dtype=torch.float32),
            'train_label': torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1),
            'test_input': torch.tensor(X_test_scaled, dtype=torch.float32),
            'test_label': torch.tensor(y_test, dtype=torch.float32).unsqueeze(1), # Not scaled for test eval yet
        }
        
        # Model definition [input_dim, 12, 1]
        model = KAN(width=[len(features), 12, 1], grid=grid_size, k=3, seed=42)
        
        # Training
        # pykan's train function expects dataset, opt, steps, etc.
        def train_acc():
            return torch.mean((model(dataset['train_input']) - dataset['train_label'])**2)
        
        def test_acc():
            preds_scaled = model(dataset['test_input']).detach().numpy()
            preds = scaler_y.inverse_transform(preds_scaled).flatten()
            return np.mean((preds - y_test)**2)
            
        logging.info("Starting KAN training loop...")
        results_hist = model.fit(
            dataset, 
            opt="LBFGS", 
            steps=epochs, 
            metrics=(train_acc, test_acc),
            loss_fn=torch.nn.MSELoss()
        )
        
        # Final Evaluation
        preds_scaled = model(dataset['test_input']).detach().numpy()
        preds = scaler_y.inverse_transform(preds_scaled).flatten()
        
        res_df = test_df.copy()
        res_df["prediction"] = preds
        res_df["residual"] = res_df[target_col] - res_df["prediction"]
        res_df["model_name"] = model_name
        
        results.append(res_df)
        
    unified_results = pd.concat(results, ignore_index=True)
    return unified_results

def compute_kan_metrics(unified_results: pd.DataFrame) -> dict[str, float]:
    metrics = {}
    for model_name in unified_results["model_name"].unique():
        subset = unified_results[unified_results["model_name"] == model_name]
        rmsd = np.sqrt(np.mean(subset["residual"]**2))
        mae = np.mean(np.abs(subset["residual"]))
        metrics[f"{model_name}_rmsd"] = float(rmsd)
        metrics[f"{model_name}_mae"] = float(mae)
    return metrics
