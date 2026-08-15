import logging
from typing import Any

import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax import nnx
from jaxkan.KAN import KAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


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
    Trains and evaluates the 4 KAN models specified in Liu et al. 2024 using jaxKAN.
    """
    target_col = params.get("target_col", "mass_excess")
    grid_size = params.get("grid_size", 30)
    epochs = params.get("epochs", 500)
    learning_rate = params.get("learning_rate", 0.01)
    model_suite = params.get("model_suite", {})
    
    results = []
    
    for model_name, features in model_suite.items():
        logger.info(f"Training {model_name} with features: {features}")
        
        # Data preparation
        X_train = train_df[features].values
        y_train = train_df[target_col].values
        X_test = test_df[features].values
        _y_test = test_df[target_col].values
        
        # Scaling
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        X_test_scaled = scaler_X.transform(X_test)
        
        X_train_jnp = jnp.array(X_train_scaled, dtype=jnp.float32)
        y_train_jnp = jnp.array(y_train_scaled, dtype=jnp.float32).reshape(-1, 1)
        X_test_jnp = jnp.array(X_test_scaled, dtype=jnp.float32)
        
        # Initialize jaxKAN model
        model = KAN(
            layer_dims=[len(features), 12, 1], 
            layer_type='Spline',
            required_parameters={"G": grid_size, "k": 3}
        )
        
        # Setup Optimizer (optax.adamw with cosine decay)
        schedule = optax.cosine_decay_schedule(learning_rate, epochs)
        optimizer = nnx.Optimizer(model, optax.adamw(schedule))
        
        @nnx.jit
        def train_step(model, optimizer, x, y):
            def loss_fn(model):
                pred = model(x)
                return jnp.mean((pred - y) ** 2)
            
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(grads)
            return loss

        # Training Loop
        logger.info("Starting KAN training loop...")
        for epoch in range(epochs):
            loss = train_step(model, optimizer, X_train_jnp, y_train_jnp)
            if epoch % 500 == 0:
                logger.info(f"Epoch {epoch}/{epochs} Loss: {loss:.4f}")
        
        # Ensure we block until the last loss is computed to avoid async exit issues
        loss.block_until_ready()
        
        # Final Evaluation
        @nnx.jit
        def predict(model, x):
            return model(x)
            
        preds_scaled = np.array(predict(model, X_test_jnp))
        preds = scaler_y.inverse_transform(preds_scaled).flatten()
        
        res_df = test_df.copy()
        res_df["prediction"] = preds
        res_df["residual"] = res_df[target_col] - res_df["prediction"]
        res_df["model_name"] = model_name
        res_df["framework"] = "jaxkan"
        
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
