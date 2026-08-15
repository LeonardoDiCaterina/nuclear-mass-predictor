from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def split_historical_data(
    df: pd.DataFrame, 
    params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into a historical training set (AME2016, 3434 samples)
    and a holdout test set (AME2020 additions, 122 samples).
    Supports WS4 subset filtering (Z >= 8, N >= 8) if ws4_subset_only is configured.
    """
    if "is_test20" in df.columns:
        test_set_def = params.get("test_set_definition", "default")
        train_df = df[~df["is_test20"]].copy()
        
        if test_set_def == "promoted" and "is_promoted_test" in df.columns:
            test_df = df[df["is_promoted_test"]].copy()
        else:
            test_df = df[df["is_test20"]].copy()
    else:
        test_size = params.get("test_size_target", 122)
        if "discovery" in df.columns:
            df = df.sort_values(by="discovery")
        test_df = df.tail(test_size)
        train_df = df.head(len(df) - test_size)

    if params.get("ws4_subset_only", False) and "is_ws4_subset" in df.columns:
        train_df = train_df[train_df["is_ws4_subset"]].copy()
        test_df = test_df[test_df["is_ws4_subset"]].copy()
    
    feature_cols = ["z", "n", "z_eo", "n_eo", "delta_z", "delta_n", "asy"]
    target_col = params.get("target_col", "binding_energy_total_mev")
    if target_col not in train_df.columns:
        target_col = "residual_energy" if "residual_energy" in train_df.columns else "binding_energy"

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()
    
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: dict[str, Any]
) ->tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:

    """
    Fits a StandardScaler on training data, transforms both splits, and exports
    pure statistical parameters (mean, scale) as a dictionary for YAML serialization.
    """
    scale_target = params.get("scale_target", False)
    scaler_type = params.get("scaler_type", "standard")

    if scaler_type == "minmax":
        x_scaler = MinMaxScaler()
    else:
        x_scaler = StandardScaler()
        
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    if scale_target:
        if scaler_type == "minmax":
            y_scaler = MinMaxScaler()
            y_train_scaled = y_scaler.fit_transform(np.asarray(y_train.values).reshape(-1, 1)).ravel()
            y_test_scaled = y_scaler.transform(np.asarray(y_test.values).reshape(-1, 1)).ravel()
            y_mean = float(y_scaler.data_min_[0])
            y_scale = float(y_scaler.data_range_[0])
        else:
            y_scaler = StandardScaler()
            y_train_scaled = y_scaler.fit_transform(np.asarray(y_train.values).reshape(-1, 1)).ravel()
            y_test_scaled = y_scaler.transform(np.asarray(y_test.values).reshape(-1, 1)).ravel()
            y_mean = float(y_scaler.mean_[0])
            y_scale = float(y_scaler.scale_[0])
    else:
        y_train_scaled = y_train.values.astype(float)
        y_test_scaled = y_test.values.astype(float)
        y_mean = 0.0
        y_scale = 1.0
    scaler_params = {
        "x_mean": x_scaler.data_min_.tolist() if scaler_type == "minmax" else x_scaler.mean_.tolist(),
        "x_scale": x_scaler.data_range_.tolist() if scaler_type == "minmax" else x_scaler.scale_.tolist(),
        "y_mean": y_mean,
        "y_scale": y_scale,
        "scale_target": scale_target,
        "feature_names": X_train.columns.tolist()
    }
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_params


import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nuclear_mass_predictor.models.pytorch_ann import (
    DynamicNuclearANN as torch_DynamicNuclearANN,
)


def train_pytorch_model(
    X_train_scaled: np.ndarray,
    y_train_scaled: pd.Series,
    scaler_params: dict[str, Any],
    params: dict[str, Any],
    model_name: str
) -> tuple[nn.Module, pd.DataFrame]:
    """
    Trains the PyTorch dynamic ANN model and tracks epoch-by-epoch MAE loss.
    """
    lr = params.get("learning_rate", 0.0001)
    beta1 = params.get("beta1", 0.9)
    beta2 = params.get("beta2", 0.999)
    epochs = params.get("epochs", 10000)
    batch_size = params.get("batch_size", 64)
    if params.get("smoke_run", False):
        epochs = 5

    features = params["model_suite"][model_name]["features"]
    hidden_dims = params["model_suite"][model_name]["hidden_dims"]
    
    all_features = scaler_params["feature_names"]
    feature_indices = [all_features.index(f) for f in features]
    X_train_scaled_filtered = X_train_scaled[:, feature_indices]
    input_dim = len(features)

    X_tensor = torch.tensor(X_train_scaled_filtered, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
    weight_decay = params.get("weight_decay", 0.0)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = torch_DynamicNuclearANN(input_dim=input_dim, hidden_dims=hidden_dims)
    model.train()
    
    criterion = nn.L1Loss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    
    scheduler = None
    if params.get("lr_schedule") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    epoch_losses = []

    for epoch in range(epochs):
        batch_losses = []
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        if scheduler:
            scheduler.step()
            
        epoch_losses.append(np.mean(batch_losses))

    loss_df = pd.DataFrame({
        "epoch": range(epochs),
        "framework": "pytorch",
        "model_name": model_name,
        "loss": epoch_losses
    })

    return model, loss_df




from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from nuclear_mass_predictor.models.jax_ann import (
    DynamicNuclearANN as jax_DynamicNuclearANN,
)


def train_jax_model(
    X_train_scaled: np.ndarray,
    y_train_scaled: pd.Series,
    scaler_params: dict[str, Any],
    params: dict[str, Any],
    model_name: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Trains the JAX dynamic ANN model with mini-batching and tracks MAE loss.
    """
    lr = params.get("learning_rate", 0.0001)
    epochs = params.get("epochs", 10000)
    batch_size = params.get("batch_size", 64)
    if params.get("smoke_run", False):
        epochs = 5
        
    features = params["model_suite"][model_name]["features"]
    hidden_dims = params["model_suite"][model_name]["hidden_dims"]
    
    all_features = scaler_params["feature_names"]
    feature_indices = [all_features.index(f) for f in features]
    X_train_scaled_filtered = X_train_scaled[:, feature_indices]

    X_jnp = jnp.array(X_train_scaled_filtered)
    y_jnp = jnp.array(y_train_scaled).reshape(-1, 1)
    num_samples = X_jnp.shape[0]

    model = jax_DynamicNuclearANN(hidden_dims=hidden_dims)
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    variables = model.init(init_rng, X_jnp[:1])

    if params.get("lr_schedule") == "cosine":
        lr_sched = optax.cosine_decay_schedule(init_value=lr, decay_steps=epochs, alpha=1e-2)
        optimizer = optax.adam(learning_rate=lr_sched, b1=0.9, b2=0.999)
    else:
        optimizer = optax.adam(learning_rate=lr, b1=0.9, b2=0.999)
        
    opt_state = optimizer.init(variables['params'])

    @jax.jit
    def loss_fn(params_tree, x, y):
        preds = model.apply({'params': params_tree}, x)
        return jnp.mean(jnp.abs(preds.flatten() - y.flatten()))

    @jax.jit
    def step_fn(params_tree, state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params_tree, x, y)
        updates, new_state = optimizer.update(grads, state, params_tree)
        new_params = optax.apply_updates(params_tree, updates)
        return new_params, new_state, loss

    params_tree = variables['params']
    epoch_losses = []

    for epoch in range(epochs):
        rng, shuffle_rng = jax.random.split(rng)
        perms = jax.random.permutation(shuffle_rng, num_samples)
        X_shuffled = X_jnp[perms]
        y_shuffled = y_jnp[perms]

        batch_losses = []
        for i in range(0, num_samples, batch_size):
            batch_x = X_shuffled[i : i + batch_size]
            batch_y = y_shuffled[i : i + batch_size]
            params_tree, opt_state, loss = step_fn(params_tree, opt_state, batch_x, batch_y)
            batch_losses.append(float(loss))

        epoch_losses.append(np.mean(batch_losses))

    loss_df = pd.DataFrame({
        "epoch": range(epochs),
        "framework": "jax",
        "model_name": model_name,
        "loss": epoch_losses
    })

    return {'params': params_tree}, loss_df




from nuclear_mass_predictor.schemas.reporting_schema import UnifiedPredictionSchema


def evaluate_all_models(
    X_test_scaled: np.ndarray,
    y_test: pd.Series,
    X_test_raw: pd.DataFrame,
    scaler_params: dict[str, Any],
    params: dict[str, Any],
    **models
) -> pd.DataFrame:
    
    results = []
    y_true = y_test.values
    y_mean = scaler_params.get("y_mean", 0.0)
    y_scale = scaler_params.get("y_scale", 1.0)
    all_features = scaler_params["feature_names"]

    for name, model_obj in models.items():
        parts = name.split('_')
        model_name = parts[0]
        framework = parts[1]
        
        features = params["model_suite"][model_name]["features"]
        feature_indices = [all_features.index(f) for f in features]
        X_test_scaled_filtered = X_test_scaled[:, feature_indices]
        
        if framework == "pytorch":
            model_obj.eval()
            with torch.no_grad():
                X_test_torch = torch.tensor(X_test_scaled_filtered, dtype=torch.float32)
                preds_scaled = model_obj(X_test_torch).numpy().flatten()
        elif framework == "jax":
            hidden_dims = params["model_suite"][model_name]["hidden_dims"]
            jax_model = jax_DynamicNuclearANN(hidden_dims=hidden_dims)
            preds_scaled = np.array(jax_model.apply(model_obj, jnp.array(X_test_scaled_filtered))).flatten()
            
        preds = (preds_scaled * y_scale) + y_mean
        
        df = X_test_raw.copy()
        df["binding_energy_true"] = y_true
        df["prediction"] = preds
        df["residual"] = df["binding_energy_true"] - df["prediction"]
        df["model_name"] = model_name
        df["framework"] = framework
        results.append(df)

    return UnifiedPredictionSchema.validate(pd.concat(results, ignore_index=True))




def compute_summary_metrics(unified_preds: pd.DataFrame) -> dict[str, float]:
    """
    Consumes the unified long-format predictions and computes scalar summary
    metrics (e.g., RMSD) for native Kedro metric tracking.
    """
    metrics = {}

    # Loop over each framework present in the unified dataframe
    for fw in unified_preds["framework"].unique():
        fw_data = unified_preds[unified_preds["framework"] == fw]

        rmsd = float(np.sqrt(np.mean(fw_data["residual"] ** 2)))
        mae = float(np.mean(np.abs(fw_data["residual"])))

        metrics[f"{fw}_test_rmsd_mev"] = rmsd
        metrics[f"{fw}_test_mae_mev"] = mae

    return metrics
