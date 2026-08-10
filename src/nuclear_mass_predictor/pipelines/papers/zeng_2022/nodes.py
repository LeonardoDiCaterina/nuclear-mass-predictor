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


import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nuclear_mass_predictor.models.pytorch_ann import NuclearANN7 as torch_NuclearANN7


def train_pytorch_model(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    params: dict[str, Any]
) -> tuple[nn.Module, pd.DataFrame]:
    """
    Trains the PyTorch ANN7 model and tracks epoch-by-epoch MAE loss.
    Returns a tuple of (model, loss_history_dataframe).
    """
    lr = params.get("learning_rate", 0.0001)
    beta1 = params.get("beta1", 0.9)
    beta2 = params.get("beta2", 0.999)
    epochs = params.get("epochs", 3000)
    batch_size = params.get("batch_size", 64)

    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = torch_NuclearANN7()
    model.train()
    
    criterion = nn.L1Loss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))

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
            
        epoch_losses.append(np.mean(batch_losses))

    # Compile the loss history into a standardized DataFrame
    loss_df = pd.DataFrame({
        "epoch": range(epochs),
        "framework": "pytorch",
        "loss": epoch_losses
    })

    return model, loss_df
def deprecated_1_train_pytorch_model(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    params: dict[str, Any]
) -> nn.Module:
    """
    Trains the PyTorch ANN7 model using the historical training set,
    Adam optimizer, and Mean Absolute Error (MAE) loss.
    """
    # 1. Hyperparameters from configuration
    lr = params.get("learning_rate", 0.0001)
    beta1 = params.get("beta1", 0.9)
    beta2 = params.get("beta2", 0.999)
    epochs = params.get("epochs", 3000)
    batch_size = params.get("batch_size", 64)

    # 2. Prepare DataLoaders
    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Initialize Model, Loss, and Optimizer
    model = torch_NuclearANN7()
    model.train()
    
    criterion = nn.L1Loss()  # Mean Absolute Error (MAE)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr, 
        betas=(beta1, beta2)
    )

    # 4. Training Loop
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

    return model



from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from nuclear_mass_predictor.models.jax_ann import NuclearANN7 as jax_NuclearANN7


def train_jax_model(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    params: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Trains the JAX ANN7 model with mini-batching and tracks MAE loss.
    Returns a tuple of (params_tree, loss_history_dataframe).
    """
    lr = params.get("learning_rate", 0.0001)
    epochs = params.get("epochs", 3000)
    batch_size = params.get("batch_size", 64)

    X_jnp = jnp.array(X_train_scaled)
    y_jnp = jnp.array(y_train.values).reshape(-1, 1)
    num_samples = X_jnp.shape[0]

    model = jax_NuclearANN7()
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    variables = model.init(init_rng, X_jnp)

    optimizer = optax.adam(learning_rate=lr, b1=0.9, b2=0.999)
    opt_state = optimizer.init(variables['params'])

    @jax.jit
    def loss_fn(params_tree, x, y):
        preds = model.apply({'params': params_tree}, x)
        return jnp.mean(jnp.abs(preds - y))

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

    # Compile the loss history into a standardized DataFrame
    loss_df = pd.DataFrame({
        "epoch": range(epochs),
        "framework": "jax",
        "loss": epoch_losses
    })

    return {'params': params_tree}, loss_df

def deprecated_1_train_jax_model(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    params: dict[str, Any]
) -> dict[str, Any]:
    """
    Trains the Flax/JAX ANN7 model using Optax for optimization and MAE loss.
    Implements random shuffling and mini-batching to match PyTorch dynamics.
    Returns the serialized parameter tree.
    """
    lr = params.get("learning_rate", 0.0001)
    epochs = params.get("epochs", 3000)
    batch_size = params.get("batch_size", 64)

    X_jnp = jnp.array(X_train_scaled)
    y_jnp = jnp.array(y_train.values).reshape(-1, 1)
    num_samples = X_jnp.shape[0]

    model = jax_NuclearANN7()
    
    # JAX requires explicit state management for random seeds
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    # Initialize parameters
    variables = model.init(init_rng, X_jnp)

    # Optimizer configuration matching PyTorch settings
    optimizer = optax.adam(learning_rate=lr, b1=0.9, b2=0.999)
    opt_state = optimizer.init(variables['params'])

    @jax.jit
    def loss_fn(params_tree, x, y):
        preds = model.apply({'params': params_tree}, x)
        return jnp.mean(jnp.abs(preds - y))  # MAE loss

    @jax.jit
    def step_fn(params_tree, state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params_tree, x, y)
        updates, new_state = optimizer.update(grads, state, params_tree)
        new_params = optax.apply_updates(params_tree, updates)
        return new_params, new_state, loss

    # Training loop
    params_tree = variables['params']
    
    for epoch in range(epochs):
        # 1. Shuffle data at the start of each epoch
        rng, shuffle_rng = jax.random.split(rng)
        perms = jax.random.permutation(shuffle_rng, num_samples)
        X_shuffled = X_jnp[perms]
        y_shuffled = y_jnp[perms]
        
        # 2. Mini-batch iteration
        for i in range(0, num_samples, batch_size):
            batch_x = X_shuffled[i : i + batch_size]
            batch_y = y_shuffled[i : i + batch_size]
            
            params_tree, opt_state, _ = step_fn(params_tree, opt_state, batch_x, batch_y)

    return {'params': params_tree}


def deprecated_train_jax_model(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    params: dict[str, Any]
) -> dict[str, Any]:
    """
    Trains the Flax/JAX ANN7 model using Optax for optimization and MAE loss.
    Returns the serialized parameter tree (variables).
    """
    lr = params.get("learning_rate", 0.0001)
    epochs = params.get("epochs", 3000)

    X_jnp = jnp.array(X_train_scaled)
    y_jnp = jnp.array(y_train.values).reshape(-1, 1)

    model = jax_NuclearANN7()
    rng = jax.random.PRNGKey(0)

    # Initialize parameters
    variables = model.init(rng, X_jnp)

    # Optimizer configuration matching PyTorch settings
    optimizer = optax.adam(learning_rate=lr, b1=0.9, b2=0.999)
    opt_state = optimizer.init(variables['params'])

    @jax.jit
    def loss_fn(params_tree, x, y):
        preds = model.apply({'params': params_tree}, x)
        return jnp.mean(jnp.abs(preds - y))  # MAE loss

    @jax.jit
    def step_fn(params_tree, state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params_tree, x, y)
        updates, new_state = optimizer.update(grads, state, params_tree)
        new_params = optax.apply_updates(params_tree, updates)
        return new_params, new_state, loss

    # Training loop
    params_tree = variables['params']
    for epoch in range(epochs):
        params_tree, opt_state, _ = step_fn(params_tree, opt_state, X_jnp, y_jnp)

    return {'params': params_tree}


from nuclear_mass_predictor.schemas.reporting_schema import UnifiedPredictionSchema


def evaluate_models(
    pytorch_model: nn.Module,
    jax_params: dict[str, Any],
    X_test_scaled: np.ndarray,
    y_test: pd.Series,
    X_test_raw: pd.DataFrame
) -> pd.DataFrame:

    # 1. Generate PyTorch Predictions
    pytorch_model.eval()
    with torch.no_grad():
        X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32)
        pt_preds = pytorch_model(X_test_torch).numpy().flatten()

    # 2. Generate JAX Predictions
    jax_model = jax_NuclearANN7()
    jax_preds = jax_model.apply(jax_params, jnp.array(X_test_scaled)).flatten()

    # 3. Standardize into long format
    results = []
    y_true = y_test.values

    for name, framework, preds in [
        ("ann7_baseline", "pytorch", pt_preds),
        ("ann7_baseline", "jax", jax_preds)
    ]:
        df = X_test_raw.copy()
        df["binding_energy_true"] = y_true
        df["prediction"] = preds
        df["residual"] = df["binding_energy_true"] - df["prediction"]
        df["model_name"] = name
        df["framework"] = framework
        results.append(df)

    return UnifiedPredictionSchema.validate(pd.concat(results, ignore_index=True))

def deprecated_evaluate_models(
    pytorch_model: nn.Module,
    jax_params: dict[str, Any],
    X_test_scaled: np.ndarray,
    y_test: pd.Series
) -> dict[str, float]:
    """
    Evaluates both PyTorch and JAX models on the holdout test set (AME2020 additions)
    and computes the Root-Mean-Square Deviation (RMSD).
    """
    # PyTorch Evaluation
    pytorch_model.eval()
    with torch.no_grad():
        X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32)
        pt_preds = pytorch_model(X_test_torch).numpy().flatten()

    y_true = y_test.values
    pt_rmsd = float(np.sqrt(np.mean((pt_preds - y_true) ** 2)))

    # JAX Evaluation
    jax_model = jax_NuclearANN7()
    jax_preds = jax_model.apply(jax_params, jnp.array(X_test_scaled)).flatten()
    jax_rmsd = float(np.sqrt(np.mean((jax_preds - y_true) ** 2)))

    metrics = {
        "pytorch_test_rmsd_mev": pt_rmsd,
        "jax_test_rmsd_mev": jax_rmsd
    }

    return metrics


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
