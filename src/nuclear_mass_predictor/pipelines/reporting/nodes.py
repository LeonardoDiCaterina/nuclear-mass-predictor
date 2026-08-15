import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def concat_unified_predictions(*preds: pd.DataFrame) -> pd.DataFrame:
    """Concatenate unified predictions from all paper pipelines."""
    return pd.concat(preds, ignore_index=True)

def compare_models_metrics(master_preds: pd.DataFrame) -> dict[str, float]:
    """Calculate summary metrics (RMSD, MAE) across all models and frameworks."""
    metrics = {}
    for (model, fw), df in master_preds.groupby(["model_name", "framework"]):
        rmsd = float(np.sqrt(np.mean(df["residual"] ** 2)))
        mae = float(np.mean(np.abs(df["residual"])))
        metrics[f"{model}_{fw}_test_rmsd_mev"] = rmsd
        metrics[f"{model}_{fw}_test_mae_mev"] = mae
    return metrics

def create_residual_plots(predictions_df: pd.DataFrame) -> plt.Figure:
    """
    Creates a multi-panel figure showing residuals vs Z, N, and A
    to diagnose heteroscedasticity and shell closure effects.
    """
    # Calculate Mass Number (A) if not explicitly passed
    if "a" not in predictions_df.columns:
        predictions_df["a"] = predictions_df["z"] + predictions_df["n"]

    # Set up the matplotlib figure
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    # Plot 1: Residuals vs Z (Proton Number)
    sns.scatterplot(
        data=predictions_df, x="z", y="residual", hue="framework", 
        alpha=0.7, edgecolor=None, ax=axes[0]
    )
    axes[0].set_title("Residuals vs Proton Number (Z)")
    axes[0].set_xlabel("Protons (Z)")
    axes[0].set_ylabel("Residual Error (MeV)")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1.5)

    # Plot 2: Residuals vs N (Neutron Number)
    sns.scatterplot(
        data=predictions_df, x="n", y="residual", hue="framework", 
        alpha=0.7, edgecolor=None, ax=axes[1]
    )
    axes[1].set_title("Residuals vs Neutron Number (N)")
    axes[1].set_xlabel("Neutrons (N)")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.5)

    # Plot 3: Residuals vs A (Mass Number)
    sns.scatterplot(
        data=predictions_df, x="a", y="residual", hue="framework", 
        alpha=0.7, edgecolor=None, ax=axes[2]
    )
    axes[2].set_title("Residuals vs Mass Number (A)")
    axes[2].set_xlabel("Mass Number (A)")
    axes[2].axhline(0, color="black", linestyle="--", linewidth=1.5)

    plt.tight_layout()
    return fig

def test_heteroscedasticity(predictions_df: pd.DataFrame) -> dict[str, float]:
    """
    Applies Spearman rank correlation to check if the magnitude of errors 
    (absolute residuals) significantly correlates with nuclear mass/size.
    """
    if "a" not in predictions_df.columns:
        predictions_df["a"] = predictions_df["z"] + predictions_df["n"]
        
    results = {}
    
    for fw in predictions_df["framework"].unique():
        fw_data = predictions_df[predictions_df["framework"] == fw]
        abs_res = fw_data["residual"].abs()
        
        # Test correlation between Mass Number (A) and Absolute Error
        corr_a, p_a = stats.spearmanr(fw_data["a"], abs_res)
        
        # Test correlation between Proton Number (Z) and Absolute Error
        corr_z, p_z = stats.spearmanr(fw_data["z"], abs_res)
        
        # Store as standard floats for Kedro MetricsDataset
        results[f"{fw}_spearman_corr_A"] = float(corr_a)
        results[f"{fw}_spearman_pvalue_A"] = float(p_a)
        results[f"{fw}_spearman_corr_Z"] = float(corr_z)
        results[f"{fw}_spearman_pvalue_Z"] = float(p_z)
        
    return results

def create_loss_curves_plot(pytorch_loss: pd.DataFrame, jax_loss: pd.DataFrame) -> plt.Figure:
    """
    Creates a side-by-side or combined convergence plot comparing
    PyTorch and JAX epoch-by-epoch MAE loss.
    """
    # Combine the two loss dataframes
    combined_loss = pd.concat([pytorch_loss, jax_loss], ignore_index=True)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.lineplot(
        data=combined_loss,
        x="epoch",
        y="loss",
        hue="framework",
        linewidth=2,
        ax=ax
    )

    ax.set_title("Training Loss Convergence (ANN7)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Absolute Error (MeV)")

    # Use a logarithmic scale if loss spans orders of magnitude during early epochs
    ax.set_yscale("log")

    plt.tight_layout()
    return fig
