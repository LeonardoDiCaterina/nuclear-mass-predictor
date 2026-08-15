import pandas as pd
from matplotlib import pyplot as plt

from nuclear_mass_predictor.pipelines.papers.zeng_2022.nodes import (
    compute_summary_metrics,
)
from nuclear_mass_predictor.pipelines.reporting.nodes import (
    create_loss_curves_plot,
    create_residual_plots,
)
from nuclear_mass_predictor.pipelines.reporting.nodes import (
    test_heteroscedasticity as compute_heteroscedasticity,
)


def test_compute_summary_metrics():
    df = pd.DataFrame({
        "framework": ["jax", "jax"],
        "residual": [3.0, -4.0]
    })
    metrics = compute_summary_metrics(df)
    
    assert "jax_test_rmsd_mev" in metrics
    assert "jax_test_mae_mev" in metrics
    assert metrics["jax_test_mae_mev"] == 3.5 # (3 + 4) / 2
    import math
    assert math.isclose(metrics["jax_test_rmsd_mev"], 3.5355339059327378)


def test_create_residual_plots():
    df = pd.DataFrame({
        "framework": ["jax", "pytorch"],
        "z": [10, 20],
        "n": [10, 20],
        "residual": [1.0, -1.0]
    })
    
    fig = create_residual_plots(df)
    assert isinstance(fig, plt.Figure)
    
def test_heteroscedasticity_metric():
    df = pd.DataFrame({
        "framework": ["jax", "jax", "jax", "jax", "jax"],
        "z": [10, 20, 30, 40, 50],
        "n": [10, 20, 30, 40, 50],
        "a": [20, 40, 60, 80, 100],
        "residual": [1.0, -2.0, 3.0, -4.0, 5.0] 
    })
    
    res = compute_heteroscedasticity(df)
    assert "jax_spearman_corr_A" in res
    import math
    assert math.isclose(res["jax_spearman_corr_A"], 1.0)

def test_create_loss_curves_plot():
    pytorch_loss = pd.DataFrame({
        "epoch": [1, 2],
        "loss": [10.0, 5.0],
        "framework": ["pytorch", "pytorch"]
    })
    jax_loss = pd.DataFrame({
        "epoch": [1, 2],
        "loss": [11.0, 4.0],
        "framework": ["jax", "jax"]
    })
    
    fig = create_loss_curves_plot(pytorch_loss, jax_loss)
    assert isinstance(fig, plt.Figure)
