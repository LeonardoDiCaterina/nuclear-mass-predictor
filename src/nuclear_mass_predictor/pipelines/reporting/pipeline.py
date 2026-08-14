from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_loss_curves_plot,
    create_residual_plots,
    test_heteroscedasticity,
    concat_unified_predictions,
    compare_models_metrics,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=concat_unified_predictions,
                inputs=["zeng_2022.unified_test_predictions", "liu_2024.unified_test_predictions"],
                outputs="master_predictions_df",
                name="concat_unified_predictions_node",
            ),
            node(
                func=compare_models_metrics,
                inputs="master_predictions_df",
                outputs="master_evaluation_metrics",
                name="compare_models_metrics_node",
            ),
            node(
                func=create_residual_plots,
                inputs="master_predictions_df",
                outputs="master_residual_plots",
                name="create_residual_plots_node",
            ),
            node(
                func=test_heteroscedasticity,
                inputs="master_predictions_df",
                outputs="master_heteroscedasticity_metrics",
                name="test_heteroscedasticity_node",
            ),
            node(
                func=create_loss_curves_plot,
                inputs=["ANN7_pytorch_loss_history", "ANN7_jax_loss_history"],
                outputs="zeng_2022.loss_convergence_plot",
                name="create_loss_curves_plot_node",
            )
        ]
    )
