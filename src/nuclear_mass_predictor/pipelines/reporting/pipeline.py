from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_residual_plots, test_heteroscedasticity


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_residual_plots,
                inputs="zeng_2022.unified_test_predictions",  # Consumes the rich DataFrame
                outputs="zeng_2022.residual_plots",          # Outputs a matplotlib figure
                name="create_residual_plots_node",
            ),
            node(
                func=test_heteroscedasticity,
                inputs="zeng_2022.unified_test_predictions",
                outputs="zeng_2022.heteroscedasticity_metrics",
                name="test_heteroscedasticity_node",
            )
        ]
    )
