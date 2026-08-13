from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    compute_kan_metrics,
    split_data_for_kan,
    train_and_evaluate_kan,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data_for_kan,
                inputs=["primary_features", "params:liu_2024_training"],
                outputs=["liu_2024_dataset", "liu_2024_train", "liu_2024_test"],
                name="split_data_for_kan_node",
            ),
            node(
                func=train_and_evaluate_kan,
                inputs=["liu_2024_train", "liu_2024_test", "params:liu_2024_training"],
                outputs="liu_2024.unified_test_predictions",
                name="train_and_evaluate_kan_node",
            ),
            node(
                func=compute_kan_metrics,
                inputs="liu_2024.unified_test_predictions",
                outputs="liu_2024.evaluation_metrics",
                name="compute_kan_metrics_node",
            ),
       ]
    )
