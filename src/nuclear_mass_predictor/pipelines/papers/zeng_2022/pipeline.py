from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    compute_summary_metrics,
    evaluate_models,
    scale_features,
    split_historical_data,
    train_jax_model,
    train_pytorch_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_historical_data,
                inputs=["primary_features", "params:zeng_2022_training"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_historical_data_node",
            ),
            node(
                func=scale_features,
                inputs=["X_train", "X_test"],
                outputs=["X_train_scaled", "X_test_scaled", "scaler_params"],
                name="scale_features_node",
            ),
            node(
                func=train_pytorch_model,
                inputs=["X_train_scaled", "y_train", "params:zeng_2022_training"],
                outputs="pytorch_model",
                name="train_pytorch_model_node",
            ),
            node(
                func=train_jax_model,
                inputs=["X_train_scaled", "y_train", "params:zeng_2022_training"],
                outputs="jax_model",
                name="train_jax_model_node",
            ),
            node(
                func=evaluate_models,
                inputs=["pytorch_model", "jax_model", "X_test_scaled", "y_test", "X_test"],
                outputs="zeng_2022.unified_test_predictions", # Outputs the rich DataFrame
                name="evaluate_models_node",
            ),
            node(
                func=compute_summary_metrics,
                inputs="zeng_2022.unified_test_predictions",  # Consumes the rich DataFrame
                outputs="zeng_2022.evaluation_metrics",       # Outputs the {str: float} dict
                name="compute_summary_metrics_node",
            ),

       ]
    )
