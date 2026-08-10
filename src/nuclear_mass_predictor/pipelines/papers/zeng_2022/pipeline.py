from kedro.pipeline import Pipeline, node, pipeline

from .nodes import scale_features, split_historical_data


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
        ]
    )
