"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.19.15
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_engineered_features, fetch_iaea_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=fetch_iaea_data,
                inputs="params:iaea_api",
                outputs="iaea_raw_data",
                name="fetch_iaea_data_node",
            ),
            node(
                func=create_engineered_features,
                inputs=["iaea_raw_data", "params:ws4_parameters"],
                outputs="primary_features",
                name="create_engineered_features_node",
            )
        ]
    )
