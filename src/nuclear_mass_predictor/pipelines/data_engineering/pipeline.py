"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.19.15
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import fetch_iaea_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=fetch_iaea_data,
                inputs="params:iaea_api",
                outputs="iaea_raw_data",
                name="fetch_iaea_data_node",
            )
        ]
    )
