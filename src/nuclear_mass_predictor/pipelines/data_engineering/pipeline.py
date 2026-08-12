from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_ame_historical_dataset,
    fetch_and_parse_ame_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=fetch_and_parse_ame_data,
                inputs="params:ame_api",
                outputs=["ame2016_raw_data", "ame2020_raw_data"],
                name="fetch_and_parse_ame_data_node",
            ),
            node(
                func=create_ame_historical_dataset,
                inputs=[
                    "ame2016_raw_data",
                    "ame2020_raw_data",
                    "params:ws4_parameters",
                    "params:baseline_params",
                ],
                outputs=[
                    "primary_features",
                    "zeng_2022.train_dataset",
                    "zeng_2022.test_dataset",
                ],
                name="create_ame_historical_dataset_node",
            ),
        ]
    )

