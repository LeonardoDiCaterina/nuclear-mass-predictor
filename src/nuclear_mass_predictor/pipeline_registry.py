from kedro.pipeline import Pipeline

from nuclear_mass_predictor.pipelines.data_engineering.pipeline import (
    create_pipeline as de_pipeline,
)
from nuclear_mass_predictor.pipelines.papers.liu_2024.pipeline import (
    create_pipeline as liu_2024_pipeline,
)
from nuclear_mass_predictor.pipelines.papers.zeng_2022.pipeline import (
    create_pipeline as zeng_2022_pipeline,
)
from nuclear_mass_predictor.pipelines.reporting.pipeline import (
    create_pipeline as reporting,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to corresponding Pipeline objects.
    """
    de = de_pipeline()
    zeng = zeng_2022_pipeline()
    liu = liu_2024_pipeline()

    return {
        "data_engineering": de,
        "zeng_2022": de + zeng,  # Chains ingestion/engineering + zeng_2022 splitting/scaling
        "liu_2024": de + liu,
        "reporting": reporting(),
        "__default__": de + zeng,
    }
