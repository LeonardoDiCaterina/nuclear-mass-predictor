import pandera.pandas as pa
from pandera.typing import Series

class UnifiedPredictionSchema(pa.DataFrameModel):
    z: Series[int] = pa.Field(ge=0)
    n: Series[int] = pa.Field(ge=0)
    binding_energy_true: Series[float]
    prediction: Series[float]
    residual: Series[float]
    model_name: Series[str]
    framework: Series[str]  # "pytorch", "jax", etc.

    class Config:
        coerce = True
        strict = False 
