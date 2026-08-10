import pandera.pandas as pa
from pandera.typing import Series


class RawNuclearSchema(pa.DataFrameModel):
    """
    Pandera schema enforcing the data contract for raw nuclear data.
    """
    z: Series[int] = pa.Field(ge=0, description="Proton number")
    n: Series[int] = pa.Field(ge=0, description="Neutron number")
    binding_energy: Series[float] = pa.Field(description="Target variable (BE)")

    class Config:
        # Automatically cast types if they are close enough (e.g., string '1' to int 1)
        coerce = True
        # Drop any extra columns that we don't care about from the API
        strict = False
