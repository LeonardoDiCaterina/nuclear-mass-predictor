from pydantic import BaseModel, Field


class NuclearInput(BaseModel):
    z: int = Field(..., description="Proton number (atomic number Z)", ge=1, le=120, json_schema_extra={"example": 20})
    n: int = Field(..., description="Neutron number (N)", ge=1, le=200, json_schema_extra={"example": 28})


class PredictionRequest(BaseModel):
    nuclei: list[NuclearInput] = Field(
        ..., 
        description="List of nuclei to evaluate",
        json_schema_extra={"example": [{"z": 20, "n": 28}, {"z": 82, "n": 126}]}
    )


class PredictionResult(BaseModel):
    z: int
    n: int
    a: int
    predicted_binding_energy_total_mev: float
    predicted_binding_energy_per_nucleon_mev: float
    model_name: str
    framework: str


class PredictionResponse(BaseModel):
    predictions: list[PredictionResult]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_name: str
    framework: str
    version: str
