import pandas as pd
import pytest
from pandera.errors import SchemaError

from nuclear_mass_predictor.schemas.reporting_schema import UnifiedPredictionSchema


def test_unified_prediction_schema_valid():
    """Test that a valid dataframe (even with extra columns) passes validation."""
    valid_df = pd.DataFrame({
        "z": [10, 20],
        "n": [12, 22],
        "binding_energy_true": [100.5, 200.5],
        "prediction": [100.0, 200.0],
        "residual": [0.5, 0.5],
        "model_name": ["ann7_baseline", "ann7_baseline"],
        "framework": ["pytorch", "jax"],
        "extra_column": ["will_be_ignored", "will_be_ignored"] # Testing strict=False
    })
    
    validated_df = UnifiedPredictionSchema.validate(valid_df)
    assert not validated_df.empty
    assert "extra_column" in validated_df.columns

def test_unified_prediction_schema_invalid():
    """Test that missing required columns raises a SchemaError."""
    invalid_df = pd.DataFrame({
        "z": [10, 20],
        # 'n' is intentionally missing
        "binding_energy_true": [100.5, 200.5],
        "prediction": [100.0, 200.0],
        "residual": [0.5, 0.5],
        "model_name": ["ann7_baseline", "ann7_baseline"],
        "framework": ["pytorch", "jax"]
    })
    
    with pytest.raises(SchemaError):
        UnifiedPredictionSchema.validate(invalid_df)

