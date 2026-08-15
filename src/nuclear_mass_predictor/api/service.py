import os
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from nuclear_mass_predictor.models.jax_ann import DynamicNuclearANN
from nuclear_mass_predictor.utils.physics_core import (
    calculate_asy,
    calculate_pairing,
    distance_to_magic,
)


class ModelInferenceService:
    """
    Inference service for predicting nuclear binding energy using the trained JAX ANN7 model.
    """

    def __init__(self, model_dir: Path | str | None = None):
        if model_dir is None:
            # Default to project data folder
            root_dir = Path(__file__).resolve().parents[3]
            model_dir = root_dir / "data" / "06_models" / "zeng_2022"

        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "ANN7_jax_model.pkl"
        self.scaler_path = self.model_dir / "scaler_params.yml"

        self.model_params: dict[str, Any] | None = None
        self.scaler_params: dict[str, Any] | None = None
        self.model: DynamicNuclearANN | None = None
        self.is_loaded: bool = False

    def load_model(self) -> None:
        """Loads model weights and scaler parameters."""
        if not self.model_path.exists() or not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found at {self.model_dir}. "
                "Please run 'kedro run --pipeline zeng_2022' first to train and save the model."
            )

        with open(self.model_path, "rb") as f:
            self.model_params = pickle.load(f)

        with open(self.scaler_path, "r") as f:
            self.scaler_params = yaml.safe_load(f)

        # ANN7 architecture uses [32, 16] hidden dims
        self.model = DynamicNuclearANN(hidden_dims=[32, 16])
        self.is_loaded = True

    def extract_features(self, z: int, n: int) -> list[float]:
        """
        Computes the 7 physical features for ANN7:
        ["z", "n", "z_eo", "n_eo", "delta_z", "delta_n", "asy"]
        """
        z_eo = calculate_pairing(z)
        n_eo = calculate_pairing(n)
        delta_z = distance_to_magic(z)
        delta_n = distance_to_magic(n)
        asy = calculate_asy(z, n, kappa=1.139, xi=1.250, fs=1.0)

        return [float(z), float(n), float(z_eo), float(n_eo), float(delta_z), float(delta_n), float(asy)]

    def predict_single(self, z: int, n: int) -> tuple[float, float]:
        """
        Predicts binding energy for a single nucleus (Z, N).
        Returns: (predicted_binding_energy_total_mev, predicted_binding_energy_per_nucleon_mev)
        """
        if not self.is_loaded or self.model is None or self.model_params is None or self.scaler_params is None:
            self.load_model()

        raw_features = np.array([self.extract_features(z, n)], dtype=np.float32)

        # Scale features using stored parameters
        x_mean = np.array(self.scaler_params["x_mean"], dtype=np.float32)
        x_scale = np.array(self.scaler_params["x_scale"], dtype=np.float32)
        scaled_features = (raw_features - x_mean) / x_scale

        # Model evaluation via JAX
        input_jnp = jnp.array(scaled_features)
        pred_scaled = self.model.apply({"params": self.model_params}, input_jnp)
        pred_scaled_val = float(np.array(pred_scaled).flatten()[0])

        # Unscale target if target was scaled during training
        if self.scaler_params.get("scale_target", False):
            y_mean = float(self.scaler_params["y_mean"])
            y_scale = float(self.scaler_params["y_scale"])
            total_be = pred_scaled_val * y_scale + y_mean
        else:
            total_be = pred_scaled_val

        a = z + n
        be_per_nucleon = total_be / a if a > 0 else 0.0

        return total_be, be_per_nucleon
