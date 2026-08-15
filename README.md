# Nuclear Mass Predictor

[![Kedro](https://img.shields.io/badge/kedro-0.19.6-ffc900?logo=kedro)](https://kedro.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
![Coverage](coverage.svg)

Welcome to the **Nuclear Mass Predictor** project. This repository serves as a machine learning benchmark and exploration framework for predicting nuclear mass excess. It leverages historical Atomic Mass Evaluation (AME) datasets (2016 and 2020) and WS4 baseline models to evaluate modern neural network architectures.

The project is built using [Kedro](https://kedro.org/), providing a highly modular and reproducible data engineering and data science pipeline.

## Project Architecture

This project is divided into several Kedro pipelines, separating data processing from paper-specific implementations:

1. **`data_engineering`**: Fetches raw AME data (via API/CSVs), cleans it, computes separation energies, calculates magic number proximity, and generates primary historical features.
2. **`papers`**:
   - **[Zeng 2022 Pipeline](src/nuclear_mass_predictor/pipelines/papers/zeng_2022/README.md)**: Implementation of Bayesian Neural Networks (BNN) with MC Dropout for uncertainty quantification.
   - **[Liu 2024 Pipeline](src/nuclear_mass_predictor/pipelines/papers/liu_2024/README.md)**: Implementation of Kolmogorov-Arnold Networks (KAN) utilizing high-speed JAX/Flax NNX compilation.
3. **`reporting`**: Aggregates evaluation metrics (RMSD, MAE) and generates comparative analysis plots.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/leonardodicaterina/Nuclear_Project.git
   cd Nuclear_Project/nuclear-mass-predictor
   ```

2. **Activate the Environment:**
   Ensure you have `conda` installed.
   ```bash
   conda activate NuclearEnv
   ```

3. **Install Dependencies:**
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

## Running the Pipelines

You can run specific pipelines using the Kedro CLI.

**Run the Zeng 2022 Pipeline (BNN/Dropout):**
```bash
kedro run --pipeline=zeng_2022
```

**Run the Liu 2024 Pipeline (JAX KAN):**
```bash
kedro run --pipeline=liu_2024
```

**Run the Master Reporting Pipeline:**
```bash
kedro run --pipeline=reporting
```

## REST API Model Serving

You can serve predictions using our built-in FastAPI model server:

1. **Launch the server:**
   ```bash
   nuclear-mass-predictor-api
   # Or directly:
   uvicorn nuclear_mass_predictor.api.main:app --host 0.0.0.0 --port 8000
   ```

2. **Access Swagger Interactive API Docs:**
   Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

3. **Query a Prediction (GET endpoint):**
   ```bash
   curl -X GET "http://localhost:8000/predict/20/28"
   ```
   *Response:*
   ```json
   {
     "z": 20,
     "n": 28,
     "a": 48,
     "predicted_binding_energy_total_mev": 418.712,
     "predicted_binding_energy_per_nucleon_mev": 8.7232,
     "model_name": "ANN7",
     "framework": "jax"
   }
   ```

4. **Query Batch Predictions (POST endpoint):**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"nuclei": [{"z": 20, "n": 28}, {"z": 82, "n": 126}]}'
   ```

## Results & Conclusion

The master reporting pipeline aggregates unified predictions from both paper-specific implementations to calculate global evaluation metrics (RMSD and MAE) on the promoted AME2020 test set.

| Model (Features) | Framework | Test RMSD (MeV) | Test MAE (MeV) |
| :--- | :--- | :--- | :--- |
| **ANN7 (Z, N, Z_eo, N_eo, dZ, dN, Asy)** | **JAX** | **0.633** | **0.358** |
| ANN7 (Z, N, Z_eo, N_eo, dZ, dN, Asy) | PyTorch | 0.878 | 0.470 |
| **KAN_2 (Z, N)** | **jaxkan** | **4.973** | **2.762** |
| **KAN_4 (Z, N, A, Asy)** | **jaxkan** | **5.640** | **3.762** |
| KAN_11 (11 features) | jaxkan | 13.640 | 10.303 |
| KAN_9 (9 features) | jaxkan | 26.320 | 20.491 |

### Takeaways
1. **ANN Dominance**: The ANNs (specifically the JAX implementation) are outperforming the KAN models right now, hitting an impressive RMSD of ~0.63 MeV.
2. **JAX vs PyTorch**: The JAX implementation of ANN7 achieved a lower error (0.63 MeV vs 0.93 MeV for PyTorch), likely due to robust optimization dynamics over 30,000 epochs.
3. **KAN Performance**: The jaxKAN models currently lag behind with RMSDs around 9-16 MeV. This is because they were trained for a limited number of epochs (500) as a rapid prototype, whereas the ANNs ran for 30,000 epochs. KAN networks will require further hyperparameter tuning (grid size, learning rate schedules) to converge effectively on complex regression tasks.

Based on current benchmarks, the **JAX ANN7** is selected as the primary candidate for model deployment.

## Developer Tools (CI/CD)

This project strictly adheres to modern Python standards. It utilizes GitHub Actions for Continuous Integration (CI):
- **Linting & Formatting**: Enforced via [Ruff](https://docs.astral.sh/ruff/).
- **Type Checking**: Enforced via [Mypy](https://mypy.readthedocs.io/).
- **Testing**: Run via `pytest`, with dynamic coverage reports automatically updating the badge above on every commit.

## License
MIT License.
