# Liu et al. (2024): Nuclear Mass Prediction via Kolmogorov-Arnold Networks

This pipeline replicates and extends the nuclear mass prediction framework based on **Kolmogorov-Arnold Networks (KAN)**, as proposed by Liu et al. (2024).

## Overview

Unlike standard Multi-Layer Perceptrons (MLPs) that place static activation functions at the nodes, KANs parameterize learnable B-splines along the computational edges. This provides two significant benefits:
1. **Interpretability**: Small KAN models (e.g., KAN-2) can often be simplified into exact, symbolic mathematical formulas.
2. **Efficiency**: KANs can sometimes achieve comparable or superior accuracy to large MLPs using far fewer parameters.

## JAX / Flax NNX Migration 🚀

The original paper implementation relied on the `pykan` (PyTorch) library, optimizing with L-BFGS. While accurate, training took approximately 40 minutes for 500 epochs on a standard machine.

In this pipeline, we have completely migrated the underlying architecture to **`jaxkan`**, a JAX/Flax NNX port of the KAN framework. 
- **Optimizer**: `optax.adamw` with a cosine decay schedule.
- **XLA Compilation**: The model step is fully `@nnx.jit` compiled. 
- **Result**: The pipeline now completes a massive **2000 epochs** for all 4 model configurations in **under 2 minutes**—an extreme, orders-of-magnitude speedup.

## Model Configurations

The pipeline trains 4 variations of the KAN model, each receiving a different feature set:

- **`KAN_2`**: Receives only proton ($Z$) and neutron ($N$) numbers. Ideal for symbolic extraction.
- **`KAN_4`**: Receives ($Z, N, A, I$), providing basic macroscopic structure context.
- **`KAN_9`**: Incorporates proximity to magic numbers, allowing the spline edges to smoothly capture shell effects.
- **`KAN_11`**: The fully-featured phenomenological macroscopic-microscopic input space (including pairing, asymmetry, etc.). Rivals black-box MLP accuracy.

## Executing the Pipeline

To run this specific paper's pipeline from the project root:

```bash
kedro run --pipeline=liu_2024
```

### Outputs
- **Models**: Saved to `data/06_models/liu_2024/`.
- **Predictions**: Unified predictions saved to `data/07_model_output/liu_2024/`.
- **Metrics**: Evaluation JSON (RMSD, MAE) saved to `data/08_reporting/liu_2024/evaluation_metrics.json`.
