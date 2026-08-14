# Zeng et al. (2022): Nuclear Mass Prediction via Bayesian Neural Networks

This pipeline reproduces the framework for predicting nuclear mass excess and calculating epistemic uncertainty using **Bayesian Neural Networks (BNNs)** via Monte Carlo (MC) Dropout, as outlined by Zeng et al. (2022).

## Overview

Unlike standard neural networks that provide a single point-estimate, BNNs output a probability distribution over predictions. This pipeline uses **MC Dropout**, a highly effective approximation for Bayesian inference, where dropout layers remain active during inference time. 

By running multiple forward passes (sampling), the network produces a mean prediction alongside standard deviation bounds, yielding robust uncertainty metrics critical for physical sciences.

## Model Architectures

The pipeline compares two primary architectures:

1. **Baseline MLP**: A standard deterministic Multi-Layer Perceptron used for establishing performance benchmarks without uncertainty estimates.
2. **MC Dropout BNN**: A network featuring dropout layers inserted between dense projections. During prediction, 100 stochastic forward passes are sampled to generate the final prediction mean and standard deviation.

## Data & Features

The models leverage a carefully engineered feature suite focusing on phenomenological properties:
- **Core Numbers**: Proton ($Z$), Neutron ($N$), Mass ($A$), Isospin Asymmetry ($I$).
- **Pairing Effects**: Parity variables to capture odd-even mass staggering.
- **Separation Energies**: Computed one- and two-nucleon separation boundaries.

The pipeline utilizes the **AME2016** dataset for training and evaluates extrapolation performance on newly discovered nuclei from the **AME2020** dataset.

## Executing the Pipeline

To run this specific paper's pipeline from the project root:

```bash
kedro run --pipeline=zeng_2022
```

### Outputs
- **Models**: Saved to `data/06_models/zeng_2022/`.
- **Predictions & Uncertainty**: Parquet datasets saved to `data/07_model_output/zeng_2022/`.
- **Plots**: Residual analysis and predictive uncertainty plots saved to `data/08_reporting/zeng_2022/`.
