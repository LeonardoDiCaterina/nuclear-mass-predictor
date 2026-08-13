import torch
from torch import nn


class DynamicNuclearANN(nn.Module):
    """
    Dynamic ANN architecture for predicting nuclear binding energies.
    Supports variable input features and hidden layer dimensions.
    """
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_features=in_dim, out_features=h_dim))
            layers.append(nn.GELU())
            in_dim = h_dim
        
        layers.append(nn.Linear(in_features=in_dim, out_features=1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.
        """
        return self.network(x)
