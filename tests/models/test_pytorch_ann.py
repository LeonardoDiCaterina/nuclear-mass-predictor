import torch
from torch import nn


class NuclearANN7(nn.Module):
    """
    ANN7 architecture for predicting nuclear binding energies.
    Uses 7 input features, two hidden layers (32 and 16 nodes), and GeLU activation.
    """
    def __init__(self):
        super().__init__()
        
        # Define the layers according to the paper's Table I specifications
        self.network = nn.Sequential(
            nn.Linear(in_features=7, out_features=32),
            nn.GELU(),
            nn.Linear(in_features=32, out_features=16),
            nn.GELU(),
            nn.Linear(in_features=16, out_features=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.
        """
        return self.network(x)
