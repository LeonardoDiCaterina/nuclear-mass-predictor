import flax.linen as nn


class NuclearANN7(nn.Module):
    """
    ANN7 architecture in Flax/Linen for predicting nuclear binding energies.
    Uses 7 input features, two hidden layers (9 and 8 nodes), and GeLU activation.
    """
    @nn.compact
    def __call__(self, x):
        # First hidden layer (9 nodes)
        x = nn.Dense(features=9)(x)
        x = nn.gelu(x)
        
        # Second hidden layer (8 nodes)
        x = nn.Dense(features=8)(x)
        x = nn.gelu(x)
        
        # Output layer (1 node for Binding Energy)
        x = nn.Dense(features=1)(x)
        
        return x
