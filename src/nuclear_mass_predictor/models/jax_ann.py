import flax.linen as nn


class NuclearANN7(nn.Module):
    """
    ANN7 architecture in Flax/Linen for predicting nuclear binding energies.
    Uses 7 input features, two hidden layers (32 and 16 nodes), and GeLU activation.
    """
    @nn.compact
    def __call__(self, x):
        # First hidden layer (32 nodes)
        x = nn.Dense(features=32)(x)
        x = nn.gelu(x)
        
        # Second hidden layer (16 nodes)
        x = nn.Dense(features=16)(x)
        x = nn.gelu(x)
        
        # Output layer (1 node for Binding Energy)
        x = nn.Dense(features=1)(x)
        
        return x
