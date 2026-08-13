import flax.linen as nn


class DynamicNuclearANN(nn.Module):
    """
    Dynamic ANN architecture in Flax/Linen for predicting nuclear binding energies.
    Supports variable hidden layer dimensions.
    """
    hidden_dims: list[int]

    @nn.compact
    def __call__(self, x):
        for h_dim in self.hidden_dims:
            x = nn.Dense(features=h_dim)(x)
            x = nn.gelu(x)
        
        # Output layer (1 node for Binding Energy)
        x = nn.Dense(features=1)(x)
        
        return x
