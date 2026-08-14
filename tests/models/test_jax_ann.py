import jax
import jax.numpy as jnp

from nuclear_mass_predictor.models.jax_ann import DynamicNuclearANN


def test_jax_nuclear_ann7_parameter_count():
    model = DynamicNuclearANN(hidden_dims=[9, 8])
    rng = jax.random.PRNGKey(0)
    
    # Flax requires dummy input to initialize the weights lazily
    dummy_input = jnp.ones((1, 7))
    variables = model.init(rng, dummy_input)
    
    # Flatten the PyTree to count total parameters
    params = variables['params']
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    
    # Model topology: 7->9->8->1 has exactly 161 parameters
    assert total_params == 161, f"Expected 161 parameters, but found {total_params}"

def test_jax_nuclear_ann7_forward_pass():
    model = DynamicNuclearANN(hidden_dims=[9, 8])
    rng = jax.random.PRNGKey(0)
    
    batch_size = 32
    dummy_input = jnp.ones((batch_size, 7))
    
    # Initialize and run forward pass
    variables = model.init(rng, dummy_input)
    output = model.apply(variables, dummy_input)
    
    # Assert shape and mathematical stability
    assert output.shape == (batch_size, 1)
    assert not jnp.isnan(output).any()
