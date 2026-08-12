import torch

from nuclear_mass_predictor.models.pytorch_ann import NuclearANN7


def test_pytorch_nuclear_ann7_parameter_count():
    model = NuclearANN7()
    total_params = sum(p.numel() for p in model.parameters())
    # Topology: (7*9 + 9) + (9*8 + 8) + (8*1 + 1) = 72 + 80 + 9 = 161 parameters
    assert total_params == 161, f"Expected 161 parameters, but found {total_params}"


def test_pytorch_nuclear_ann7_forward_pass():
    model = NuclearANN7()
    model.eval()
    
    batch_size = 32
    dummy_input = torch.ones((batch_size, 7), dtype=torch.float32)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape == (batch_size, 1)
    assert not torch.isnan(output).any()

