
import torch
import pytest
import torch.utils.data
import numpy as np

from networks import WeightLearningNetwork
from learning import hebbian_rule, train


@pytest.mark.parametrize('num_mutations', [0, 100])
def test_weight_updates(num_mutations):
    """Check that only weights of existing connections change during backprop and hebbian."""
    net = WeightLearningNetwork(5, 2, 0.5)
    for _ in range(num_mutations):  # add some mutations to get depth
        net.mutate()
    net.create_torch_layers()

    # Run a simple input forward through the network.
    batch_size = 5
    input = torch.randn(batch_size, 5)
    target = torch.randn(batch_size, 2)  # make random target so we never have the correct output
    output, activities = net.forward(input)

    # Store old weight matrices for comparison.
    old_weight_matrices = [layer.weight_matrix.detach().clone() for layer in net.torch_layers]

    # Do one backprop step and check that same weights as before are zero/non-zero.
    optimizer = torch.optim.SGD(net.parameters(), lr=10)
    loss = ((output - target) ** 2).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for layer, old_weight_matrix in zip(net.torch_layers, old_weight_matrices):
        assert torch.all(layer.weight_matrix.nonzero() == old_weight_matrix.nonzero())

    # Do one hebbian step and check that same weights as before are zero/non-zero.
    hebbian_rule(net, activities)
    for layer, old_weight_matrix in zip(net.torch_layers, old_weight_matrices):
        assert torch.all(layer.weight_matrix.nonzero() == old_weight_matrix.nonzero())
