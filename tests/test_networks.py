import numpy as np
import torch
import matplotlib.pyplot as plt
import pytest
import copy

from networks import WeightAgnosticNetwork, activation_functions_dict, \
    torch_activation_functions_dict, WeightLearningNetwork

show_plots = False


def get_network_stats(net):
    """Return the number of connections, neurons and layers."""
    return net.get_num_connections(), net.num_neurons, len(net.neurons_in_layer)


# ------------------------------------ Activation Functions ----------------------------------------

def test_plot_activation_functions():
    """Plot all activation functions for inspection.."""
    x = np.arange(-2, 2, 0.1)
    for name, f in activation_functions_dict.items():
        plt.plot(x, f(x), label=name)
    plt.title('Numpy activation functions')
    plt.legend()
    if show_plots:
        plt.show()


def test_plot_torch_activation_functions():
    """Plot all torch activation functions for inspection."""
    x = np.arange(-2, 2, 0.1)
    x = torch.from_numpy(x)
    for name, f in torch_activation_functions_dict.items():
        plt.plot(x.numpy(), f(x).numpy(), label=name)
    plt.title('Torch activation functions')
    plt.legend()
    if show_plots:
        plt.show()


# ---------------------------------- WeightAgnosticNetwork -----------------------------------------

@pytest.mark.parametrize('num_mutations', [0, 50])
def test_add_node(num_mutations):
    """Assert that the add_node mutation adds one neuron, one connection and 
    zero or one layers."""
    net = WeightAgnosticNetwork(10, 2, 0.5)
    for _ in range(num_mutations):
        net.mutate()

    num_connections_pre, num_neurons_pre, num_layers_pre = get_network_stats(net)
    net.add_node()
    assert net.get_num_connections() == num_connections_pre + 1
    assert net.num_neurons == num_neurons_pre + 1
    assert len(net.neurons_in_layer) == num_layers_pre or len(
        net.neurons_in_layer) == num_layers_pre + 1


@pytest.mark.parametrize('num_mutations', [0, 50])
def test_add_connection(num_mutations):
    """Assert that the add_connection mutation adds one connection, zero neurons 
    and zero layers."""
    net = WeightAgnosticNetwork(10, 2, 0.5)
    for _ in range(num_mutations):
        net.mutate()

    # Raw net without hidden layers.
    num_connections_pre, num_neurons_pre, num_layers_pre = get_network_stats(net)
    net.add_connection()
    assert net.get_num_connections() == num_connections_pre + 1
    assert net.num_neurons == num_neurons_pre
    assert len(net.neurons_in_layer) == num_layers_pre


@pytest.mark.parametrize('num_mutations', [0, 50])
def test_change_activation(num_mutations):
    """Assert that the change_activation mutation changes one activation 
    function and doesn't add neurons, connections or layers."""
    net = WeightAgnosticNetwork(10, 2, 0.5)
    for _ in range(num_mutations):
        net.mutate()

    while len(net.neurons_in_layer) == 2:  # add ad least one hidden neuron
        net.add_node()

    num_connections_pre, num_neurons_pre, num_layers_pre = get_network_stats(net)
    activation_names_pre = np.array(net.activation_names)
    net.change_activation()
    assert net.get_num_connections() == num_connections_pre
    assert net.num_neurons == num_neurons_pre
    assert len(net.neurons_in_layer) == num_layers_pre
    assert np.sum(activation_names_pre != np.array(net.activation_names)) == 1


@pytest.mark.parametrize('num_mutations', [0, 50])
def test_activation_names(num_mutations):
    """Assert that the number of activation functions matches the number of 
    neurons at all times."""
    net = WeightAgnosticNetwork(10, 2, 0.5)
    for _ in range(num_mutations):
        net.mutate()

    assert len(net.activation_names) == net.num_neurons


def test_wann_forward():
    """Test the forward function of WeightAgnosticNetwork with a manually constructed network."""
    net = WeightAgnosticNetwork(2, 2, 0)  # zero initial connections

    # Insert one hidden node manually.
    net.neurons_in_layer.insert(1, [4])
    net.num_neurons += 1
    net.activation_names.append('sine')

    # Add some connections manually, so that y1 == x1 and y2 == sin(pi *(x1+x2))
    # (where x1/x2 are input neurons and y1/y2 are output neurons).
    net.connections.append([0, 2])
    net.connections.append([0, 4])
    net.connections.append([1, 4])
    net.connections.append([4, 3])

    # Zero input -> zero output.
    input = np.zeros((2, 2))
    output = net.forward(input, 1)
    assert np.all(output == np.zeros(2))

    # Defined input and output.
    input = np.arange(10).reshape(5, 2)
    output = net.forward(input, 1)
    # y1 == x1, see above
    assert np.all(output[:, 0] == input[:, 0])
    # y2 == sin(pi * (x1 + x2))
    assert np.all(output[:, 1] == activation_functions_dict['sine'](input[:, 0] + input[:, 1]))


# ----------------------------------- WeightLearningNetwork ----------------------------------------

@pytest.mark.parametrize('num_mutations', [0, 100])
def test_create_torch_layers(num_mutations):
    """Test create_torch_layers by verifying that weight matrices and weight masks correspond and
    that weight matrices are zero for non-existing connections and non-zero for existing
    connections."""
    net = WeightLearningNetwork(10, 10, 0.5)
    for _ in range(num_mutations):  # add some mutations to get depth
        net.mutate()
    net.create_torch_layers()

    # Check that weight_matrix and weight_mask correspond.
    for layer in net.torch_layers:
        assert torch.all(layer.weight_matrix.nonzero() == layer.weight_mask.nonzero())

    # Check that weights in weight matrices match connections.
    for from_neuron in range(net.num_neurons):
        for to_neuron in range(net.num_neurons):

            # Try to extract the weight from all layers.
            weight = 0
            for layer in net.torch_layers:
                try:
                    weight = layer.get_weight(from_neuron, to_neuron)
                    break
                except ValueError:
                    pass  # layer doesn't contain this weight, search the other ones

            # Check that weight is non-zero if connection exists and non-zero if not.
            if [from_neuron, to_neuron] in net.connections:
                assert weight != 0
            else:
                assert weight == 0


def test_create_delete_torch_layers():
    """Test the methods create_torch_layers and delete_torch_layers by checking that weights
    before and after calling these two methods are equal."""
    net = WeightLearningNetwork(10, 10, 0.5)

    assert net.torch_layers is None

    # Store initial connectivity.
    weights_before = copy.copy(net.weights)
    connections_before = copy.copy(net.connections)

    # Create weight matrices, then delete them again.
    net.create_torch_layers()
    assert net.torch_layers is not None
    assert net.weights is None
    net.delete_torch_layers()

    # Check that connections and weights are as before.
    assert np.allclose(weights_before, net.weights)
    assert connections_before == net.connections


@pytest.mark.parametrize('num_mutations', [0, 50, 500])
def test_restructure_layers(num_mutations):
    """Test restructure_layers by comparing outputs and the number of layers before and after
    restructuring."""

    input = torch.rand(8, 5)
    net = WeightLearningNetwork(5, 2, 0.5)
    for _ in range(num_mutations):
        net.mutate()

    num_layers_before = len(net.neurons_in_layer)

    # Compute network output before restructuring.
    net.create_torch_layers()
    old_output, old_activities = net.forward(input)
    net.delete_torch_layers()

    net.restructure_layers()

    # Compute network output after restructuring. Note that this is using the same weights as above
    # (they were stored via create_weights).
    net.create_torch_layers()
    new_output, new_activities = net.forward(input)
    net.delete_torch_layers()

    # Compare network outputs. We are only moving neurons so this shouldn't change the computation.
    assert torch.allclose(old_output, new_output)
    assert torch.allclose(old_activities, new_activities)

    # Check that network is still strictly feedforward.
    for from_neuron, to_neuron in net.connections:
        from_layer = net.find_layer(from_neuron)
        to_layer = net.find_layer(to_neuron)
        assert from_layer < to_layer

    # Check that the network has less or the same number of layers.
    assert len(net.neurons_in_layer) <= num_layers_before


@pytest.mark.parametrize('num_mutations', [0, 50])
def test_reshuffle_connections(num_mutations):
    """Test reshuffle_connections by checking that the overall architecture of the network doesn't
    change (number of neurons, layers, position of neurons in layers)."""

    # WeightAngosticNetwork.
    net = WeightAgnosticNetwork(5, 2, 0.5)
    for _ in range(num_mutations):
        net.mutate()

    # Compute network stats before reshuffling.
    neurons_in_layer_before = net.neurons_in_layer
    connections_before = net.connections

    net.reshuffle_connections()

    # Compare overall architecture to pre-reshuffling.
    assert (neurons_in_layer_before == net.neurons_in_layer)
    assert (len(connections_before) == len(net.connections))

    # WeightLearningNetwork.
    net = WeightLearningNetwork(5, 2, 0.5, p_add_node=0.5, p_add_connection=0.5)
    for _ in range(num_mutations):
        net.mutate()

    # Compute network stats before reshuffling.
    neurons_in_layer_before = net.neurons_in_layer
    connections_before = net.connections

    net.reshuffle_connections()

    # Compare overall architecture to pre-reshuffling.
    assert neurons_in_layer_before == net.neurons_in_layer
    assert len(connections_before) == len(net.connections)


@pytest.mark.parametrize('num_mutations', [0, 50])
def test_save_load(tmp_path, num_mutations):
    """Test save and load by saving and loading a network and checking that all attributes are
    equal. """

    def assert_networks_equal(net, loaded_net):
        assert net.in_size == loaded_net.in_size
        assert net.out_size == loaded_net.out_size
        assert net.p_initial_connection_enabled == loaded_net.p_initial_connection_enabled
        assert net.p_add_connection == loaded_net.p_add_connection
        assert net.p_add_node == loaded_net.p_add_node
        assert net.inherit_weights == loaded_net.inherit_weights
        assert net.activation_name == loaded_net.activation_name
        assert net.neurons_in_layer == loaded_net.neurons_in_layer
        assert net.connections == loaded_net.connections
        assert np.allclose(net.weights, loaded_net.weights)

    net = WeightLearningNetwork(5, 2, 0.5)
    for _ in range(num_mutations):
        net.mutate()

    # Save network, load it and check that everything is still correct.
    print(tmp_path / 'network.json')
    net.save(tmp_path / 'network.json')
    loaded_net = WeightLearningNetwork.load(tmp_path / 'network.json')
    assert_networks_equal(net, loaded_net)

    # Do this again after creating and deleting weight matrices.
    net.create_torch_layers()
    net.delete_torch_layers()

    net.save(tmp_path / 'network2.json')
    loaded_net = WeightLearningNetwork.load(tmp_path / 'network2.json')
    assert_networks_equal(net, loaded_net)

    # Check that we can still create weight matrices on loaded net.
    loaded_net.create_torch_layers()
