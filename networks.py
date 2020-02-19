"""
Network implementations.

All networks can do mutations that can be used in evolution. Includes weight agnostic network and a
normal learning network that can be used with different learning algorithms.
"""

import numpy as np
import torch
import logging
from layers import LinearLayer, LinearLayerFA
import random
import json


activation_functions_dict = {
    'linear': lambda x: x,
    'step': lambda x: 1 * (x > 0),
    'sine': lambda x: np.sin(np.pi * x),
    'cosine': lambda x: np.cos(np.pi * x),
    'gaussian': lambda x: np.exp(-x ** 2 / 2),
    'tanh': np.tanh,
    'sigmoid': lambda x: (np.tanh(x / 2) + 1) / 2,
    'inverse': lambda x: -x,
    'abs': np.abs,
    'relu': lambda x: x * (x > 0)
}

torch_activation_functions_dict = {
    'linear': lambda x: x,
    'step': lambda x: 1 * (x > 0),
    'sine': lambda x: torch.sin(np.pi * x),
    'cosine': lambda x: torch.cos(np.pi * x),
    'gaussian': lambda x: torch.exp(-x ** 2 / 2),
    'tanh': torch.tanh,
    'sigmoid': torch.sigmoid,
    'inverse': lambda x: -x,
    'abs': torch.abs,
    'relu': torch.relu
}


def random_activation_name(exclude=None):
    """
    Return the name of a random activation function (see dict above).

    Args:
        exclude (str): Do not return this activation function.
    """
    activation_function = np.random.choice(list(activation_functions_dict.keys()))
    if activation_function == exclude:
        return random_activation_name(exclude=exclude)
    else:
        return activation_function


def flatten(l):
    """Flatten a list of lists."""
    return [item for sublist in l for item in sublist]


class WeightAgnosticNetwork:
    """
    Network class for a weight agnostic network. All values share the same weight value during the
    forward pass. The architecture of the network can be mutated by adding nodes, adding connections
    or changing activation function.

    Attributes:
        num_neurons (int): The number of neurons.
        neurons_in_layer (list): Each element is a list with the neuron indices of this layer.
        activation_names (list): Each element is the (str) name of the activation function
            for the neuron at this index.
        connections (list): Each element is a list [from_neuron, to_neuron].
    """

    def __init__(self, in_size, out_size, p_initial_connection_enabled,
                 p_change_activation=0.5, p_add_connection=0.25, p_add_node=0.25,
                 use_torch=False, add_only_hidden_connections=False):
        """
        Initialize a weight agnostic network with input/output layer and some random connections
        in between.

        Args:
            in_size (int): The number of input neurons.
            out_size (int): The number of output neurons.
            p_initial_connection_enabled (float): Probability to enable a connection
                between input and output at start.
            p_change_activation (float, optional): Probability for the "change activation"
                mutation (default: 0.5).
            p_add_connection (float, optional): Probability for the "add connection" mutation
                (default: 0.25).
            p_add_node (float, optional): Probability for the "add node" mutation (default: 0.25).
            use_torch (boolean, optional): Whether to use torch or numpy backend (default:
                False).
            add_only_hidden_connections (boolean, optional): If True, the add_node mutation will not
                add connections to output neurons (default: False).
        """

        if p_add_connection + p_add_node + p_change_activation != 1:
            raise ValueError('p_add_connection and p_add_node and p_change_activation must sum '
                             'to 1')

        self.in_size = in_size
        self.out_size = out_size
        self.p_change_activation = p_change_activation
        self.p_add_connection = p_add_connection
        self.p_add_node = p_add_node
        self.use_torch = use_torch
        self.p_initial_connection_enabled = p_initial_connection_enabled
        self.add_only_hidden_connections = add_only_hidden_connections

        # These attributes will be set in self.reset.
        self.num_neurons = None
        self.neurons_in_layer = None
        self.activation_names = None
        self.connections = None
        self.reset()

    def reset(self):
        """
        Reset the state of the network.

        Initialize input and output layer with linear activation functions. Create some random
        connections between these two layers.
        """
        self.num_neurons = self.in_size + self.out_size
        self.neurons_in_layer = [list(range(self.in_size)),
                                  list(range(self.in_size, self.num_neurons))]
        self.activation_names = ['linear'] * self.num_neurons

        # Create some random connections.
        self.connections = []  # elements are: [from_neuron, to_neuron]
        for from_neuron in self.neurons_in_layer[0]:
            for to_neuron in self.neurons_in_layer[-1]:
                if np.random.rand() < self.p_initial_connection_enabled:
                    self.connections.append([from_neuron, to_neuron])

    def reshuffle_connections(self):
        """
        Reshuffle the connections.

        This function reshuffles the connections while maintaining the same 
        number of neurons and their location within the layers. Specifically,
        given an existing connection, it replaces it by a connection from a 
        neuron of the same pre-synaptic layer to a neuron in the same post-
        synaptic layer. This function is mostly for control experiments.
        """
        old_connections = self.connections

        self.connections = []
        for i in range(len(old_connections)):
            from_neuron, to_neuron = old_connections[i]
            from_layer = self.find_layer(from_neuron)
            to_layer = self.find_layer(to_neuron)
            from_neuron = random.choice(self.neurons_in_layer[from_layer])
            to_neuron = random.choice(self.neurons_in_layer[to_layer])
            self.connections.append((from_neuron, to_neuron))

    def forward(self, input, weight_value):
        """
        Pass a batch of samples through the network and return its output.

        Args:
            input (array, shape: batch size x in size): Batch of input samples.
            weight_value (int): The shared weight value of all connections.

        Returns:
            array (shape: batch size x out size): The batched network output.
        """
        batch_size = len(input)

        # Make an array to store the activity of each neuron for each batch.
        activities = np.zeros((batch_size, self.num_neurons)) if not \
            self.use_torch else torch.zeros((batch_size, self.num_neurons))
        activities[:, :self.in_size] = input

        # Make an array to store the weighted input of the current neuron for
        # each batch element. Reuse this in the loop below.
        weighted_input = np.zeros(batch_size) if not self.use_torch \
            else torch.zeros(batch_size)

        # Iterate over all hidden and output neurons and calculate their activity.
        for neurons in self.neurons_in_layer[1:]:
            for neuron in neurons:
                weighted_input[:] = 0
                for i, j in self.connections:
                    if j == neuron:
                        weighted_input += weight_value * activities[:, i]
                if self.use_torch:
                    activities[:, neuron] = torch_activation_functions_dict[
                        self.activation_names[neuron]](
                        weighted_input)
                else:
                    activities[:, neuron] = activation_functions_dict[
                        self.activation_names[neuron]](weighted_input)

        output = activities[:, self.in_size:self.in_size + self.out_size]
        return output

    def add_node(self):
        """Mutate the network by splitting a random connection and adding a node in between."""
        # print('Adding node')
        # print('Layers before:', self.layers)
        # print('Connections before:', self.connections)

        if len(self.connections) > 0:
            # Pick a random connection.
            connection_idx = np.random.randint(len(self.connections))

            # Change existing connection and add new connection.
            from_neuron, to_neuron = self.connections[connection_idx]
            self.connections[connection_idx][1] = self.num_neurons
            self.connections.append([self.num_neurons, to_neuron])

            # Add new neuron to intermediate layer (existing one or new one).
            from_layer = self.find_layer(from_neuron)
            to_layer = self.find_layer(to_neuron)
            if to_layer > from_layer + 1:
                self.neurons_in_layer[from_layer + 1].append(self.num_neurons)
            else:
                self.neurons_in_layer.insert(from_layer + 1, [self.num_neurons])

            self.activation_names.append(random_activation_name())
            self.num_neurons += 1

            # print('Added node for connection', from_neuron, '->', to_neuron)
            # print('Layers after:', self.layers)
            # print('Connections after:', self.connections)
            # print()

    def add_connection(self):
        """Mutate the network by adding a feedforward connection between two random neurons."""
        # print('Adding connection')
        # print('Layers before:', self.layers)
        # print('Connections before:', self.connections)

        for i in range(10000):  # max 10k tries to find non-existing connection
            # Pick a random source neuron from all layers but the output layer.
            # .item() is necessary here to get standard int type (instead of numpy's int64 type) for
            # json serializing in save method.
            if self.add_only_hidden_connections:
                if len(self.neurons_in_layer) <= 2:
                    logging.info(
                        'Wanted to add a new connection but could not find a valid one '
                        '(add_only_hidden_connections is set to True but network does not contain '
                        'hidden neurons)')
                    break
                from_layers = self.neurons_in_layer[:-2]
            else:
                from_layers = self.neurons_in_layer[:-1]
            from_neuron = np.random.choice(flatten(from_layers)).item()
            from_layer = self.find_layer(from_neuron)

            # Find a destination neuron in the layers above it.
            if self.add_only_hidden_connections:
                to_layers = self.neurons_in_layer[from_layer+1:len(self.neurons_in_layer)-1]
                # print('start', from_layer+1)
                # print('end', len(self.neurons_in_layer)-2)
            else:
                to_layers = self.neurons_in_layer[from_layer+1:]
            # for neurons in self.neurons_in_layer:
            #     print(neurons)
            # print(from_layer)
            # print(to_layers)
            # print('--')
            to_neuron = np.random.choice(flatten(to_layers)).item()
            new_connection = [from_neuron, to_neuron]

            # Add the new connection.
            if new_connection not in self.connections:
                self.connections.append(new_connection)

                # Debug.
                # if new_connection[1] in self.neurons_in_layer[-1]:
                #     print('added output connection')
                # else:
                #     print('added hidden connection')

                break
            if i == 9999:
                logging.info('Wanted to add a new connection but could not find a valid one (tried '
                             '10k times)')

        # print('Added connection', new_connection[0], '->', new_connection[1])
        # print('Layers after:', self.layers)
        # print('Connections after:', self.connections)
        # print()

    def change_activation(self):
        """Mutate the network by changing the activation function of a random hidden neuron."""
        # print('Changing activation')
        # # print('Layers before:', self.layers)
        # # print('Connections before:', self.connections)
        # print('Activations before:', self.activation_functions)

        if self.num_neurons > self.in_size + self.out_size:  # only hidden neurons
            neuron = np.random.randint(self.in_size + self.out_size, self.num_neurons)
            self.activation_names[neuron] = random_activation_name(
                exclude=self.activation_names[neuron])

            # print('Changed activation for neuron', neuron)
            # # print('Layers after:', self.layers)
            # # print('Connections after:', self.connections)
            # print('Activations after:', self.activation_functions)
        # else:
        #     print('No hidden neurons, could not change activation')
        # print()

    def mutate(self):
        """Mutate the network by applying a random mutation."""
        mutation = np.random.choice(
            [self.change_activation, self.add_connection, self.add_node],
            p=[self.p_change_activation, self.p_add_connection, self.p_add_node])
        mutation()

    def get_num_connections(self):
        """Return the number of connections in the network."""
        return len(self.connections)

    def __str__(self):
        return f'{self.num_neurons} neurons, {self.get_num_connections()} connections, ' \
               f'{len(self.neurons_in_layer)} layers'

    def find_layer(self, neuron):
        """
        Find the layer in which a neuron is located.

        Args:
            neuron (int): Index of the neuron.

        Returns:
            int: Index of the layer that the neuron is in.
        """
        for i, neurons in enumerate(self.neurons_in_layer):
            if neuron in neurons:
                return i


class WeightLearningNetwork(WeightAgnosticNetwork):
    """
    Network whose architecture can be evolved (similar to WeightAgnosticNetwork) but which learns
    its weights via learning algorithms (e.g. backpropagation). Makes use of pytorch for learning.

    This net basically exists in two states: By default, weights are stored as a plain list in
    self.weights. This state is compact and easy to mutate but doesn't allow for any computations
    (e.g. forward or backward pass). If you want to start training, call create_torch_layers. This
    will turn self.weights into weight matrices on a specified device and store them as proper torch
    layers in self.torch_layer. The network can now do all of its computations. After training has
    finished and you may want to do mutations or save the network to file, call delete_torch_layers
    to re-create self.weights and get rid of self.torch_layers.
    Note that in this class, all weight matrices are compact (no all-zero rows or columns exist) in
    order to speed up the training.

    Attributes:
        weights (list): A list of weight values for all connections. Will be deleted once
            create_torch_layers is called. Call delete_torch_layers to re-create it.
        torch_layers (list): The torch layers with weight matrices used for forward
            computation. Initialize these with create_torch_layers.

    See WeightAgnosticNetwork for further attributes.
    """

    def __init__(self, in_size, out_size, p_initial_connection_enabled, p_add_connection=0.5,
                 p_add_node=0.5, activation_name='relu', inherit_weights=False,
                 weight_init_std=0.1, train_only_outputs=False, use_random_feedback=False,
                 add_only_hidden_connections=False):
        """
        Initialize a WeightLearningNetwork with random connections and weights.

        Args:
            in_size (int): The number of input neurons.
            out_size (int): The number of output neurons.
            p_initial_connection_enabled (float): Probability to enable a connection between input
                and output at start.
            p_add_connection (float, optional): Probability for the "add connection" mutation
                (default: 0.5).
            p_add_node (float, optional): Probability for the "add node" mutation (default: 0.5).
            activation_name (str, optional): The name of the activation function for all neurons
                (default: 'relu').
            inherit_weights (bool, optional): Retain existing weights after a mutation (if False,
                they will be reset after a mutation) (default: False).
            weight_init_std (float, optional): Standard deviation for weight initialization
                (default: 0.1).
            train_only_outputs (bool, optional): If True, only the weights to the output
                layer will be trained. Else, all weights will be trained.
            use_random_feedback (bool, optional): If True, use random feedback connections. This is
                used for training with feedback alignment.
            add_only_hidden_connections (bool, optional): If True, the add_connection mutation will
                not add connections to output neurons.
        """
        # We do not use p_change_activation here (it will be set to 0), so the other two
        # probabilities must sum to 1.
        if p_add_connection + p_add_node != 1:
            raise ValueError('p_add_connection and p_add_node must sum to 1')

        # Always set use_torch=True because this network is built with torch.
        super().__init__(in_size, out_size, p_initial_connection_enabled, p_add_node=p_add_node,
                         p_add_connection=p_add_connection, p_change_activation=0, use_torch=True,
                         add_only_hidden_connections=add_only_hidden_connections)

        self.activation_name = activation_name
        self.activation = torch_activation_functions_dict[activation_name]
        self.inherit_weights = inherit_weights
        self.train_only_outputs = train_only_outputs
        self.use_random_feedback = use_random_feedback

        self.weights = None  # list of weight values for all connections
        self.torch_layers = None  # list of torch layers, initialized with create_weight_matrices

        # Create initial weights.
        self.weight_init_std = weight_init_std
        self.reset_weights()

    def reset_weights(self):
        """
        Reset the weights of the network.

        Weights are sampled from a normal distribution and stored as a list in self.weights. Any
        existing torch layers are deleted when this function is called.
        """
        # TODO: Maybe use xavier initialization instead.
        self.delete_torch_layers()
        weights = np.random.randn(len(self.connections)) * self.weight_init_std
        self.weights = weights.tolist()

    def parameters(self):
        if self.torch_layers is None:
            raise RuntimeError('Torch parameters are not built, call create_torch_layers before '
                               'this method')
        if self.train_only_outputs:
            return [self.torch_layers[-1].weight_matrix]
        else:
            return [layer.weight_matrix for layer in self.torch_layers]

    def create_torch_layers(self, device=None):
        """
        Create torch layers in self.torch_layers (with weight matrices and masks) based on
        self.connections and self.weights.

        Each layer has one weight matrix/weight mask. This function also deletes self.weights (the
        list of connection strengths), because it might get out of sync with the weight matrices
        once training starts. To get self.weights back (and get rid of torch layers), call
        delete_torch_layers.

        Args:
            device (str or torch.device, optional): The device to put weight matrices and masks on.
        """
        if self.torch_layers is not None:
            raise RuntimeError('Torch layers already exist. If you want to re-create them '
                               '(e.g. on a different device), call delete_torch_layers before')

        # Create weight matrices and masks.
        self.torch_layers = []
        for i in range(1, len(self.neurons_in_layer)):  # no torch layer for input neurons

            # Find all neurons connecting to this layer.
            neurons_connecting_to_layer = set()  # avoid duplicates
            for from_neuron, to_neuron in self.connections:
                to_layer = self.find_layer(to_neuron)
                if to_layer == i:
                    neurons_connecting_to_layer.add(from_neuron)
            neurons_connecting_to_layer = sorted(list(neurons_connecting_to_layer))

            weight_matrix, weight_mask = self.create_weight_matrix(
                self.connections, self.weights, from_neurons=neurons_connecting_to_layer,
                to_neurons=self.neurons_in_layer[i], device=device)
            if not self.train_only_outputs or i == len(self.neurons_in_layer)-1:
                weight_matrix.requires_grad = True

            if self.use_random_feedback:  # feedback alignment
                backward_weight_matrix = torch.randn_like(weight_matrix, requires_grad=False)
                backward_weight_matrix *= weight_mask
                self.torch_layers.append(LinearLayerFA(weight_matrix, backward_weight_matrix,
                                                       weight_mask,
                                                       from_neurons=neurons_connecting_to_layer,
                                                       to_neurons=self.neurons_in_layer[i]))
            else:  # normal backpropagation
                self.torch_layers.append(LinearLayer(weight_matrix, weight_mask,
                                                     from_neurons=neurons_connecting_to_layer,
                                                     to_neurons=self.neurons_in_layer[i]))

        # Delete self.weights so that it doesn't get out of sync with weight matrices during
        # training.
        self.weights = None

    def delete_torch_layers(self):
        """Delete torch layers and store all weights in self.weights."""
        if self.torch_layers is not None:
            # Retrieve weights from torch layers and store them as a list in self.weights
            # (corresponding to the order of self.connections).
            self.weights = []
            for from_neuron, to_neuron in self.connections:
                to_layer = self.find_layer(to_neuron)
                weight = self.torch_layers[to_layer - 1].get_weight(from_neuron, to_neuron)
                self.weights.append(weight)

        # Delete torch layers and the helper list neurons_connecting_to_layer.
        self.torch_layers = None

    def reshuffle_connections(self):
        """Reshuffle connections and recreate torch layers if necessary."""

        # TODO: Re-create layers on the same device as before.
        # TODO: Maybe rather delete torch layers completely here (similar to mutation methods).
        create_layers = False
        if self.torch_layers is not None:
            create_layers = True
            self.delete_torch_layers()

        super().reshuffle_connections()

        if create_layers:
            self.create_torch_layers()

    def create_weight_matrix(self, connections, weights, from_neurons, to_neurons, device=None):
        """
        Create weight matrix and weight mask for a specific layer and return both.

        Args:
            connections (list): Each element should be one connection [pre-synaptic neuron,
                post-synaptic neuron].
            weights (list, same length as connections): A list of connection strengths for all
                connections.
            from_neurons (list): Indices of neurons having connections to this layer.
            to_neurons (list): Indices of neurons in this layer.
            device (str or torch.cuda, optional): The device to put the weight matrix on.

        Returns:
            torch.Tensor: The weight matrix of shape num neurons in the layer x num neurons
                connecting to the layer. First dimension is post-synaptic neuron, second dimension
                is pre-synaptic neuron.
            torch.Tensor: The corresponding weight mask for the weight matrix (same shape). 1 where
                a connection exists and 0 if the connection doesn't exist.
        """
        weight_matrix = np.zeros((len(to_neurons), len(from_neurons)))
        weight_mask = np.zeros((len(to_neurons), len(from_neurons)))

        # Iterate through self.connections and self.weights and set the weights in the weight
        # matrix.
        for (from_neuron, to_neuron), weight in zip(connections, weights):
            if to_neuron in to_neurons:
                from_neuron_index = from_neurons.index(from_neuron)
                to_neuron_index = to_neurons.index(to_neuron)
                weight_matrix[to_neuron_index, from_neuron_index] = weight
                weight_mask[to_neuron_index, from_neuron_index] = 1

        # Convert to torch and move to device.
        weight_matrix = torch.from_numpy(weight_matrix).float().to(device)
        weight_mask = torch.from_numpy(weight_mask).float().to(device)

        return weight_matrix, weight_mask

    def count_nonzero_weights(self):
        """
        Return the number of non-zero weights in all weight matrices.

        Returns:
            int: Number of nonzero weights in all weight matrices.
        """
        num_nonzero_weights = 0
        for layer in self.torch_layers:
            num_nonzero_weights += (layer.weight_matrix != 0).sum().item()
        return num_nonzero_weights

    def forward(self, input):
        """
        Compute output of the network for a given batch.

        Args:
            input (torch.Tensor, shape: batch size x num input neurons): The batch of samples.

        Returns:
            torch.Tensor (shape: batch size x num output neurons): The output for the batch.
            torch.Tensor (shape: batch size x num neurons): The activities of all neurons in the
                network for the batch.
        """
        if self.torch_layers is None:
            raise RuntimeError('torch layers do not exist, call create_weight_matrices before this')

        batch_size = len(input)

        # Make a tensor which stores the activities of all neurons in the network for each batch.
        activities = torch.zeros((batch_size, self.num_neurons), device=input.device)
        activities[:, self.neurons_in_layer[0]] = input

        for i, layer in enumerate(self.torch_layers):
            # Compute activities of neurons in current layer (in contrast to activities, this is a
            # compact tensor, which only contains the activities for neurons in this layer!).
            # Shapes: (batch_size x len layer) = (batch_size x num neurons connecting to layer)
            #                                    * (num neurons connecting to layer x len layer)

            # Compute the total inputs to a given layer.
            layer_activities = layer.forward(activities[:, layer.from_neurons])

            # Apply activation function (not for last layer).
            is_output_layer = (i == len(self.torch_layers) - 1)
            if not is_output_layer:
                layer_activities = self.activation(layer_activities)

            # Update the matrix of neuron activations by adding those of the current layer.
            activities_new = activities.clone()
            activities_new[:, layer.to_neurons] = activities[:, layer.to_neurons] + layer_activities
            activities = activities_new

        output = activities[:, self.neurons_in_layer[-1]]
        return output, activities

    def add_node(self):
        """
        When a node is added, a random connection between a from_neuron
        and a to_neuron is selected. A new node of index self.num_neurons is
        then added in this connection. The old weight is maintained for the
        connection to the new neuron, while a random one is selected for the 
        second connection.
        Therefore an initial connection: 
            from_neuron -> to_neuron (old_weight)
        Is replaced by two new connections:
            from_neuron -> new_neuron (old_weight)
            new_neuron -> to_neuron (new random weight)

        If self.inherit_weights=False, all weights are reset after adding the new node using 
        self.reset_weights. This step automatically creates a self.weights vector that has one 
        more value than before the mutation. If self.inherit_weights=True, a random weight 
        value is appended to the existing self.weights list.
        """
        self.delete_torch_layers()
        super().add_node()

        if self.inherit_weights:
            # Append random weight for the new connection (if one was added).
            if len(self.connections) > 0:  # if no connection exists, no node was added
                new_weight = np.random.randn() * self.weight_init_std
                self.weights.append(new_weight)
        else:
            self.reset_weights()

        # Restructure layers to make the network as compact as possible. Adding a node sometimes
        # introduces unnecessary layers that can be avoided by arranging neurons differently.
        self.restructure_layers()

    def add_connection(self):
        """
        Mutate the network by adding a (feedforward) connection between two
        random neurons.

        Note that this will delete any existing weight matrices. If self.inherit_weights is True, a
        new random weight will be added if a connection was added. Otherwise, all weights will be
        reset after the mutation.
        """
        self.delete_torch_layers()
        super().add_connection()

        if self.inherit_weights:
            # Append random weight for the new connection.
            new_weight = np.random.randn() * self.weight_init_std
            self.weights.append(new_weight)
        else:
            self.reset_weights()

    def restructure_layers(self):
        """
        Make the layers of the network as compact as possible.

        After a lot of mutations, the layers can be arranged quite inefficiently (e.g. if a node is
        added between input and first hidden layer). This costs time and memory during training
        because we have one weight matrix per layer. This method goes through each layer and moves
        neurons to the layer below if possible, until the layer structure is fully optimized. This
        doesn't change connections or the network computation/output, only layer arrangement.
        """
        if self.torch_layers is not None:
            raise RuntimeError('Found weight matrices, delete them with delete_weight_matrices '
                               'before calling restructure_layers')

        is_sorted = False  # this will turn True if the network cannot be optimized any more
        while not is_sorted:
            is_sorted = True
            for i_layer in range(2, len(
                    self.neurons_in_layer) - 1):  # not input/first hidden/output layer
                for neuron in self.neurons_in_layer[i_layer]:

                    # Go through all connections and find out if the neuron has a connection from
                    # the layer directly below.
                    has_connection_from_layer_below = False
                    for from_neuron, to_neuron in self.connections:
                        if to_neuron == neuron and from_neuron in self.neurons_in_layer[
                            i_layer - 1]:
                            has_connection_from_layer_below = True
                            break

                    # If there is no such connection, move the neuron one layer lower.
                    if not has_connection_from_layer_below:
                        self.neurons_in_layer[i_layer - 1].append(neuron)
                        self.neurons_in_layer[i_layer].remove(neuron)
                        is_sorted = False  # if something has changed, start optimizing again

            # Remove empty layers (needs to be done after iterating through layers!).
            self.neurons_in_layer = [x for x in self.neurons_in_layer if x != []]

    def save(self, filename):
        """
        Save the network as a json file.

        Stores self.connections, self.weights and self.neurons_in_layer in the file, along with
        parameters given to constructor (e.g. mutation probabilities). This is safer than saving the
        entire network object as a pickle file because the pickled object may not work any more when
        the implementation here changes.
        """
        if self.torch_layers is not None:
            raise RuntimeError('Found weight matrices, delete them with delete_weight_matrices '
                               'before calling save')

        json_dict = {
            'in_size': self.in_size,
            'out_size': self.out_size,
            'num_neurons': self.num_neurons,
            'p_initial_connection_enabled': self.p_initial_connection_enabled,
            'p_add_connection': self.p_add_connection,
            'p_add_node': self.p_add_node,
            'inherit_weights': self.inherit_weights,
            'activation_name': self.activation_name,
            'neurons_in_layer': self.neurons_in_layer,
            'connections': self.connections,
            'weights': self.weights
        }

        with open(filename, 'w') as f:
            json.dump(json_dict, f)

    @staticmethod
    def load(filename):
        """Load network from a json file."""
        with open(filename, 'r') as f:
            json_dict = json.load(f)

        net = WeightLearningNetwork(
            json_dict['in_size'], json_dict['out_size'], json_dict['p_initial_connection_enabled'],
            json_dict['p_add_connection'], json_dict['p_add_node'], json_dict['activation_name'],
            json_dict['inherit_weights'])

        net.num_neurons = json_dict['num_neurons']
        net.neurons_in_layer = json_dict['neurons_in_layer']
        net.connections = json_dict['connections']
        net.weights = json_dict['weights']

        return net
