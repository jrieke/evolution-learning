#!/usr/bin/env python3
"""
Train network on a classification dataset with one of several learning algorithms (without any
evolution loop!).

This serves as a baseline for all scripts involving evolution, and to check results and runtimes in
a simple setting.
"""
import torch.utils.data
import logging

from networks import WeightLearningNetwork
from datasets import load_preprocessed_dataset
from learning import train
import utils


# Set up parameters and output dir.
params = utils.load_params(mode='wlnn')  # based on terminal input
params['script'] = 'run-wlnn-mnist-no-evolution.py'
writer, out_dir = utils.init_output(params)

if params['use_cuda'] and not torch.cuda.is_available():
    logging.info('use_cuda was set but cuda is not available, running on cpu')
    params['use_cuda'] = False

device = 'cuda' if params['use_cuda'] else 'cpu'


# Load dataset.
train_images, train_labels, test_images, test_labels = load_preprocessed_dataset(
    params['dataset'], flatten_images=True, use_torch=True)
train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)


# Create network.
#utils.seed_all(0)
train_only_outputs = (params['learning_rule'] == 'hebbian')
use_random_feedback = (params['learning_rule'] == 'feedback_alignment')
net = WeightLearningNetwork(params['num_inputs'], params['num_outputs'],
                            params['p_initial_connection_enabled'], p_add_connection=0.8,
                            p_add_node=0.2, train_only_outputs=train_only_outputs,
                            use_random_feedback=use_random_feedback)
logging.info(f'Initial network has {net}')


# Option 1: Apply some random mutations to make the network deeper.
num_mutations = 0
for _ in range(num_mutations):
    net.mutate()
    #net.add_connection()
logging.info(f'Applied {num_mutations} mutations, network now has {net}')
#net.restructure_layers()
#logging.info('Restructured layers')


# Option 2: Manually add a hidden layer with 500 neurons.
# num_hidden = 500
# net.neurons_per_layer.insert(1, list(range(net.num_neurons, net.num_neurons + num_hidden)))
# net.connections = []  # erase any connections between input and output
# for from_neuron in net.neurons_per_layer[0]:  # add connections between input and hidden
#     for to_neuron in net.neurons_per_layer[1]:
#         net.connections.append([from_neuron, to_neuron])
# for from_neuron in net.neurons_per_layer[1]:  # add connections between hidden and output
#     for to_neuron in net.neurons_per_layer[2]:
#         net.connections.append([from_neuron, to_neuron])
# net.num_neurons += num_hidden
# net.reset_weights()  # need to call this manually due to manual changes to architecture
# logging.info(f'Added a hidden layer with {num_hidden} neurons, network now has {net}')


net.create_torch_layers(device)
train(net, train_dataset, params, device=device, verbose=2, test_dataset=test_dataset)
