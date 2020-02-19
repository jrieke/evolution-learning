"""
Learning rules and helper functions for training.
"""

import torch
import numpy as np
import logging
import time
import torch.nn as nn
import GPUtil


def hebbian_rule(net, activities, lr=1e-3):
    """
    Update the weights in all hidden layers of the network using Oja's rule.

    Oja's rule is a simple Hebbian learning rule (activity of pre-synaptic neuron * activity of
    post-synaptic neuron) with an additional normalization term. This term normalizes the vector of
    incoming weights to a neuron, so that its Euclidean norm is one. Itt stabilizes the weight
    update so that the weights don't vanish or explode.

    Note that the last layer is not trained in this function but should be trained with normal
    gradient descent.

    Args:
        net (WeightLearningNetwork): The network to update. Needs to have its torch layers
            initialized.
        activities (torch.Tensor, shape: batch size x num neurons): Activities of all neurons across
            for all samples in the batch.
        lr (float, optional): Learning rate (default: 1e-3).
    """
    with torch.no_grad():  # disable gradient tracking to prevent memory leaks

        batch_size = len(activities)

        # Obtain coactivities (= pre * post) for all neurons, averaged over the batch.
        pre_activities = activities
        post_activities = activities.t()
        coactivities = torch.mm(post_activities, pre_activities) / batch_size

        # Update weights in each layer based on these coactivities.
        # Leave out last layer as it will be trained by gradient.
        for layer in net.torch_layers[:-1]:

            # Compute gradient for the layer's weight matrix.
            layer_coactivities = coactivities[layer.to_neurons][:, layer.from_neurons]
            squared_layer_activities = (activities[:, layer.to_neurons]**2).sum(0) / batch_size

            # Oja's rule.
            grad = lr * (layer_coactivities - squared_layer_activities[:, None]
                         * layer.weight_matrix)

            # Backward hook: zero-out non existing connections
            grad = grad * layer.weight_mask

            # Log very low or high values in the gradient.
            if grad.abs().max() > 10:
                logging.info('Gradient is unusually low or high:')
                logging.info('Gradient average: {:.3f} +- {:.3f} (min: {:.3f}, '
                             'max: {:.3f})'.format(grad.mean().item(), grad.std().item(),
                                                   grad.min().item(), grad.max().item()))

            # Update weight matrix with gradient.
            layer.weight_matrix += grad


def yield_batch(dataset, batch_size):
    """
    Generator that yields batches (with random samples) from a torch TensorDataset.

    Note that this is a lot faster than torch's built-in DataLoader because DataLoader slices and
    concatenates tensors at each step (which we avoid here).
    """
    permuted_indices = np.random.permutation(len(dataset))
    num_batches = int(len(dataset) / batch_size)
    for i in range(num_batches):
        batch_indices = permuted_indices[i * batch_size:(i + 1) * batch_size]
        yield dataset.tensors[0][batch_indices], dataset.tensors[1][batch_indices]


def test(net, test_dataset, params, device='cpu'):
    """
    Evaluate net on test_dataset.

    Args:
        net: net (networks.WeightLearningNetwork): The network to train. Weight matrices need to be
            created outside of this method.
        test_dataset (torch.utils.data.Dataset): The dataset to test on.
        params (dict): The parameter dictionary. Needs to contain the field 'batch_size'.
        device (str): The device to train on. Note that the weight matrices of the network need to
            be on the same device (default: cpu).

    Returns:
        float: Loss averaged over the test dataset.
        float: Accuracy averaged over the test dataset.
    """
    loss_function = torch.nn.CrossEntropyLoss()

    batch_loss = 0
    batch_acc = 0

    with torch.no_grad():

        # Store a list of all real and predicted labels to get confusion matrix.
        all_labels = []
        all_pred_labels = []

        for batch, (images, labels) in enumerate(yield_batch(test_dataset, params['batch_size'])):
            # Forward pass.
            images, labels = images.to(device), labels.to(device)
            output, _ = net.forward(images)

            # Calculate loss and accuracy.
            loss = loss_function(output, labels)
            batch_loss += loss.item()
            pred_labels = output.argmax(1)
            acc = (pred_labels == labels).float().mean()
            batch_acc += acc.item()

            all_labels.extend(labels)
            all_pred_labels.extend(pred_labels)

    test_loss = batch_loss / (batch + 1)
    test_acc = batch_acc / (batch + 1)

    return test_loss, test_acc


def train(net, train_dataset, params, device='cpu', verbose=0, test_dataset=None):
    """
    Train net on train_dataset for multiple epochs.

    Args:
        net (networks.WeightLearningNetwork): The network to train. Note that the weight matrices of
            this network will be created directly in this method.
        train_dataset (torch.utils.data.Dataset): The dataset to train on.
        params (dict): The parameter dictionary. Needs to contain the fields 'optimizer', 'lr',
            'batch_size', 'num_epochs', 'learning_rule'.
        device (str, optional): The device to train on. Note that the weight matrices of the network
            will also be created on this device (default: cpu).
        verbose (int, optional): Verbose mode. 0: Log nothing, 1: Log loss and accuracy per epoch,
            2: Log loss, accuracy and weight statistics per epoch and if a large weight
            (absolute value > 1000) was found (default: 0).
        test_dataset (torch.utils.data.Dataset, optional): If not None, evaluate on this dataset
            after every epoch. Note that these results will only be printed in verbose mode
            (default: None).

    Returns:
        (list): List of losses over all epochs, where each element is the list of losses in each
            batch for a given epoch.
        (list): List of accuracies over all epochs, where each element is the list of accuracies 
            in each batch for a given epoch.
    """
    loss_function = torch.nn.CrossEntropyLoss()

    # Set up optimizer.
    # If hebbian learning is enabled, this optimizer will only train the last layer.
    if params['optimizer'] == 'sgd':
        opt = torch.optim.SGD(net.parameters(), lr=params['lr'])
    elif params['optimizer'] == 'adam':
        opt = torch.optim.Adam(net.parameters(), lr=params['lr'])
    elif params['optimizer'] == 'adadelta':
        opt = torch.optim.Adadelta(net.parameters(), lr=params['lr'])
    elif params['optimizer'] is None or params['optimizer'] == 'None':
        opt = None
    else:
        raise ValueError('optimizer not recognized')

    # Training loop.
    batch_losses_per_epoch = []
    batch_accs_per_epoch = []
    for epoch in range(params['num_epochs']):

        start_time_epoch = time.time()
        batch_losses = []
        batch_accs = []

        if verbose >= 2:
            log_weight_stats(net)
            logging.info('')

        for batch, (images, labels) in enumerate(yield_batch(train_dataset, params['batch_size'])):

            # Sanity check 1: Train on zero input and dataset target
            #                 (should give chance results).
            # images = torch.zeros_like(images)

            # Sanity check 2: Train on dataset input and zero target
            #                 (should give perfect results).
            # labels = torch.zeros_like(labels)

            # Sanity check 3: Overfit on a single batch
            #                 (should give almost perfect results).
            # if epoch == 0 and batch == 0:
            #     global fixed_images, fixed_labels
            #     fixed_images = images
            #     fixed_labels = labels
            # else:
            #     images = fixed_images
            #     labels = fixed_labels

            if verbose >= 2:
                # Print stats about weights if an extremely small (< -1000) or large (> 1000) weight
                # was found.
                for layer in net.torch_layers:
                    if layer.weight_matrix.abs().max() > 1000:
                        logging.info('Found extreme weight:')
                        log_weight_stats(net)
                        logging.info('')
                        break

            # Forward pass.
            images, labels = images.to(device), labels.to(device)
            output, activities = net.forward(images)

            # Calculate loss and accuracy.
            loss = loss_function(output, labels)
            batch_losses.append(loss.item())
            pred_labels = output.argmax(1)
            acc = (pred_labels == labels).float().mean()
            batch_accs.append(acc.item())

            # logging.info('Batch {}: loss: {}, acc: {}'.format(batch,
            #                                                   loss.item(),
            #                                                   acc.item()))

            # Backward pass.
            if params['lr'] != 0:
                if params['learning_rule'] in ['backprop', 'feedback_alignment']:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                elif params['learning_rule'] == 'hebbian':
                    # This will only train the last layer of the network using an error.
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    # This will train all layers below the last one with a hebbian rule.
                    hebbian_rule(net, activities, lr=params['lr'])
                else:
                    raise ValueError('learning_rule not recognized:', params['learning_rule'])

        batch_losses_per_epoch.append(batch_losses)
        batch_accs_per_epoch.append(batch_accs)

        time_epoch = time.time() - start_time_epoch

        if verbose >= 1:
            if test_dataset is not None:
                start_time_test = time.time()
                test_loss, test_acc = test(net, test_dataset, params, device)
                time_test = time.time() - start_time_test
                logging.info(
                    f'Epoch {epoch + 1} / {params["num_epochs"]}: train set: '
                    f'{np.mean(batch_losses):.3f} (acc: {np.mean(batch_accs):.3f}), test set: '
                    f'{test_loss:.3f} (acc: {test_acc:.3f}) (train: {time_epoch:.1f} s, '
                    f'test: {time_test:.1f} s)')
            else:
                logging.info(
                    f'Epoch {epoch} / {params["num_epochs"]}: train set: '
                    f'{np.mean(batch_losses):.3f} (acc: {np.mean(batch_accs):.3f}) '
                    f'({time_epoch:.1f} s)')

        # If the learning rate is zero, the values won't change.
        if params['lr'] == 0:
            break

    return batch_losses_per_epoch, batch_accs_per_epoch


def log_weight_stats(net):
    """Print stats and histogram counts of weight values of existing connections."""
    weight_values = [layer.weight_matrix[layer.weight_mask.byte()] for layer in net.torch_layers]
    weight_values = np.concatenate([w.detach().cpu().numpy().flatten() for w in weight_values])

    logging.info(f'Weight average: {np.mean(weight_values):.3f} +- {np.std(weight_values):.3f} '
                 f'(min: {np.min(weight_values):.3f}, max: {np.max(weight_values):.3f})')

    # Print histogram counts from -1 to 1.
    counts, _ = np.histogram(weight_values, bins=9, range=(-1, 1))
    logging.info(f'Weight histogram (-1 to 1): {counts}')

    # Print ASCII representation of histogram.
    # width = 5
    # print('|', end='')
    # for count in counts:
    #     num_ticks = int(np.round(count / np.max(counts) * width))
    #     print('∎'*num_ticks, end='')
    #     print(' '*(width-num_ticks), end='')
    #     print('|', end='')
    # print('')


def train_and_evaluate(net, train_dataset, test_dataset, params, verbose=0, save_net=False,
                       filename=None):
    """
    Train net and return the negative loss (= reward) and accuracy, averaged 
    over the last epoch.

    The loss returned here is averaged over the last epoch while training is 
    ongoing, so it might be worse than the actual loss on the train set 
    (evaluated after training).
    If this is run with multiple workers, the weight matrices will not be 
    retained outside of this function because each worker gets a copy of the 
    network.

    Args:
        net (WeightLearningNetwork): The network to train/evaluate.
        train_dataset (tensor.Dataset): The training dataset.
        test_dataset (tensor.Dataset): The testing dataset.
        params (dict): The parameters governing the simulation.
        verbose (int, optional): Verbose mode. Same levels as learning.train, 
            but evaluate on test set as well if >= 1) (default: 0).
        save_net (bool, optional): Save the net to json file after training (default: False).
        filename (str, optional): The filename where to store the net if save_net is True
            (default: None).

    Returns:
        float: The reward. Depending on params['evaluation_period'], it 
            corresponds to either the value in the last epoch, the last batch, 
            or across the entirety of training.
        float: The accuracy. Depending on params['evaluation_period'], it 
            corresponds to either the value in the last epoch, the last batch, 
            or across the entirety of training.
    """
    # Handle device allocation.
    if params['use_cuda']:
        # Use a random cuda device with min. 20 % free memory.
        device_id = GPUtil.getAvailable(order='random', maxLoad=1,
                                        maxMemory=0.8)[0]
        device = 'cuda:{}'.format(device_id)
    else:
        device = 'cpu'

    # Train the network.
    net.create_torch_layers(device=device)
    batch_losses_per_epoch, batch_accs_per_epoch = train(
        net, train_dataset, params, device=device, verbose=verbose,
        test_dataset=test_dataset if verbose >= 1 else None)
    net.delete_torch_layers()

    # Store network as json.
    if save_net:
        net.save(filename)

    # Take negative training loss as reward
    training_loss = get_performance_value(batch_losses_per_epoch,
                                          period=params['evaluation_period'])
    training_acc = get_performance_value(batch_accs_per_epoch,
                                         period=params['evaluation_period'])

    return -training_loss, training_acc


def get_performance_value(performance_values, period='last_epoch'):
    """Extract the average performance value given a period of evaluation.

    Specifically, this function takes in a list of performance values, such as
    the list of losses or accuracies during training, and returns an average
    performance value computed in the desired evaluation period.

    Args:
        performance_values (list): The performance values across epochs and 
            batches.
        period (str): The training period where networks should be evaluated.

    Return:
        (float): The performance value averaged across the desired period
    """

    # Take negative training loss as reward.
    if period == 'last_epoch':
        avg_performance_value = np.mean(performance_values[-1])
    elif period == 'last_batch':
        avg_performance_value = performance_values[-1][-1]
    elif period == 'last_ten_batches':
        avg_performance_value = np.mean(performance_values[-1][-10:])
    elif period == 'integral':
        avg_performance_value = np.mean(performance_values)
    elif period == 'first_ten_batches':
        avg_performance_value = np.mean(performance_values[-1][:10])

    return avg_performance_value
