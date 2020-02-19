#!/usr/bin/env python3
"""
Evolve a weight-agnostic network on a classification dataset.
"""

import numpy as np
import joblib
import time
import logging
import copy
import os
import pickle
import torch

from networks import WeightAgnosticNetwork
from evolution import reproduce_tournament, rank_by_dominance
from datasets import load_preprocessed_dataset
import utils


# Set up parameters and output dir.
train_weight_values = [-2, -1, -0.5, 0.5, 1, 2]

params = utils.load_params(mode='wann')  # based on terminal input
params['script'] = 'run-wann-classification.py'
writer, out_dir = utils.init_output(params)


# Load dataset.
train_images, train_labels, test_images, test_labels = load_preprocessed_dataset(
    params['dataset'], flatten_images=True)


def evaluate(net, weight_values, images, labels, return_std=False, verbose=False):
    """
    Evaluate the fitness of the network by running it on a batch of images.

    Args:
        net (WeightAgnosticNetwork): The network to evaluate.
        weight_values (iterable of float): The weight values to evaluate on.
        images (array, shape batch_size x pixels): The batch of images to evaluate on.
        labels (array, shape batch_size): The correct labels for the images.
        return_std (bool, optional): If True, return the standard deviation across weight values
            in addition to the mean across weight values (default: False).
        verbose (bool, optional): Verbose mode (default: False).

    Returns:
        Reward (= negative cross entropy loss) and accuracy in mean (mean of all weight values) and
        max (best weight value) setting. If return_std is True, the standard deviations across
        weight values (only for mean setting) are returned as well.
    """
    # Sanity check 1: Train on zero input and dataset target.
    #images = np.zeros_like(images)

    # Sanity check 2: Train on dataset input and zero target.
    #labels = np.zeros_like(labels)

    # Sanity check 3: Overfit on a single batch.
    #images = fixed_batch_images
    #batch_labels = fixed_batch_labels

    losses = np.zeros(len(weight_values))
    accs = np.zeros(len(weight_values))

    for i, weight_value in enumerate(weight_values):
        output = net.forward(images, weight_value)
        # TODO: Maybe remove torch stuff here, we don't use it anyway for WANN.
        if params['use_torch']:
            loss = torch.nn.functional.cross_entropy(output, labels).item()
            # The output here isn't softmaxed, but it doesn't matter because we only
            # take the argmax and softmax doesn't change the maximum.
            pred_labels = output.argmax(1)
            acc = (pred_labels == labels).float().mean()
        else:
            loss = torch.nn.functional.cross_entropy(torch.from_numpy(output),
                                                     torch.from_numpy(labels)).item()
            pred_labels = output.argmax(1)
            acc = (pred_labels == labels).mean()
        losses[i] = loss
        accs[i] = acc

    mean_reward = -losses.mean()  # reward is negative cross-entropy loss
    mean_reward_std = losses.std()
    max_reward = (-losses).max()
    mean_acc = accs.mean()
    mean_acc_std = accs.std()
    max_acc = accs[np.argmax(-losses)]  # use the run with maximal reward

    if verbose:
        logging.info(f'Evaluating net ({net}): mean reward: {mean_reward:.3f} '
                     f'+- {mean_reward_std:.3f} (acc: {mean_acc:.3f} +- {mean_acc_std:.3f}), '
                     f'max reward: {max_reward:.3f} (acc: {max_acc:.3f})')

    if return_std:
        return mean_reward, max_reward, mean_acc, max_acc, mean_reward_std, mean_acc_std
    else:
        return mean_reward, max_reward, mean_acc, max_acc


# Create initial population.
population = [
    WeightAgnosticNetwork(params['num_inputs'], params['num_outputs'],
                          params['p_initial_connection_enabled'], params['p_change_activation'],
                          params['p_add_connection'], params['p_add_node'],
                          use_torch=params['use_torch'])
    for _ in range(params['population_size'])]


# Keep track of best network of all generations (= champion).
champion = {'mean_reward': -np.inf}


# Evolution loop.
with joblib.Parallel(n_jobs=params['num_workers']) as parallel:
    for generation in range(params['num_generations']):
        start_time_generation = time.time()

        # Pick a batch from MNIST. Use the same batch for all networks in this generation. This
        # makes it easier to compare different mutations and significantly improves performance.
        indices = np.random.randint(len(train_images), size=params['batch_size'])
        batch_images = train_images[indices]
        batch_labels = train_labels[indices]

        # Evaluate fitness of all networks.
        start_time_evaluation = time.time()
        objectives = parallel(joblib.delayed(evaluate)(net, train_weight_values, batch_images,
                                                       batch_labels) for net in population)

        objectives = np.array(objectives)  # shape: population_size, 4
        mean_rewards = objectives[:, 0]
        max_rewards = objectives[:, 1]
        mean_accs = objectives[:, 2]
        max_accs = objectives[:, 3]
        complexities = np.array([net.get_num_connections() for net in population])
        complexities = np.maximum(complexities, 1)  # prevent 0 division
        time_evaluation = time.time() - start_time_evaluation

        # Pick best net from this generation (based on mean reward) and check
        # if it's better than the previously observed best net (= champion).
        start_time_champion_evaluation = time.time()
        best_index = mean_rewards.argmax()
        if mean_rewards[best_index] > champion['mean_reward']:
            # Check again on entire training set to make sure it wasn't a lucky shot.
            # TODO: See how long this takes, maybe parallelize across weight values,
            #  similar to run-wann-gym.py.
            exact_results = evaluate(population[best_index], train_weight_values, train_images,
                                     train_labels, return_std=True)
            if exact_results[0] > champion['mean_reward']:
                champion = {'net': copy.deepcopy(population[best_index]),
                            'mean_reward': exact_results[0],
                            'max_reward': exact_results[1],
                            'mean_acc': exact_results[2],
                            'max_acc': exact_results[3],
                            'mean_reward_std': exact_results[4],
                            'mean_acc_std': exact_results[5]}
                # Save new champion net to file.
                with open(os.path.join(out_dir, 'champion_network.pkl'), 'wb') as f:
                    pickle.dump(champion['net'], f)
        time_champion_evaluation = time.time() - start_time_champion_evaluation

        # Write metrics to log and tensorboard.
        logging.info(
            f'{generation} - Best net: mean reward: {mean_rewards[best_index]:.3f} '
            f'(acc: {mean_accs[best_index]:.3f}), max reward: {max_rewards[best_index]:.3f} '
            f'(acc: {max_accs[best_index]:.3f}) - evaluation: {time_evaluation:.1f} s, '
            f'champion evaluation: {time_champion_evaluation:.1f} s')
        writer.add_scalar('best/mean_reward', mean_rewards[best_index], generation)
        writer.add_scalar('best/mean_acc', mean_accs[best_index], generation)
        writer.add_scalar('best/max_reward', max_rewards[best_index], generation)
        writer.add_scalar('best/max_acc', max_accs[best_index], generation)

        if generation % params['test_every'] == 0:
            # Run champion net on test set on more weight values
            if 'test_mean_reward' not in champion:

                # Test evaluation 1: Each weight used in training.
                champion['test_mean_reward'], champion['test_max_reward'], \
                    champion['test_mean_acc'], champion['test_max_acc'], \
                    champion['test_mean_reward_std'], champion['test_mean_acc_std'] = evaluate(
                    champion['net'], train_weight_values, test_images, test_labels, return_std=True)

                # Test evaluation 2: 10 linearly spaced weights.
                test_mean_reward_2, test_max_reward_2, test_mean_acc_2, test_max_acc_2, \
                    test_mean_reward_std_2, test_mean_acc_std_2 = evaluate(
                        champion['net'], np.linspace(-2, 2, 10), test_images, test_labels,
                        return_std=True)

            logging.info(f'All-time champion net on train set: '
                         f'mean reward: {champion["mean_reward"]:.3f} '
                         f'+- {champion["mean_reward_std"]:.3f} '
                         f'(acc: {champion["mean_acc"]:.3f} +- {champion["mean_acc_std"]:.3f}), '
                         f'max reward: {champion["max_reward"]:.3f} '
                         f'(acc: {champion["max_acc"]:.3f})')
            logging.info(f'                    ...on test set: '
                         f'mean reward: {champion["test_mean_reward"]:.3f} '
                         f'+- {champion["test_mean_reward_std"]:.3f} '
                         f'(acc: {champion["test_mean_acc"]:.3f} '
                         f'+- {champion["test_mean_acc_std"]:.3f}), '
                         f'max reward: {champion["test_max_reward"]:.3f} '
                         f'(acc: {champion["test_max_acc"]:.3f})')

            logging.info(f'                    ...on test set with 10 lin. spaced weight values: '
                         f'mean reward: {test_mean_reward_2:.3f} '
                         f'+- {test_mean_reward_std_2:.3f} '
                         f'(acc: {test_mean_acc_2:.3f} '
                         f'+- {test_mean_acc_std_2:.3f}), '
                         f'max reward: {test_max_reward_2:.3f} '
                         f'(acc: {test_max_acc_2:.3f})')

            writer.add_scalar('champion/mean_reward', champion['mean_reward'], generation)
            writer.add_scalar('champion/mean_acc', champion['mean_acc'], generation)
            writer.add_scalar('champion/max_reward', champion['max_reward'], generation)
            writer.add_scalar('champion/max_acc', champion['max_acc'], generation)
            writer.add_scalar('champion/test_mean_reward', champion['test_mean_reward'], generation)
            writer.add_scalar('champion/test_mean_acc', champion['test_mean_acc'], generation)
            writer.add_scalar('champion/test_max_reward', champion['test_max_reward'], generation)
            writer.add_scalar('champion/test_max_acc', champion['test_max_acc'], generation)
            writer.add_scalar('champion/connections', champion['net'].get_num_connections(),
                              generation)
            writer.add_scalar('champion/neurons', champion['net'].num_neurons, generation)
            writer.add_scalar('champion/layers', len(champion['net'].neurons_in_layer), generation)
            utils.write_networks_stats(writer, population, generation)

            utils.log_network_stats(population, writer, generation)
            logging.info('')

        # Rank networks based on the evaluation metrics.
        start_time_ranking = time.time()
        ranks = rank_by_dominance(mean_rewards, max_rewards, complexities,
                                  p_complexity_objective=params['p_complexity_objective'])
        time_ranking = time.time() - start_time_ranking

        # Make new population by picking networks via tournament selection and mutating them.
        start_time_reproduction = time.time()
        new_population = reproduce_tournament(population, ranks, params['tournament_size'],
                                              cull_ratio=params['cull_ratio'],
                                              elite_ratio=params['elite_ratio'],
                                              num_mutations=params['num_mutations_per_generation'])
        population = new_population
        time_reproduction = time.time() - start_time_reproduction

        time_generation = time.time() - start_time_generation
        writer.add_scalar('times/complete_generation', time_generation, generation)
        writer.add_scalar('times/evaluation', time_evaluation, generation)
        writer.add_scalar('times/champion_evaluation', time_champion_evaluation, generation)
        writer.add_scalar('times/ranking', time_ranking, generation)
        writer.add_scalar('times/reproduction', time_reproduction, generation)

writer.close()
