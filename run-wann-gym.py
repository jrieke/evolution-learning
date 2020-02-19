#!/usr/bin/env python3
"""
Evolve a weight-agnostic network on a gym environment.
"""

import numpy as np
import joblib
import time
import logging
import copy
import os
import pickle

from networks import WeightAgnosticNetwork
from evolution import reproduce_tournament, rank_by_dominance
from datasets import make_env
import utils


# Set up parameters and output dir.
train_weight_values = [-2, -1, -0.5, 0.5, 1, 2]

params = utils.load_params(mode='wann')  # based on terminal input
params['script'] = 'run-wann-gym.py'
writer, out_dir = utils.init_output(params)


# Define evaluate function used during evolution.
def evaluate(net, weight_values, num_trials, env_seed=None, return_std=False,
             verbose=False, render=False):
    """
    Evaluate the fitness of the network by running it on a gym environment
    for multiple trials.

    Args:
        net (WeightAgnosticNetwork): The network to evaluate.
        weight_values (iterable of float): The weight values to evaluate on.
        num_trials (int): How often to run on the environment per weight value.
        env_seed (int, optional): The base seed for the gym environment. If given, each trial will
            still use a different seed, but synced across weight values. I.e. the first trials for
            all weight values will use the same seed, the second trials will use another same seed,
            etc. If None, the environments aren't seeded (default: None).
        return_std (bool, optional): If True, return the standard deviations across trials
            in addition to the averages across trials (default: False).
        verbose (bool, optional): Verbose mode (default: False).
        render (bool, optional): Render the gym environments (default: False).

    Returns:
        Cumulative reward in mean (mean of all weight values) and max (best weight value) setting,
        averaged over trials. If return_std is True, the standard deviations across trials (again
        in mean and max setting) are returned as well.
    """
    env = make_env(params['env_name'])

    # For each weight value, run multiple trials and record all (cumulative) rewards.
    rewards = np.zeros((len(weight_values), num_trials))
    for i, weight_value in enumerate(weight_values):
        for trial in range(num_trials):
            if env_seed is not None:
                env.seed(env_seed + trial)  # different for trials but synced across weight values
            observation = env.reset()
            done = False
            while not done:
                if render:
                    env.render()
                output = net.forward(observation.reshape(1, -1),
                                     weight_value)  # add (empty) batch dimension
                output = output.reshape(-1)  # get rid of batch dimension again
                if params['take_argmax_action']:
                    action = np.argmax(output)
                else:
                    action = output
                observation, reward, done, info = env.step(action)
                rewards[i, trial] += reward
                if done:
                    env.close()

    #print(rewards)

    # Average rewards over trials.
    reward_per_weight_value = rewards.mean(axis=1)
    reward_per_weight_value_std = rewards.std(axis=1)

    # Mean setting: Average over all weight values.
    mean_reward = reward_per_weight_value.mean()
    mean_reward_std = reward_per_weight_value_std.mean()

    # Max setting: Use best performing weight value.
    max_index = reward_per_weight_value.argmax()
    max_reward = reward_per_weight_value[max_index]
    max_reward_std = reward_per_weight_value_std[max_index]

    if verbose:
        logging.info(
            f'Evaluating net ({net}): mean reward: {mean_reward:.2f} +- {mean_reward_std:.2f}, '
            f'max reward: {max_reward:.2f} +- {max_reward_std:.2f}')

    if return_std:
        return mean_reward, max_reward, mean_reward_std, max_reward_std
    else:
        return mean_reward, max_reward


# Create initial population.
population = [
    WeightAgnosticNetwork(params['num_inputs'], params['num_outputs'],
                          params['p_initial_connection_enabled'], params['p_change_activation'],
                          params['p_add_connection'], params['p_add_node'])
    for _ in range(params['population_size'])]


# Keep track of best network of all generations (= champion).
champion = {'mean_reward': -np.inf}


# Evolution loop.
with joblib.Parallel(n_jobs=params['num_workers']) as parallel:
    for generation in range(params['num_generations']):
        start_time_generation = time.time()

        # Evaluate fitness of all networks. Use the same environment seed for each network. This
        # makes it easier to compare different mutations and significantly improves performance.
        start_time_evaluation = time.time()
        objectives = parallel(
            joblib.delayed(evaluate)(net, train_weight_values, params['num_trials'],
                                     env_seed=np.random.randint(1000))
            for net in population)

        objectives = np.array(objectives)  # shape: population_size, 2
        mean_rewards = objectives[:, 0]
        max_rewards = objectives[:, 1]
        complexities = np.array([net.get_num_connections() for net in population])
        complexities = np.maximum(complexities, 1)  # prevent 0 division
        time_evaluation = time.time() - start_time_evaluation

        # Pick best net from this generation (based on mean reward) and check
        # if it's better than the previously observed best net (= champion).
        start_time_champion_evaluation = time.time()
        best_index = mean_rewards.argmax()
        if mean_rewards[best_index] > champion['mean_reward']:
            # Check again by running for 96 trials (16 per weight value) to make sure it wasn't a
            # lucky shot. This is a hacky implementation to parallelize across weight values.
            # TODO: Use seed here or not?
            # TODO: Think about making this nicer.
            exact_rewards = parallel(
                joblib.delayed(evaluate)(population[best_index], [weight_value], 16,
                                         return_std=True) for weight_value in train_weight_values)
            exact_rewards = np.array(exact_rewards)
            exact_mean_reward = exact_rewards[:, 0].mean()

            # if exact_mean_reward > champion_mean_reward:
            if exact_mean_reward > champion['mean_reward']:
                max_index = exact_rewards[:, 0].argmax()
                champion = {'net': copy.deepcopy(population[best_index]),
                            'mean_reward': exact_mean_reward,
                            'mean_reward_std': exact_rewards[:, 2].mean(),
                            'max_reward': exact_rewards[max_index, 0],
                            'max_reward_std': exact_rewards[max_index, 2]}
                # Save new champion net to file.
                with open(os.path.join(out_dir, 'champion_network.pkl'), 'wb') as f:
                    pickle.dump(champion['net'], f)
        time_champion_evaluation = time.time() - start_time_champion_evaluation

        # Write metrics to log and tensorboard.
        logging.info(
            f'{generation} - Best net: mean reward: {mean_rewards[best_index]:.2f}, '
            f'max reward: {max_rewards[best_index]:.2f} - evaluation: {time_evaluation:.1f} s, '
            f'champion evaluation: {time_champion_evaluation:.1f} s')
        writer.add_scalar('best/mean_reward', mean_rewards[best_index], generation)
        writer.add_scalar('best/max_reward', max_rewards[best_index], generation)

        if generation % params['test_every'] == 0:
            # Run champion net in "test" setting, i.e. on more weight values and trials.
            if 'test_mean_reward' not in champion:
                # Test evaluation: Each weight used in training, 100 trials per weight value.
                champion['test_mean_reward'], champion['test_max_reward'], \
                    champion['test_mean_reward_std'], champion['test_max_reward_std'] = evaluate(
                    champion['net'], train_weight_values, 100, return_std=True)

            logging.info(f'All-time champion net in train setting: '
                         f'mean reward: {champion["mean_reward"]:.2f} '
                         f'+- {champion["mean_reward_std"]:.2f}, '
                         f'max reward: {champion["max_reward"]:.2f} '
                         f'+- {champion["max_reward_std"]:.2f}')
            logging.info(f'       ...in test setting (100 trials): '
                         f'mean reward: {champion["test_mean_reward"]:.2f} '
                         f'+- {champion["test_mean_reward_std"]:.2f}, '
                         f'max reward: {champion["test_max_reward"]:.2f} '
                         f'+- {champion["test_max_reward_std"]:.2f}')

            writer.add_scalar('champion/mean_reward', champion['mean_reward'], generation)
            writer.add_scalar('champion/max_reward', champion['max_reward'], generation)
            writer.add_scalar('champion/test_mean_reward', champion['test_mean_reward'], generation)
            writer.add_scalar('champion/test_max_reward', champion['test_max_reward'], generation)
            writer.add_scalar('champion/connections', champion['net'].get_num_connections(),
                              generation)
            writer.add_scalar('champion/neurons', champion['net'].num_neurons, generation)
            writer.add_scalar('champion/layers', len(champion['net'].neurons_per_layer), generation)

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
