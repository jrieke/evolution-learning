#!/usr/bin/env python3
"""
Evolve network architecture on a classification dataset, while at the same time training the weights
with one of several learning algorithms.
"""
import joblib
import time
import torch.utils.data
import logging
import numpy as np
import copy
import os
import pickle

from networks import WeightLearningNetwork
from evolution import rank_by_dominance, reproduce_tournament
from datasets import load_preprocessed_dataset
from learning import train, test, train_and_evaluate, get_performance_value
import utils


# Set up parameters and output dir.
params = utils.load_params(mode='wlnn')  # based on terminal input
params['script'] = 'run-wlnn-mnist.py'
writer, out_dir = utils.init_output(params, overwrite=params['overwrite_output'])
os.makedirs(os.path.join(out_dir, 'networks'))  # dir to store all networks

if params['use_cuda'] and not torch.cuda.is_available():
    logging.info('use_cuda was set but cuda is not available, running on cpu')
    params['use_cuda'] = False
device = 'cuda' if params['use_cuda'] else 'cpu'


# Ensure deterministic computation.
utils.seed_all(0)

### Ensure that runs are reproducible even on GPU. Note, this slows down training!
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Load dataset.
train_images, train_labels, test_images, test_labels = load_preprocessed_dataset(
    params['dataset'], flatten_images=True, use_torch=True)
train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

# Create initial population.
# TODO: Make train_only_outputs a learning_rule.
train_only_outputs = (params['train_only_outputs'] or params['learning_rule'] == 'hebbian')
use_random_feedback = (params['learning_rule'] == 'feedback_alignment')
population = [
    WeightLearningNetwork(params['num_inputs'], params['num_outputs'],
                          params['p_initial_connection_enabled'],
                          p_add_connection=params['p_add_connection'],
                          p_add_node=params['p_add_node'],
                          inherit_weights=params['inherit_weights'],
                          train_only_outputs=train_only_outputs,
                          use_random_feedback=use_random_feedback,
                          add_only_hidden_connections=True)
    for _ in range(params['population_size'])]

# Add some nodes manually at the beginning.
for net in population:
    for _ in range(net.get_num_connections()):
        if np.random.rand() < 0.5:
            net.add_node()

# Evaluate the networks before doing any evolution or learning.
for net in population:
    net.create_torch_layers(device=device)
with joblib.Parallel(n_jobs=params['num_workers']) as parallel:
    # Select champion based on training set for consistency with evolution loop.
    objectives = parallel(joblib.delayed(test)(net, \
        train_dataset, params, device=device) for net in population)
    objectives = np.array(objectives)
    rewards = -objectives[:, 0]
    accs = objectives[:, 1]
    best_index = rewards.argmax()
    champion = {'net': copy.deepcopy(population[best_index]),
                'reward': rewards[best_index],
                'acc': accs[best_index],
                'connections': population[best_index].get_num_connections()}
logging.info(f'Pre-evolution and training champion net on test set: '
             f'reward: {champion["reward"]:.3f} '
             f'(acc: {champion["acc"]:.3f})')
for net in population:
    net.delete_torch_layers()

# Store the current champion network.
champion['net'].delete_torch_layers()
champion['net'].save(os.path.join(out_dir, 'champion_network.json'))

# Evolution loop.
generation = -1 # necessary for logging info when there are 0 generations
with joblib.Parallel(n_jobs=params['num_workers']) as parallel:
    for generation in range(params['num_generations']):
        start_time_generation = time.time()

        # Evaluate fitness of all networks.
        start_time_evaluation = time.time()
        objectives = parallel(joblib.delayed(train_and_evaluate)(
            net, train_dataset, test_dataset, params, verbose=0, save_net=(generation % 100 == 0),
            filename=os.path.join(out_dir, 'networks', f'generation{generation}-net{i}.json'))
                              for i, net in enumerate(population))
        objectives = np.array(objectives)  # shape: population_size, 2
        rewards = objectives[:, 0]
        accs = objectives[:, 1]
        complexities = np.array([net.get_num_connections() for net in population])
        complexities = np.maximum(complexities, 1)  # prevent 0 division
        time_evaluation = time.time() - start_time_evaluation

        # Pick best net from this generation (based on reward) and check
        # if it's better than the previously observed best net (= champion).
        start_time_champion_evaluation = time.time()
        best_index = rewards.argmax()
        if rewards[best_index] > champion['reward']:
            # In contrast to run-wann-mnist.py, we don't have to check on the 
            # entire training set because the network was already evaluated on 
            # the complete set.
            # TODO: Maybe train champion net on more epochs already here (it's 
            # done below right now) and compare against results of previous 
            # champion net. This would take quite a bit of time though because 
            # I probably need to do it at almost every generation.
            champion = {'net': copy.deepcopy(population[best_index]),
                        'reward': rewards[best_index],
                        'acc': accs[best_index],
                        'connections': population[best_index].get_num_connections()}
            # Save new champion net to file. Note that this net doesn't have weight_matrices when
            # using multiple workers (weight_matrices is only created within the worker process).
            champion['net'].delete_torch_layers()
            champion['net'].save(os.path.join(out_dir, 'champion_network.json'))
        time_champion_evaluation = time.time() - start_time_champion_evaluation

        # Write metrics to log and tensorboard.
        logging.info(f'{generation} - Best net: reward: {rewards[best_index]:.3f} '
             f'(acc: {accs[best_index]:.3f}) - evaluation: {time_evaluation:.1f} s, '
             f'champion evaluation: {time_champion_evaluation:.1f} s')
        writer.add_scalar('best/reward', rewards[best_index], generation)
        writer.add_scalar('best/acc', accs[best_index], generation)

        if generation % 20 == 0:
            if 'long_training_reward' not in champion:

                # Train champion net for more epochs.
                # TODO: Do this more elegantly. Maybe make an additional 
                # parameter num_epochs_long.
                long_params = params.copy()
                long_params['num_epochs'] = 10
                champion['net'].create_torch_layers(device)
                loss, acc = train(champion['net'], train_dataset, long_params, device=device)
                champion['long_training_reward'] = - get_performance_value(loss, period='last_epoch')
                champion['long_training_acc'] = get_performance_value(acc,  period='last_epoch')

                # Evaluate this long trained net on test set.
                loss, acc = test(champion['net'], test_dataset, params, device=device)
                champion['test_reward'] = -loss
                champion['test_acc'] = acc

                # Manually delete weight matrices, so they don't block memory 
                # (important on cuda).
                champion['net'].delete_torch_layers()

            utils.log_champion_info(champion)
            utils.write_champion_info(writer, champion, generation)
            utils.write_networks_stats(writer, population, generation)



            utils.log_network_stats(population, writer, generation)
            logging.info('')

        # TODO: Is this necessary?
        #writer.add_histogram('final_acc', accs, generation)
        writer.add_histogram('population/acc', accs, generation)
        writer.add_histogram('population/connections', [net.get_num_connections() for net
                                                        in population], generation)

        # Store all accuracies and connections (for learning rate plots).
        for i, (net, acc) in enumerate(zip(population, accs)):
            writer.add_scalar(f'population/net{i}_acc', acc, generation)
            writer.add_scalar(f'population/net{i}_connections', net.get_num_connections(), generation)

        # Rank networks based on the evaluation metrics.
        start_time_ranking = time.time()
        # TODO: This is a dirty hack, I am using rewards for both mean_rewards 
        # and max_rewards for now. Think about how to make this better. Also, 
        # should maybe adapt parameters of how often complexity is used vs. 
        # reward.
        ranks = rank_by_dominance(rewards, rewards, complexities,
                                  p_complexity_objective=params['p_complexity_objective'])
        time_ranking = time.time() - start_time_ranking

        # Make new population by picking parent networks via tournament
        # selection and mutating them.
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

# Log final results and close writer.
logging.info('\nResults at the end of evolution:')
utils.log_champion_info(champion)
utils.write_networks_stats(writer, population, generation)
utils.log_network_stats(population, writer, generation)
writer.close()

# Store performance summary.
utils.store_performance(objectives, out_dir=params['out_dir'])
