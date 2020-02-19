import numpy as np
import torch
import logging
import sys
import os
from datetime import datetime
import copy
import argparse
import tensorboardX
from tensorboardX import SummaryWriter
import shutil
import pickle
import pandas as pd
import joblib
import yaml
import random


def seed_all(seed=None):
    """Set seed for numpy, random, torch."""
    np.random.seed(seed)
    random.seed(seed)

    # This will seed both cpu and cuda.
    if seed is None:
        torch.seed()
    else:
        torch.manual_seed(seed)


def timestamp():
    """Return a string timestamp."""
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def init_logging(to_file=False, filename=None):
    """
    Initialize the logging module to print to stdout and (optionally) a file.

    Call this only once at the beginning of your script.
    To use the logging module, do `import logging` at the top of your file and use `logging.info`
    instead of `print.
    Note: Doesn't work properly inside parallel processes spawned via joblib.

    Args:
        to_file (bool, optional): Whether to write to a file in addition to the console
            (default: False).
        filename (str, optional): The filename to store the log to. If None, use the current
            timestamp.log (default: None).
    """
    if to_file:
        if filename is None:
            filename = timestamp() + '.log'
        logging.basicConfig(level=logging.INFO, format='%(message)s', filename=filename)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # write to stdout + file
        print('Logging to:', filename)
    else:
        logging.basicConfig(level=logging.INFO, format='%(message)s')


def requires_grad(x):
    """Recursively check whether an object, of any type, requires grad 
    computation for its elements or not.
    Args:
        x: object of any type
    Returns:
        boolean inidicating whether the elements of x require grad"""
    if isinstance(x, torch.Tensor):
        return x.requires_grad
    elif isinstance(x, list) and not (x == []):
        return requires_grad(x[0])
    else:
        return False


def state_requires_grad(x):
    """Recursively turn requires_grad to True to the elements of an object x.
    Args:
        x: object of any type
    Returns:
        x: object where the elements require grad"""
    if isinstance(x, torch.Tensor):
        x.requires_grad = True
    elif isinstance(x, list):
        for i in range(len(x)):
            x[i] = state_requires_grad(x[i])
    return x


def detach(x):
    """Recursively detach the grads of elements in x.
    Args:
        x: object of any type
    Returns:
        x_detach: object where all elements have been detached"""
    if isinstance(x, torch.Tensor):
        return x.detach()
    elif isinstance(x, list):
        for i in range(len(x)):
            x[i] = detach(x[i])
        return x


def deepcopy(item):
    """Deepcopy function that can deal with classes with attributes that 
    require_grad. It detaches those variables from its grad, and then sets again
    requires_grad to true
    Args:
        item: object with arbitrary attributes
    Returnd:
        item: equivalent object with all of its gradients detached"""

    try:
        return copy.deepcopy(item)
    except:
        # Detach grad when necessary
        key_requires_grad = []
        for key, value in zip(item.__dict__.keys(), item.__dict__.values()):
            if requires_grad(value):
                value_detached = detach(value)
                setattr(item, key, value_detached)
                key_requires_grad.append(key)

        # Set requires_grad to True when necessary
        item_copy = copy.deepcopy(item)
        for key in key_requires_grad:
            value = getattr(item_copy, key)
            setattr(item_copy, key, state_requires_grad(value))
        return item_copy


def str2bool(v):
    """
    Convert str to bool for use in boolean argparse options.

    From: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_params_file(filename):
    """Load parameters from yaml file and return as dictionary."""
    with open(filename, 'r') as f:
        params = yaml.safe_load(f)
    return params


def load_params(mode):
    """Load parameters from params file and terminal input. 

    Args:
        mode (str): The mode of the experiment; either 'wann' or 'wlnn'.

    """

    # Set up argparse.
    parser = argparse.ArgumentParser()

    # Evolution options.
    egroup = parser.add_argument_group('evolution options')
    egroup.add_argument('--cull_ratio', type=float,
                        help='Fraction of worst networks to leave out of the breeding pool')
    egroup.add_argument('--elite_ratio', type=float,
                        help='Fraction of best networks to pass on to the new population directly')
    egroup.add_argument('--num_generations', type=int, help='Number of generations for evolution')
    egroup.add_argument('--population_size', type=int, help='Number of networks in the population')
    egroup.add_argument('--p_add_connection', type=float,
                        help='Probability of adding a connection during mutation')
    egroup.add_argument('--p_add_node', type=float,
                        help='Probability of adding a node during mutation')
    if mode == 'wann':
        egroup.add_argument('--p_change_activation', type=float,
                            help='Probability of changing the activation of a node during mutation')
    egroup.add_argument('--tournament_size', type=int,
                        help='Number of networks that compete during tournament selection')
    egroup.add_argument('--inherit_weights', type=str2bool,
                        help='Keep weights after mutating a network')
    egroup.add_argument('--num_mutations_per_generation', type=int,
                       help='The number of mutations to carry out at each generation')
    egroup.add_argument('--p_complexity_objective', type=float,
                        help='The fraction of generations to rank according to complexity and '
                             'reward (otherwise only reward)')

    # Evaluation options.
    vgroup = parser.add_argument_group('evaluation options')
    vgroup.add_argument('--batch_size', type=int,
                        help='Batch size for evaluation (in run-wann-classification.py) or '
                             'learning (in run-wlnn-mnist-no-evolution.py and run-wlnn-mnist.py)')
    vgroup.add_argument('--batch_size_eval', type=int,
                        help='Batch size for test set evaluation of learned networks')
    if mode == 'wlnn':
        vgroup.add_argument('--evaluation_period', type=str, choices=['integral', 'last_batch', \
                                 'last_ten_batches', 'first_ten_batches', 'last_epoch'],
                            default='last_epoch',
                            help='Which training period should be used to evaluate networks. The ' +
                                 'options are: "integral" the mean across the whole training ' +
                                 'duration, "last_batch" the values in the final batch, ' +
                                 '"last_ten_batches" the values in the last ten batches, ' +
                                 '"first_ten_batches" the values in the first ten batches and ' +
                                 '"last_epoch" the values averaged in the last epoch only.')
    vgroup.add_argument('--num_trials', type=int,
                        help='How often to run the gym environment during evaluation')

    # Training options.
    if mode == 'wlnn':
        tgroup = parser.add_argument_group('training options')
        tgroup.add_argument('--learning_rule', type=str,
                            help='Learning rule to train network')
        tgroup.add_argument('--lr', type=float, help='Learning rate')
        tgroup.add_argument('--num_epochs', type=int, help='Number of epochs for learning')
        tgroup.add_argument('--optimizer', type=str,
                            help='Optimizer to train network (sgd, adam or adadelta)')
        tgroup.add_argument('--train_only_outputs', action='store_true',
                            help='If this option is selected, only the weights to the output ' +
                                 'units will be learned. Else, all weights will be learned.')

    # Architecture options.
    agroup = parser.add_argument_group('architecture options')
    agroup.add_argument('--num_inputs', type=int, help='Number of input neurons')
    agroup.add_argument('--num_outputs', type=int, help='Number of output neurons')
    agroup.add_argument('--p_initial_connection_enabled', type=float,
                        help='Probability of enabling a connection between input and output layer '
                             'at the start of evolution')

    # Task and dataset options.
    dgroup = parser.add_argument_group('task and dataset options')
    dgroup.add_argument('params_file', type=str,
                        help='A yaml file with parameters (see folder params for examples)')
    dgroup.add_argument('--dataset', type=str, help='Dataset for classification (digits or mnist)')
    dgroup.add_argument('--env_name', type=str, help='Name of the gym environment')

    # Computational options.
    cgroup = parser.add_argument_group('computational options')
    cgroup.add_argument('--num_workers', type=int, help='Number of workers to run on')
    cgroup.add_argument('--take_argmax_action', type=str2bool,
                        help='Use argmax of the network output as the action in the environment')
    cgroup.add_argument('--use_cuda', type=str2bool, help='Use cuda devices if available')
    cgroup.add_argument('--use_torch', type=str2bool, help='Use torch instead of numpy')

    # Miscellaneous options.
    mgroup = parser.add_argument_group('miscellaneous options')
    mgroup.add_argument('--out_dir', type=str, default='',
                        help='The path to the output directory')
    mgroup.add_argument('--overwrite_output', action='store_true',
                        help='Overwrite data in the output folder if it already exists.')

    
    args = parser.parse_args()

    # Read params from yaml file.
    params = load_params_file(args.params_file)

    # Update with any direct input from the terminal.
    params_args = vars(args)
    del params_args['params_file']
    for key, value in params_args.items():
        if value is not None:
            params[key] = value

    # If no out_dir path is provided through the command line arguments,
    # generate a path based on the current time.
    if params['out_dir'] == '':
        params['out_dir'] = create_out_dir_name(params)

    return params


def create_out_dir_name(params):
    """
    Create output directory name for the experiment based on the current date 
    and time.

    Args:
        params (dict): The parameters of the experiment.

    Returns:
        str: The path to the output directory.
    """

    current_timestamp = timestamp()
    out_dir = os.path.join('out', current_timestamp)
    return out_dir


def init_output(params, script_name=None, overwrite=False):
    """
    Initialize all output stuff for an experiment, namely:

    - create output folder based on the current date and time
    - store the params in this output folder
    - set up a log in this output folder (can be used via logging.info from 
        everywhere)
    - set up a tensorboard writer in this output folder

    Args:
        params (dict): The parameters of the experiment.
        script_name (str): The name of the script, will be written to the log. 
            If None, this will be inferred from terminal input (default: None).
        overwrite (bool, optional): If True, overwrite the output dir. If False, 
            raise an exception if the output dir exists (default: False).

    Returns:
        tensorboardX.SummaryWriter: The tensorboard writer.
        str: The path to the output directory.
    """

    out_dir = params['out_dir']

    # Make output folder.
    if os.path.exists(out_dir):
        if overwrite:
            shutil.rmtree(out_dir)
            print('Output dir exists, deleting it (overwrite is set to True):', out_dir)
        else:
            raise IOError('Output dir already exists, set overwrite=True to overwrite:', out_dir)
    os.makedirs(out_dir)
    print('Created output dir:', out_dir)

    # Save params to out dir as pickle and yaml.
    # TODO: Do we use the pickled params somewhere or should we only store yaml?
    with open(os.path.join(out_dir, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)
    with open(os.path.join(out_dir, 'params.yaml'), 'w') as f:
        yaml.dump(params, f, default_flow_style=False)

    # Initialize tensorboard summary writer.
    if not hasattr(tensorboardX, '__version__'):
        writer = SummaryWriter(log_dir=os.path.join(out_dir, 'tensorboard_summary'))
    else:
        writer = SummaryWriter(logdir=os.path.join(out_dir, 'tensorboard_summary'))

    # Initialize logging.
    init_logging(to_file=True, filename=os.path.join(out_dir, os.path.basename(out_dir) + '.log'))

    # Write header to log (including script name and parameters).
    if script_name is None:
        script_name = os.path.basename(sys.argv[0])
    logging.info(f'Running {script_name}')
    logging.info(f'{joblib.cpu_count()} cpu core(s) and '
                 f'{torch.cuda.device_count()} cuda devices available')
    logging.info('-' * 80)
    logging.info('Parameters:')
    for key, value in params.items():
        logging.info(f'{key}: {value}')
    logging.info('-' * 80)

    return writer, out_dir


def store_performance(results, out_dir='', name='results_summary'):
    """Store a summary of the performance for the current run in a .csv file.
    Args:
        results: np.array of dimension (population_size, 4), where each column
            contains, for each network in the population, the final:
            * mean_rewards
            * max_rewards
            * mean_accuracies
            * max_accuracies
            across all different random seeds
    """

    results_file = os.path.join(out_dir, name + '.csv')

    results_summary = {
        'pop_mean_accuracies': ['%.2f' % (100 * np.mean(results[:, 1]))],
        'pop_max_accuracies': ['%.2f' % (100 * np.max(results[:, 1]))],
        'pop_mean_rewards': [np.mean(results[:, 0])],
        'pop_max_rewards': [np.max(results[:, 0])],
    }

    df = pd.DataFrame.from_dict(results_summary)

    if os.path.isfile(results_file):
        old_df = pd.read_csv(results_file, sep=';')
        df = pd.concat([old_df, df], sort=True)
    df.to_csv(results_file, sep=';', index=False)


def log_network_stats(population, writer=None, iteration=None):
    """
    Write statistics about networks in the population (neurons, connections, 
    layers) to logging.info and tensorboard.

    Args:
        population (iterable): The population of networks.
        writer (tensorboardX.SummaryWriter, optional): The tensorboard writer 
            (default: None).
        iteration (int, optional): Store the values for this iteration in 
            tensorboard (default: None).
    """
    num_connections = [net.get_num_connections() for net in population]
    num_neurons = [net.num_neurons for net in population]
    num_layers = [len(net.neurons_in_layer) for net in population]

    logging.info(
        f'Connections: {np.mean(num_connections):.0f} +- {np.std(num_connections):.0f} '
        f'({np.min(num_connections)}-{np.max(num_connections)}), '
        f'Neurons: {np.mean(num_neurons):.0f} +- {np.std(num_neurons):.0f} '
        f'({np.min(num_neurons)}-{np.max(num_neurons)}), '
        f'Layers: {np.mean(num_layers):.0f} +- {np.std(num_layers):.0f} '
        f'({np.min(num_layers)}-{np.max(num_layers)})')


def write_networks_stats(writer, population, generation):
    """Add stats about the networks to the tensorboard writer.

    Args:
        writer: The tensorboard writer.
        population (list): The networks in the population.
        generation (int): The current generation.

    """

    g = generation
    num_connections = [net.get_num_connections() for net in population]
    num_neurons = [net.num_neurons for net in population]
    num_layers = [len(net.neurons_in_layer) for net in population]

    writer.add_scalar('stats/connections_mean', np.mean(num_connections), g)
    writer.add_scalar('stats/connections_std', np.std(num_connections), g)
    writer.add_scalar('stats/neurons_mean', np.mean(num_neurons), g)
    writer.add_scalar('stats/neurons_std', np.std(num_neurons), g)
    writer.add_scalar('stats/layers_mean', np.mean(num_layers), g)
    writer.add_scalar('stats/layers_std', np.std(num_layers), g)


def write_champion_info(writer, champion, generation):
    """Add evolution data to the tensorboard writer.

    Args:
        writer: The tensorboard writer.
        champion (dict): The data about the champion network.
        generation (int): The current generation.

    """

    writer.add_scalar('champion/reward', champion['reward'], generation)
    writer.add_scalar('champion/acc', champion['acc'], generation)
    writer.add_scalar('champion/long_training_reward', champion['long_training_reward'], generation)
    writer.add_scalar('champion/long_training_acc', champion['long_training_acc'], generation)
    writer.add_scalar('champion/test_reward', champion['test_reward'], generation)
    writer.add_scalar('champion/test_acc', champion['test_acc'], generation)
    writer.add_scalar('champion/connections', champion['connections'], generation)


def log_champion_info(champion):
    """Log the info about the champion in the current generation.

    Args:        
        champion (dict): The data about the champion network.
    """

    logging.info(f'All-time champion net on train set (short training): '
                 f'reward: {champion["reward"]:.3f} '
                 f'(acc: {champion["acc"]:.3f})')
    if 'long_training_reward' in champion:
        logging.info(f'                   ...on train set (long training): '
                     f'reward: {champion["long_training_reward"]:.3f} '
                     f'(acc: {champion["long_training_acc"]:.3f})')
    if 'test_reward' in champion:
        logging.info(f'                   ...on test set (long training): '
                     f'reward: {champion["test_reward"]:.3f} '
                     f'(acc: {champion["test_acc"]:.3f})')
